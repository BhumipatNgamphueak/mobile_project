#!/usr/bin/env python3
"""
path_planning_node.py
=====================
TEB-style local planner — ROS plumbing layer.

Owns the ROS interface (subs, pubs, timer) and converts the local costmap
into polygon obstacles. Delegates the elastic-band optimisation to
`path_planning.elastic_band.ElasticBandPlanner`.

Pipeline (one cycle):
  1. Costmap -> polygon obstacles
       BFS-cluster lethal cells in a window around the robot, then run
       MCCH (multi-convex-hull) decomposition on each cluster. MCCH
       splits concave shapes (L, U, T) into multiple convex hulls so the
       hull's diagonal edges don't fill in real free space — fixes the
       "robot pinned inside fake polygon at L-shape inner corner" case.
       (nav2 analog: costmap_converter::CostmapToPolygonsDBSMCCH)
  2. Local segment extraction
       Prune behind the robot, take a lookahead window of the global path.
  3. Elastic-band optimisation
       ElasticBandPlanner.optimize(robot_xy, segment, polygons).
  4. Holonomic velocity extraction
       Vector pursuit on the optimised band -> (vx, vy, omega=0).

Inputs
------
/odom           nav_msgs/Odometry       robot pose + velocity
/global_path    nav_msgs/Path           reference path from global planner
/local_costmap  nav_msgs/OccupancyGrid  obstacle grid in odom frame

Outputs
-------
/cmd_vel             geometry_msgs/TwistStamped       velocity command
/planned_path        nav_msgs/Path                    optimised band (RViz)
/obstacle_polygons   visualization_msgs/MarkerArray   extracted polygons (RViz)
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from path_planning.elastic_band import (
    ElasticBandConfig,
    ElasticBandPlanner,
    mcch_decompose,
)


def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── General parameters ────────────────────────────────────────────
        self.planning_freq          = 10     # Hz
        self.lookahead_dist         = 15.0   # m — local segment length
        self.min_waypoints_required = 2
        self.goal_tolerance         = 0.4    # m
        self.v_max                  = 1.5    # m/s

        # ── Elastic-band planner ──────────────────────────────────────────
        self.eb = ElasticBandPlanner(ElasticBandConfig(
            n_iter    = 60,
            w_obs     = 50.0,
            w_smooth  = 2.0,
            inflation = 1.0,
            step      = 0.05,
            max_delta = 0.10,
            skip_dist = 0.4,
        ))

        # ── Costmap-to-polygons parameters ────────────────────────────────
        self.obstacle_threshold = 60     # cost >= this -> obstacle cell
        self.cluster_min_cells  = 3      # filter noise clusters
        self.search_margin      = 8.0    # m beyond lookahead to scan
        self.mcch_split_thresh  = 0.3    # m — MCCH fictitious-edge threshold

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,       '/odom',          self._odom_callback,          10)
        self.create_subscription(Path,           '/global_path',   self._global_path_callback,   10)
        self.create_subscription(OccupancyGrid,  '/local_costmap', self._local_costmap_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel',           10)
        self.path_pub    = self.create_publisher(Path,         '/planned_path',      10)
        self.polygon_pub = self.create_publisher(MarkerArray,  '/obstacle_polygons', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose: Pose | None             = None
        self._robot_vel                           = None
        self._global_path: Path | None            = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int                    = 0
        self._polygons: list[list[tuple[float, float]]] = []

        self.create_timer(1.0 / self.planning_freq, self._replan)
        self.get_logger().info('PathPlanningNode (TEB polygon-style) started')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose
        self._robot_vel  = msg.twist.twist

    def _local_costmap_callback(self, msg: OccupancyGrid):
        self._local_costmap = msg

    def _global_path_callback(self, msg: Path):
        if self._global_path is not None and len(msg.poses) > 0:
            if (msg.header.stamp.sec == self._global_path.header.stamp.sec and
                msg.header.stamp.nanosec == self._global_path.header.stamp.nanosec):
                return

            new_goal = msg.poses[-1].pose.position
            old_goal = self._global_path.poses[-1].pose.position
            same_goal = (abs(new_goal.x - old_goal.x) < 0.01 and
                         abs(new_goal.y - old_goal.y) < 0.01)
            if same_goal:
                # Geometry changed but goal is the same — keep warm state.
                self._global_path = msg
                self._prune_index = 0
                return

        self._global_path = msg
        self._prune_index = 0
        self.eb.reset_warm_start()
        self.get_logger().info(f'New goal received ({len(msg.poses)} waypoints)')

    # ──────────────────────────────────────────────────────────────────────
    # Planning loop
    # ──────────────────────────────────────────────────────────────────────

    def _replan(self):
        if self._robot_pose is None or self._global_path is None or self._local_costmap is None:
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            return

        if euclidean(self._robot_pose, self._global_path.poses[-1].pose) < self.goal_tolerance:
            self._pub_stop()
            return

        # 1. Costmap -> polygon obstacles (BFS clustering + MCCH split).
        self._polygons = self._costmap_to_polygons()
        self._publish_polygons()

        # 2. Reference segment from global path.
        self._prune_index = self._find_closest_index()
        segment = self._extract_local_segment()
        if len(segment) < self.min_waypoints_required:
            return
        segment_xys = [(p.pose.position.x, p.pose.position.y) for p in segment]

        # 3. Elastic-band optimisation.
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        opt_xs, opt_ys = self.eb.optimize((rx, ry), segment_xys, self._polygons)

        # 4. Publish optimised path & velocity.
        self._publish_path(opt_xs, opt_ys, segment[0].header.frame_id)
        self.cmd_vel_pub.publish(self._compute_cmd_vel(opt_xs, opt_ys))

    # ──────────────────────────────────────────────────────────────────────
    # Costmap -> polygons (BFS clustering + MCCH per cluster)
    # ──────────────────────────────────────────────────────────────────────

    def _costmap_to_polygons(self) -> list[list[tuple[float, float]]]:
        """
        BFS-cluster lethal cells with 8-connectivity, then run MCCH
        decomposition on each cluster's world-frame cell centres.

        Costmap is in odom frame, so polygons are returned in odom frame.
        """
        cm   = self._local_costmap
        info = cm.info
        w, h = info.width, info.height
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        data = cm.data

        # Restrict scan to a window around the robot for performance.
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        margin_cells = int((self.lookahead_dist + self.search_margin) / res)
        cx_robot = int((rx - ox) / res)
        cy_robot = int((ry - oy) / res)
        i_lo = max(0, cx_robot - margin_cells)
        i_hi = min(w, cx_robot + margin_cells + 1)
        j_lo = max(0, cy_robot - margin_cells)
        j_hi = min(h, cy_robot + margin_cells + 1)

        # 1. Collect occupied cells in the window.
        threshold = self.obstacle_threshold
        occupied: set[tuple[int, int]] = set()
        for j in range(j_lo, j_hi):
            base = j * w
            for i in range(i_lo, i_hi):
                if data[base + i] >= threshold:
                    occupied.add((i, j))

        if not occupied:
            return []

        # 2. BFS clustering with 8-connectivity.
        polygons: list[list[tuple[float, float]]] = []
        visited: set[tuple[int, int]] = set()
        neighbors = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

        for start in occupied:
            if start in visited:
                continue
            cluster = []
            stack = [start]
            while stack:
                c = stack.pop()
                if c in visited:
                    continue
                visited.add(c)
                cluster.append(c)
                ci, cj = c
                for di, dj in neighbors:
                    n = (ci + di, cj + dj)
                    if n in occupied and n not in visited:
                        stack.append(n)

            if len(cluster) < self.cluster_min_cells:
                continue

            # 3. MCCH decomposition (handles L-, U-shape concavity).
            world_pts = [(ox + (i + 0.5) * res, oy + (j + 0.5) * res)
                         for i, j in cluster]
            for hull in mcch_decompose(world_pts, self.mcch_split_thresh):
                if len(hull) >= 2:
                    polygons.append(hull)

        return polygons

    # ──────────────────────────────────────────────────────────────────────
    # Velocity extraction (holonomic vector pursuit)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_cmd_vel(self,
                         opt_xs: list[float],
                         opt_ys: list[float]) -> TwistStamped:
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if not opt_xs or len(opt_xs) < 2:
            return cmd

        rx  = self._robot_pose.position.x
        ry  = self._robot_pose.position.y
        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 1. Lookahead point on the optimised band.
        lookahead = 0.5
        tx, ty = opt_xs[-1], opt_ys[-1]
        for ox, oy in zip(opt_xs, opt_ys):
            if math.hypot(ox - rx, oy - ry) >= lookahead:
                tx, ty = ox, oy
                break

        # 2. Body-frame vector to lookahead.
        dxw = tx - rx
        dyw = ty - ry
        xl =  dxw * cos_y + dyw * sin_y
        yl = -dxw * sin_y + dyw * cos_y
        L  = math.hypot(xl, yl)
        if L < 1e-6:
            return cmd

        # 3. Decelerate near goal.
        d_goal   = euclidean(self._robot_pose, self._global_path.poses[-1].pose)
        v_target = min(self.v_max, max(0.05, d_goal * 1.5))

        # 4. Holonomic distribution (slide directly toward lookahead).
        cmd.twist.linear.x  = v_target * (xl / L)
        cmd.twist.linear.y  = v_target * (yl / L)
        cmd.twist.angular.z = 0.0
        return cmd

    def _pub_stop(self):
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(cmd)

    # ──────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────

    def _publish_polygons(self):
        ma = MarkerArray()
        frame = self._local_costmap.header.frame_id
        stamp = self.get_clock().now().to_msg()

        # Clear previous polygons before drawing new ones.
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for idx, poly in enumerate(self._polygons):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = stamp
            m.ns              = 'obstacle_polygons'
            m.id              = idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.05
            m.color           = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            m.pose.orientation.w = 1.0
            for (px, py) in poly:
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.0
                m.points.append(pt)
            if len(poly) >= 3:
                first = Point()
                first.x, first.y, first.z = poly[0][0], poly[0][1], 0.0
                m.points.append(first)
            ma.markers.append(m)

        self.polygon_pub.publish(ma)

    def _publish_path(self, xs: list[float], ys: list[float], frame: str):
        n = len(xs)
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = frame
        for i in range(n):
            theta = math.atan2(
                ys[min(i + 1, n - 1)] - ys[max(i - 1, 0)],
                xs[min(i + 1, n - 1)] - xs[max(i - 1, 0)]
            )
            ps = PoseStamped()
            ps.header.frame_id    = frame
            ps.header.stamp       = stamp
            ps.pose.position.x    = xs[i]
            ps.pose.position.y    = ys[i]
            ps.pose.orientation.z = math.sin(theta / 2.0)
            ps.pose.orientation.w = math.cos(theta / 2.0)
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    # ──────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────

    def _find_closest_index(self) -> int:
        wp = self._global_path.poses
        idx, best = self._prune_index, float('inf')
        for i in range(self._prune_index, len(wp)):
            d = euclidean(self._robot_pose, wp[i].pose)
            if d < best:
                best, idx = d, i
        return idx

    def _extract_local_segment(self) -> list[PoseStamped]:
        wp = self._global_path.poses
        seg = []
        for i in range(self._prune_index, len(wp)):
            d = euclidean(self._robot_pose, wp[i].pose)
            if d <= self.lookahead_dist:
                seg.append(wp[i])
            else:
                break
        return seg


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
