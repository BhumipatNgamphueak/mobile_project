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
       Vector pursuit on the optimised band -> (vx, vy, omega)
       with proportional heading control to face direction of travel.

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

import csv
import math
import os
import time
from datetime import datetime

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

        # ── Declare Param
        self.declare_parameter('cost_map_topic', '/local_costmap')
        self.declare_parameter('lookahead_dist', 15.0)
        self.declare_parameter('log_dir',    os.path.expanduser('~/robot_logs'))
        self.declare_parameter('world_name', 'unknown')
        self.cost_map_topic = self.get_parameter('cost_map_topic').value

        # ── General parameters ────────────────────────────────────────────
        self.planning_freq          = 10     # Hz
        self.lookahead_dist         = self.get_parameter('lookahead_dist').value
        self.min_waypoints_required = 2
        self.goal_tolerance         = 0.4    # m
        self.v_max                  = 1.5    # m/s
        self.w_max                  = 1.2    # rad/s — max yaw rate
        self.yaw_kp                 = 1.5    # proportional gain for heading control
        self.cmd_lookahead          = 0.5    # m — lookahead for velocity extraction
        self.sensor_timeout         = 0.5    # s — max age of odom/costmap before stop

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
        self.obstacle_threshold = 30     # cost >= this -> obstacle cell
        self.cluster_min_cells  = 3      # filter noise clusters
        self.mcch_split_thresh  = 0.3    # m — MCCH fictitious-edge threshold

        # ── CSV path logger ───────────────────────────────────────────────
        self._log_dir    = self.get_parameter('log_dir').value
        self._world_name = self.get_parameter('world_name').value
        os.makedirs(self._log_dir, exist_ok=True)
        self._csv_file:   object       = None
        self._csv_writer: object       = None
        self._current_goal_x: float | None = None
        self._current_goal_y: float | None = None

        # Per-cycle planning stats (updated in _replan, written in odom rows)
        self._last_replan_ms:      float = 0.0
        self._last_n_polygons:     int   = 0
        self._last_seg_len_m:      float = 0.0
        self._last_min_obs_dist_m: float = 0.0

        # Run-level accumulators (reset on each new goal)
        self._path_length_m:   float       = 0.0
        self._goal_start_time: float | None = None
        self._goal_start_x:    float | None = None
        self._goal_start_y:    float | None = None
        self._last_logged_pos: tuple | None = None
        self._last_logged_t:   float | None = None
        self._goal_reached_logged: bool     = False

        # Derived velocity (finite-difference from odom positions, body frame)
        self._derived_vx:    float = 0.0
        self._derived_vy:    float = 0.0
        self._derived_omega: float = 0.0
        self._last_yaw:      float | None = None

        # Session summary CSV (one row per goal attempt, opened once)
        session_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path  = os.path.join(
            self._log_dir,
            f'run_summary_{self._world_name}_{session_stamp}.csv')
        self._summary_file   = open(summary_path, 'w', newline='')
        self._summary_writer = csv.writer(self._summary_file)
        self._summary_writer.writerow([
            'world', 'goal_x', 'goal_y',
            'start_x', 'start_y', 'straight_line_m',
            'path_length_m', 'travel_time_s',
            'final_pos_error_m', 'goal_reached',
        ])
        self.get_logger().info(f'Run summary → {summary_path}')

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,       '/odom',          self._odom_callback,          10)
        self.create_subscription(Path,           '/global_path',   self._global_path_callback,   10)
        self.create_subscription(OccupancyGrid,  self.cost_map_topic, self._local_costmap_callback, 10)
        self.create_subscription(PoseStamped,    '/goal_pose',     self._goal_pose_callback,     10)

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
        self._last_odom_time: float | None        = None
        self._last_costmap_time: float | None     = None

        self.create_timer(1.0 / self.planning_freq, self._replan)
        self.get_logger().info('PathPlanningNode (TEB polygon-style) started')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self._robot_pose     = msg.pose.pose
        self._robot_vel      = msg.twist.twist
        self._last_odom_time = self.get_clock().now().nanoseconds * 1e-9

        cx  = msg.pose.pose.position.x
        cy  = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        now = self._last_odom_time

        if self._last_logged_pos is not None and self._last_logged_t is not None:
            dt = now - self._last_logged_t
            if dt > 1e-6:
                dx_w = cx - self._last_logged_pos[0]
                dy_w = cy - self._last_logged_pos[1]
                c, s = math.cos(yaw), math.sin(yaw)
                self._derived_vx    = ( c * dx_w + s * dy_w) / dt
                self._derived_vy    = (-s * dx_w + c * dy_w) / dt
                d_yaw = self._last_yaw if self._last_yaw is not None else yaw
                dyaw  = math.atan2(math.sin(yaw - d_yaw), math.cos(yaw - d_yaw))
                self._derived_omega = dyaw / dt
            self._path_length_m += math.hypot(cx - self._last_logged_pos[0],
                                              cy - self._last_logged_pos[1])

        self._last_logged_pos = (cx, cy)
        self._last_logged_t   = now
        self._last_yaw        = yaw

        if self._csv_writer is None:
            return

        self._csv_writer.writerow([
            f'{now:.4f}',
            f'{cx:.4f}',
            f'{cy:.4f}',
            f'{yaw:.4f}',
            f'{self._derived_vx:.4f}',
            f'{self._derived_vy:.4f}',
            f'{self._derived_omega:.4f}',
            f'{self._current_goal_x:.4f}' if self._current_goal_x is not None else '',
            f'{self._current_goal_y:.4f}' if self._current_goal_y is not None else '',
            f'{self._last_replan_ms:.2f}',
            f'{self._last_n_polygons}',
            f'{self._last_seg_len_m:.4f}',
            f'{self._last_min_obs_dist_m:.4f}',
            f'{self._path_length_m:.4f}',
            f'{self.lookahead_dist:.2f}',
        ])

    def _goal_pose_callback(self, msg: PoseStamped):
        # Finalise previous run before starting new one
        if self._goal_start_time is not None and not self._goal_reached_logged:
            self._write_summary(goal_reached=False)

        self._current_goal_x = msg.pose.position.x
        self._current_goal_y = msg.pose.position.y

        # Reset run accumulators
        rx = self._robot_pose.position.x if self._robot_pose else 0.0
        ry = self._robot_pose.position.y if self._robot_pose else 0.0
        self._goal_start_time      = time.perf_counter()
        self._goal_start_x         = rx
        self._goal_start_y         = ry
        self._path_length_m        = 0.0
        self._last_logged_pos      = (rx, ry)
        self._last_logged_t        = None
        self._derived_vx           = 0.0
        self._derived_vy           = 0.0
        self._derived_omega        = 0.0
        self._goal_reached_logged  = False

        # Close previous path log if open
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()

        stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(
            self._log_dir,
            f'robot_path_{self._world_name}_{stamp}.csv')
        self._csv_file   = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'time_s', 'x', 'y', 'yaw_rad', 'vx', 'vy', 'omega',
            'goal_x', 'goal_y',
            'replan_ms', 'n_polygons', 'seg_len_m', 'min_obs_dist_m', 'path_len_m',
            'lookahead_dist',
        ])
        self.get_logger().info(
            f'Goal set: x={self._current_goal_x:.3f} y={self._current_goal_y:.3f}'
            f' — logging to {csv_path}')

    def _local_costmap_callback(self, msg: OccupancyGrid):
        self._local_costmap     = msg
        self._last_costmap_time = self.get_clock().now().nanoseconds * 1e-9

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
                # Geometry changed but goal is the same — keep warm state and
                # progress index so we don't re-scan already-passed waypoints.
                self._global_path = msg
                return

        self._global_path = msg
        self._prune_index = 0
        self.eb.reset_warm_start()
        self.get_logger().info(f'New goal received ({len(msg.poses)} waypoints)')

    # ──────────────────────────────────────────────────────────────────────
    # Planning loop
    # ──────────────────────────────────────────────────────────────────────

    def _replan(self):
        _t0 = time.perf_counter()
        if self._robot_pose is None or self._local_costmap is None:
            return

        now_s = self.get_clock().now().nanoseconds * 1e-9
        if (self._last_odom_time is not None
                and now_s - self._last_odom_time > self.sensor_timeout):
            self.get_logger().warn('Odom data stale — stopping')
            self._pub_stop()
            return
        if (self._last_costmap_time is not None
                and now_s - self._last_costmap_time > self.sensor_timeout):
            self.get_logger().warn('Costmap data stale — stopping')
            self._pub_stop()
            return

        # 1. Costmap -> polygon obstacles (BFS clustering + MCCH split).
        # Run every tick — independent of goal/motion — so the hulls
        # follow moving obstacles even when the robot is idle or at goal.
        self._polygons = self._costmap_to_polygons()
        self._publish_polygons()
        self._last_n_polygons     = len(self._polygons)
        self._last_min_obs_dist_m = self._min_obstacle_dist()

        if self._global_path is None:
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            return

        if euclidean(self._robot_pose, self._global_path.poses[-1].pose) < self.goal_tolerance:
            if not self._goal_reached_logged:
                self._goal_reached_logged = True
                self._write_summary(goal_reached=True)
            self._pub_stop()
            return

        # 2. Reference segment from global path.
        self._prune_index = self._find_closest_index()
        segment = self._extract_local_segment()
        if len(segment) < self.min_waypoints_required:
            return
        segment_xys = [(p.pose.position.x, p.pose.position.y) for p in segment]
        self._last_seg_len_m = sum(
            math.hypot(segment_xys[i+1][0] - segment_xys[i][0],
                       segment_xys[i+1][1] - segment_xys[i][1])
            for i in range(len(segment_xys) - 1))

        # 3. Elastic-band optimisation.
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        opt_xs, opt_ys = self.eb.optimize((rx, ry), segment_xys, self._polygons)

        # 4. Publish optimised path & velocity.
        self._publish_path(opt_xs, opt_ys, segment[0].header.frame_id)
        self.cmd_vel_pub.publish(self._compute_cmd_vel(opt_xs, opt_ys))
        self._last_replan_ms = (time.perf_counter() - _t0) * 1000.0

    # ──────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────────

    def _min_obstacle_dist(self) -> float:
        if not self._polygons or self._robot_pose is None:
            return float('inf')
        from path_planning.elastic_band import point_polygon_signed_dist
        rx  = self._robot_pose.position.x
        ry  = self._robot_pose.position.y
        min_d = float('inf')
        for poly in self._polygons:
            d, _, _, inside = point_polygon_signed_dist(rx, ry, poly)
            if inside:
                return 0.0
            if d < min_d:
                min_d = d
        return min_d

    def _write_summary(self, goal_reached: bool) -> None:
        if self._goal_start_time is None or self._robot_pose is None:
            return
        travel_time = time.perf_counter() - self._goal_start_time
        sx = self._goal_start_x or 0.0
        sy = self._goal_start_y or 0.0
        gx = self._current_goal_x or 0.0
        gy = self._current_goal_y or 0.0
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        straight = math.hypot(gx - sx, gy - sy)
        pos_err  = math.hypot(rx - gx, ry - gy)
        self._summary_writer.writerow([
            self._world_name,
            f'{gx:.4f}', f'{gy:.4f}',
            f'{sx:.4f}', f'{sy:.4f}',
            f'{straight:.4f}',
            f'{self._path_length_m:.4f}',
            f'{travel_time:.2f}',
            f'{pos_err:.4f}',
            int(goal_reached),
        ])
        self._summary_file.flush()
        self.get_logger().info(
            f'Run summary: reached={int(goal_reached)} '
            f'dist={self._path_length_m:.2f}m '
            f'time={travel_time:.1f}s err={pos_err:.3f}m')

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

        # 1. Collect all occupied cells in the costmap.
        threshold = self.obstacle_threshold
        occupied: set[tuple[int, int]] = set()
        for j in range(h):
            base = j * w
            for i in range(w):
                if data[base + i] >= threshold:
                    occupied.add((i, j))

        if not occupied:
            return []

        # 2. DFS clustering with 8-connectivity.
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
        q   = self._robot_pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 1. Lookahead point on the optimised band.
        tx, ty = opt_xs[-1], opt_ys[-1]
        for bx, by in zip(opt_xs, opt_ys):
            if math.hypot(bx - rx, by - ry) >= self.cmd_lookahead:
                tx, ty = bx, by
                break

        # 2. Body-frame vector to lookahead.
        dxw = tx - rx
        dyw = ty - ry
        xl =  dxw * cos_y + dyw * sin_y
        yl = -dxw * sin_y + dyw * cos_y
        L  = math.hypot(xl, yl)
        if L < 1e-6:
            return cmd

        # 3. Decelerate near goal: speed tracks distance 1:1, capped at v_max.
        d_goal   = euclidean(self._robot_pose, self._global_path.poses[-1].pose)
        v_target = min(self.v_max, max(0.05, d_goal))

        # 4. Holonomic motion — fixed heading, no rotation.
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
            if euclidean(self._robot_pose, wp[i].pose) <= self.lookahead_dist:
                seg.append(wp[i])
        return seg


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._csv_file is not None:
            node._csv_file.flush()
            node._csv_file.close()
        if node._goal_start_time is not None and not node._goal_reached_logged:
            node._write_summary(goal_reached=False)
        node._summary_file.flush()
        node._summary_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
