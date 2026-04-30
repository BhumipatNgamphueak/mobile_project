#!/usr/bin/env python3
"""
path_planning_node.py
=====================
TEB-style local planner with point obstacles.

Pipeline (one cycle):
  1. Costmap -> point obstacles
       Extract occupied cells in a window around the robot.
  2. Local segment extraction
       Prune behind the robot, then take a lookahead window of the global
       path. Warm-started from the previous optimised band so the trajectory
       evolves smoothly across cycles.
  3. TEB optimisation
       Gradient descent on the elastic band against
         - point obstacles (outward force inside an inflation zone)
         - smoothness (pull toward midpoint of neighbours)
       Endpoints fixed.
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
/obstacle_points     visualization_msgs/MarkerArray   extracted points (RViz)
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── General parameters ────────────────────────────────────────────
        self.planning_freq          = 10     # Hz
        self.lookahead_dist         = 4.0   # m — local segment length
        self.min_waypoints_required = 2
        self.goal_tolerance         = 0.4    # m

        # ── TEB parameters ────────────────────────────────────────────────
        self.teb_n_iter      = 60      # gradient descent iterations / cycle
        self.teb_w_obs       = 50.0 # 50.0    # obstacle repulsion weight
        self.teb_w_smooth    = 2.0     # elastic band weight
        self.teb_inflation   = 1.0 # 0.5     # m — repulsion radius around points
        self.teb_step        = 0.05    # gradient step
        self.teb_max_delta   = 0.10    # m — clamp per-iteration movement
        self.v_max           = 1.5     # m/s

        self.teb_w_time      = 1.0     # weight to minimize total time
        self.teb_w_max_vel   = 10.0    # weight to penalize exceeding v_max
        self.teb_max_delta_t = 0.1     # clamp per-iteration dt movement

        # ── Costmap-to-points parameters ────────────────────────────────
        self.obstacle_threshold = 60    # cost >= this -> obstacle cell
        self.search_margin      = 8.0   # m beyond lookahead to scan

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,       '/odom',          self._odom_callback,          10)
        self.create_subscription(Path,           '/global_path',   self._global_path_callback,   10)
        self.create_subscription(OccupancyGrid,  '/local_costmap', self._local_costmap_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel',           10)
        self.path_pub    = self.create_publisher(Path,         '/planned_path',      10)
        self.obstacle_pub = self.create_publisher(MarkerArray,  '/obstacle_points',   10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose: Pose | None             = None
        self._robot_vel                           = None
        self._global_path: Path | None            = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int                    = 0

        # Warm start from previous optimised band.
        self._warm_xs: list[float] | None = None
        self._warm_ys: list[float] | None = None
        self._warm_dts: list[float] | None = None

        # Points extracted from current costmap (set each cycle).
        self._obstacle_points: list[tuple[float, float]] = []

        self.create_timer(1.0 / self.planning_freq, self._replan)
        self.get_logger().info('PathPlanningNode (TEB point-style) started')

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
        self._warm_xs     = None
        self._warm_ys     = None
        self._warm_dts    = None
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

        # 1. Costmap -> point obstacles
        self._obstacle_points = self._costmap_to_points()
        self._publish_points()

        # 2. Reference segment from global path
        self._prune_index = self._find_closest_index()
        segment = self._extract_local_segment()
        if len(segment) < self.min_waypoints_required:
            return

        # 3. TEB optimisation
        opt_xs, opt_ys, opt_dts = self._teb_optimize(segment)

        # 4. Cache for next cycle's warm start
        self._warm_xs = opt_xs[:]
        self._warm_ys = opt_ys[:]
        self._warm_dts = opt_dts[:]

        # 5. Publish optimised path & velocity
        self._publish_path(opt_xs, opt_ys, segment[0].header.frame_id)
        self.cmd_vel_pub.publish(self._compute_cmd_vel(opt_xs, opt_ys, opt_dts))

    # ──────────────────────────────────────────────────────────────────────
    # Costmap -> Points
    # ──────────────────────────────────────────────────────────────────────

    def _costmap_to_points(self) -> list[tuple[float, float]]:
        """
        Extract occupied cells as point obstacles.
        Costmap is in odom frame, so points are returned in odom frame.
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

        threshold = self.obstacle_threshold
        points = []
        for j in range(j_lo, j_hi):
            base = j * w
            for i in range(i_lo, i_hi):
                if data[base + i] >= threshold:
                    wx = ox + (i + 0.5) * res
                    wy = oy + (j + 0.5) * res
                    points.append((wx, wy))
        return points

    # ──────────────────────────────────────────────────────────────────────
    # Point distance / obstacle force
    # ──────────────────────────────────────────────────────────────────────

    def _obstacle_force(self, x: float, y: float) -> tuple[float, float]:
        """
        Returns the repulsive force on a waypoint from the closest point.
        """
        if not self._obstacle_points:
            return 0.0, 0.0

        best_d = float('inf')
        best_px = 0.0
        best_py = 0.0

        for px, py in self._obstacle_points:
            d = math.hypot(x - px, y - py)
            if d < best_d:
                best_d = d
                best_px = px
                best_py = py

        if best_d >= self.teb_inflation:
            return 0.0, 0.0

        # Push away from boundary.
        vx, vy = x - best_px, y - best_py
        m = math.hypot(vx, vy)
        if m < 1e-9:
            return 0.0, 0.0
        mag = self.teb_inflation - best_d
        return mag * vx / m, mag * vy / m

    # ──────────────────────────────────────────────────────────────────────
    # TEB optimisation
    # ──────────────────────────────────────────────────────────────────────

    def _teb_optimize(self,
                      segment: list[PoseStamped]
                      ) -> tuple[list[float], list[float], list[float]]:
        """
        Elastic band optimisation against point obstacles.

        Anchoring:
          The band is anchored at the *actual* robot pose (xs[0]) — not at
          the closest global-path waypoint. This matters when the robot has
          deviated laterally to avoid an obstacle: anchoring at the global
          path would put xs[0] inside the inflation zone and the smoothness
          force would keep dragging the band back into the obstacle.

        Warm start:
          Across cycles the previous solution is re-projected onto the new
          band by *arc-length* parameterisation, not nearest-neighbour. This
          preserves the avoidance shape even when the lateral deviation is
          larger than the global-path waypoint spacing — preventing the
          left/right flip-flop that produces in-and-out motion.

        Endpoints (start = robot, end = lookahead goal) are fixed; only
        interior nodes move. Per-iteration motion is clamped so the band
        cannot teleport across narrow gaps.
        """
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y

        # Initial band: robot pose + global-path waypoints.
        # Skip waypoints that would create a degenerate first segment.
        xs = [rx]
        ys = [ry]
        skip_dist = 0.4
        for p in segment:
            px, py = p.pose.position.x, p.pose.position.y
            if math.hypot(px - xs[-1], py - ys[-1]) > skip_dist:
                xs.append(px)
                ys.append(py)

        n = len(xs)
        if n < 2:
            return xs, ys, [0.1]

        # Initialize time intervals based on max velocity
        dts = []
        for i in range(n - 1):
            dist = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            dts.append(max(0.01, dist / self.v_max))

        # ── Warm start: arc-length re-projection of previous band ────────
        if self._warm_xs and self._warm_ys and self._warm_dts and len(self._warm_xs) >= 2:
            warm_xs, warm_ys, warm_dts = self._warm_xs, self._warm_ys, self._warm_dts
            warm_arc = [0.0]
            for k in range(1, len(warm_xs)):
                warm_arc.append(warm_arc[-1] + math.hypot(
                    warm_xs[k] - warm_xs[k - 1],
                    warm_ys[k] - warm_ys[k - 1]))
            L_warm = warm_arc[-1] # old length

            new_arc = [0.0]
            for k in range(1, n):
                new_arc.append(new_arc[-1] + math.hypot(
                    xs[k] - xs[k - 1], ys[k] - ys[k - 1]))
            L_new = new_arc[-1]  # new length

            # Interpolating new length
            if L_warm > 1e-3 and L_new > 1e-3:
                for i in range(1, n - 1):
                    target = (new_arc[i] / L_new) * L_warm
                    for k in range(len(warm_arc) - 1):
                        if warm_arc[k + 1] >= target:
                            span = warm_arc[k + 1] - warm_arc[k]
                            if span < 1e-9:
                                xs[i], ys[i] = warm_xs[k], warm_ys[k]
                            else:
                                a = (target - warm_arc[k]) / span
                                xs[i] = warm_xs[k] + a * (warm_xs[k + 1] - warm_xs[k])
                                ys[i] = warm_ys[k] + a * (warm_ys[k + 1] - warm_ys[k])
                            break
                # Simple interpolation for dt based on relative lengths
                len_ratio = L_new / L_warm
                for i in range(min(len(dts), len(warm_dts))):
                    dts[i] = warm_dts[i] * len_ratio

        step        = self.teb_step
        max_delta   = self.teb_max_delta
        max_delta_t = self.teb_max_delta_t
        w_obs       = self.teb_w_obs
        w_smooth    = self.teb_w_smooth
        w_time      = self.teb_w_time
        w_max_vel   = self.teb_w_max_vel
        v_max       = self.v_max

        for _ in range(self.teb_n_iter):
            # Accumulate forces
            fx = [0.0] * n
            fy = [0.0] * n
            fdt = [0.0] * (n - 1)

            # 1. Obstacle & Smoothness forces
            for i in range(1, n - 1):
                fx_obs, fy_obs = self._obstacle_force(xs[i], ys[i])

                # Smoothness force (elastic band): pull toward midpoint.
                f_sx = (xs[i - 1] + xs[i + 1]) / 2.0 - xs[i]
                f_sy = (ys[i - 1] + ys[i + 1]) / 2.0 - ys[i]

                fx[i] += w_obs * fx_obs + w_smooth * f_sx
                fy[i] += w_obs * fy_obs + w_smooth * f_sy

            # 2. Velocity & Time forces
            for i in range(n - 1):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                dist = math.hypot(dx, dy)
                dt = dts[i]
                v = dist / dt

                # Time force (fastest path objective)
                fdt[i] -= w_time

                # Velocity limit penalty
                if v > v_max:
                    delta_v = v - v_max
                    # Force on dt: derivative of C_v w.r.t dt is -weight * delta_v * (v / dt)
                    # We negate it for gradient descent update
                    fdt[i] += w_max_vel * delta_v * (v / dt)

                    if v > 1e-6:
                        # Force on x, y: derivative of C_v w.r.t x_i
                        # We negate it for gradient descent update
                        fx_vel = w_max_vel * delta_v * (dx / (v * dt * dt))
                        fy_vel = w_max_vel * delta_v * (dy / (v * dt * dt))
                        fx[i] += fx_vel
                        fy[i] += fy_vel
                        fx[i + 1] -= fx_vel
                        fy[i + 1] -= fy_vel

            # 3. Apply updates
            for i in range(1, n - 1):
                dx = step * fx[i]
                dy = step * fy[i]
                mag = math.hypot(dx, dy)
                if mag > max_delta:
                    dx *= max_delta / mag
                    dy *= max_delta / mag
                xs[i] += dx
                ys[i] += dy

            for i in range(n - 1):
                ddt = step * fdt[i]
                if ddt > max_delta_t:
                    ddt = max_delta_t
                elif ddt < -max_delta_t:
                    ddt = -max_delta_t
                dts[i] += ddt
                dts[i] = max(0.01, dts[i]) # clamp to prevent negative time

        return xs, ys, dts

    # ──────────────────────────────────────────────────────────────────────
    # Velocity extraction (True TEB execution)
    # ──────────────────────────────────────────────────────────────────────
    def _compute_cmd_vel(self,
                         opt_xs: list[float],
                         opt_ys: list[float],
                         opt_dts: list[float]) -> TwistStamped:
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if not opt_xs or len(opt_xs) < 2 or not opt_dts:
            return cmd

        rx  = self._robot_pose.position.x
        ry  = self._robot_pose.position.y
        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 1. Decelerate near goal.
        d_goal = euclidean(self._robot_pose, self._global_path.poses[-1].pose)
        
        # 2. True TEB velocity: velocity of the first segment
        # In a real TEB, the robot is meant to follow the trajectory encoded in the first segment
        dxw = opt_xs[1] - opt_xs[0]
        dyw = opt_ys[1] - opt_ys[0]
        dt = opt_dts[0]

        # Convert world-frame velocity to body-frame velocity
        vx_world = dxw / dt
        vy_world = dyw / dt

        vx_body =  vx_world * cos_y + vy_world * sin_y
        vy_body = -vx_world * sin_y + vy_world * cos_y

        v_mag = math.hypot(vx_body, vy_body)

        # 3. Apply deceleration near goal and ensure we do not exceed v_max
        v_target_limit = min(self.v_max, max(0.05, d_goal * 1.5))
        if v_mag > v_target_limit:
            vx_body *= v_target_limit / v_mag
            vy_body *= v_target_limit / v_mag

        cmd.twist.linear.x  = vx_body
        cmd.twist.linear.y  = vy_body
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

    def _publish_points(self):
        ma = MarkerArray()
        frame = self._local_costmap.header.frame_id
        stamp = self.get_clock().now().to_msg()

        # Clear previous points before drawing new ones.
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        if self._obstacle_points:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = stamp
            m.ns              = 'obstacle_points'
            m.id              = 0
            m.type            = Marker.POINTS
            m.action          = Marker.ADD
            m.scale.x         = 0.05
            m.scale.y         = 0.05
            m.color           = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            m.pose.orientation.w = 1.0
            
            for (px, py) in self._obstacle_points:
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.0
                m.points.append(pt)
            ma.markers.append(m)

        self.obstacle_pub.publish(ma)

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
