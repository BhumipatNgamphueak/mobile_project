#!/usr/bin/env python3
"""
path_planning_node.py
=====================
TEB (Timed Elastic Band) local planner.

How it differs from restarting TEB each cycle (which causes jitter):
  Real nav2 teb_local_planner keeps a *persistent* trajectory between calls.
  Each cycle it only prunes the start and extends the end, then runs a few
  optimisation steps on the *existing* band — not from a fresh straight line.
  We replicate this with a warm-start: interior waypoints are initialised
  from the previous cycle's solution, so the band evolves smoothly.

Inputs
------
/odom           nav_msgs/Odometry       robot pose + velocity
/global_path    nav_msgs/Path           reference path from global planner
/local_costmap  nav_msgs/OccupancyGrid  obstacle grid (robot frame, 0.1 m/cell)

Outputs
-------
/cmd_vel        geometry_msgs/TwistStamped   velocity command → mecanum robot
/planned_path   nav_msgs/Path                optimised path (RViz visualisation)
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped


def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.planning_freq          = 10    # Hz
        self.lookahead_dist         = 10.0   # m — segment length fed to TEB
        self.prune_dist             = 1.0   # m — drop waypoints closer than this
        self.min_waypoints_required = 2
        self.goal_tolerance         = 0.4   # m — stop when within this of goal

        # ── TEB parameters ────────────────────────────────────────────────
        self.teb_n_iter   = 100   # optimisation iterations per cycle
        self.teb_w_obs    = 3.0   # obstacle repulsion weight
        self.teb_w_smooth = 2.0   # smoothness (elastic band) weight
        self.teb_w_time   = 0.1   # time-minimisation weight
        self.v_max        = 1.5   # m/s — mecanum max linear speed

        # ── Subscribers ───────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)
        self.global_path_sub = self.create_subscription(
            Path, '/global_path', self._global_path_callback, 10)
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, '/local_costmap', self._local_costmap_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.path_pub    = self.create_publisher(Path, '/planned_path', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose: Pose | None        = None
        self._robot_vel                      = None   # geometry_msgs/Twist
        self._global_path: Path | None       = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int               = 0

        # Warm-start state: interior waypoints from the previous optimised band.
        # Set to None whenever a new global path arrives so TEB re-initialises.
        self._warm_xs: list[float] | None = None
        self._warm_ys: list[float] | None = None

        # ── Timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.planning_freq, self._replan)

        self.get_logger().info('PathPlanningNode started')

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
            # Ignore exact same message republished by timer
            if (msg.header.stamp.sec == self._global_path.header.stamp.sec and
                msg.header.stamp.nanosec == self._global_path.header.stamp.nanosec):
                return
                
            new_goal = msg.poses[-1].pose.position
            old_goal = self._global_path.poses[-1].pose.position
            same_goal = (abs(new_goal.x - old_goal.x) < 0.01 and
                         abs(new_goal.y - old_goal.y) < 0.01)
            if same_goal:
                # Path changed geometrically but goal is the same (e.g. replanning).
                # Reset prune_index so we find the correct closest point on the new path,
                # but KEEP warm state (Euclidean matching handles it safely) to avoid jitter.
                self._global_path = msg
                self._prune_index = 0
                return

        # New goal → full reset so TEB starts fresh from the new reference path
        self._global_path = msg
        self._prune_index = 0
        self._warm_xs     = None
        self._warm_ys     = None
        self.get_logger().info(f'New goal received — resetting planner ({len(msg.poses)} waypoints)')

    # ──────────────────────────────────────────────────────────────────────
    # Planning loop
    # ──────────────────────────────────────────────────────────────────────

    def _replan(self):
        if self._robot_pose is None or self._global_path is None \
                or self._local_costmap is None:
            print("No Data")
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            print("No Enough Point")
            return

        # Goal reached → stop
        goal_pose = self._global_path.poses[-1].pose
        if euclidean(self._robot_pose, goal_pose) < self.goal_tolerance:
            print("Reach Goal")
            self._pub_stop()
            return

        # 1. Advance prune index to closest global-path waypoint
        closest_idx = self._find_closest_index()
        self._prune_passes_waypoints(closest_idx)

        # 2. Extract local segment (reference band for TEB)
        segment = self._extract_local_segment()
        if len(segment) < self.min_waypoints_required:
            return

        # 3. TEB optimisation with warm start
        opt_xs, opt_ys = self._teb_optimize(segment)

        # 4. Save warm state for next cycle
        self._warm_xs = opt_xs[:]
        self._warm_ys = opt_ys[:]
        
        # 5. Publish optimised path (RViz)
        self._publish_path(opt_xs, opt_ys, segment[0].header.frame_id)

        # 6. Compute and publish velocity command
        cmd = self._compute_cmd_vel(opt_xs, opt_ys)
        self.cmd_vel_pub.publish(cmd)

    # ──────────────────────────────────────────────────────────────────────
    # TEB core
    # ──────────────────────────────────────────────────────────────────────

    def _teb_optimize(self,
                      segment: list[PoseStamped]
                      ) -> tuple[list[float], list[float]]:
        """
        Timed Elastic Band — adapted from nav2 teb_local_planner behaviour.

        nav2 TEB key ideas replicated here:
          1. Warm start  — interior nodes initialised from previous solution
             so the band evolves incrementally (no restart jitter).
          2. Elastic band — smoothness force pulls nodes toward midpoint of
             neighbours; obstacle force pushes them away.
          3. Time update  — dt_i shrinks toward dist/v_max each iteration.
          4. Fixed endpoints — start and goal are never moved.

        Returns (xs, ys): optimised waypoint positions in odom frame.
        """
        n = len(segment)
        if n < 2:
            return ([p.pose.position.x for p in segment],
                    [p.pose.position.y for p in segment])

        xs = [p.pose.position.x for p in segment]
        ys = [p.pose.position.y for p in segment]

        # ── Warm start ────────────────────────────────────────────────────
        # For each interior node in the new segment, find the closest node
        # in the previous optimised band and use it as the starting position.
        # This is equivalent to nav2 TEB's "updateAndPruneTEB" step: the old
        # band is re-used as the initial guess rather than a straight line.
        if self._warm_xs and self._warm_ys:
            for i in range(1, n - 1):
                best_d = float('inf')
                bx, by = xs[i], ys[i]
                for wx, wy in zip(self._warm_xs, self._warm_ys):
                    d = math.hypot(xs[i] - wx, ys[i] - wy)
                    if d < best_d:
                        best_d, bx, by = d, wx, wy
                if best_d < 0.8:   # only warm-start if previous node is nearby
                    xs[i], ys[i] = bx, by

        # ── Initial time intervals ────────────────────────────────────────
        dts = []
        for i in range(n - 1):
            dist = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            dts.append(max(0.05, dist / self.v_max))

        # ── Pre-compute robot-frame transform (constant during optimisation) ─
        # Avoids repeating atan2 + trig inside the inner loop.
        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rx    = self._robot_pose.position.x
        ry    = self._robot_pose.position.y

        step      = 0.05   # gradient step size
        eps       = 0.15   # > costmap resolution (0.1 m) → samples different cells
        max_delta = 0.10   # max movement per waypoint per iteration (m)

        for _ in range(self.teb_n_iter):

            # ── A. Elastic band update ─────────────────────────────────
            for i in range(1, n - 1):

                # Obstacle gradient (numerical, using cached transform)
                c_px = self._costmap_cost_fast(xs[i]+eps, ys[i],   rx, ry, cos_y, sin_y)
                c_mx = self._costmap_cost_fast(xs[i]-eps, ys[i],   rx, ry, cos_y, sin_y)
                c_py = self._costmap_cost_fast(xs[i],     ys[i]+eps, rx, ry, cos_y, sin_y)
                c_my = self._costmap_cost_fast(xs[i],     ys[i]-eps, rx, ry, cos_y, sin_y)
                grad_x = (c_px - c_mx) / (2.0 * eps)
                grad_y = (c_py - c_my) / (2.0 * eps)

                # Smoothness force (elastic): pull toward midpoint of neighbours
                f_sx = (xs[i-1] + xs[i+1]) / 2.0 - xs[i]
                f_sy = (ys[i-1] + ys[i+1]) / 2.0 - ys[i]

                dx = step * (-self.teb_w_obs * grad_x + self.teb_w_smooth * f_sx)
                dy = step * (-self.teb_w_obs * grad_y + self.teb_w_smooth * f_sy)

                # Clamp to prevent instability
                mag = math.hypot(dx, dy)
                if mag > max_delta:
                    dx = dx * max_delta / mag
                    dy = dy * max_delta / mag

                xs[i] += dx
                ys[i] += dy

            # ── B. Time interval update ────────────────────────────────
            for i in range(n - 1):
                dist   = math.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i])
                dt_min = dist / self.v_max
                dts[i] = max(dt_min, dts[i] - self.teb_w_time * step)

        return xs, ys

    # ──────────────────────────────────────────────────────────────────────
    # Velocity extraction
    # ──────────────────────────────────────────────────────────────────────

    def _compute_cmd_vel(self,
                         opt_xs: list[float],
                         opt_ys: list[float]) -> TwistStamped:
        """
        Strict Holonomic Velocity Extraction — produces vx, vy, omega.
        
        Instead of using Pure Pursuit curvature to steer the nose, this draws a 
        direct vector to the lookahead point in the robot's body frame and applies 
        linear velocity along that exact vector (sliding).
        """
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

        # ── 1. Select lookahead point ──────────────────────────────────
        lookahead = 0.5   # m
        tx, ty = opt_xs[-1], opt_ys[-1]
        for ox, oy in zip(opt_xs, opt_ys):
            if math.hypot(ox - rx, oy - ry) >= lookahead:
                tx, ty = ox, oy
                break

        # ── 2. Transform lookahead point to robot body frame ───────────
        dx_w = tx - rx
        dy_w = ty - ry
        xl   =  dx_w * cos_y + dy_w * sin_y   # Forward distance to target
        yl   = -dx_w * sin_y + dy_w * cos_y   # Lateral distance to target
        L    = math.hypot(xl, yl)             # Direct distance to lookahead
        
        if L < 1e-6:
            return cmd

        # ── 3. Forward speed: decelerate near goal ─────────────────────
        goal   = self._global_path.poses[-1].pose
        d_goal = euclidean(self._robot_pose, goal)
        v_target = min(self.v_max, max(0.05, d_goal * 1.5))

        # ── 4. Holonomic Velocity Distribution ─────────────────────────
        # Normalize the body-frame vector (xl, yl) and multiply by target speed.
        # This makes the robot slide directly toward the point.
        vx = v_target * (xl / L)
        vy = v_target * (yl / L)

        # ── 5. Angular velocity (Yaw) ──────────────────────────────────
        # Holonomic robots can slide. User requested pure lateral movement without yawing.
        omega = 0.0

        cmd.twist.linear.x  = vx
        cmd.twist.linear.y  = vy
        cmd.twist.angular.z = omega
        return cmd

    def _pub_stop(self):
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(cmd)

    # ──────────────────────────────────────────────────────────────────────
    # Costmap helpers
    # ──────────────────────────────────────────────────────────────────────

    def _costmap_cost_fast(self, x: float, y: float,
                           rx: float, ry: float,
                           cos_y: float, sin_y: float) -> float:
        """
        Obstacle penalty [0–1] with pre-computed robot-frame transform.
        Called thousands of times per cycle — avoids repeated trig.
        """
        if self._local_costmap is None:
            return 0.0
        dx = x - rx
        dy = y - ry
        lx =  dx * cos_y + dy * sin_y
        ly = -dx * sin_y + dy * cos_y
        info = self._local_costmap.info
        gx = int((lx - info.origin.position.x) / info.resolution)
        gy = int((ly - info.origin.position.y) / info.resolution)
        if not (0 <= gx < info.width and 0 <= gy < info.height):
            return 0.0
        raw = float(max(0, self._local_costmap.data[gy * info.width + gx]))
        if raw <= 20.0:
            return 0.0
        return (raw - 20.0) / 80.0

    def _costmap_cost(self, x: float, y: float) -> float:
        """Obstacle penalty [0–1] — convenience wrapper used outside TEB."""
        if self._local_costmap is None or self._robot_pose is None:
            return 0.0
        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        return self._costmap_cost_fast(x, y,
                                       self._robot_pose.position.x,
                                       self._robot_pose.position.y,
                                       math.cos(yaw), math.sin(yaw))

    # ──────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────

    def _publish_path(self, xs: list[float], ys: list[float], frame: str):
        n = len(xs)
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = frame
        for i in range(n):
            theta = math.atan2(
                ys[min(i+1, n-1)] - ys[max(i-1, 0)],
                xs[min(i+1, n-1)] - xs[max(i-1, 0)]
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

    def _find_closest_index(self) -> int:
        waypoints    = self._global_path.poses
        closest_idx  = self._prune_index
        closest_dist = float('inf')
        for i in range(self._prune_index, len(waypoints)):
            d = euclidean(self._robot_pose, waypoints[i].pose)
            if d < closest_dist:
                closest_dist = d
                closest_idx  = i
        return closest_idx

    def _prune_passes_waypoints(self, closest_idx: int):
        self._prune_index = closest_idx

    def _extract_local_segment(self) -> list[PoseStamped]:
        waypoints = self._global_path.poses
        segment   = []
        for i in range(self._prune_index, len(waypoints)):
            d = euclidean(self._robot_pose, waypoints[i].pose)
            if d <= self.lookahead_dist:
                segment.append(waypoints[i])
            else:
                break
        return segment


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
