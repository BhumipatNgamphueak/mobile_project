#!/usr/bin/env python3
"""
pure_pursuit_node.py  —  Regulated Pure Pursuit (RPP)
======================================================
Implements the Regulated Pure Pursuit algorithm (Macenski et al., 2023,
arXiv:2305.20026) adapted for a Mecanum (holonomic) wheeled robot.

Key additions over classic Pure Pursuit:
  - Adaptive lookahead distance proportional to commanded speed (APP)
  - Curvature heuristic: slow before sharp turns
  - Proximity heuristic: slow near obstacles
  - Predictive collision detection along the commanded arc
  - Goal-approach deceleration
  - Mecanum lateral correction (v_y = K_lat * cross-track error)

Inputs
------
/planned_path   nav_msgs/Path          path from path_planning_node
/odom           nav_msgs/Odometry      current robot pose + velocity
/real_map       nav_msgs/OccupancyGrid static + inflated obstacle map

Outputs
-------
/cmd_vel        geometry_msgs/TwistStamped   (TwistStamped — required by
                                              mecanum_drive_controller)

Parameters
----------
lt              (float, 1.0)    lookahead time gain [s]
lt_min          (float, 0.25)   minimum lookahead distance [m]
lt_max          (float, 1.2)    maximum lookahead distance [m]
v_max           (float, 0.8)    maximum linear speed [m/s]
v_min           (float, 0.05)   minimum speed (stall prevention) [m/s]
omega_max       (float, 3.2)    maximum angular speed [rad/s]
r_min           (float, 0.9)    min turn radius for curvature heuristic [m]
d_prox          (float, 0.7)    proximity activation distance [m]
alpha           (float, 0.7)    proximity gain (≤ 1.0)
t_collision     (float, 1.5)    predictive collision horizon [s]
k_lat           (float, 0.5)    lateral correction gain (Mecanum only)
v_y_max         (float, 0.3)    max lateral strafe speed [m/s]
goal_tolerance  (float, 0.15)   stop within this distance of goal [m]
d_goal_approach (float, 1.0)    begin decelerating at this distance [m]
control_hz      (float, 20.0)   control loop frequency [Hz]
odom_frame      (str,  'odom')
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from std_msgs.msg import Header


class RegulatedPurePursuitNode(Node):

    def __init__(self):
        super().__init__('Regulated_Pure_Pursuit')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('lt',              1.0)
        self.declare_parameter('lt_min',          0.25)
        self.declare_parameter('lt_max',          1.2)
        self.declare_parameter('v_max',           0.8)
        self.declare_parameter('v_min',           0.05)
        self.declare_parameter('omega_max',       3.2)
        self.declare_parameter('r_min',           0.9)
        self.declare_parameter('d_prox',          0.7)
        self.declare_parameter('alpha',           0.7)
        self.declare_parameter('t_collision',     1.5)
        self.declare_parameter('k_lat',           0.5)
        self.declare_parameter('v_y_max',         0.3)
        self.declare_parameter('goal_tolerance',  0.15)
        self.declare_parameter('d_goal_approach', 1.0)
        self.declare_parameter('control_hz',      50.0)
        self.declare_parameter('odom_frame',      'odom')

        self.lt              = self.get_parameter('lt').value
        self.lt_min          = self.get_parameter('lt_min').value
        self.lt_max          = self.get_parameter('lt_max').value
        self.v_max           = self.get_parameter('v_max').value
        self.v_min           = self.get_parameter('v_min').value
        self.omega_max       = self.get_parameter('omega_max').value
        self.r_min           = self.get_parameter('r_min').value
        self.d_prox          = self.get_parameter('d_prox').value
        self.alpha           = self.get_parameter('alpha').value
        self.t_collision     = self.get_parameter('t_collision').value
        self.k_lat           = self.get_parameter('k_lat').value
        self.v_y_max         = self.get_parameter('v_y_max').value
        self.goal_tolerance  = self.get_parameter('goal_tolerance').value
        self.d_goal_approach = self.get_parameter('d_goal_approach').value
        control_hz           = self.get_parameter('control_hz').value
        self.odom_frame      = self.get_parameter('odom_frame').value

        # ── Subscribers ───────────────────────────────────────────────────
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Path,          '/planned_path', self._path_callback,  10)
        self.create_subscription(Odometry,      '/odom',         self._odom_callback,  10)
        self.create_subscription(OccupancyGrid, '/real_map',     self._map_callback,   map_qos)

        # ── Publisher ─────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._path:       list[PoseStamped]    = []
        self._robot_pose: Pose | None          = None
        self._robot_yaw:  float                = 0.0
        self._costmap:    OccupancyGrid | None = None
        self._v_cmd:      float                = 0.0   # last commanded speed

        self.create_timer(1.0 / control_hz, self._control_loop)
        self.get_logger().info('Regulated_Pure_Pursuit node started')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _path_callback(self, msg: Path):
        self._path = list(msg.poses)   # mutable copy — pruned in-place each cycle
        self.get_logger().debug(f'New path received: {len(self._path)} poses')

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose
        self._robot_yaw  = self._yaw_from_quaternion(msg.pose.pose.orientation)

    def _map_callback(self, msg: OccupancyGrid):
        self._costmap = msg

    # ──────────────────────────────────────────────────────────────────────
    # Main control loop
    # ──────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        if self._robot_pose is None or not self._path:
            self.get_logger().warn(
                f'Waiting — pose={self._robot_pose is not None}, '
                f'path_len={len(self._path)}',
                throttle_duration_sec=3.0)
            self._publish_zero()
            return

        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y

        # Goal check ───────────────────────────────────────────────────────
        goal = self._path[-1].pose.position
        dist_goal = math.hypot(goal.x - rx, goal.y - ry)
        if dist_goal < self.goal_tolerance:
            self._publish_zero()
            self._v_cmd = 0.0
            self.get_logger().info('Goal reached', throttle_duration_sec=2.0)
            return

        # Step 1 — Prune past path points ─────────────────────────────────
        self._prune_path()
        if not self._path:
            self.get_logger().warn('Path empty after pruning', throttle_duration_sec=3.0)
            self._publish_zero()
            return

        # Step 2 — Adaptive lookahead distance (APP) ──────────────────────
        v_ref = self._v_cmd if self._v_cmd >= self.v_min else self.v_max
        lt    = max(self.lt_min, min(self.lt_max, v_ref * self.lt))

        # Step 3 — Select lookahead point ─────────────────────────────────
        lx_w, ly_w = self._select_lookahead(lt)

        # Transform lookahead point to robot body frame ───────────────────
        _, yl = self._world_to_robot(lx_w, ly_w)

        # Step 4 — Path curvature in body frame ───────────────────────────
        kappa = (2.0 * yl) / (lt * lt) if lt > 1e-6 else 0.0

        # Step 5 — Regulate linear velocity ───────────────────────────────
        v_regulated = self._regulate_velocity(kappa)

        # Step 8 (optional) — Goal approach deceleration ──────────────────
        if dist_goal < self.d_goal_approach:
            v_regulated = max(self.v_min,
                              v_regulated * dist_goal / self.d_goal_approach)

        # Step 6 — Compute velocity commands (Mecanum) ────────────────────
        vx    = v_regulated
        vy    = max(-self.v_y_max, min(self.v_y_max, self.k_lat * yl))
        omega = max(-self.omega_max, min(self.omega_max, v_regulated * kappa))

        # Step 7 — Predictive collision detection ─────────────────────────
        if self._collision_ahead(vx, vy, omega):
            self.get_logger().warn(
                'Collision predicted — stopping', throttle_duration_sec=1.0)
            self._publish_zero()
            self._v_cmd = 0.0
            return

        self._v_cmd = vx
        self.get_logger().info(
            f'CMD vx={vx:.3f} vy={vy:.3f} w={omega:.3f} | '
            f'kappa={kappa:.3f} lt={lt:.2f} yl={yl:.3f} dist_goal={dist_goal:.2f}',
            throttle_duration_sec=2.0)
        self.cmd_pub.publish(self._make_twist(vx, vy, omega))

    # ──────────────────────────────────────────────────────────────────────
    # Step 1 — Path pruning
    # ──────────────────────────────────────────────────────────────────────

    def _prune_path(self):
        """Remove all path points that are behind the closest point to the robot."""
        if not self._path:
            return
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        min_d, closest = float('inf'), 0
        for i, ps in enumerate(self._path):
            d = math.hypot(ps.pose.position.x - rx, ps.pose.position.y - ry)
            if d < min_d:
                min_d, closest = d, i
        if closest > 0:
            self._path = self._path[closest:]

    # ──────────────────────────────────────────────────────────────────────
    # Step 3 — Lookahead point selection
    # ──────────────────────────────────────────────────────────────────────

    def _select_lookahead(self, lt: float) -> tuple[float, float]:
        """Return the first path point at or beyond lookahead distance lt."""
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        for ps in self._path:
            if math.hypot(ps.pose.position.x - rx, ps.pose.position.y - ry) >= lt:
                return ps.pose.position.x, ps.pose.position.y
        last = self._path[-1].pose.position
        return last.x, last.y

    # ──────────────────────────────────────────────────────────────────────
    # Step 5 — Velocity regulation (curvature + proximity heuristics)
    # ──────────────────────────────────────────────────────────────────────

    def _regulate_velocity(self, kappa: float) -> float:
        # 5a — curvature heuristic: slow before sharp turns
        t_kappa = 1.0 / self.r_min
        if abs(kappa) > t_kappa:
            v_curv = self.v_max / (self.r_min * abs(kappa))
        else:
            v_curv = self.v_max

        # 5b — proximity heuristic: slow near obstacles
        d_obs = self._nearest_obstacle_dist()
        if d_obs is not None and d_obs <= self.d_prox:
            v_prox = self.v_max * self.alpha * (d_obs / self.d_prox)
        else:
            v_prox = self.v_max

        # 5c — combine: min of both (most restrictive), then clamp
        return max(self.v_min, min(v_curv, v_prox))

    # ──────────────────────────────────────────────────────────────────────
    # Step 7 — Predictive collision detection
    # ──────────────────────────────────────────────────────────────────────

    def _collision_ahead(self, vx: float, vy: float, omega: float) -> bool:
        """
        Forward-integrate Mecanum kinematics over t_collision seconds.
        Returns True if any sampled position is in an occupied costmap cell.
        """
        if self._costmap is None:
            return False

        cm  = self._costmap
        res = cm.info.resolution
        if res <= 0.0:
            return False

        ox = cm.info.origin.position.x
        oy = cm.info.origin.position.y
        w  = cm.info.width
        h  = cm.info.height

        speed = math.hypot(vx, vy)
        dt    = res / max(speed, 0.01)
        steps = max(1, int(self.t_collision / dt))
        dt    = self.t_collision / steps

        x   = self._robot_pose.position.x
        y   = self._robot_pose.position.y
        yaw = self._robot_yaw

        for _ in range(steps):
            # Full Mecanum forward integration in world frame
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            x   += (cos_y * vx - sin_y * vy) * dt
            y   += (sin_y * vx + cos_y * vy) * dt
            yaw += omega * dt

            ci = int((x - ox) / res)
            cj = int((y - oy) / res)
            if 0 <= ci < w and 0 <= cj < h:
                if cm.data[cj * w + ci] > 50:
                    return True

        return False

    # ──────────────────────────────────────────────────────────────────────
    # Proximity helper — nearest obstacle distance from costmap
    # ──────────────────────────────────────────────────────────────────────

    def _nearest_obstacle_dist(self) -> float | None:
        """Return distance to nearest occupied costmap cell, or None if no map."""
        if self._costmap is None:
            return None

        cm  = self._costmap
        res = cm.info.resolution
        ox  = cm.info.origin.position.x
        oy  = cm.info.origin.position.y
        w   = cm.info.width
        h   = cm.info.height

        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        ri = int((rx - ox) / res)
        rj = int((ry - oy) / res)

        search_r = max(1, int(self.d_prox / res) + 1)
        min_d = float('inf')

        for dj in range(-search_r, search_r + 1):
            for di in range(-search_r, search_r + 1):
                ci, cj = ri + di, rj + dj
                if 0 <= ci < w and 0 <= cj < h:
                    if cm.data[cj * w + ci] > 50:
                        d = math.hypot(di * res, dj * res)
                        if d < min_d:
                            min_d = d

        return min_d if min_d < float('inf') else None

    # ──────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────

    def _world_to_robot(self, wx: float, wy: float) -> tuple[float, float]:
        """Transform a world-frame point into the robot body frame."""
        dx = wx - self._robot_pose.position.x
        dy = wy - self._robot_pose.position.y
        c  = math.cos(-self._robot_yaw)
        s  = math.sin(-self._robot_yaw)
        return c * dx - s * dy, s * dx + c * dy

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ──────────────────────────────────────────────────────────────────────
    # Message construction
    # ──────────────────────────────────────────────────────────────────────

    def _make_twist(self, vx: float, vy: float, wz: float) -> TwistStamped:
        vx = max(-self.v_max,     min(self.v_max,     vx))
        vy = max(-self.v_y_max,   min(self.v_y_max,   vy))
        wz = max(-self.omega_max, min(self.omega_max, wz))
        cmd = TwistStamped()
        cmd.header = Header(
            frame_id=self.odom_frame,
            stamp=self.get_clock().now().to_msg())
        cmd.twist.linear.x  = vx
        cmd.twist.linear.y  = vy
        cmd.twist.angular.z = wz
        return cmd

    def _publish_zero(self):
        self.cmd_pub.publish(self._make_twist(0.0, 0.0, 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = RegulatedPurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
