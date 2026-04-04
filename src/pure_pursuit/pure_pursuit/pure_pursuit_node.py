#!/usr/bin/env python3
"""
pure_pursuit_node.py  —  SKELETON
=====================================
Track the planned path using a Pure Pursuit (or similar) controller
and publish velocity commands to the mecanum drive controller.

The mecanum_drive_controller expects geometry_msgs/TwistStamped on
/cmd_vel (remapped from /mecanum_drive_controller/reference).

Inputs
------
/planned_path   nav_msgs/Path        path from path_planning_node
/odom           nav_msgs/Odometry    current robot pose + velocity

Outputs
-------
/cmd_vel        geometry_msgs/TwistStamped   velocity command (TwistStamped!)

Parameters
----------
lookahead_dist   (float, default 1.0)    pure pursuit lookahead distance [m]
max_linear_vel   (float, default 0.5)    max forward speed [m/s]
max_angular_vel  (float, default 1.0)    max yaw rate [rad/s]
max_lateral_vel  (float, default 0.5)    max lateral speed [m/s]  ← mecanum only
goal_tolerance   (float, default 0.15)   stop when closer than this [m]
control_hz       (float, default 20.0)   control loop frequency [Hz]
odom_frame       (str,   default 'odom')
use_sim_time     (bool,  default True)

Notes
-----
- IMPORTANT: /cmd_vel must be TwistStamped, NOT Twist.
  The mecanum_drive_controller was launched with the reference remapped
  to /cmd_vel and expects stamped messages.
- The mecanum robot can move sideways (linear.y ≠ 0).  A standard Pure
  Pursuit only controls linear.x and angular.z.  Consider extending it
  to also command linear.y for smoother lateral tracking.
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from std_msgs.msg import Header


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('lookahead_dist',  1.0)
        self.declare_parameter('max_linear_vel',  0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('max_lateral_vel', 0.5)
        self.declare_parameter('goal_tolerance',  0.15)
        self.declare_parameter('control_hz',      20.0)
        self.declare_parameter('odom_frame',      'odom')

        self.lookahead_dist  = self.get_parameter('lookahead_dist').value
        self.max_linear_vel  = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.max_lateral_vel = self.get_parameter('max_lateral_vel').value
        self.goal_tolerance  = self.get_parameter('goal_tolerance').value
        control_hz           = self.get_parameter('control_hz').value
        self.odom_frame      = self.get_parameter('odom_frame').value

        # ── Subscribers ───────────────────────────────────────────────────
        self.path_sub = self.create_subscription(
            Path, '/planned_path',
            self._path_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 10)

        # ── Publisher ─────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._path:       list[PoseStamped] = []
        self._robot_pose: Pose | None       = None
        self._robot_yaw:  float             = 0.0

        # ── Control loop timer ────────────────────────────────────────────
        self._control_timer = self.create_timer(
            1.0 / control_hz, self._control_loop)

        self.get_logger().info('PurePursuitNode started (skeleton)')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _path_callback(self, msg: Path):
        self._path = msg.poses
        self.get_logger().debug(f'New path received: {len(self._path)} poses')

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose
        self._robot_yaw  = self._yaw_from_quaternion(msg.pose.pose.orientation)

    # ──────────────────────────────────────────────────────────────────────
    # Control loop  ← IMPLEMENT THIS
    # ──────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        """
        Main control loop, called at control_hz.

        Pure Pursuit algorithm (classic)
        ---------------------------------
        1. Find the lookahead point: the first path point that is at least
           `lookahead_dist` metres ahead along the path.
        2. Transform the lookahead point into the robot's local frame.
        3. Compute curvature κ = 2·y_local / L²  (L = lookahead_dist).
        4. angular_vel = v · κ
        5. Publish TwistStamped.

        Mecanum extension
        -----------------
        The robot can also move sideways.  After computing the angular
        command, you can set linear.y proportional to the cross-track error
        to reduce lateral drift without large yaw corrections.

        Stop condition
        --------------
        If the robot is within goal_tolerance of the last path pose, stop.
        """
        if self._robot_pose is None or not self._path:
            self._publish_zero()
            return

        # Check if goal reached
        goal = self._path[-1].pose
        dist_to_goal = math.hypot(
            goal.position.x - self._robot_pose.position.x,
            goal.position.y - self._robot_pose.position.y)

        if dist_to_goal < self.goal_tolerance:
            self._publish_zero()
            self.get_logger().info('Goal reached', throttle_duration_sec=2.0)
            return

        # TODO: replace stub with real Pure Pursuit (or other) controller
        cmd = self._stub_control()
        self.cmd_pub.publish(cmd)

    def _stub_control(self) -> TwistStamped:
        """
        Stub controller — publishes zero velocity.
        Replace with your Pure Pursuit implementation.
        """
        return self._make_twist(0.0, 0.0, 0.0)

    # ──────────────────────────────────────────────────────────────────────
    # Helper — lookahead search
    # ──────────────────────────────────────────────────────────────────────

    def _find_lookahead_point(self) -> tuple[float, float] | None:
        """
        Walk the path from the closest point forward until a pose at least
        `lookahead_dist` away is found.

        Returns (x, y) of the lookahead point in the odom frame, or None if
        the path is shorter than lookahead_dist.
        """
        if self._robot_pose is None or not self._path:
            return None

        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y

        # Find closest path index
        min_dist = float('inf')
        closest_idx = 0
        for i, ps in enumerate(self._path):
            d = math.hypot(ps.pose.position.x - rx, ps.pose.position.y - ry)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Walk forward until lookahead distance
        for i in range(closest_idx, len(self._path)):
            d = math.hypot(self._path[i].pose.position.x - rx,
                           self._path[i].pose.position.y - ry)
            if d >= self.lookahead_dist:
                return self._path[i].pose.position.x, self._path[i].pose.position.y

        # Lookahead beyond end of path → use last point
        last = self._path[-1].pose.position
        return last.x, last.y

    def _world_to_robot(self, wx: float, wy: float) -> tuple[float, float]:
        """
        Transform world (odom) coordinates into robot body frame.

        Returns (x_local, y_local) where x_local is forward, y_local is left.
        """
        dx = wx - self._robot_pose.position.x
        dy = wy - self._robot_pose.position.y
        cos_yaw = math.cos(-self._robot_yaw)
        sin_yaw = math.sin(-self._robot_yaw)
        x_local =  cos_yaw * dx - sin_yaw * dy
        y_local =  sin_yaw * dx + cos_yaw * dy
        return x_local, y_local

    # ──────────────────────────────────────────────────────────────────────
    # Helper — message constructors
    # ──────────────────────────────────────────────────────────────────────

    def _make_twist(self, vx: float, vy: float, wz: float) -> TwistStamped:
        """Build a TwistStamped with clamped velocities."""
        vx = max(-self.max_linear_vel,  min(self.max_linear_vel,  vx))
        vy = max(-self.max_lateral_vel, min(self.max_lateral_vel, vy))
        wz = max(-self.max_angular_vel, min(self.max_angular_vel, wz))

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

    # ──────────────────────────────────────────────────────────────────────
    # Helper — quaternion → yaw
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        """Extract yaw (rotation about Z) from a geometry_msgs Quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
