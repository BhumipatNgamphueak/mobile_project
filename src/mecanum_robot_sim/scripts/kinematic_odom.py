#!/usr/bin/env python3
"""
kinematic_odom.py
=================
Integrates /cmd_vel (TwistStamped) to produce nav_msgs/Odometry on
/odom and the odom → base_link TF transform.

Because Ignition's VelocityControl plugin applies the commanded body
velocity exactly (kinematic, no wheel-slip), integrating cmd_vel gives
accurate odometry without needing wheel encoder feedback.

The integration uses a fixed-rate timer so that zero-velocity periods
(when no cmd_vel arrives) are handled correctly.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class KinematicOdom(Node):

    def __init__(self):
        super().__init__('kinematic_odom')

        self.declare_parameter('publish_hz', 50.0)
        hz = self.get_parameter('publish_hz').value

        # Current pose (odom frame)
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0

        # Last commanded velocity (body frame)
        self._vx    = 0.0
        self._vy    = 0.0
        self._omega = 0.0

        self._last_t = None   # rclpy.Time

        self._br  = TransformBroadcaster(self)
        self._pub = self.create_publisher(Odometry, '/odom', 10)

        self.create_subscription(TwistStamped, '/cmd_vel', self._cmd_cb, 10)
        self.create_timer(1.0 / hz, self._update)

        self.get_logger().info('kinematic_odom ready')

    # ──────────────────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: TwistStamped):
        self._vx    = msg.twist.linear.x
        self._vy    = msg.twist.linear.y
        self._omega = msg.twist.angular.z

    def _update(self):
        now = self.get_clock().now()

        if self._last_t is None:
            self._last_t = now
            return

        dt = (now - self._last_t).nanoseconds * 1e-9
        self._last_t = now

        if dt <= 0.0:
            return

        # Euler integration in world frame
        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        self._x   += (c * self._vx - s * self._vy) * dt
        self._y   += (s * self._vx + c * self._vy) * dt
        self._yaw += self._omega * dt

        # Quaternion from yaw
        cy = math.cos(self._yaw * 0.5)
        sy = math.sin(self._yaw * 0.5)

        stamp = now.to_msg()

        # Publish TF
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = self._x
        tf.transform.translation.y = self._y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = sy
        tf.transform.rotation.w = cy
        self._br.sendTransform(tf)

        # Publish Odometry
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x  = self._x
        odom.pose.pose.position.y  = self._y
        odom.pose.pose.orientation.z = sy
        odom.pose.pose.orientation.w = cy
        odom.twist.twist.linear.x  = self._vx
        odom.twist.twist.linear.y  = self._vy
        odom.twist.twist.angular.z = self._omega
        self._pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = KinematicOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
