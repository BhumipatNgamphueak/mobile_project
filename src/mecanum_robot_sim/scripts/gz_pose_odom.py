#!/usr/bin/env python3
"""
gz_pose_odom.py
===============
Converts Ignition Gazebo ground-truth poses into ROS odometry.

Subscribes to /gz_dynamic_poses (tf2_msgs/TFMessage, bridged from
/world/<world>/dynamic_pose/info) and extracts the mecanum_robot transform.
Converts world-frame pose → odom-frame pose (subtract spawn offset), then
publishes nav_msgs/Odometry on /odom and broadcasts odom→base_link TF.

Because the message only arrives after the robot is actually spawned, path
tracking naturally starts at the right time.

Velocity is estimated by finite-differencing successive poses.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster


class GzPoseOdom(Node):

    def __init__(self):
        super().__init__('gz_pose_odom')

        # Spawn offsets (world → odom frame)
        self.declare_parameter('spawn_x', -7.0)
        self.declare_parameter('spawn_y',  4.0)
        self._ox = self.get_parameter('spawn_x').value
        self._oy = self.get_parameter('spawn_y').value

        self._br  = TransformBroadcaster(self)
        self._pub = self.create_publisher(Odometry, '/odom', 10)

        # Previous pose for velocity estimation
        self._prev_x   = None
        self._prev_y   = None
        self._prev_yaw = None
        self._prev_t   = None

        self.create_subscription(TFMessage, '/gz_dynamic_poses', self._pose_cb, 10)
        self.get_logger().info(
            f'gz_pose_odom ready  spawn_offset=({self._ox}, {self._oy})')

    # ──────────────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: TFMessage):
        for tf in msg.transforms:
            if tf.child_frame_id == 'mecanum_robot':
                self._handle(tf)
                return

    def _handle(self, tf: TransformStamped):
        # World-frame position
        wx = tf.transform.translation.x
        wy = tf.transform.translation.y

        # Quaternion → yaw
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        yaw = 2.0 * math.atan2(qz, qw)

        # Convert to odom frame
        ox = wx - self._ox
        oy = wy - self._oy

        now = self.get_clock().now()
        stamp = now.to_msg()

        # Velocity by finite difference
        vx = vy = omega = 0.0
        if self._prev_t is not None:
            dt = (now - self._prev_t).nanoseconds * 1e-9
            if dt > 0.0:
                # World-frame delta → body-frame velocity
                dx_w = ox - self._prev_x
                dy_w = oy - self._prev_y
                c = math.cos(yaw)
                s = math.sin(yaw)
                vx    = ( c * dx_w + s * dy_w) / dt
                vy    = (-s * dx_w + c * dy_w) / dt
                dyaw  = yaw - self._prev_yaw
                # Wrap to [-π, π]
                dyaw  = math.atan2(math.sin(dyaw), math.cos(dyaw))
                omega = dyaw / dt

        self._prev_x   = ox
        self._prev_y   = oy
        self._prev_yaw = yaw
        self._prev_t   = now

        # Quaternion from yaw
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        # Publish TF odom → base_link
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'
        t.transform.translation.x = ox
        t.transform.translation.y = oy
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy
        self._br.sendTransform(t)

        # Publish Odometry
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x  = ox
        odom.pose.pose.position.y  = oy
        odom.pose.pose.orientation.z = sy
        odom.pose.pose.orientation.w = cy
        odom.twist.twist.linear.x  = vx
        odom.twist.twist.linear.y  = vy
        odom.twist.twist.angular.z = omega
        self._pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = GzPoseOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
