#!/usr/bin/env python3
"""
odom_tf_broadcaster.py
======================
Re-publishes the /odom nav_msgs/Odometry as an odom → base_link
TF transform so that RViz and other tools can track the robot pose.

The mecanum_drive_controller reports odometry correctly on the /odom
topic but does not publish the corresponding TF in this installation.
This node bridges that gap.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTFBroadcaster(Node):

    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self._br = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.get_logger().info('odom_tf_broadcaster ready')

    def _odom_cb(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp    = msg.header.stamp
        t.header.frame_id = msg.header.frame_id          # 'odom'
        t.child_frame_id  = msg.child_frame_id            # 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation
        self._br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
