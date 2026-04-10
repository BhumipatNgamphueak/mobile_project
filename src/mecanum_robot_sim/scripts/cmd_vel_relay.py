#!/usr/bin/env python3
"""
cmd_vel_relay.py
================
Strips the header from /cmd_vel (TwistStamped) and republishes the
inner Twist on /cmd_vel_gz, which is then bridged into Ignition Gazebo
for the VelocityControl plugin.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class CmdVelRelay(Node):

    def __init__(self):
        super().__init__('cmd_vel_relay')
        self._pub = self.create_publisher(Twist, '/cmd_vel_gz', 10)
        self.create_subscription(TwistStamped, '/cmd_vel', self._cb, 10)
        self.get_logger().info('cmd_vel_relay ready')

    def _cb(self, msg: TwistStamped):
        # VelocityControl applies body-frame y opposite to ROS convention —
        # negate lateral velocity so positive vy moves the robot left (north).
        t = msg.twist
        out = Twist()
        out.linear.x  =  t.linear.x
        out.linear.y  = -t.linear.y
        out.linear.z  =  t.linear.z
        out.angular.x =  t.angular.x
        out.angular.y =  t.angular.y
        out.angular.z =  t.angular.z
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
