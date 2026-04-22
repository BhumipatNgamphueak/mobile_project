#!/usr/bin/env python3
"""
test_cmd_vel_node.py
====================
Sends a fixed sequence of raw TwistStamped commands to /cmd_vel to verify
that the mecanum robot responds correctly to vx, vy, and omega independently.

Sequence (each phase lasts PHASE_DURATION seconds):
  1. Forward  (vx=+0.3, vy=0, omega=0)
  2. Backward (vx=-0.3, vy=0, omega=0)
  3. Strafe right (vx=0, vy=-0.3, omega=0)   -- body-frame right = negative vy
  4. Strafe left  (vx=0, vy=+0.3, omega=0)
  5. Rotate CW    (vx=0, vy=0, omega=-0.5)
  6. Rotate CCW   (vx=0, vy=0, omega=+0.5)
  7. Diagonal     (vx=+0.3, vy=+0.3, omega=0)
  8. STOP         (all zero)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

PHASE_DURATION = 2.0   # seconds per phase
PUBLISH_HZ     = 20    # Hz

PHASES = [
    ('Forward',       0.3,  0.0,  0.0),
    ('Backward',     -0.3,  0.0,  0.0),
    ('Strafe right',  0.0, -0.3,  0.0),
    ('Strafe left',   0.0,  0.3,  0.0),
    ('Rotate CW',     0.0,  0.0, -0.5),
    ('Rotate CCW',    0.0,  0.0,  0.5),
    ('Diagonal',      0.3,  0.3,  0.0),
    ('STOP',          0.0,  0.0,  0.0),
]


class TestCmdVelNode(Node):
    def __init__(self):
        super().__init__('test_cmd_vel_node')

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        self._phase_idx   = 0
        self._phase_ticks = 0
        self._ticks_per_phase = int(PHASE_DURATION * PUBLISH_HZ)

        self.get_logger().info(f'TestCmdVelNode started — phase 0: {PHASES[0][0]}')

    def _tick(self):
        if self._phase_idx >= len(PHASES):
            return

        name, vx, vy, omega = PHASES[self._phase_idx]

        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x  = vx
        cmd.twist.linear.y  = vy
        cmd.twist.angular.z = omega
        self.pub.publish(cmd)

        self._phase_ticks += 1
        if self._phase_ticks >= self._ticks_per_phase:
            self._phase_idx  += 1
            self._phase_ticks = 0
            if self._phase_idx < len(PHASES):
                next_name, *_ = PHASES[self._phase_idx]
                self.get_logger().info(
                    f'Phase {self._phase_idx}: {next_name}  '
                    f'vx={PHASES[self._phase_idx][1]:.2f} '
                    f'vy={PHASES[self._phase_idx][2]:.2f} '
                    f'omega={PHASES[self._phase_idx][3]:.2f}'
                )
            else:
                self.get_logger().info('All phases done.')


def main(args=None):
    rclpy.init(args=args)
    node = TestCmdVelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
