#!/usr/bin/env python3
"""
path_evaluator.py
=================
Evaluates the Regulated Pure Pursuit controller by comparing the desired
path with the robot's actual trajectory.

RViz output
-----------
/visualization/actual_path   nav_msgs/Path   actual trajectory (red in RViz)

The desired path is already visible via /planned_path (blue).

Graph output  (saved when goal is reached or Ctrl+C)
------------
  /tmp/rpp_evaluation_<timestamp>.png

  Panel 1 (top-left)  : XY plot — desired path vs actual path
  Panel 2 (top-right) : Cross-track error over time
  Panel 3 (bottom-left): Linear speed (v_x) over time
  Panel 4 (bottom-right): Angular velocity over time

Parameters
----------
  goal_tolerance  (float, 0.15)  metres — stop recording when this close to goal
  use_sim_time    (bool,  true)
"""

import math
import os
import time as _time
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Header


class PathEvaluatorNode(Node):

    def __init__(self):
        super().__init__('path_evaluator')

        self.declare_parameter('goal_tolerance', 0.15)
        self._goal_tol = self.get_parameter('goal_tolerance').value

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Path,         '/planned_path', self._path_cb,  10)
        self.create_subscription(Odometry,     '/odom',         self._odom_cb,  50)
        self.create_subscription(TwistStamped, '/cmd_vel',      self._cmd_cb,   50)

        # ── Publisher — actual path for RViz ──────────────────────────────
        self._actual_pub = self.create_publisher(Path, '/visualization/actual_path', 10)

        # ── State ─────────────────────────────────────────────────────────
        self._desired:    list[tuple[float, float]]            = []  # (x, y)
        self._actual:     list[tuple[float, float, float]]     = []  # (x, y, t)
        self._cmds:       list[tuple[float, float, float, float]] = []  # (t, vx, vy, wz)
        self._recording:  bool              = False
        self._t0:         float | None      = None
        self._goal:       tuple[float, float] | None = None
        self._saved:      bool              = False

        self._actual_path_msg        = Path()
        self._actual_path_msg.header.frame_id = 'odom'

        self.get_logger().info('PathEvaluatorNode ready — waiting for /planned_path')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        if not msg.poses:
            return

        new_goal = (msg.poses[-1].pose.position.x,
                    msg.poses[-1].pose.position.y)

        # Only reset if goal changed (test_path_publisher republishes same path at 1 Hz)
        if self._goal is not None:
            if math.hypot(new_goal[0] - self._goal[0],
                          new_goal[1] - self._goal[1]) < 0.05:
                return   # same path — ignore

        self._goal    = new_goal
        self._desired = [(ps.pose.position.x, ps.pose.position.y)
                         for ps in msg.poses]
        self._actual.clear()
        self._cmds.clear()
        self._actual_path_msg.poses.clear()
        self._t0       = None
        self._recording = True
        self._saved     = False
        self.get_logger().info(
            f'Recording started — goal ({new_goal[0]:.2f}, {new_goal[1]:.2f}), '
            f'{len(self._desired)} desired poses')

    def _odom_cb(self, msg: Odometry):
        if not self._recording:
            return

        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = t

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._actual.append((x, y, t - self._t0))

        # Update RViz actual path
        ps = PoseStamped()
        ps.header.stamp    = stamp
        ps.header.frame_id = 'odom'
        ps.pose = msg.pose.pose
        self._actual_path_msg.poses.append(ps)
        self._actual_path_msg.header.stamp = stamp
        self._actual_pub.publish(self._actual_path_msg)

        # Check goal
        if self._goal and not self._saved:
            if math.hypot(x - self._goal[0], y - self._goal[1]) < self._goal_tol:
                self.get_logger().info('Goal reached — saving evaluation plots')
                self._save_plots()
                self._recording = False

    def _cmd_cb(self, msg: TwistStamped):
        if not self._recording or self._t0 is None:
            return
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9 - self._t0
        self._cmds.append((t,
                            msg.twist.linear.x,
                            msg.twist.linear.y,
                            msg.twist.angular.z))

    # ──────────────────────────────────────────────────────────────────────
    # Cross-track error helper
    # ──────────────────────────────────────────────────────────────────────

    def _cross_track_error(self, x: float, y: float) -> float:
        """Minimum perpendicular distance from (x, y) to the desired path."""
        if len(self._desired) < 2:
            return 0.0
        min_d = float('inf')
        for i in range(len(self._desired) - 1):
            x0, y0 = self._desired[i]
            x1, y1 = self._desired[i + 1]
            dx, dy = x1 - x0, y1 - y0
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                d = math.hypot(x - x0, y - y0)
            else:
                t  = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / l2))
                px = x0 + t * dx
                py = y0 + t * dy
                d  = math.hypot(x - px, y - py)
            if d < min_d:
                min_d = d
        return min_d

    # ──────────────────────────────────────────────────────────────────────
    # Plot generation
    # ──────────────────────────────────────────────────────────────────────

    def _save_plots(self):
        if len(self._actual) < 5:
            self.get_logger().warn('Too few data points to plot')
            return

        try:
            import matplotlib
            matplotlib.use('Agg')       # headless — no display needed
            import matplotlib.pyplot as plt
        except ImportError:
            self.get_logger().error('matplotlib not installed — cannot save plots')
            return

        # ── Unpack data ───────────────────────────────────────────────────
        act_x  = [p[0] for p in self._actual]
        act_y  = [p[1] for p in self._actual]
        act_t  = [p[2] for p in self._actual]
        des_x  = [p[0] for p in self._desired]
        des_y  = [p[1] for p in self._desired]
        cte    = [self._cross_track_error(p[0], p[1]) for p in self._actual]
        cmd_t  = [c[0] for c in self._cmds]
        cmd_vx = [c[1] for c in self._cmds]
        cmd_vy = [c[2] for c in self._cmds]
        cmd_wz = [c[3] for c in self._cmds]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Regulated Pure Pursuit — Evaluation', fontsize=14, fontweight='bold')

        # ── Panel 1: XY trajectory ────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(des_x, des_y, '--',  color='#2196F3', linewidth=2.0,
                label='Desired path')
        ax.plot(act_x, act_y, '-',   color='#F44336', linewidth=1.5,
                label='Actual path')
        ax.plot(des_x[0],  des_y[0],  'gs', markersize=8, label='Start')
        ax.plot(des_x[-1], des_y[-1], 'r*', markersize=12, label='Goal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('XY Trajectory')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # ── Panel 2: Cross-track error ────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(act_t, cte, color='#FF9800', linewidth=1.5)
        ax.axhline(y=sum(cte) / len(cte), color='gray', linestyle='--',
                   linewidth=1.0, label=f'Mean: {sum(cte)/len(cte):.3f} m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cross-track error (m)')
        ax.set_title('Cross-Track Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # ── Panel 3: Linear speed ─────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(cmd_t, cmd_vx, color='#4CAF50', linewidth=1.5, label='v_x (forward)')
        ax.plot(cmd_t, cmd_vy, color='#9C27B0', linewidth=1.2,
                label='v_y (lateral)', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Commanded Linear Velocity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Panel 4: Angular velocity ─────────────────────────────────────
        ax = axes[1, 1]
        ax.plot(cmd_t, cmd_wz, color='#795548', linewidth=1.5)
        ax.axhline(y=0, color='gray', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ω (rad/s)')
        ax.set_title('Commanded Angular Velocity')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = int(_time.time())
        out_path  = f'/tmp/rpp_evaluation_{timestamp}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        max_cte  = max(cte)
        mean_cte = sum(cte) / len(cte)
        duration = act_t[-1] if act_t else 0.0
        self.get_logger().info(
            f'Plot saved → {out_path}\n'
            f'  Duration : {duration:.1f} s\n'
            f'  CTE mean : {mean_cte:.3f} m\n'
            f'  CTE max  : {max_cte:.3f} m\n'
            f'  Points   : {len(self._actual)} odom, {len(self._cmds)} cmd_vel')
        self._saved = True

    def destroy_node(self):
        if self._recording and not self._saved and len(self._actual) > 5:
            self.get_logger().info('Shutdown detected — saving partial evaluation plots')
            self._save_plots()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
