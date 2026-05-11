#!/usr/bin/env python3
"""
human_marker_publisher.py

Computes human actor positions analytically from the SDF trajectory waypoints
using /clock (sim time).  Ignition Fortress does NOT publish actor poses in
any world pose topic, so reading the clock and re-playing the known trajectory
is the only reliable method.

Publishes:
  /visualization/humans          MarkerArray – cylinder body + label + velocity arrow
  /visualization/human_paths     MarkerArray – predicted future dots (3 s look-ahead)
  /visualization/robot_trail     nav_msgs/Path – robot odometry history
  /visualization/obstacles       MarkerArray – static obstacle outlines
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


# ── Human trajectory definitions (mirrors crossing_humans.sdf exactly) ────
# Each waypoint: (time_s, world_x, world_y)
# delay_start = 1.0 s for both actors; loop period = 28.6 s
_H1_WP = [
    ( 0.0, -8.0,  1.0),
    (13.3,  8.0,  1.0),
    (14.3,  8.0,  1.0),
    (27.6, -8.0,  1.0),
    (28.6, -8.0,  1.0),
]
_H2_WP = [
    ( 0.0,  8.0, -1.0),
    (13.3, -8.0, -1.0),
    (14.3, -8.0, -1.0),
    (27.6,  8.0, -1.0),
    (28.6,  8.0, -1.0),
]
DELAY_START = 1.0   # seconds before trajectory begins
LOOP_TIME   = 28.6  # seconds per full loop

HUMANS = {
    'Human 1': {'waypoints': _H1_WP, 'color': (1.0, 0.45, 0.0)},  # orange
    'Human 2': {'waypoints': _H2_WP, 'color': (0.1, 0.55, 1.0)},  # blue
}

# Static obstacles: (world_x, world_y, half_sx, half_sy, half_sz, r, g, b)
OBSTACLES = [
    ( 4.0,  5.0, 1.0, 1.0, 1.0, 0.8, 0.2, 0.2),
    (-4.0, -5.0, 1.0, 1.0, 1.0, 0.2, 0.5, 0.8),
    ( 2.0, -4.0, 1.0, 1.0, 1.0, 0.2, 0.8, 0.3),
]

PREDICT_STEPS = 8     # future prediction dots
PREDICT_DT    = 0.4   # seconds per step  →  3.2 s look-ahead
TRAIL_MAX     = 600


def _interp(waypoints, t: float):
    """Linear interpolation of (x, y) at time t within a waypoint list."""
    if t <= waypoints[0][0]:
        return waypoints[0][1], waypoints[0][2]
    if t >= waypoints[-1][0]:
        return waypoints[-1][1], waypoints[-1][2]
    for i in range(len(waypoints) - 1):
        t0, x0, y0 = waypoints[i]
        t1, x1, y1 = waypoints[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return x0, y0
            alpha = (t - t0) / (t1 - t0)
            return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)
    return waypoints[-1][1], waypoints[-1][2]


def _human_world_pos(waypoints, sim_time_s: float):
    """Return (world_x, world_y) for an actor at sim_time_s."""
    effective = max(0.0, sim_time_s - DELAY_START)
    t = effective % LOOP_TIME
    return _interp(waypoints, t)


def _human_velocity(waypoints, sim_time_s: float, dt: float = 0.1):
    """Finite-difference velocity estimate (world frame)."""
    x0, y0 = _human_world_pos(waypoints, sim_time_s - dt)
    x1, y1 = _human_world_pos(waypoints, sim_time_s + dt)
    return (x1 - x0) / (2 * dt), (y1 - y0) / (2 * dt)


class HumanMarkerPublisher(Node):

    def __init__(self):
        super().__init__('human_marker_publisher')

        # spawn offset: robot spawn in world coords  (odom origin = world spawn)
        self.spawn_x = self.declare_parameter('spawn_x', -7.0).value
        self.spawn_y = self.declare_parameter('spawn_y',  4.0).value

        self._sim_time_s: float = 0.0
        self._trail = Path()
        self._trail.header.frame_id = 'odom'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.create_subscription(Clock,    '/clock', self._clock_cb, qos)
        self.create_subscription(Odometry, '/odom',  self._odom_cb, 10)

        self._pub_humans  = self.create_publisher(MarkerArray, '/visualization/humans',     10)
        self._pub_hpaths  = self.create_publisher(MarkerArray, '/visualization/human_paths', 10)
        self._pub_trail   = self.create_publisher(Path,        '/visualization/robot_trail', 10)
        self._pub_obs     = self.create_publisher(MarkerArray, '/visualization/obstacles',   10)

        self.create_timer(0.1, self._publish_humans)   # 10 Hz
        self.create_timer(1.0, self._publish_obstacles)
        self.get_logger().info(
            f'human_marker_publisher ready  (spawn offset: {self.spawn_x}, {self.spawn_y})')

    # ── helpers ────────────────────────────────────────────────────────────

    def _w2o(self, wx: float, wy: float):
        """World → odom frame."""
        return wx - self.spawn_x, wy - self.spawn_y

    # ── callbacks ──────────────────────────────────────────────────────────

    def _clock_cb(self, msg: Clock):
        self._sim_time_s = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _odom_cb(self, msg: Odometry):
        now = self.get_clock().now().to_msg()
        ps = PoseStamped()
        ps.header.stamp    = now
        ps.header.frame_id = 'odom'
        ps.pose = msg.pose.pose
        self._trail.poses.append(ps)
        if len(self._trail.poses) > TRAIL_MAX:
            self._trail.poses.pop(0)
        self._trail.header.stamp = now
        self._pub_trail.publish(self._trail)

    # ── periodic publish ───────────────────────────────────────────────────

    def _publish_humans(self):
        now   = self.get_clock().now().to_msg()
        t     = self._sim_time_s
        bodies  = MarkerArray()
        predict = MarkerArray()
        uid = 0

        for name, cfg in HUMANS.items():
            wp    = cfg['waypoints']
            r, g, b = cfg['color']

            wx, wy = _human_world_pos(wp, t)
            ox, oy = self._w2o(wx, wy)
            vx, vy = _human_velocity(wp, t)

            # ── cylinder body ────────────────────────────────────
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = now
            m.ns, m.id = 'human_body', uid;  uid += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = ox
            m.pose.position.y = oy
            m.pose.position.z = 0.9
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.5
            m.scale.z = 1.8
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.85
            m.lifetime = Duration(seconds=0.3).to_msg()
            bodies.markers.append(m)

            # ── name label ──────────────────────────────────────
            lbl = Marker()
            lbl.header = m.header
            lbl.ns, lbl.id = 'human_label', uid;  uid += 1
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = ox
            lbl.pose.position.y = oy
            lbl.pose.position.z = 2.3
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.35
            lbl.color.r, lbl.color.g, lbl.color.b, lbl.color.a = r, g, b, 1.0
            lbl.text = name
            lbl.lifetime = Duration(seconds=0.3).to_msg()
            bodies.markers.append(lbl)

            # ── velocity arrow ──────────────────────────────────
            speed = math.hypot(vx, vy)
            if speed > 0.05:
                yaw = math.atan2(vy, vx)
                arr = Marker()
                arr.header = m.header
                arr.ns, arr.id = 'human_vel', uid;  uid += 1
                arr.type = Marker.ARROW
                arr.action = Marker.ADD
                arr.pose.position.x = ox
                arr.pose.position.y = oy
                arr.pose.position.z = 1.8
                arr.pose.orientation.z = math.sin(yaw / 2)
                arr.pose.orientation.w = math.cos(yaw / 2)
                arr.scale.x = min(speed * 0.7, 1.5)
                arr.scale.y = arr.scale.z = 0.12
                arr.color.r, arr.color.g, arr.color.b, arr.color.a = r, g, b, 0.9
                arr.lifetime = Duration(seconds=0.3).to_msg()
                bodies.markers.append(arr)

            # ── predicted future dots ───────────────────────────
            for step in range(1, PREDICT_STEPS + 1):
                ft  = t + step * PREDICT_DT
                fwx, fwy = _human_world_pos(wp, ft)
                fox, foy = self._w2o(fwx, fwy)
                size = max(0.06, 0.28 - step * 0.025)
                fade = max(0.10, 0.80 - step * 0.09)
                sp = Marker()
                sp.header = m.header
                sp.ns, sp.id = 'human_pred', uid;  uid += 1
                sp.type = Marker.SPHERE
                sp.action = Marker.ADD
                sp.pose.position.x = fox
                sp.pose.position.y = foy
                sp.pose.position.z = 0.15
                sp.pose.orientation.w = 1.0
                sp.scale.x = sp.scale.y = sp.scale.z = size
                sp.color.r, sp.color.g, sp.color.b, sp.color.a = r, g, b, fade
                sp.lifetime = Duration(seconds=0.3).to_msg()
                predict.markers.append(sp)

        self._pub_humans.publish(bodies)
        self._pub_hpaths.publish(predict)

    def _publish_obstacles(self):
        arr   = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (wx, wy, hx, hy, hz, r, g, b) in enumerate(OBSTACLES):
            ox, oy = self._w2o(wx, wy)
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = stamp
            m.ns, m.id = 'obstacles', i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = ox
            m.pose.position.y = oy
            m.pose.position.z = hz
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = hx*2, hy*2, hz*2
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.40
            arr.markers.append(m)
        self._pub_obs.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = HumanMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
