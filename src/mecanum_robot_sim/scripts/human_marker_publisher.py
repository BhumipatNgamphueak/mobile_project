#!/usr/bin/env python3
"""
human_marker_publisher.py
=========================
Ground-truth world visualizer.  Parses the active world SDF once at
startup for static geometry, then renders humans from real Gazebo
ground-truth pose data.

Sources
  * Walls + obstacles  ← SDF <model>/<pose> + <geometry>     (static, parsed once)
  * Humans (primary)   ← /gz_dynamic_poses                   (live; kinematic
                                                              human models are
                                                              physics-bound and
                                                              get teleported by
                                                              human_controller)
  * Humans (fallback)  ← SDF <waypoint> linear interpolation (only if no live
                                                              pose seen in 0.5 s)
  * Robot trail        ← /odom (from gz_pose_odom)

Coordinate frames
  Gazebo world frame origin = arena centre (0, 0)
  RViz fixed frame `odom`   = robot spawn point (spawn_x, spawn_y)
  world→odom                = subtract spawn offset

Topics published
  /visualization/world        MarkerArray   walls + obstacles + humans
  /visualization/human_paths  MarkerArray   short-horizon velocity prediction
  /visualization/robot_trail  Path          robot path history (odom frame)
"""

import math
import os
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point, PoseStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray


# Cycled in name order so each human gets a distinct, stable colour
_PALETTE = [
    (1.00, 0.45, 0.00),
    (0.10, 0.55, 1.00),
    (1.00, 0.20, 0.20),
    (0.00, 0.80, 0.30),
    (0.80, 0.00, 0.80),
    (0.85, 0.80, 0.00),
    (0.00, 0.80, 0.80),
    (0.95, 0.50, 0.10),
]

PREDICT_STEPS = 8
PREDICT_DT    = 0.4   # seconds per step → ~3.2 s horizon
TRAIL_MAX     = 600


# ──────────────────────────────────────────────────────────────────────────
#  SDF parsing
# ──────────────────────────────────────────────────────────────────────────

def _floats(text: str) -> list:
    return [float(p) for p in text.strip().split()]


def _classify(name: str) -> str:
    n = name.lower()
    if 'ground' in n or 'plane' in n:
        return 'skip'
    if 'mecanum' in n or n == 'robot':
        return 'skip'
    if n.startswith('human_') or n == 'human':
        return 'skip'   # humans are tracked separately via /gz_dynamic_poses
    if n.startswith('wall_') or n == 'wall' or n.endswith('_wall'):
        return 'wall'
    return 'obstacle'


def _color_from_material(material) -> tuple:
    """Pull RGB from <material><diffuse> or <ambient>; bump dim values."""
    if material is None:
        return 0.55, 0.55, 0.55
    for tag in ('diffuse', 'ambient'):
        el = material.find(tag)
        if el is not None and el.text:
            vals = _floats(el.text)
            if len(vals) >= 3:
                r, g, b = vals[0], vals[1], vals[2]
                # Brighten very dark colours so RViz markers are readable
                m = max(r, g, b)
                if m < 0.35:
                    s = 0.55 / m if m > 0 else 1.0
                    r, g, b = min(1.0, r * s), min(1.0, g * s), min(1.0, b * s)
                return r, g, b
    return 0.55, 0.55, 0.55


def _parse_actor_trajectory(actor) -> dict:
    """
    Extract full trajectory data:
      {
        'delay':     float,                     # delay_start
        'loop':      bool,                      # script/loop
        'waypoints': [(t, x, y), ...] sorted    # by time
      }
    """
    delay = 0.0
    loop  = True
    script = actor.find('script')
    if script is not None:
        d = script.find('delay_start')
        if d is not None and d.text:
            try:    delay = float(d.text)
            except ValueError: pass
        l = script.find('loop')
        if l is not None and l.text:
            loop = l.text.strip().lower() in ('1', 'true')

    waypoints = []
    for wp in actor.iter('waypoint'):
        t_el = wp.find('time')
        p_el = wp.find('pose')
        if t_el is None or p_el is None or not p_el.text:
            continue
        try:
            t = float(t_el.text)
            xs = _floats(p_el.text)
            if len(xs) >= 2:
                waypoints.append((t, xs[0], xs[1]))
        except ValueError:
            continue
    waypoints.sort(key=lambda w: w[0])
    return {'delay': delay, 'loop': loop, 'waypoints': waypoints}


def _interp_trajectory(traj: dict, sim_t: float) -> tuple:
    """Position (x, y) at sim time `sim_t` along the trajectory."""
    wps = traj['waypoints']
    if not wps:
        return 0.0, 0.0

    rel = sim_t - traj['delay']
    if rel <= wps[0][0]:
        return wps[0][1], wps[0][2]

    period = wps[-1][0] - wps[0][0]
    if traj['loop'] and period > 0.0:
        rel = wps[0][0] + ((rel - wps[0][0]) % period)
    elif rel >= wps[-1][0]:
        return wps[-1][1], wps[-1][2]

    for i in range(len(wps) - 1):
        t0, x0, y0 = wps[i]
        t1, x1, y1 = wps[i + 1]
        if t0 <= rel <= t1:
            dt = t1 - t0
            if dt <= 0.0:
                return x0, y0
            a = (rel - t0) / dt
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return wps[-1][1], wps[-1][2]


def parse_world_sdf(sdf_path: str) -> dict:
    """
    Returns:
      {
        'name':    <world name>,
        'statics': [(name, kind, shape, x, y, z, sx, sy, sz, r, g, b), ...],
        'actors':  {name: (init_x, init_y), ...},
      }
    where (sx, sy, sz) are full sizes; (x, y, z) is centre.
    """
    out = {'name': '', 'statics': [], 'actors': {}}

    try:
        root = ET.parse(sdf_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return out

    world_el = root.find('world')
    if world_el is None:
        return out
    out['name'] = world_el.get('name', '')

    # Static models
    for model in world_el.findall('model'):
        name = model.get('name', '')
        kind = _classify(name)
        if kind == 'skip':
            continue

        pose_el = model.find('pose')
        if pose_el is None or not pose_el.text:
            x, y, z = 0.0, 0.0, 0.0
        else:
            xs = _floats(pose_el.text)
            x = xs[0] if len(xs) > 0 else 0.0
            y = xs[1] if len(xs) > 1 else 0.0
            z = xs[2] if len(xs) > 2 else 0.0

        link = model.find('link')
        if link is None:
            continue

        visual = link.find('visual') or link.find('collision')
        if visual is None:
            continue
        geom = visual.find('geometry')
        if geom is None:
            continue

        shape = None
        sx = sy = sz = 1.0
        box = geom.find('box')
        cyl = geom.find('cylinder')
        sph = geom.find('sphere')

        if box is not None:
            size_el = box.find('size')
            if size_el is None or not size_el.text:
                continue
            vals = _floats(size_el.text)
            if len(vals) < 3:
                continue
            sx, sy, sz = vals[0], vals[1], vals[2]
            shape = 'box'
        elif cyl is not None:
            r_el = cyl.find('radius')
            l_el = cyl.find('length')
            if r_el is None or l_el is None:
                continue
            r = float(r_el.text)
            L = float(l_el.text)
            sx = sy = 2.0 * r
            sz = L
            shape = 'cylinder'
        elif sph is not None:
            r_el = sph.find('radius')
            if r_el is None:
                continue
            r = float(r_el.text)
            sx = sy = sz = 2.0 * r
            shape = 'sphere'
        else:
            continue

        col = _color_from_material(visual.find('material'))
        out['statics'].append((name, kind, shape, x, y, z, sx, sy, sz, *col))

    # Humans:
    #  (a) New format: <model name="human_*"> with <plugin filename="__waypoints__">
    #  (b) Legacy format: <actor name="human_*">
    for model in world_el.findall('model'):
        name = model.get('name', '')
        if not name.lower().startswith('human'):
            continue
        wp_plugin = None
        for plugin in model.findall('plugin'):
            if plugin.get('filename') == '__waypoints__':
                wp_plugin = plugin
                break
        if wp_plugin is None:
            # Human model without waypoint plugin — seed at its <pose>
            pose_el = model.find('pose')
            if pose_el is not None and pose_el.text:
                xs = _floats(pose_el.text)
                if len(xs) >= 2:
                    out['actors'][name] = {
                        'delay': 0.0, 'loop': False,
                        'waypoints': [(0.0, xs[0], xs[1])]}
            continue
        # Reuse the actor trajectory parser by adapting plugin children
        delay = 0.0
        loop  = True
        d = wp_plugin.find('delay_start')
        if d is not None and d.text:
            try:    delay = float(d.text)
            except ValueError: pass
        l = wp_plugin.find('loop')
        if l is not None and l.text:
            loop = l.text.strip().lower() in ('1', 'true')
        wps = []
        for wp in wp_plugin.findall('waypoint'):
            t_el = wp.find('time')
            p_el = wp.find('pose')
            if t_el is None or p_el is None or not p_el.text:
                continue
            try:
                t = float(t_el.text)
                xs = _floats(p_el.text)
                if len(xs) >= 2:
                    wps.append((t, xs[0], xs[1]))
            except ValueError:
                continue
        wps.sort(key=lambda w: w[0])
        out['actors'][name] = {'delay': delay, 'loop': loop, 'waypoints': wps}

    for actor in world_el.findall('actor'):
        name = actor.get('name', '')
        if not name or name in out['actors']:
            continue
        # `*_anim` actors are visual-only siblings of kinematic models in
        # the hybrid SDF format — the kinematic model has already been
        # registered above, and tracking the actor too would double-count.
        if name.endswith('_anim'):
            continue
        out['actors'][name] = _parse_actor_trajectory(actor)

    return out


# ──────────────────────────────────────────────────────────────────────────
#  Node
# ──────────────────────────────────────────────────────────────────────────

class HumanMarkerPublisher(Node):

    def __init__(self):
        super().__init__('human_marker_publisher')

        self._spawn_x    = float(self.declare_parameter('spawn_x',     -8.0).value)
        self._spawn_y    = float(self.declare_parameter('spawn_y',      0.0).value)
        self._time_scale = float(self.declare_parameter('time_scale',   1.0).value)
        world            = self.declare_parameter('world', 'crossing_humans').value

        share = get_package_share_directory('mecanum_robot_sim')
        sdf_path = os.path.join(share, 'worlds', f'{world}.sdf')

        self._world = parse_world_sdf(sdf_path)
        self.get_logger().info(
            f'Loaded "{world}" from {sdf_path}: '
            f'{len(self._world["statics"])} statics, '
            f'{len(self._world["actors"])} actors')
        for s in self._world['statics']:
            self.get_logger().info(
                f'  {s[1]:8s} {s[0]:14s} pos=({s[3]:+.1f},{s[4]:+.1f},{s[5]:+.1f}) '
                f'size=({s[6]:.1f},{s[7]:.1f},{s[8]:.1f})')
        for n, traj in self._world['actors'].items():
            wp = traj['waypoints']
            self.get_logger().info(
                f'  actor    {n:14s} waypoints={len(wp)}  '
                f'delay={traj["delay"]:.1f}s  loop={traj["loop"]}  '
                f'period={(wp[-1][0] - wp[0][0]) if len(wp) >= 2 else 0:.1f}s')

        # Live actor state (world frame, computed from SDF trajectory + sim time)
        self._actor_pos:   dict[str, tuple] = {}  # name → (wx, wy)
        self._actor_vel:   dict[str, tuple] = {}  # name → (vx, vy)
        self._actor_prev:  dict[str, tuple] = {}  # name → (wx, wy, sim_t)
        self._actor_color: dict[str, tuple] = {}
        for i, n in enumerate(sorted(self._world['actors'])):
            self._actor_color[n] = _PALETTE[i % len(_PALETTE)]

        self.get_logger().info(
            f'time_scale={self._time_scale:.2f}  '
            f'(<1 slower, >1 faster; tune to match Gazebo actor pace)')

        # Live ground truth from /gz_dynamic_poses (kinematic human models
        # are physics-bound, so they appear here once human_controller
        # starts teleporting them).  Falls back to SDF interpolation when
        # no live pose has arrived yet for a given human.
        self._actor_live_t: dict[str, float] = {}   # name → last sim_t we saw it

        # Robot trail
        self._trail = Path()
        self._trail.header.frame_id = 'odom'

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.create_subscription(TFMessage, '/gz_dynamic_poses',
                                 self._dynamic_poses_cb, qos_be)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self._pub_world  = self.create_publisher(MarkerArray, '/visualization/world',       10)
        self._pub_paths  = self.create_publisher(MarkerArray, '/visualization/human_paths', 10)
        self._pub_trail  = self.create_publisher(Path,        '/visualization/robot_trail', 10)

        self.create_timer(0.05, self._update_actors)   # 20 Hz
        self.create_timer(0.1,  self._publish)         # 10 Hz
        self.create_timer(2.0,  self._log)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _w2o(self, wx: float, wy: float) -> tuple:
        return wx - self._spawn_x, wy - self._spawn_y

    def _sim_seconds(self) -> float | None:
        """
        Sim time in seconds, taken straight from the node's clock (which is
        Gazebo's /clock when use_sim_time=True).  Wall-time leakage before
        /clock has connected is filtered out by the upper bound check.
        """
        now_msg = self.get_clock().now().to_msg()
        t = now_msg.sec + now_msg.nanosec * 1e-9
        # 0 = clock not yet connected;  >1e8 ≈ 3 yrs → must be wall time
        if t <= 0.0 or t > 1.0e8:
            return None
        return t

    def _dynamic_poses_cb(self, msg: TFMessage) -> None:
        """
        Real ground-truth poses from Gazebo (kinematic human models appear
        here once human_controller starts teleporting them).  Updates
        position + finite-difference velocity for any tracked actor name.
        """
        sim_t = self._sim_seconds()
        if sim_t is None:
            return
        for tf in msg.transforms:
            name = tf.child_frame_id
            if not name or name not in self._actor_color:
                continue
            wx = tf.transform.translation.x
            wy = tf.transform.translation.y
            if name in self._actor_prev:
                px, py, pt = self._actor_prev[name]
                dt = sim_t - pt
                if dt > 0.05:
                    self._actor_vel[name] = ((wx - px) / dt, (wy - py) / dt)
            self._actor_prev[name] = (wx, wy, sim_t)
            self._actor_pos[name]  = (wx, wy)
            self._actor_live_t[name] = sim_t

    def _update_actors(self) -> None:
        """
        Recompute actor positions.  Prefers live /gz_dynamic_poses data
        (real Gazebo ground truth from kinematic human models).  Falls
        back to SDF waypoint interpolation for any actor whose live
        pose is stale (>0.5 s) or never received.
        """
        sim_t = self._sim_seconds()
        if sim_t is None:
            return
        traj_t  = sim_t * self._time_scale
        sim_dt  = 0.05
        traj_dt = sim_dt * self._time_scale
        for name, traj in self._world['actors'].items():
            last = self._actor_live_t.get(name)
            if last is not None and (sim_t - last) < 0.5:
                continue   # live data is fresh — _dynamic_poses_cb owns it
            wx,  wy  = _interp_trajectory(traj, traj_t)
            wx2, wy2 = _interp_trajectory(traj, traj_t + traj_dt)
            self._actor_pos[name] = (wx, wy)
            self._actor_vel[name] = ((wx2 - wx) / sim_dt, (wy2 - wy) / sim_dt)
            self._actor_prev[name] = (wx, wy, sim_t)

    def _odom_cb(self, msg: Odometry) -> None:
        ps = PoseStamped()
        ps.header.stamp    = msg.header.stamp
        ps.header.frame_id = 'odom'
        ps.pose = msg.pose.pose
        self._trail.poses.append(ps)
        if len(self._trail.poses) > TRAIL_MAX:
            self._trail.poses.pop(0)
        self._trail.header.stamp = msg.header.stamp
        self._pub_trail.publish(self._trail)

    # ── Publishers ───────────────────────────────────────────────────────

    def _publish(self) -> None:
        arr  = MarkerArray()
        pred = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # ── Statics: walls + obstacles ─────────────────────────────────
        # Stable per-namespace IDs so RViz overwrites the same marker
        # every tick (no ghost trail of ID-shifted leftovers).
        wall_idx = 0
        obs_idx  = 0
        for entry in self._world['statics']:
            name, kind, shape, wx, wy, wz, sx, sy, sz, r, g, b = entry
            ox, oy = self._w2o(wx, wy)

            if kind == 'wall':
                ns, idx = 'walls', wall_idx
                wall_idx += 1
                alpha = 0.55
            else:
                ns, idx = 'obstacles', obs_idx
                obs_idx += 1
                alpha = 0.70

            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = stamp
            m.ns, m.id = ns, idx
            m.type = (Marker.CYLINDER if shape == 'cylinder' else
                      Marker.SPHERE   if shape == 'sphere'   else
                      Marker.CUBE)
            m.action = Marker.ADD
            m.pose.position.x = ox
            m.pose.position.y = oy
            m.pose.position.z = wz
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, alpha
            m.lifetime = Duration(seconds=2.0).to_msg()
            arr.markers.append(m)

            # Outline edge for walls so they remain crisp at low alpha
            if kind == 'wall':
                edge = Marker()
                edge.header = m.header
                edge.ns, edge.id = 'wall_edges', idx
                edge.type   = Marker.LINE_LIST
                edge.action = Marker.ADD
                edge.pose.orientation.w = 1.0
                edge.scale.x = 0.04
                edge.color.r, edge.color.g, edge.color.b, edge.color.a = \
                    0.85, 0.85, 0.85, 1.0
                hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
                cx, cy, cz = ox, oy, wz
                top = [(cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
                       (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
                bot = [(x, y, cz - hz) for (x, y, _) in top]
                pts = []
                for i in range(4):
                    pts += [top[i], top[(i + 1) % 4]]
                    pts += [bot[i], bot[(i + 1) % 4]]
                    pts += [top[i], bot[i]]
                for (x, y, z) in pts:
                    p = Point(); p.x, p.y, p.z = x, y, z
                    edge.points.append(p)
                edge.lifetime = Duration(seconds=2.0).to_msg()
                arr.markers.append(edge)

        # ── Humans: stable per-actor IDs across namespaces ─────────────
        for actor_idx, name in enumerate(sorted(self._actor_color)):
            if name not in self._actor_pos:
                continue
            wx, wy = self._actor_pos[name]
            ox, oy = self._w2o(wx, wy)
            r, g, b = self._actor_color[name]
            vx, vy = self._actor_vel.get(name, (0.0, 0.0))
            speed  = math.hypot(vx, vy)

            # Body
            body = Marker()
            body.header.frame_id = 'odom'
            body.header.stamp    = stamp
            body.ns, body.id = 'humans', actor_idx
            body.type   = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = ox
            body.pose.position.y = oy
            body.pose.position.z = 0.9
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = 0.5
            body.scale.z = 1.8
            body.color.r, body.color.g, body.color.b, body.color.a = r, g, b, 0.85
            body.lifetime = Duration(seconds=0.5).to_msg()
            arr.markers.append(body)

            # Label
            lbl = Marker()
            lbl.header = body.header
            lbl.ns, lbl.id = 'human_labels', actor_idx
            lbl.type   = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = ox
            lbl.pose.position.y = oy
            lbl.pose.position.z = 2.3
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.4
            lbl.color.r, lbl.color.g, lbl.color.b, lbl.color.a = r, g, b, 1.0
            lbl.text = name
            lbl.lifetime = Duration(seconds=0.5).to_msg()
            arr.markers.append(lbl)

            # Velocity arrow — same ID every tick.  When the human is
            # almost stopped we still emit the marker but with action=DELETE
            # so RViz removes any previous arrow at the old pose (otherwise
            # it would linger for the lifetime duration → ghost arrow).
            av = Marker()
            av.header = body.header
            av.ns, av.id = 'human_vel', actor_idx
            if speed > 0.08:
                yaw = math.atan2(vy, vx)
                av.type   = Marker.ARROW
                av.action = Marker.ADD
                av.pose.position.x = ox
                av.pose.position.y = oy
                av.pose.position.z = 1.85
                av.pose.orientation.z = math.sin(yaw / 2.0)
                av.pose.orientation.w = math.cos(yaw / 2.0)
                av.scale.x = min(speed * 0.7, 1.5)
                av.scale.y = av.scale.z = 0.12
                av.color.r, av.color.g, av.color.b, av.color.a = r, g, b, 0.9
                av.lifetime = Duration(seconds=0.5).to_msg()
            else:
                av.action = Marker.DELETE
            arr.markers.append(av)

            # Predicted path (dotted) — IDs unique per (actor, step)
            base = actor_idx * PREDICT_STEPS
            for step in range(PREDICT_STEPS):
                dt = (step + 1) * PREDICT_DT
                size = max(0.07, 0.28 - (step + 1) * 0.025)
                fade = max(0.10, 0.80 - (step + 1) * 0.09)
                sp = Marker()
                sp.header = body.header
                sp.ns, sp.id = 'human_pred', base + step
                sp.type   = Marker.SPHERE
                sp.action = Marker.ADD
                sp.pose.position.x = ox + vx * dt
                sp.pose.position.y = oy + vy * dt
                sp.pose.position.z = 0.15
                sp.pose.orientation.w = 1.0
                sp.scale.x = sp.scale.y = sp.scale.z = size
                sp.color.r, sp.color.g, sp.color.b, sp.color.a = r, g, b, fade
                sp.lifetime = Duration(seconds=0.5).to_msg()
                pred.markers.append(sp)

        self._pub_world.publish(arr)
        self._pub_paths.publish(pred)

    def _log(self) -> None:
        sim_t = self._sim_seconds()
        sim_str = f'{sim_t:.2f}s' if sim_t is not None else '<waiting>'
        self.get_logger().info(f'sim_t={sim_str}  actors={len(self._actor_color)}')
        for name in sorted(self._actor_color):
            wx, wy = self._actor_pos.get(name, (float('nan'), float('nan')))
            vx, vy = self._actor_vel.get(name, (0.0, 0.0))
            self.get_logger().info(
                f'  {name}  world=({wx:+.2f},{wy:+.2f})  v=({vx:+.2f},{vy:+.2f})')


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
