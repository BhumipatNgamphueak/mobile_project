#!/usr/bin/env python3
"""
human_controller.py
===================
Drives the kinematic human models in the active world along their
SDF-embedded waypoints, using Gazebo's /world/<w>/set_pose service
bridged through ros_gz_bridge as ros_gz_interfaces/SetEntityPose.

Calls are fully async (call_async with a no-op done callback) so the
timer never blocks.  This replaces the earlier subprocess-based
implementation which generated `Host unreachable` log spam from
ign-transport whenever a CLI helper exited before Gazebo finished its
response handshake.
"""

import math
import os
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


def _floats(text: str) -> list:
    return [float(p) for p in text.strip().split()]


def _parse_waypoints_plugin(plugin) -> dict:
    delay = 0.0
    loop  = True
    d = plugin.find('delay_start')
    if d is not None and d.text:
        try:    delay = float(d.text)
        except ValueError: pass
    l = plugin.find('loop')
    if l is not None and l.text:
        loop = l.text.strip().lower() in ('1', 'true')

    wps = []
    for wp in plugin.findall('waypoint'):
        t_el = wp.find('time')
        p_el = wp.find('pose')
        if t_el is None or p_el is None or not p_el.text:
            continue
        try:
            t = float(t_el.text)
            xs = _floats(p_el.text)
            x   = xs[0] if len(xs) > 0 else 0.0
            y   = xs[1] if len(xs) > 1 else 0.0
            z   = xs[2] if len(xs) > 2 else 0.0
            yaw = xs[5] if len(xs) > 5 else 0.0
            wps.append((t, x, y, z, yaw))
        except ValueError:
            continue
    wps.sort(key=lambda w: w[0])
    return {'delay': delay, 'loop': loop, 'waypoints': wps}


def parse_human_models(sdf_path: str) -> dict:
    """
    Returns {model_name: trajectory_dict, ...} for every <model> that
    contains a <plugin filename="__waypoints__"> child.
    """
    out = {}
    try:
        root = ET.parse(sdf_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return out
    world = root.find('world')
    if world is None:
        return out
    for model in world.findall('model'):
        name = model.get('name', '')
        for plugin in model.findall('plugin'):
            if plugin.get('filename') == '__waypoints__':
                out[name] = _parse_waypoints_plugin(plugin)
                break
    return out


def interp(traj: dict, sim_t: float) -> tuple:
    wps = traj['waypoints']
    if not wps:
        return 0.0, 0.0, 0.0, 0.0
    rel = sim_t - traj['delay']
    if rel <= wps[0][0]:
        return wps[0][1], wps[0][2], wps[0][3], wps[0][4]
    period = wps[-1][0] - wps[0][0]
    if traj['loop'] and period > 0.0:
        rel = wps[0][0] + ((rel - wps[0][0]) % period)
    elif rel >= wps[-1][0]:
        return wps[-1][1], wps[-1][2], wps[-1][3], wps[-1][4]
    for i in range(len(wps) - 1):
        t0, x0, y0, z0, yw0 = wps[i]
        t1, x1, y1, z1, yw1 = wps[i + 1]
        if t0 <= rel <= t1:
            dt = t1 - t0
            if dt <= 0:
                return x0, y0, z0, yw0
            a = (rel - t0) / dt
            # Shortest-arc yaw interpolation
            dyaw = math.atan2(math.sin(yw1 - yw0), math.cos(yw1 - yw0))
            return (x0 + a * (x1 - x0),
                    y0 + a * (y1 - y0),
                    z0 + a * (z1 - z0),
                    yw0 + a * dyaw)
    return wps[-1][1], wps[-1][2], wps[-1][3], wps[-1][4]


class HumanController(Node):

    def __init__(self):
        super().__init__('human_controller')

        world      = self.declare_parameter('world',      'crossing_humans').value
        rate_hz    = float(self.declare_parameter('rate_hz',     20.0).value)
        time_scale = float(self.declare_parameter('time_scale',   1.0).value)

        self._world      = world
        self._service    = f'/world/{world}/set_pose'
        self._time_scale = time_scale

        self._client = self.create_client(SetEntityPose, self._service)
        self.get_logger().info(
            f'Waiting for {self._service} (bridged from Ignition)…')

        share = get_package_share_directory('mecanum_robot_sim')
        sdf_path = os.path.join(share, 'worlds', f'{world}.sdf')
        self._humans = parse_human_models(sdf_path)
        if not self._humans:
            self.get_logger().warn(f'No human models found in {sdf_path}')
        else:
            for name, traj in self._humans.items():
                wp = traj['waypoints']
                self.get_logger().info(
                    f'human "{name}": {len(wp)} waypoints, '
                    f'delay={traj["delay"]}s, loop={traj["loop"]}')

        self.get_logger().info(
            f'service={self._service}  rate={rate_hz}Hz  time_scale={time_scale}')

        self.create_timer(1.0 / rate_hz, self._tick)

    # ──────────────────────────────────────────────────────────────────────

    def _sim_seconds(self) -> float | None:
        msg = self.get_clock().now().to_msg()
        t = msg.sec + msg.nanosec * 1e-9
        return t if 0.0 < t < 1.0e8 else None

    def _set_pose(self, name: str, x: float, y: float, z: float, yaw: float):
        if not self._client.service_is_ready():
            return  # bridge / Gazebo not up yet
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = float(math.sin(yaw * 0.5))
        req.pose.orientation.w = float(math.cos(yaw * 0.5))
        # Fire-and-forget: attach a no-op done-callback so rclpy releases
        # the future when Gazebo responds (avoids leaking pending futures).
        future = self._client.call_async(req)
        future.add_done_callback(lambda _f: None)

    def _tick(self):
        sim_t = self._sim_seconds()
        if sim_t is None:
            return
        traj_t = sim_t * self._time_scale
        for name, traj in self._humans.items():
            x, y, z, yaw = interp(traj, traj_t)
            self._set_pose(name, x, y, z, yaw)


def main(args=None):
    rclpy.init(args=args)
    node = HumanController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
