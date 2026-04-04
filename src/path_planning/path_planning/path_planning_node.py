#!/usr/bin/env python3
"""
path_planning_node.py  —  SKELETON
=====================================
Receive a goal pose and the current robot pose, then compute a
collision-free path while accounting for detected dynamic obstacles
(humans).  Also publishes the known static map so RViz can overlay
"what the robot knows" vs. the real world.

Inputs
------
/odom                   nav_msgs/Odometry          robot pose in odom frame
/goal_pose              geometry_msgs/PoseStamped  target given by operator / mission node
/detected_human_poses   geometry_msgs/PoseArray    dynamic obstacles from human_detection

Outputs
-------
/planned_path           nav_msgs/Path              sequence of poses for pure_pursuit
/real_map               nav_msgs/OccupancyGrid     static map of the known environment

Parameters
----------
odom_frame          (str,   default 'odom')
map_width_m         (float, default 20.0)   world width  [m]
map_height_m        (float, default 20.0)   world height [m]
map_resolution      (float, default 0.1)    metres per cell
robot_radius        (float, default 0.35)   inflation radius around obstacles [m]
human_radius        (float, default 0.6)    inflation radius around detected humans [m]
replan_hz           (float, default 2.0)    replanning frequency [Hz]
use_sim_time        (bool,  default True)
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('odom_frame',     'odom')
        self.declare_parameter('map_width_m',    20.0)
        self.declare_parameter('map_height_m',   20.0)
        self.declare_parameter('map_resolution', 0.1)
        self.declare_parameter('robot_radius',   0.35)
        self.declare_parameter('human_radius',   0.6)
        self.declare_parameter('replan_hz',      2.0)

        self.odom_frame     = self.get_parameter('odom_frame').value
        self.map_width_m    = self.get_parameter('map_width_m').value
        self.map_height_m   = self.get_parameter('map_height_m').value
        self.map_resolution = self.get_parameter('map_resolution').value
        self.robot_radius   = self.get_parameter('robot_radius').value
        self.human_radius   = self.get_parameter('human_radius').value
        replan_hz           = self.get_parameter('replan_hz').value

        # ── Subscribers ───────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 10)

        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose',
            self._goal_callback, 10)

        self.human_sub = self.create_subscription(
            PoseArray, '/detected_human_poses',
            self._human_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.path_pub = self.create_publisher(Path,          '/planned_path', 10)
        self.map_pub  = self.create_publisher(OccupancyGrid, '/real_map',     10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose:   Pose      | None = None
        self._goal_pose:    Pose      | None = None
        self._human_poses:  list[Pose]       = []

        # Static obstacle list (centre_x, centre_y, half_size) — mirrors SDF
        # Update if world geometry changes.
        self._static_obstacles: list[tuple[float, float, float]] = [
            # (cx_world, cy_world, half_size)
            # world → odom: odom = world - spawn_offset(-7, 4)
            ( 4 - (-7),  5 - 4, 1.0),   # obstacle_A
            (-4 - (-7), -5 - 4, 1.0),   # obstacle_B
            ( 2 - (-7), -4 - 4, 1.0),   # obstacle_C
        ]

        # ── Replanning timer ──────────────────────────────────────────────
        self._replan_timer = self.create_timer(
            1.0 / replan_hz, self._replan)

        # Publish static map once at startup (latched via transient_local QoS)
        from rclpy.qos import QoSProfile, DurabilityPolicy
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/real_map', map_qos)
        self._publish_static_map()

        self.get_logger().info('PathPlanningNode started (skeleton)')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose

    def _goal_callback(self, msg: PoseStamped):
        self._goal_pose = msg.pose
        self.get_logger().info(
            f'New goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
        self._replan()   # replan immediately on new goal

    def _human_callback(self, msg: PoseArray):
        self._human_poses = msg.poses

    # ──────────────────────────────────────────────────────────────────────
    # Planning pipeline  ← IMPLEMENT THIS
    # ──────────────────────────────────────────────────────────────────────

    def _replan(self):
        """
        Called at replan_hz.  Compute a path from current robot pose to goal.

        Suggested algorithms
        --------------------
        A*       : discretise odom grid, inflate static + dynamic obstacles,
                   run A* search, smooth with B-spline or bezier.
        RRT/RRT* : sample-based, good for cluttered environments.
        DWA      : dynamic window approach, handles moving obstacles natively.
        TEB      : timed-elastic-band, good for narrow passages.

        Output contract
        ---------------
        Publish a nav_msgs/Path on /planned_path.
        Each PoseStamped in path.poses must be in the odom frame.
        """
        if self._robot_pose is None or self._goal_pose is None:
            return

        # TODO: replace stub with real planner
        path = self._stub_plan()
        if path is not None:
            self.path_pub.publish(path)

    def _stub_plan(self) -> Path | None:
        """
        Stub planner — straight line from robot to goal, no collision check.
        Replace with your algorithm.
        """
        if self._robot_pose is None or self._goal_pose is None:
            return None

        stamp = self.get_clock().now().to_msg()
        header = Header(frame_id=self.odom_frame, stamp=stamp)

        path = Path(header=header)

        # Linear interpolation: N steps between robot and goal
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        gx = self._goal_pose.position.x
        gy = self._goal_pose.position.y

        N = max(2, int(math.hypot(gx - rx, gy - ry) / 0.2))   # one pose every 0.2 m
        for i in range(N + 1):
            t = i / N
            ps = PoseStamped(header=header)
            ps.pose.position.x = rx + t * (gx - rx)
            ps.pose.position.y = ry + t * (gy - ry)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        return path

    # ──────────────────────────────────────────────────────────────────────
    # Static map publisher
    # ──────────────────────────────────────────────────────────────────────

    def _publish_static_map(self):
        """
        Build and publish an OccupancyGrid representing the known static world.

        The map origin is set so that (0,0) in odom aligns with cell (0,0).
        Spawn offset: robot was spawned at world (-7, 4), so odom=(0,0) == world=(-7,4).
        Map covers [-10, 10] in both world axes → odom x∈[-3,17], y∈[-14,6].

        Obstacle cells are inflated by robot_radius to give conservative clearance.
        """
        res = self.map_resolution
        w_cells = int(self.map_width_m  / res)   # 200 cells @ 0.1 m/cell
        h_cells = int(self.map_height_m / res)

        # Map origin = odom coordinate of cell (0,0) bottom-left corner
        # World boundary: x∈[-10,10], y∈[-10,10]
        # odom = world - spawn_offset; spawn_offset = (-7, 4)
        # → odom_origin_x = -10 - (-7) = -3
        # → odom_origin_y = -10 -   4  = -14
        origin_x = -3.0
        origin_y = -14.0

        data = [0] * (w_cells * h_cells)   # 0 = free

        def mark_obstacle(cx_odom, cy_odom, half_size, inflate):
            """Mark an axis-aligned box obstacle + inflation ring."""
            lo_x = cx_odom - half_size - inflate
            hi_x = cx_odom + half_size + inflate
            lo_y = cy_odom - half_size - inflate
            hi_y = cy_odom + half_size + inflate
            for gx in range(w_cells):
                wx = origin_x + gx * res
                if wx < lo_x or wx > hi_x:
                    continue
                for gy in range(h_cells):
                    wy = origin_y + gy * res
                    if wy < lo_y or wy > hi_y:
                        continue
                    data[gy * w_cells + gx] = 100

        # Boundary walls (0.2 m thick, world ±10 m)
        wall_half = 10.0
        wall_thick = 0.1 + self.robot_radius
        for cx_odom, cy_odom, half_w, half_h in [
            ( 7.0,  6.0, wall_half, wall_thick),   # north  (world y=10)
            ( 7.0, -14.0, wall_half, wall_thick),  # south  (world y=-10)
            ( 17.0, -4.0, wall_thick, wall_half),  # east   (world x=10)
            (-3.0,  -4.0, wall_thick, wall_half),  # west   (world x=-10)
        ]:
            mark_obstacle(cx_odom, cy_odom, max(half_w, half_h), 0.0)

        # Static obstacles from SDF
        for cx, cy, hs in self._static_obstacles:
            mark_obstacle(cx, cy, hs, self.robot_radius)

        stamp = self.get_clock().now().to_msg()
        grid = OccupancyGrid()
        grid.header = Header(frame_id=self.odom_frame, stamp=stamp)
        grid.info = MapMetaData(
            map_load_time=stamp,
            resolution=res,
            width=w_cells,
            height=h_cells,
        )
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = data

        self.map_pub.publish(grid)
        self.get_logger().info('Static map published on /real_map')


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
