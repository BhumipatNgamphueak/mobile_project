#!/usr/bin/env python3
"""
path_planning_node.py
=====================
Receive a goal pose and the current robot pose, then compute a
collision-free path while accounting for detected dynamic obstacles
(humans). Also publishes the known static map so RViz can overlay it.

Inputs
------
/odom                   nav_msgs/Odometry          robot pose in odom frame


Outputs
-------
/planned_path           nav_msgs/Path              sequence of poses for pure_pursuit

Parameters
----------


Path contract with pure_pursuit
--------------------------------
- path.header.frame_id must be 'odom'
- Waypoint spacing <= 0.05 m recommended
- Re-publish at replan_hz — pure_pursuit always needs a fresh copy
- All waypoints must be outside inflated obstacle zones
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)

class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.planning_freq = 10 # Hz
        self.lookahead_dist         = 5.0    # meters — how far ahead to extract
        self.prune_dist             = 1.0    # meters — prune waypoints behind robot
        self.min_waypoints_required = 2      # don't publish if too few points

        # ── TEB parameters ────────────────────────────────────────────────
        self.teb_n_iter   = 300   # optimisation iterations per replan cycle
        self.teb_w_obs    = 3.0  # obstacle repulsion weight
        self.teb_w_smooth = 2.0  # smoothness (elastic) weight
        self.teb_w_time   = 0.1  # time minimisation weight
        self.v_max        = 0.5  # m/s — max linear speed (mecanum any direction)

        # ── Subscribers ───────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 10)
        self.global_path_sub = self.create_subscription(
            Path, '/global_path', self._global_path_callback, 10)
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, '/local_costmap', self._local_costmap_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose: Pose | None = None
        self._robot_vel = None               # geometry_msgs/Twist (vx, vy, omega)
        self._global_path: Path | None = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int = 0           # ← tracks how far we've consumed

        # ── Replanning timer ──────────────────────────────────────────────
        self._replan_timer = self.create_timer(
            1.0 / self.planning_freq, self._replan)

        self.get_logger().info('PathPlanningNode started')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose
        self._robot_vel  = msg.twist.twist

    def _local_costmap_callback(self, msg:OccupancyGrid):
        self._local_costmap = msg

    def _global_path_callback(self, msg: Path):
        """
        Called every time a new global path arrives.
        Reset the prune index so we start consuming from the beginning.
        """
        self._global_path = msg
        self._prune_index = 0
        self.get_logger().info(f'New global path received: {len(msg.poses)} waypoints')

    
    # ──────────────────────────────────────────────────────────────────────
    # Planning 
    # ──────────────────────────────────────────────────────────────────────
    
    def _replan(self):
        # Guard: wait until all inputs are ready
        if self._robot_pose is None or self._global_path is None or self._local_costmap is None:
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            return

        # Advance prune index to robot's current position on global path
        closest_idx = self._find_closest_index()
        self._prune_passes_waypoints(closest_idx)

        # Extract the relevant slice of the global path ahead of the robot
        local_segment = self._extract_local_segment()
        if len(local_segment) < self.min_waypoints_required:
            return

        # TEB: deform local_segment to avoid obstacles and minimise time
        optimized = self._teb_optimize(local_segment)

        # Publish the optimised path for pure_pursuit
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        path_msg.poses = optimized
        self.path_pub.publish(path_msg)

    # ──────────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────────
    def _costmap_cost(self, x: float, y: float) -> float:
        """
        Return obstacle penalty [0–1] at odom-frame position (x, y).

        0   = free space (raw cost ≤ 20)
        0–1 = inflated zone (raw cost 20–100)
        1   = lethal obstacle (raw cost 100)

        The local costmap is built in the robot/laser frame (centred on the
        robot), so we must transform odom → robot frame before indexing.
        """
        if self._local_costmap is None or self._robot_pose is None:
            return 0.0

        # Extract robot yaw from quaternion (rotation around z-axis only)
        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)

        # Translate then rotate: odom → robot frame
        dx = x - self._robot_pose.position.x
        dy = y - self._robot_pose.position.y
        lx =  dx * math.cos(yaw) + dy * math.sin(yaw)
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)

        # Robot-frame (lx, ly) → grid cell indices
        info = self._local_costmap.info
        gx = int((lx - info.origin.position.x) / info.resolution)
        gy = int((ly - info.origin.position.y) / info.resolution)

        if not (0 <= gx < info.width and 0 <= gy < info.height):
            return 0.0  # outside costmap window → assume free

        raw = float(max(0, self._local_costmap.data[gy * info.width + gx]))
        threshold = 20.0
        if raw <= threshold:
            return 0.0
        return (raw - threshold) / (100.0 - threshold)  # normalised to [0, 1]

    def _teb_optimize(self, segment: list[PoseStamped]) -> list[PoseStamped]:
        """
        Timed Elastic Band optimisation.

        Each iteration does two things:
          A) Elastic band — move intermediate waypoints to
               - reduce obstacle cost  (repulsion from obstacles)
               - stay smooth           (attraction toward midpoint of neighbours)
          B) Time adjustment — shrink each dt interval as fast as the
               velocity limit allows

        Start and goal waypoints are held fixed throughout.
        """
        n = len(segment)
        if n < 2:
            return segment

        # Mutable copies of waypoint positions (start/end will stay fixed)
        xs = [p.pose.position.x for p in segment]
        ys = [p.pose.position.y for p in segment]

        # Initial time intervals: dist / v_max gives the fastest feasible dt
        dts = []
        for i in range(n - 1):
            dist = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            dts.append(max(0.05, dist / self.v_max))

        step      = 0.05  # position update step size
        eps       = 0.15  # must be > costmap resolution (0.1 m) to sample different cells
        max_delta = 0.1   # max waypoint movement per iteration (m) — prevents explosion

        for _ in range(self.teb_n_iter):

            # ── A. Update intermediate waypoint positions ──────────────
            for i in range(1, n - 1):   # index 0 and n-1 are fixed

                # Obstacle repulsion — numerical gradient of normalised cost [0,1]
                # eps > cell size ensures the two samples land in different cells
                c_px = self._costmap_cost(xs[i] + eps, ys[i])
                c_mx = self._costmap_cost(xs[i] - eps, ys[i])
                c_py = self._costmap_cost(xs[i],       ys[i] + eps)
                c_my = self._costmap_cost(xs[i],       ys[i] - eps)
                grad_obs_x = (c_px - c_mx) / (2.0 * eps)
                grad_obs_y = (c_py - c_my) / (2.0 * eps)

                # Smoothness — pull waypoint toward midpoint of its neighbours
                f_smooth_x = (xs[i - 1] + xs[i + 1]) / 2.0 - xs[i]
                f_smooth_y = (ys[i - 1] + ys[i + 1]) / 2.0 - ys[i]

                # Combined update: repel from obstacles, attract to smooth line
                delta_x = step * (-self.teb_w_obs * grad_obs_x + self.teb_w_smooth * f_smooth_x)
                delta_y = step * (-self.teb_w_obs * grad_obs_y + self.teb_w_smooth * f_smooth_y)

                # Clamp movement magnitude to keep optimisation stable
                mag = math.hypot(delta_x, delta_y)
                if mag > max_delta:
                    delta_x = delta_x * max_delta / mag
                    delta_y = delta_y * max_delta / mag

                xs[i] += delta_x
                ys[i] += delta_y

            # ── B. Adjust time intervals ───────────────────────────────
            for i in range(n - 1):
                dist   = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                dt_min = dist / self.v_max          # kinematic lower bound
                # Time cost shrinks dt; kinematic constraint floors it
                dts[i] = max(dt_min, dts[i] - self.teb_w_time * step)

        # Rebuild PoseStamped list with forward-facing orientation
        stamp = self.get_clock().now().to_msg()
        frame = segment[0].header.frame_id
        result = []
        for i in range(n):
            # Heading: direction from previous to next waypoint
            theta = math.atan2(
                ys[min(i + 1, n - 1)] - ys[max(i - 1, 0)],
                xs[min(i + 1, n - 1)] - xs[max(i - 1, 0)]
            )
            ps = PoseStamped()
            ps.header.frame_id    = frame
            ps.header.stamp       = stamp
            ps.pose.position.x    = xs[i]
            ps.pose.position.y    = ys[i]
            ps.pose.orientation.z = math.sin(theta / 2.0)
            ps.pose.orientation.w = math.cos(theta / 2.0)
            result.append(ps)
        return result

    def _find_closest_index(self) -> int:
        """
        Find the index in the global path closest to the robot,
        starting from the current prune index (never go backwards).

        Global path:   0──1──2──3──[4]──5──6──7──8
                                    ↑
                               closest index
        """
        waypoints = self._global_path.poses

        closest_idx = self._prune_index
        closest_dist = float('inf')

        # Loop all waypoints from the current prune index
        for i in range(self._prune_index, len(waypoints)):
            d = euclidean(self._robot_pose, waypoints[i].pose)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i
        return closest_idx

    def _prune_passes_waypoints(self, closest_idx: int):
        """
        Prune the robot path, start planning at closest point in global path
        Before:  0──1──2──[Robot]──3──4──5
        After:   prune_index moves to 3
        """
        self._prune_index = closest_idx
    
    def _extract_local_segment(self) -> list[PoseStamped]:
        """
        Extract waypoints within lookahead_dist from the robot, starting from prune_index.
        Global path:  ...──[prune]──A──B──C──D──E──...
                                    |←── lookahead ──→|
        Local segment return:       A──B──C──D
        """
        waypoints = self._global_path.poses
        local_segment = []

        for i in range(self._prune_index, len(waypoints)):
            d = euclidean(self._robot_pose, waypoints[i].pose)
            if d <= self.lookahead_dist:
                local_segment.append(waypoints[i])
            else:
                break  # waypoints are ordered, no need to continue

        return local_segment


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
