#!/usr/bin/env python3
"""
path_planning_node.py
=====================
TEB-style local planner with point obstacles.

Pipeline (one cycle):
  1. Costmap -> point obstacles
       Extract occupied cells in a window around the robot.
  2. Local segment extraction
       Prune behind the robot, then take a lookahead window of the global
       path. Warm-started from the previous optimised band so the trajectory
       evolves smoothly across cycles.
  3. TEB optimisation
       Gradient descent on the elastic band against
         - point obstacles (outward force inside an inflation zone)
         - smoothness (pull toward midpoint of neighbours)
       Endpoints fixed.
  4. Holonomic velocity extraction
       Vector pursuit on the optimised band -> (vx, vy, omega=0).

Inputs
------
/odom           nav_msgs/Odometry       robot pose + velocity
/global_path    nav_msgs/Path           reference path from global planner
/local_costmap  nav_msgs/OccupancyGrid  obstacle grid in odom frame

Outputs
-------
/cmd_vel             geometry_msgs/TwistStamped       velocity command
/planned_path        nav_msgs/Path                    optimised band (RViz)
/obstacle_points     visualization_msgs/MarkerArray   extracted points (RViz)
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from path_planning.dbscan import (
    cluster_obstacles,
    Cluster,
    point_to_convex_polygon,
    segment_intersects_polygon,
)


def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── General parameters ────────────────────────────────────────────
        self.planning_freq          = 10     # Hz
        self.lookahead_dist         = 4.0   # m — local segment length
        self.min_waypoints_required = 0
        self.goal_tolerance         = 0.4    # m

        # ── TEB parameters ────────────────────────────────────────────────
        self.teb_n_iter      = 60      # gradient descent iterations / cycle
        self.teb_w_obs       = 50.0 # 50.0    # obstacle repulsion weight
        self.teb_w_smooth    = 2.0     # elastic band weight
        self.teb_inflation   = 1.0 # 0.5     # m — repulsion radius around points
        self.teb_step        = 0.05    # gradient step
        self.teb_max_delta   = 0.10    # m — clamp per-iteration movement
        self.v_max           = 1.5     # m/s

        # ── Costmap-to-points parameters ────────────────────────────────
        self.obstacle_threshold = 60    # cost >= this -> obstacle cell
        self.search_margin      = 8.0   # m beyond lookahead to scan

        # ── DBSCAN clustering parameters ────────────────────────────────
        # Group raw obstacle cells into obstacle entities. Noise (isolated
        # cells from sensor flicker) is dropped before being fed to the EB.
        self.use_dbscan          = True
        self.dbscan_eps          = 0.20  # m  — neighbour radius (~4 cells @0.05)
        self.dbscan_min_samples  = 3     # min neighbours for a core point
        # Optional second-tier threshold: cluster only the *high-cost* cells
        # (e.g. inflation peaks / lethal core), useful when the inflated
        # costmap is dense and we want clusters to track obstacle cores.
        self.dbscan_cost_threshold = self.obstacle_threshold
        # Concavity guard: clusters with solidity below this threshold (L /
        # U / ring shapes whose convex hull bridges across empty space) are
        # marked unreliable and fall back to point-based repulsion in
        # _obstacle_force, so the robot is never falsely pushed out from
        # the phantom interior of a bridging hull.
        self.dbscan_solidity_threshold = 0.5

        # ── Homotopy correction parameters ──────────────────────────────
        # The EB is a *local* optimiser — gradient descent can't move the
        # band across an obstacle, so if the warm-started or reference
        # path threads through a cluster polygon the band gets stuck on
        # the wrong homotopy class. We use the DBSCAN hulls to detect
        # this (waypoint inside polygon, or segment crossing polygon)
        # and snap the offending waypoints to the outside before/during
        # gradient descent, restoring a valid topology.
        self.teb_homotopy_correction = True
        self.teb_homotopy_every      = 5     # apply every K iterations
        self.teb_homotopy_margin     = 0.30  # m past hull edge after snap

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,       '/odom',          self._odom_callback,          10)
        self.create_subscription(Path,           '/global_path',   self._global_path_callback,   10)
        self.create_subscription(OccupancyGrid,  '/local_costmap', self._local_costmap_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel',           10)
        self.path_pub    = self.create_publisher(Path,         '/planned_path',      10)
        self.obstacle_pub = self.create_publisher(MarkerArray,  '/obstacle_points',   10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose: Pose | None             = None
        self._robot_vel                           = None
        self._global_path: Path | None            = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int                    = 0

        # Warm start from previous optimised band.
        self._warm_xs: list[float] | None = None
        self._warm_ys: list[float] | None = None

        # Points extracted from current costmap (set each cycle).
        # _obstacle_points is what the EB actually uses for repulsion; after
        # clustering it equals the union of member points across clusters
        # (noise removed). _obstacle_clusters holds per-entity summaries.
        self._obstacle_points:   list[tuple[float, float]] = []
        self._obstacle_clusters: list[Cluster]             = []

        self.create_timer(1.0 / self.planning_freq, self._replan)
        self.get_logger().info('PathPlanningNode (TEB point-style) started')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose
        self._robot_vel  = msg.twist.twist

    def _local_costmap_callback(self, msg: OccupancyGrid):
        # Perception runs at sensor rate, decoupled from the planning timer.
        # Every new costmap → one DBSCAN pass → fresh clusters + RViz markers,
        # whether or not a goal exists and whether or not the robot is moving.
        self._local_costmap = msg
        self._update_obstacles()

    def _global_path_callback(self, msg: Path):
        if self._global_path is not None and len(msg.poses) > 0:
            if (msg.header.stamp.sec == self._global_path.header.stamp.sec and
                msg.header.stamp.nanosec == self._global_path.header.stamp.nanosec):
                return

            new_goal = msg.poses[-1].pose.position
            old_goal = self._global_path.poses[-1].pose.position
            same_goal = (abs(new_goal.x - old_goal.x) < 0.01 and
                         abs(new_goal.y - old_goal.y) < 0.01)
            if same_goal:
                # Geometry changed but goal is the same — keep warm state.
                self._global_path = msg
                self._prune_index = 0
                return

        self._global_path = msg
        self._prune_index = 0
        self._warm_xs     = None
        self._warm_ys     = None
        self.get_logger().info(f'New goal received ({len(msg.poses)} waypoints)')

    # ──────────────────────────────────────────────────────────────────────
    # Planning loop
    # ──────────────────────────────────────────────────────────────────────

    def _update_obstacles(self):
        """
        Refresh the obstacle representation from the latest costmap.

        Runs once per costmap arrival (sensor rate), independent of the
        planning timer, the existence of a goal, or the robot's motion.
        Populates ``self._obstacle_points`` (consumed by the EB) and
        ``self._obstacle_clusters`` (used for visualisation and any
        per-entity logic), then publishes the marker array.
        """
        if self._robot_pose is None or self._local_costmap is None:
            return

        raw_points = self._costmap_to_points(self.obstacle_threshold)
        self._obstacle_clusters = self._cluster_points(raw_points)
        if self.use_dbscan and self._obstacle_clusters:
            # Feed the EB only the points that survived clustering. Same
            # force semantics, but isolated noise cells no longer pull the
            # band around.
            self._obstacle_points = [p for c in self._obstacle_clusters
                                       for p in c.points]
        else:
            self._obstacle_points = raw_points
        self._publish_points()

    def _replan(self):
        if self._robot_pose is None or self._global_path is None or self._local_costmap is None:
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            return

        if euclidean(self._robot_pose, self._global_path.poses[-1].pose) < self.goal_tolerance:
            self._pub_stop()
            return

        # Obstacles already refreshed in _local_costmap_callback; just consume
        # the cached self._obstacle_points / self._obstacle_clusters.

        # 1. Reference segment from global path
        self._prune_index = self._find_closest_index()
        segment = self._extract_local_segment()
        if len(segment) < self.min_waypoints_required:
            return

        # 2. TEB optimisation (uses the latest cached obstacle points)
        opt_xs, opt_ys = self._teb_optimize(segment)

        # 3. Cache for next cycle's warm start
        self._warm_xs = opt_xs[:]
        self._warm_ys = opt_ys[:]

        # 4. Publish optimised path & velocity
        self._publish_path(opt_xs, opt_ys, segment[0].header.frame_id)
        self.cmd_vel_pub.publish(self._compute_cmd_vel(opt_xs, opt_ys))

    # ──────────────────────────────────────────────────────────────────────
    # Costmap -> Points
    # ──────────────────────────────────────────────────────────────────────

    def _costmap_to_points(self, threshold: int | None = None
                           ) -> list[tuple[float, float]]:
        """
        Extract occupied cells as point obstacles.
        Costmap is in odom frame, so points are returned in odom frame.

        Parameters
        ----------
        threshold : int, optional
            Cost cutoff. Cells with cost >= threshold are emitted. Defaults
            to ``self.obstacle_threshold``. Pass a higher value to extract
            only the high-cost cores (used by clustering).
        """
        if threshold is None:
            threshold = self.obstacle_threshold

        cm   = self._local_costmap
        info = cm.info
        w, h = info.width, info.height
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        data = cm.data

        # Restrict scan to a window around the robot for performance.
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        margin_cells = int((self.lookahead_dist + self.search_margin) / res)
        cx_robot = int((rx - ox) / res)
        cy_robot = int((ry - oy) / res)
        i_lo = max(0, cx_robot - margin_cells)
        i_hi = min(w, cx_robot + margin_cells + 1)
        j_lo = max(0, cy_robot - margin_cells)
        j_hi = min(h, cy_robot + margin_cells + 1)

        points = []
        for j in range(j_lo, j_hi):
            base = j * w
            for i in range(i_lo, i_hi):
                if data[base + i] >= threshold:
                    wx = ox + (i + 0.5) * res
                    wy = oy + (j + 0.5) * res
                    points.append((wx, wy))
        return points

    # ──────────────────────────────────────────────────────────────────────
    # DBSCAN clustering
    # ──────────────────────────────────────────────────────────────────────

    def _cluster_points(self,
                        points: list[tuple[float, float]]
                        ) -> list[Cluster]:
        """
        Group raw obstacle points into obstacle entities via DBSCAN.

        Returns an empty list when clustering is disabled or no points are
        supplied. Noise points (isolated cells, sensor flicker) are dropped.
        """
        if not self.use_dbscan or not points:
            return []
        # Optional second-tier threshold: re-extract using a stricter cost
        # cut so clustering tracks obstacle cores rather than the inflation
        # halo. Only applies when the cluster threshold is strictly higher
        # than the EB obstacle threshold (otherwise we'd waste a scan).
        if self.dbscan_cost_threshold > self.obstacle_threshold:
            points = self._costmap_to_points(self.dbscan_cost_threshold)
            if not points:
                return []
        # Pass the costmap resolution so cluster_obstacles can compute
        # solidity per cluster and flag concave shapes whose convex hull
        # would bridge over free space.
        cell_size = (self._local_costmap.info.resolution
                     if self._local_costmap is not None else None)
        return cluster_obstacles(
            points,
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            cell_size=cell_size,
            solidity_threshold=self.dbscan_solidity_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Point distance / obstacle force
    # ──────────────────────────────────────────────────────────────────────

    def _obstacle_force(self, x: float, y: float) -> tuple[float, float]:
        """
        Repulsive force on a band waypoint, computed per cluster.

        Every cluster contributes via its convex-hull polygon — the hull
        is the obstacle's *range*, so polygon-edge distance is the right
        signal whether the cluster is solid or not. The reliability flag
        only changes what happens *inside* the hull:

        Reliable clusters (solid / line / convex-enough; ``c.reliable``):
            Signed-distance polygon repulsion. Edges within
            ``teb_inflation`` contribute ``(inflation - d) * outward_normal``;
            inside-polygon points (``d < 0``) produce the strongest push.

        Unreliable clusters (concave shapes — L, U, ring; ``not c.reliable``):
            Polygon-edge force when the query is OUTSIDE the hull
            (``d >= 0``) — the hull is still an honest over-approximation
            of where the obstacle's extent ends, so we get the proper
            "range" force. When the query is INSIDE the hull we cannot
            trust it (might be free space in the L's mouth), so we fall
            back to per-cell nearest-member repulsion against this
            cluster's points only.

        Forces from all clusters sum, so the band balances correctly when
        squeezed between mixed reliable / unreliable obstacles.
        """
        inflation = self.teb_inflation
        fx = 0.0
        fy = 0.0

        if self._obstacle_clusters:
            for c in self._obstacle_clusters:
                # Polygon path: full signed-distance for reliable
                # clusters; outside-only for unreliable ones.
                if len(c.hull) >= 2:
                    d, nx, ny = point_to_convex_polygon(x, y, c.hull)
                    if c.reliable or d >= 0.0:
                        if d < inflation:
                            mag = inflation - d
                            fx += mag * nx
                            fy += mag * ny
                        continue
                    # Unreliable + inside hull: per-cell fallback below.
                # Per-cell fallback: degenerate hull, or unreliable
                # cluster with the query inside its hull.
                best_d  = float('inf')
                best_px = 0.0
                best_py = 0.0
                for px, py in c.points:
                    d = math.hypot(x - px, y - py)
                    if d < best_d:
                        best_d, best_px, best_py = d, px, py
                if best_d >= inflation:
                    continue
                vx, vy = x - best_px, y - best_py
                m = math.hypot(vx, vy)
                if m < 1e-9:
                    continue
                mag = inflation - best_d
                fx += mag * vx / m
                fy += mag * vy / m
            return fx, fy

        # ── Final fallback when DBSCAN is disabled entirely ──────────────
        if not self._obstacle_points:
            return 0.0, 0.0
        best_d  = float('inf')
        best_px = 0.0
        best_py = 0.0
        for px, py in self._obstacle_points:
            d = math.hypot(x - px, y - py)
            if d < best_d:
                best_d, best_px, best_py = d, px, py
        if best_d >= inflation:
            return 0.0, 0.0
        vx, vy = x - best_px, y - best_py
        m = math.hypot(vx, vy)
        if m < 1e-9:
            return 0.0, 0.0
        mag = inflation - best_d
        return mag * vx / m, mag * vy / m

    # ──────────────────────────────────────────────────────────────────────
    # Homotopy correction
    # ──────────────────────────────────────────────────────────────────────

    def _apply_homotopy_correction(self,
                                   xs: list[float],
                                   ys: list[float]) -> None:
        """
        Snap the band out of any reliable cluster polygon it has entered.

        The Elastic Band is a *local* optimiser: gradient descent can
        slide waypoints along free space, but it cannot move the band
        across an obstacle. So if the warm-started band — or the global
        reference path that seeds it — threads through a cluster
        polygon, the band is on the wrong homotopy class and no amount
        of iteration will fix it (it just sticks against opposing
        edges, sometimes flipping sides cycle-to-cycle).

        DBSCAN gives us each obstacle's hull, which is exactly the
        polygon we need to test homotopy against:

          (1) **Inside-snap.** Any interior waypoint that lies *inside*
              a hull is teleported along the outward normal until it sits
              ``teb_homotopy_margin`` beyond the nearest edge.
          (2) **Crossing-snap.** Any band edge whose endpoints are both
              outside the hull but whose interior crosses it (the
              classic "tunnels through the wall" failure) is repaired
              by lifting its midpoint out via the same outward normal,
              propagated to whichever endpoint(s) are interior band
              waypoints (the band's own start/end are fixed).

        Only ``reliable`` clusters participate — for concave (low-
        solidity) clusters the hull bridges free space, so a segment
        crossing it is *not* a homotopy violation. Those clusters fall
        back to per-cell repulsion in :meth:`_obstacle_force` instead.
        """
        if not self._obstacle_clusters:
            return
        n = len(xs)
        if n < 3:
            return
        margin = self.teb_homotopy_margin

        for c in self._obstacle_clusters:
            if not c.reliable or len(c.hull) < 3:
                continue
            hull = c.hull

            # (1) Inside-snap: lift any interior waypoint sitting inside
            # the hull straight out along the outward normal.
            for i in range(1, n - 1):
                d, nx, ny = point_to_convex_polygon(xs[i], ys[i], hull)
                if d < 0.0:
                    push = (-d) + margin
                    xs[i] += push * nx
                    ys[i] += push * ny

            # (2) Crossing-snap: any band segment that still cuts through
            # the hull is repaired by displacing its interior endpoint(s)
            # along the outward normal at the segment midpoint.
            for i in range(n - 1):
                p1 = (xs[i],     ys[i])
                p2 = (xs[i + 1], ys[i + 1])
                if not segment_intersects_polygon(p1, p2, hull):
                    continue
                mx = 0.5 * (p1[0] + p2[0])
                my = 0.5 * (p1[1] + p2[1])
                d_m, nx_m, ny_m = point_to_convex_polygon(mx, my, hull)
                push = max(0.0, -d_m) + margin
                # Endpoints of the band itself (i==0 / i==n-1) are
                # anchored; only push interior waypoints.
                if 0 < i < n - 1:
                    xs[i]     += push * nx_m
                    ys[i]     += push * ny_m
                if 0 < i + 1 < n - 1:
                    xs[i + 1] += push * nx_m
                    ys[i + 1] += push * ny_m

    # ──────────────────────────────────────────────────────────────────────
    # TEB optimisation
    # ──────────────────────────────────────────────────────────────────────

    def _teb_optimize(self,
                      segment: list[PoseStamped]
                      ) -> tuple[list[float], list[float], list[float]]:
        """
        Elastic band optimisation against point obstacles.

        Anchoring:
          The band is anchored at the *actual* robot pose (xs[0]) — not at
          the closest global-path waypoint. 

        Warm start:
          Across cycles the previous solution is re-projected onto the new
          band by *arc-length* parameterisation, not nearest-neighbour. This
          preserves the avoidance shape even when the lateral deviation is
          larger than the global-path waypoint spacing — preventing the
          left/right flip-flop that produces in-and-out motion.

        Endpoints (start = robot, end = lookahead goal) are fixed; only
        interior nodes move. Per-iteration motion is clamped so the band
        cannot teleport across narrow gaps.
        """
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y

        # Initial band: robot pose + global-path waypoints.
        # Skip waypoints that would create a degenerate first segment.
        xs = [rx]
        ys = [ry]
        skip_dist = 0.4
        for p in segment:
            px, py = p.pose.position.x, p.pose.position.y
            if math.hypot(px - xs[-1], py - ys[-1]) > skip_dist:
                xs.append(px)
                ys.append(py)

        n = len(xs)
        if n < 2:
            return xs, ys, [0.1]

        # Initialize time intervals based on max velocity
        dts = []
        for i in range(n - 1):
            dist = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            dts.append(max(0.01, dist / self.v_max))

        # ── Warm start: arc-length re-projection of previous band ────────
        if self._warm_xs and self._warm_ys and len(self._warm_xs) >= 2:
            warm_xs, warm_ys = self._warm_xs, self._warm_ys
            warm_arc = [0.0]
            for k in range(1, len(warm_xs)):
                warm_arc.append(warm_arc[-1] + math.hypot(
                    warm_xs[k] - warm_xs[k - 1],
                    warm_ys[k] - warm_ys[k - 1]))
            L_warm = warm_arc[-1] # old length

            new_arc = [0.0]
            for k in range(1, n):
                new_arc.append(new_arc[-1] + math.hypot(
                    xs[k] - xs[k - 1], ys[k] - ys[k - 1]))
            L_new = new_arc[-1]  # new length

            # Interpolating new length
            if L_warm > 1e-3 and L_new > 1e-3:
                for i in range(1, n - 1):
                    target = (new_arc[i] / L_new) * L_warm
                    for k in range(len(warm_arc) - 1):
                        if warm_arc[k + 1] >= target:
                            span = warm_arc[k + 1] - warm_arc[k]
                            if span < 1e-9:
                                xs[i], ys[i] = warm_xs[k], warm_ys[k]
                            else:
                                a = (target - warm_arc[k]) / span
                                xs[i] = warm_xs[k] + a * (warm_xs[k + 1] - warm_xs[k])
                                ys[i] = warm_ys[k] + a * (warm_ys[k + 1] - warm_ys[k])
                            break

        step        = self.teb_step
        max_delta   = self.teb_max_delta
        w_obs       = self.teb_w_obs
        w_smooth    = self.teb_w_smooth

        # Initial homotopy fix on the warm-started band — if the
        # reference / warm trajectory threads a cluster polygon, snap it
        # out *before* gradient descent starts so iterations converge in
        # a valid topology rather than fighting an unreachable minimum.
        if self.teb_homotopy_correction:
            self._apply_homotopy_correction(xs, ys)

        for it in range(self.teb_n_iter):
            # Accumulate forces
            fx = [0.0] * n
            fy = [0.0] * n

            # 1. Obstacle & Smoothness forces
            for i in range(1, n - 1):
                fx_obs, fy_obs = self._obstacle_force(xs[i], ys[i])

                # Smoothness force (elastic band): pull toward midpoint.
                f_sx = (xs[i - 1] + xs[i + 1]) / 2.0 - xs[i]
                f_sy = (ys[i - 1] + ys[i + 1]) / 2.0 - ys[i]

                fx[i] += w_obs * fx_obs + w_smooth * f_sx
                fy[i] += w_obs * fy_obs + w_smooth * f_sy

            # 3. Apply updates
            for i in range(1, n - 1):
                dx = step * fx[i]
                dy = step * fy[i]
                mag = math.hypot(dx, dy)
                if mag > max_delta:
                    dx *= max_delta / mag
                    dy *= max_delta / mag
                xs[i] += dx
                ys[i] += dy

            # 4. Periodic homotopy correction. Gradient descent may
            #    nudge a waypoint into a polygon as the band is being
            #    pulled toward smoothness; re-snap so that the next
            #    iterations don't try to optimise an invalid topology.
            if (self.teb_homotopy_correction and
                self.teb_homotopy_every > 0 and
                (it + 1) % self.teb_homotopy_every == 0):
                self._apply_homotopy_correction(xs, ys)

        # Final pass: guarantee the published band is outside every
        # reliable cluster polygon, even if the last gradient step
        # nudged a waypoint just inside.
        if self.teb_homotopy_correction:
            self._apply_homotopy_correction(xs, ys)

        return xs, ys

    # ──────────────────────────────────────────────────────────────────────
    # Velocity extraction (True TEB execution)
    # ──────────────────────────────────────────────────────────────────────
    def _compute_cmd_vel(self,
                         opt_xs: list[float],
                         opt_ys: list[float]) -> TwistStamped:
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if not opt_xs or len(opt_xs) < 2:
            return cmd

        rx  = self._robot_pose.position.x
        ry  = self._robot_pose.position.y
        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 1. Decelerate near goal.
        d_goal = euclidean(self._robot_pose, self._global_path.poses[-1].pose)
        
        # 2. True TEB velocity: velocity of the first segment
        # In a real TEB, the robot is meant to follow the trajectory encoded in the first segment
        dxw = opt_xs[1] - opt_xs[0]
        dyw = opt_ys[1] - opt_ys[0]

        # Convert world-frame velocity to body-frame velocity
        vx_body =  dxw * cos_y + dyw * sin_y
        vy_body = -dxw * sin_y + dyw * cos_y

        # 3. Apply deceleration near goal and ensure we do not exceed v_max
        v_target_limit = min(self.v_max, max(0.05, d_goal * 1.5))
        v_mag = math.hypot(vx_body, vy_body)
        if v_mag > v_target_limit:
            vx_body *= v_target_limit / v_mag
            vy_body *= v_target_limit / v_mag

        cmd.twist.linear.x  = vx_body
        cmd.twist.linear.y  = vy_body
        cmd.twist.angular.z = 0.0
        return cmd

    def _pub_stop(self):
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(cmd)

    # ──────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────

    def _publish_points(self):
        ma = MarkerArray()
        frame = self._local_costmap.header.frame_id
        stamp = self.get_clock().now().to_msg()

        # Clear previous points before drawing new ones.
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # Clustered view: one LINE_STRIP per cluster traces its convex
        # hull (the actual obstacle primitive consumed by the EB), plus a
        # SPHERE_LIST of centroids in yellow. For 1- or 2-point clusters
        # (degenerate hulls) we still emit a small POINTS marker so the
        # debug view stays consistent.
        if self._obstacle_clusters:
            centroids = Marker()
            centroids.header.frame_id = frame
            centroids.header.stamp    = stamp
            centroids.ns              = 'cluster_centroid'
            centroids.id              = 0
            centroids.type            = Marker.SPHERE_LIST
            centroids.action          = Marker.ADD
            centroids.scale.x = centroids.scale.y = centroids.scale.z = 0.15
            centroids.color           = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
            centroids.pose.orientation.w = 1.0

            for k, cluster in enumerate(self._obstacle_clusters):
                color = self._cluster_color(k)
                if cluster.reliable and len(cluster.hull) >= 2:
                    # Reliable: draw the convex hull (line segment for n=2,
                    # closed polygon for n>=3). This is the actual obstacle
                    # primitive the EB sees.
                    m = Marker()
                    m.header.frame_id = frame
                    m.header.stamp    = stamp
                    m.ns              = 'cluster_hull'
                    m.id              = k
                    m.type            = Marker.LINE_STRIP
                    m.action          = Marker.ADD
                    m.scale.x         = 0.04
                    m.color           = color
                    m.pose.orientation.w = 1.0
                    for (hx, hy) in cluster.hull:
                        p = Point(); p.x, p.y, p.z = hx, hy, 0.0
                        m.points.append(p)
                    if len(cluster.hull) >= 3:
                        # Close the polygon loop.
                        h0x, h0y = cluster.hull[0]
                        p = Point(); p.x, p.y, p.z = h0x, h0y, 0.0
                        m.points.append(p)
                    ma.markers.append(m)
                else:
                    # Unreliable (concave, low solidity) — render member
                    # cells instead of the bridging hull, so it's visually
                    # obvious that this cluster is repelling per-cell.
                    m = Marker()
                    m.header.frame_id = frame
                    m.header.stamp    = stamp
                    m.ns              = 'cluster_points'
                    m.id              = k
                    m.type            = Marker.POINTS
                    m.action          = Marker.ADD
                    m.scale.x = m.scale.y = 0.06
                    m.color           = color
                    m.pose.orientation.w = 1.0
                    for (px, py) in cluster.points:
                        p = Point(); p.x, p.y, p.z = px, py, 0.0
                        m.points.append(p)
                    ma.markers.append(m)

                cx, cy = cluster.centroid
                cpt = Point()
                cpt.x, cpt.y, cpt.z = cx, cy, 0.0
                centroids.points.append(cpt)
            ma.markers.append(centroids)

        elif self._obstacle_points:
            # DBSCAN disabled or produced no clusters — fall back to the
            # original flat red point cloud so the node stays usable.
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = stamp
            m.ns              = 'obstacle_points'
            m.id              = 0
            m.type            = Marker.POINTS
            m.action          = Marker.ADD
            m.scale.x         = 0.05
            m.scale.y         = 0.05
            m.color           = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            m.pose.orientation.w = 1.0
            for (px, py) in self._obstacle_points:
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.0
                m.points.append(pt)
            ma.markers.append(m)

        self.obstacle_pub.publish(ma)

    @staticmethod
    def _cluster_color(k: int) -> ColorRGBA:
        """Cycle through a small palette so adjacent clusters look distinct."""
        palette = (
            (1.0, 0.2, 0.2),  # red
            (0.2, 0.8, 1.0),  # cyan
            (0.2, 1.0, 0.4),  # green
            (1.0, 0.6, 0.1),  # orange
            (0.8, 0.4, 1.0),  # purple
            (1.0, 1.0, 0.2),  # yellow
        )
        r, g, b = palette[k % len(palette)]
        return ColorRGBA(r=r, g=g, b=b, a=1.0)

    def _publish_path(self, xs: list[float], ys: list[float], frame: str):
        n = len(xs)
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = frame
        for i in range(n):
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
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    # ──────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────

    def _find_closest_index(self) -> int:
        wp = self._global_path.poses
        idx, best = self._prune_index, float('inf')
        for i in range(self._prune_index, len(wp)):
            d = euclidean(self._robot_pose, wp[i].pose)
            if d < best:
                best, idx = d, i
        return idx

    def _extract_local_segment(self) -> list[PoseStamped]:
        wp = self._global_path.poses
        seg = []
        for i in range(self._prune_index, len(wp)):
            d = euclidean(self._robot_pose, wp[i].pose)
            if d <= self.lookahead_dist:
                seg.append(wp[i])
            else:
                break
        return seg


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
