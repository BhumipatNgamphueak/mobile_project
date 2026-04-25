#!/usr/bin/env python3
"""
path_planning_node.py
=====================
TEB-style local planner with polygon obstacles.

Pipeline (one cycle):
  1. Costmap -> polygon obstacles
       BFS-cluster lethal cells in a window around the robot, then take the
       convex hull of each cluster's world-frame centers.
       (nav2 analog: costmap_converter::CostmapToPolygonsDBSMCCH)
  2. Local segment extraction
       Prune behind the robot, then take a lookahead window of the global
       path. Warm-started from the previous optimised band so the trajectory
       evolves smoothly across cycles.
  3. TEB optimisation
       Gradient descent on the elastic band against
         - polygon obstacle edges (outward force inside an inflation zone)
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
/obstacle_polygons   visualization_msgs/MarkerArray   extracted polygons (RViz)
"""

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── General parameters ────────────────────────────────────────────
        self.planning_freq          = 10     # Hz
        self.lookahead_dist         = 15.0   # m — local segment length
        self.min_waypoints_required = 2
        self.goal_tolerance         = 0.4    # m

        # ── TEB parameters ────────────────────────────────────────────────
        self.teb_n_iter      = 60      # gradient descent iterations / cycle
        self.teb_w_obs       = 50.0 # 50.0    # obstacle repulsion weight
        self.teb_w_smooth    = 2.0     # elastic band weight
        self.teb_inflation   = 1.0 # 0.5     # m — repulsion radius around polygons
        self.teb_step        = 0.05    # gradient step
        self.teb_max_delta   = 0.10    # m — clamp per-iteration movement
        self.v_max           = 1.5     # m/s

        # ── Costmap-to-polygons parameters ────────────────────────────────
        self.obstacle_threshold = 60    # cost >= this -> obstacle cell
        self.cluster_min_cells  = 3     # filter noise clusters
        self.search_margin      = 8.0   # m beyond lookahead to scan

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,       '/odom',          self._odom_callback,          10)
        self.create_subscription(Path,           '/global_path',   self._global_path_callback,   10)
        self.create_subscription(OccupancyGrid,  '/local_costmap', self._local_costmap_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel',           10)
        self.path_pub    = self.create_publisher(Path,         '/planned_path',      10)
        self.polygon_pub = self.create_publisher(MarkerArray,  '/obstacle_polygons', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._robot_pose: Pose | None             = None
        self._robot_vel                           = None
        self._global_path: Path | None            = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int                    = 0

        # Warm start from previous optimised band.
        self._warm_xs: list[float] | None = None
        self._warm_ys: list[float] | None = None

        # Polygons extracted from current costmap (set each cycle).
        self._polygons: list[list[tuple[float, float]]] = []

        self.create_timer(1.0 / self.planning_freq, self._replan)
        self.get_logger().info('PathPlanningNode (TEB polygon-style) started')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        self._robot_pose = msg.pose.pose
        self._robot_vel  = msg.twist.twist

    def _local_costmap_callback(self, msg: OccupancyGrid):
        self._local_costmap = msg

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

    def _replan(self):
        if self._robot_pose is None or self._global_path is None or self._local_costmap is None:
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            return

        if euclidean(self._robot_pose, self._global_path.poses[-1].pose) < self.goal_tolerance:
            self._pub_stop()
            return

        # 1. Costmap -> polygon obstacles
        self._polygons = self._costmap_to_polygons()
        self._publish_polygons()

        # 2. Reference segment from global path
        self._prune_index = self._find_closest_index()
        segment = self._extract_local_segment()
        if len(segment) < self.min_waypoints_required:
            return

        # 3. TEB optimisation
        opt_xs, opt_ys = self._teb_optimize(segment)

        # 4. Cache for next cycle's warm start
        self._warm_xs = opt_xs[:]
        self._warm_ys = opt_ys[:]

        # 5. Publish optimised path & velocity
        self._publish_path(opt_xs, opt_ys, segment[0].header.frame_id)
        self.cmd_vel_pub.publish(self._compute_cmd_vel(opt_xs, opt_ys))

    # ──────────────────────────────────────────────────────────────────────
    # Costmap -> Polygons (costmap_converter style)
    # ──────────────────────────────────────────────────────────────────────

    def _costmap_to_polygons(self) -> list[list[tuple[float, float]]]:
        """
        Cluster lethal cells with 8-connected BFS, then convex-hull each
        cluster's world-frame centers. Equivalent in spirit to nav2's
        costmap_converter::CostmapToPolygonsDBSMCCH (DBSCAN + convex hull),
        with grid connectivity instead of DBSCAN for speed in pure Python.

        Costmap is in odom frame, so polygons are returned in odom frame.
        """
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

        # 1. Collect occupied cells in the window.
        threshold = self.obstacle_threshold
        occupied: set[tuple[int, int]] = set()
        for j in range(j_lo, j_hi):
            base = j * w
            for i in range(i_lo, i_hi):
                if data[base + i] >= threshold:
                    occupied.add((i, j))

        if not occupied:
            return []

        # 2. BFS clustering with 8-connectivity.
        polygons: list[list[tuple[float, float]]] = []
        visited: set[tuple[int, int]] = set()
        neighbors = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

        for start in occupied:
            if start in visited:
                continue
            cluster = []
            stack = [start]
            while stack:
                c = stack.pop()
                if c in visited:
                    continue
                visited.add(c)
                cluster.append(c)
                ci, cj = c
                for di, dj in neighbors:
                    n = (ci + di, cj + dj)
                    if n in occupied and n not in visited:
                        stack.append(n)

            if len(cluster) < self.cluster_min_cells:
                continue

            # 3. Convex hull of cluster (world frame).
            world_pts = [(ox + (i + 0.5) * res, oy + (j + 0.5) * res)
                         for i, j in cluster]
            hull = self._convex_hull(world_pts)
            if len(hull) >= 2:
                polygons.append(hull)

        return polygons

    @staticmethod
    def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Andrew's monotone chain. Returns CCW hull (open polyline, no repeat)."""
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    # ──────────────────────────────────────────────────────────────────────
    # Polygon distance / obstacle force
    # ──────────────────────────────────────────────────────────────────────

    def _obstacle_force(self, x: float, y: float) -> tuple[float, float]:
        """
        Returns the repulsive force on a waypoint from the closest polygon.
        Mirrors TEB's PolygonObstacle obstacle edge:
          - force = 0 outside the inflation radius
          - force grows linearly with penetration depth
          - direction: away from nearest polygon-boundary point
        Inside a polygon -> strong push toward the nearest boundary (escape).
        """
        if not self._polygons:
            return 0.0, 0.0

        best_d      = float('inf')
        best_cx     = 0.0
        best_cy     = 0.0
        best_inside = False

        for poly in self._polygons:
            d, cx, cy, inside = self._point_polygon_signed_dist(x, y, poly)
            # Once we know we're inside any polygon, prefer escaping it.
            if inside:
                if not best_inside or d < best_d:
                    best_d, best_cx, best_cy, best_inside = d, cx, cy, True
            elif not best_inside and d < best_d:
                best_d, best_cx, best_cy = d, cx, cy

        if best_inside:
            # Escape direction: toward the nearest boundary point.
            vx, vy = best_cx - x, best_cy - y
            m = math.hypot(vx, vy)
            if m < 1e-9:
                return self.teb_inflation, 0.0
            return self.teb_inflation * vx / m, self.teb_inflation * vy / m

        if best_d >= self.teb_inflation:
            return 0.0, 0.0

        # Outside, but inside inflation zone: push away from boundary.
        vx, vy = x - best_cx, y - best_cy
        m = math.hypot(vx, vy)
        if m < 1e-9:
            return 0.0, 0.0
        mag = self.teb_inflation - best_d
        return mag * vx / m, mag * vy / m

    @staticmethod
    def _point_polygon_signed_dist(
            x: float, y: float,
            poly: list[tuple[float, float]]
    ) -> tuple[float, float, float, bool]:
        """
        Returns (distance_to_boundary, closest_x, closest_y, inside).
        Distance is unsigned distance to polygon boundary.
        `inside` is True when (x, y) is strictly inside the polygon.
        """
        n = len(poly)
        if n == 0:
            return float('inf'), x, y, False
        if n == 1:
            return math.hypot(x - poly[0][0], y - poly[0][1]), poly[0][0], poly[0][1], False

        # Closest point on polygon boundary.
        best_d = float('inf')
        best_x = x
        best_y = y
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            l2 = dx * dx + dy * dy
            if l2 < 1e-12:
                cx, cy = x1, y1
            else:
                t = ((x - x1) * dx + (y - y1) * dy) / l2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                cx, cy = x1 + t * dx, y1 + t * dy
            d = math.hypot(x - cx, y - cy)
            if d < best_d:
                best_d, best_x, best_y = d, cx, cy

        # Inside test (ray casting). Skip for degenerate polygons.
        inside = False
        if n >= 3:
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > y) != (yj > y)) and \
                   (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
                j = i

        return best_d, best_x, best_y, inside

    # ──────────────────────────────────────────────────────────────────────
    # Homotopy pre-routing
    # ──────────────────────────────────────────────────────────────────────

    def _preroute_around_polygons(self,
                                  xs: list[float],
                                  ys: list[float]) -> None:
        """
        For each polygon containing band nodes, displace the affected
        nodes to one consistent side of the polygon (the side with the
        shorter perpendicular detour). This forces the band into a valid
        homotopy class so the local optimiser doesn't have to choose
        between left/right under symmetric obstacle forces.

        nav2 TEB does this implicitly by exploring multiple homotopy
        classes in parallel; here we approximate it with a single, stable
        choice per polygon per cycle.
        """
        n = len(xs)
        if n < 3:
            return

        for poly in self._polygons:
            if len(poly) < 3:
                continue

            # Find interior band nodes inside this polygon (skip endpoints).
            inside_idx = []
            for i in range(1, n - 1):
                _, _, _, inside = self._point_polygon_signed_dist(xs[i], ys[i], poly)
                if inside:
                    inside_idx.append(i)
            if not inside_idx:
                continue

            i_first = inside_idx[0]
            i_last  = inside_idx[-1]
            i_pre   = max(0,     i_first - 1)
            i_post  = min(n - 1, i_last  + 1)

            # Band direction across the polygon.
            bx = xs[i_post] - xs[i_pre]
            by = ys[i_post] - ys[i_pre]
            bm = math.hypot(bx, by)
            if bm < 1e-6:
                continue
            bx, by = bx / bm, by / bm

            # Perpendicular candidates: (-by, bx) and (by, -bx).
            # Reference point: midpoint between first and last inside nodes.
            cxr = (xs[i_first] + xs[i_last]) / 2.0
            cyr = (ys[i_first] + ys[i_last]) / 2.0

            max_left  = 0.0   # extent in direction (-by, bx)
            max_right = 0.0   # extent in direction ( by,-bx)
            for vx, vy in poly:
                d_left  = (vx - cxr) * (-by) + (vy - cyr) * bx
                d_right = (vx - cxr) * by   + (vy - cyr) * (-bx)
                if d_left  > max_left:  max_left  = d_left
                if d_right > max_right: max_right = d_right

            if max_left <= max_right:
                perp_x, perp_y, extent = -by, bx, max_left
            else:
                perp_x, perp_y, extent =  by, -bx, max_right

            offset = extent + self.teb_inflation + 0.1
            for i in inside_idx:
                xs[i] += offset * perp_x
                ys[i] += offset * perp_y

    # ──────────────────────────────────────────────────────────────────────
    # TEB optimisation
    # ──────────────────────────────────────────────────────────────────────

    def _teb_optimize(self,
                      segment: list[PoseStamped]
                      ) -> tuple[list[float], list[float]]:
        """
        Elastic band optimisation against polygon obstacles.

        Anchoring:
          The band is anchored at the *actual* robot pose (xs[0]) — not at
          the closest global-path waypoint. This matters when the robot has
          deviated laterally to avoid an obstacle: anchoring at the global
          path would put xs[0] inside the inflation zone and the smoothness
          force would keep dragging the band back into the obstacle.

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
            return xs, ys

        # ── Warm start: arc-length re-projection of previous band ────────
        if self._warm_xs and self._warm_ys and len(self._warm_xs) >= 2:
            warm_xs, warm_ys = self._warm_xs, self._warm_ys
            warm_arc = [0.0]
            for k in range(1, len(warm_xs)):
                warm_arc.append(warm_arc[-1] + math.hypot(
                    warm_xs[k] - warm_xs[k - 1],
                    warm_ys[k] - warm_ys[k - 1]))
            L_warm = warm_arc[-1]

            new_arc = [0.0]
            for k in range(1, n):
                new_arc.append(new_arc[-1] + math.hypot(
                    xs[k] - xs[k - 1], ys[k] - ys[k - 1]))
            L_new = new_arc[-1]

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

        # ── Homotopy pre-routing ─────────────────────────────────────────
        # Local optimization cannot escape a polygon that fully contains
        # the band — the obstacle force balances on both sides and the
        # band gets pinned. Push any band nodes still inside a polygon
        # to one consistent side of that polygon (whichever has the
        # shorter detour) before optimising.
        self._preroute_around_polygons(xs, ys)

        step      = self.teb_step
        max_delta = self.teb_max_delta
        w_obs     = self.teb_w_obs
        w_smooth  = self.teb_w_smooth

        for _ in range(self.teb_n_iter):
            for i in range(1, n - 1):
                fx_obs, fy_obs = self._obstacle_force(xs[i], ys[i])

                # Smoothness force (elastic band): pull toward midpoint.
                f_sx = (xs[i - 1] + xs[i + 1]) / 2.0 - xs[i]
                f_sy = (ys[i - 1] + ys[i + 1]) / 2.0 - ys[i]

                dx = step * (w_obs * fx_obs + w_smooth * f_sx)
                dy = step * (w_obs * fy_obs + w_smooth * f_sy)

                mag = math.hypot(dx, dy)
                if mag > max_delta:
                    dx *= max_delta / mag
                    dy *= max_delta / mag

                xs[i] += dx
                ys[i] += dy

        return xs, ys

    # ──────────────────────────────────────────────────────────────────────
    # Velocity extraction (holonomic vector pursuit)
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

        # 1. Lookahead point on the optimised band.
        lookahead = 0.5
        tx, ty = opt_xs[-1], opt_ys[-1]
        for ox, oy in zip(opt_xs, opt_ys):
            if math.hypot(ox - rx, oy - ry) >= lookahead:
                tx, ty = ox, oy
                break

        # 2. Body-frame vector to lookahead.
        dxw = tx - rx
        dyw = ty - ry
        xl =  dxw * cos_y + dyw * sin_y
        yl = -dxw * sin_y + dyw * cos_y
        L  = math.hypot(xl, yl)
        if L < 1e-6:
            return cmd

        # 3. Decelerate near goal.
        d_goal   = euclidean(self._robot_pose, self._global_path.poses[-1].pose)
        v_target = min(self.v_max, max(0.05, d_goal * 1.5))

        # 4. Holonomic distribution (slide directly toward lookahead).
        cmd.twist.linear.x  = v_target * (xl / L)
        cmd.twist.linear.y  = v_target * (yl / L)
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

    def _publish_polygons(self):
        ma = MarkerArray()
        frame = self._local_costmap.header.frame_id
        stamp = self.get_clock().now().to_msg()

        # Clear previous polygons before drawing new ones.
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for idx, poly in enumerate(self._polygons):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = stamp
            m.ns              = 'obstacle_polygons'
            m.id              = idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.05
            m.color           = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            m.pose.orientation.w = 1.0
            for (px, py) in poly:
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.0
                m.points.append(pt)
            if len(poly) >= 3:
                first = Point()
                first.x, first.y, first.z = poly[0][0], poly[0][1], 0.0
                m.points.append(first)
            ma.markers.append(m)

        self.polygon_pub.publish(ma)

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
