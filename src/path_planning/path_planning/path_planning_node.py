#!/usr/bin/env python3
"""
path_planning_node.py
=====================
Pure-Python TEB local planner for a holonomic mobile robot with fixed yaw.

This is a Python port of the timed-elastic-band approach
(rst-tu-dortmund/teb_local_planner). The original uses g2o for sparse
non-linear least squares; here we model the same hyper-graph (poses +
time-diffs as vertices, soft penalties as edges) and solve it with
scipy.optimize.least_squares (trust-region reflective) using an analytic
sparse Jacobian.

Yaw is fixed (omega = 0). Vertex is (x, y) only — no theta, no kinematic
rolling constraint. The robot moves laterally (mecanum-style); cmd_vel is
the world-frame velocity of the first segment, rotated into base_link.

State vector (n poses, endpoints fixed):
    [x_1, y_1, ..., x_{n-2}, y_{n-2}, dt_0, ..., dt_{n-2}]
    size = 2*(n-2) + (n-1)

Residuals (one per "edge" in the hyper-graph):
    time-optimal   : r = w_t   * dt_i                       per segment
    shortest-path  : r = w_s   * ||p_{i+1} - p_i||          per segment
    obstacle (hard): r = w_o   * max(0, d_min     - d_i)    per interior pose
    obstacle (inf) : r = w_inf * max(0, d_inflate - d_i)    per interior pose

Bounds enforced by the solver:
    dt_i in [dt_min, dt_max]
    poses unbounded.

v_max is enforced post-hoc as a clamp on the extracted cmd_vel — the user
elected to skip explicit velocity edges in the first cut.

Topics:
    /odom           nav_msgs/Odometry        in
    /global_path    nav_msgs/Path            in
    /local_costmap  nav_msgs/OccupancyGrid   in
    /cmd_vel        geometry_msgs/TwistStamped       out
    /planned_path   nav_msgs/Path                    out
    /obstacle_points visualization_msgs/MarkerArray  out
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


# ─────────────────────────────────────────────────────────────────────────
# TEB optimizer (pure numpy/scipy — no ROS deps)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class TebParams:
    # Weights
    w_time: float       = 5.0     # time-optimal
    w_path: float       = 1.0     # shortest path
    w_obs:  float       = 50.0    # hard obstacle
    w_inf:  float       = 10.0    # inflation

    # Obstacle distances
    d_min:     float    = 0.30    # m — hard min clearance
    d_inflate: float    = 0.80    # m — soft inflation radius

    # Time-diff bounds (segment duration)
    dt_min: float       = 0.05
    dt_max: float       = 1.50
    dt_ref: float       = 0.30    # auto-resize target
    dt_hyst: float      = 0.10    # auto-resize hysteresis

    # Trajectory size
    n_min: int          = 4
    n_max: int          = 60

    # Solver
    max_iter: int       = 8       # LM iterations / cycle (kept tight for 10 Hz)
    ftol:     float     = 1e-3
    xtol:     float     = 1e-3

    # Obstacle pairing (per-pose top-K nearest, sum-of-forces)
    obs_top_k:    int   = 6       # max obstacles per pose included in cost
    obs_margin:   float = 1.5     # multiplier on d_inflate for inclusion radius

    # Kinematic
    v_max:    float     = 1.5     # m/s — output cmd_vel clamp


class TebOptimizer:
    """One-shot TEB optimization of an elastic band against point obstacles.

    The hyper-graph is rebuilt every call. State is held externally (the
    ROS node passes in xs/ys/dts and gets back the optimized values), so
    this class is stateless across calls except for the parameters.
    """

    def __init__(self, params: TebParams):
        self.p = params

    # ───── public entrypoint ─────────────────────────────────────────────
    def optimize(self,
                 xs: list[float],
                 ys: list[float],
                 dts: list[float],
                 obstacles: np.ndarray
                 ) -> tuple[list[float], list[float], list[float]]:
        """Run one round of LM on the band.

        xs, ys: pose lists, len = n. xs[0]/ys[0] and xs[-1]/ys[-1] held fixed.
        dts:    n-1 segment durations.
        obstacles: (M, 2) array of point obstacles (world frame).
        """
        n = len(xs)
        if n < self.p.n_min:
            return xs, ys, dts

        # Pack into state vector
        n_pose_vars = 2 * (n - 2)
        n_dt_vars   = n - 1
        n_vars      = n_pose_vars + n_dt_vars

        x0 = np.empty(n_vars)
        for i in range(1, n - 1):
            x0[2 * (i - 1)]     = xs[i]
            x0[2 * (i - 1) + 1] = ys[i]
        x0[n_pose_vars:] = dts

        # Bounds: poses unbounded, dt in [dt_min, dt_max]
        lb = np.full(n_vars, -np.inf)
        ub = np.full(n_vars,  np.inf)
        lb[n_pose_vars:] = self.p.dt_min
        ub[n_pose_vars:] = self.p.dt_max
        # Clip current dt to be inside bounds (otherwise scipy raises)
        x0[n_pose_vars:] = np.clip(x0[n_pose_vars:], self.p.dt_min, self.p.dt_max)

        # Cache fixed start/goal
        x_start, y_start = xs[0],  ys[0]
        x_goal,  y_goal  = xs[-1], ys[-1]

        # Pre-compute (pose, obstacle) pairs from the *initial* band so the
        # residual structure is fixed during the LM run. Each interior pose
        # contributes residuals from up to K nearest obstacles within
        # obs_margin * d_inflate. Summing all of them (not just argmin)
        # gives a smooth gradient field — eliminates the "nearest-flips"
        # oscillation that produces alternating path shapes across cycles.
        obs_pairs = self._collect_obstacle_pairs(xs, ys, n, obstacles)

        try:
            result = least_squares(
                fun=self._residuals,
                x0=x0,
                jac=self._jacobian,
                bounds=(lb, ub),
                method='trf',
                max_nfev=self.p.max_iter,
                ftol=self.p.ftol, xtol=self.p.xtol, gtol=1e-8,
                args=(n, x_start, y_start, x_goal, y_goal, obstacles, obs_pairs),
            )
            x_opt = result.x
        except Exception:
            x_opt = x0

        # Unpack
        out_xs = [x_start]
        out_ys = [y_start]
        for i in range(1, n - 1):
            out_xs.append(float(x_opt[2 * (i - 1)]))
            out_ys.append(float(x_opt[2 * (i - 1) + 1]))
        out_xs.append(x_goal)
        out_ys.append(y_goal)
        out_dts = [float(v) for v in x_opt[n_pose_vars:]]
        return out_xs, out_ys, out_dts

    # ───── unpack helpers ────────────────────────────────────────────────
    def _unpack(self, state: np.ndarray, n: int,
                x_start: float, y_start: float,
                x_goal:  float, y_goal:  float):
        n_pose_vars = 2 * (n - 2)
        xs = np.empty(n); ys = np.empty(n)
        xs[0], ys[0]   = x_start, y_start
        xs[-1], ys[-1] = x_goal,  y_goal
        for i in range(1, n - 1):
            xs[i] = state[2 * (i - 1)]
            ys[i] = state[2 * (i - 1) + 1]
        dts = state[n_pose_vars:]
        return xs, ys, dts

    def _pose_var_index(self, i: int, n: int) -> tuple[int, int] | None:
        """Index into state for pose i; None if i is fixed (i=0 or i=n-1)."""
        if i <= 0 or i >= n - 1:
            return None
        return 2 * (i - 1), 2 * (i - 1) + 1

    # ───── obstacle pair selection ───────────────────────────────────────
    def _collect_obstacle_pairs(self,
                                xs: list[float], ys: list[float], n: int,
                                obstacles: np.ndarray) -> list[tuple[int, int]]:
        """For each interior pose, return up to K obstacle indices within
        (obs_margin * d_inflate). The list of (pose_i, obs_j) pairs becomes
        the obstacle edges of the hyper-graph for this optimization run.
        """
        pairs: list[tuple[int, int]] = []
        if obstacles.size == 0 or n < 3:
            return pairs
        p = self.p
        margin = p.obs_margin * p.d_inflate
        K = p.obs_top_k
        ox = obstacles[:, 0]; oy = obstacles[:, 1]
        for i in range(1, n - 1):
            dx = xs[i] - ox
            dy = ys[i] - oy
            d  = np.sqrt(dx * dx + dy * dy)
            within = np.where(d < margin)[0]
            if within.size == 0:
                continue
            if within.size > K:
                # K nearest among "within"
                kth = np.argpartition(d[within], K)[:K]
                within = within[kth]
            for j in within:
                pairs.append((i, int(j)))
        return pairs

    # ───── residuals ─────────────────────────────────────────────────────
    def _residuals(self, state, n, x_start, y_start, x_goal, y_goal,
                   obstacles, obs_pairs):
        p = self.p
        xs, ys, dts = self._unpack(state, n, x_start, y_start, x_goal, y_goal)

        n_seg   = n - 1
        n_pairs = len(obs_pairs)
        # Layout: time(n_seg) | path(n_seg) | obs(n_pairs) | inf(n_pairs)
        r = np.zeros(2 * n_seg + 2 * n_pairs)

        # Time-optimal & shortest-path
        for i in range(n_seg):
            r[i] = p.w_time * dts[i]
            dx = xs[i + 1] - xs[i]; dy = ys[i + 1] - ys[i]
            r[n_seg + i] = p.w_path * math.hypot(dx, dy)

        # Obstacle edges — one per (pose, obstacle) pair
        for k, (i, j) in enumerate(obs_pairs):
            dx = xs[i] - obstacles[j, 0]
            dy = ys[i] - obstacles[j, 1]
            d  = math.hypot(dx, dy)
            if d < p.d_min:
                r[2 * n_seg + k] = p.w_obs * (p.d_min - d)
            if d < p.d_inflate:
                r[2 * n_seg + n_pairs + k] = p.w_inf * (p.d_inflate - d)
        return r

    # ───── analytic Jacobian ─────────────────────────────────────────────
    def _jacobian(self, state, n, x_start, y_start, x_goal, y_goal,
                  obstacles, obs_pairs):
        p = self.p
        xs, ys, dts = self._unpack(state, n, x_start, y_start, x_goal, y_goal)
        n_seg       = n - 1
        n_pairs     = len(obs_pairs)
        n_pose_vars = 2 * (n - 2)
        n_vars      = n_pose_vars + n_seg
        n_res       = 2 * n_seg + 2 * n_pairs

        J = lil_matrix((n_res, n_vars))

        # 1. Time-optimal: r_t_i = w_t * dt_i
        for i in range(n_seg):
            J[i, n_pose_vars + i] = p.w_time

        # 2. Shortest-path: r_s_i = w_s * ||p_{i+1} - p_i||
        for i in range(n_seg):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            d  = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            row = n_seg + i
            ux = p.w_path * dx / d
            uy = p.w_path * dy / d
            idx_i = self._pose_var_index(i, n)
            if idx_i is not None:
                cx, cy = idx_i
                J[row, cx] = -ux
                J[row, cy] = -uy
            idx_ip1 = self._pose_var_index(i + 1, n)
            if idx_ip1 is not None:
                cx, cy = idx_ip1
                J[row, cx] = ux
                J[row, cy] = uy

        # 3. Obstacle edges — gradient of d w.r.t. pose_i, summed implicitly
        #    via separate residual rows (LM combines them in normal eq)
        for k, (i, j) in enumerate(obs_pairs):
            idx = self._pose_var_index(i, n)
            if idx is None:
                continue
            dx = xs[i] - obstacles[j, 0]
            dy = ys[i] - obstacles[j, 1]
            d  = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            gx = dx / d
            gy = dy / d
            cx, cy = idx
            # r_hard = w_o * (d_min - d)  →  ∂r/∂x = -w_o * gx
            if d < p.d_min:
                J[2 * n_seg + k, cx] = -p.w_obs * gx
                J[2 * n_seg + k, cy] = -p.w_obs * gy
            if d < p.d_inflate:
                J[2 * n_seg + n_pairs + k, cx] = -p.w_inf * gx
                J[2 * n_seg + n_pairs + k, cy] = -p.w_inf * gy

        return J.tocsr()



# ─────────────────────────────────────────────────────────────────────────
# Trajectory utilities (init, warm-start, auto-resize)
# ─────────────────────────────────────────────────────────────────────────

def init_band_from_segment(start_xy: tuple[float, float],
                           segment: list[tuple[float, float]],
                           v_max: float,
                           dt_ref: float,
                           min_step: float = 0.4,
                           ) -> tuple[list[float], list[float], list[float]]:
    """Build an initial band: robot pose + sufficiently-spaced waypoints.
    dts initialised from arc length / v_max, clipped at dt_ref scale.
    """
    xs = [start_xy[0]]
    ys = [start_xy[1]]
    for px, py in segment:
        if math.hypot(px - xs[-1], py - ys[-1]) > min_step:
            xs.append(px)
            ys.append(py)
    if len(xs) < 2 and segment:
        xs.append(segment[-1][0])
        ys.append(segment[-1][1])
    dts = []
    for i in range(len(xs) - 1):
        d = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        dts.append(max(0.05, d / max(v_max, 1e-3)))
    return xs, ys, dts


def warm_start_reproject(new_xs: list[float], new_ys: list[float],
                         new_dts: list[float],
                         warm_xs: list[float], warm_ys: list[float],
                         warm_dts: list[float]):
    """Re-project previous band onto current arc-length param of new band.
    Mutates new_xs/ys/dts in place. Preserves avoidance shape across cycles.
    """
    if not warm_xs or len(warm_xs) < 2:
        return
    n = len(new_xs)
    if n < 2:
        return

    warm_arc = [0.0]
    for k in range(1, len(warm_xs)):
        warm_arc.append(warm_arc[-1] + math.hypot(
            warm_xs[k] - warm_xs[k - 1],
            warm_ys[k] - warm_ys[k - 1]))
    L_warm = warm_arc[-1]

    new_arc = [0.0]
    for k in range(1, n):
        new_arc.append(new_arc[-1] + math.hypot(
            new_xs[k] - new_xs[k - 1],
            new_ys[k] - new_ys[k - 1]))
    L_new = new_arc[-1]

    if L_warm < 1e-3 or L_new < 1e-3:
        return

    for i in range(1, n - 1):
        target = (new_arc[i] / L_new) * L_warm
        for k in range(len(warm_arc) - 1):
            if warm_arc[k + 1] >= target:
                span = warm_arc[k + 1] - warm_arc[k]
                if span < 1e-9:
                    new_xs[i], new_ys[i] = warm_xs[k], warm_ys[k]
                else:
                    a = (target - warm_arc[k]) / span
                    new_xs[i] = warm_xs[k] + a * (warm_xs[k + 1] - warm_xs[k])
                    new_ys[i] = warm_ys[k] + a * (warm_ys[k + 1] - warm_ys[k])
                break

    ratio = L_new / L_warm
    for i in range(min(len(new_dts), len(warm_dts))):
        new_dts[i] = max(0.05, warm_dts[i] * ratio)


def auto_resize(xs: list[float], ys: list[float], dts: list[float],
                params: TebParams) -> tuple[list[float], list[float], list[float]]:
    """Insert/remove poses so each dt stays close to dt_ref ± dt_hyst.

    - dt_i too large  → split: insert midpoint pose, halve the dt.
    - dt_i too small  → merge: drop pose i+1 (unless it's the goal),
                                concatenate dt_i and dt_{i+1}.
    Endpoints (start/goal) are never removed. Total size kept in [n_min, n_max].
    """
    p = params
    upper = p.dt_ref + p.dt_hyst
    lower = max(p.dt_min, p.dt_ref - p.dt_hyst)

    # Split (one pass)
    i = 0
    while i < len(dts) and len(xs) < p.n_max:
        if dts[i] > upper:
            mx = 0.5 * (xs[i] + xs[i + 1])
            my = 0.5 * (ys[i] + ys[i + 1])
            xs.insert(i + 1, mx)
            ys.insert(i + 1, my)
            half = 0.5 * dts[i]
            dts[i] = half
            dts.insert(i + 1, half)
            # don't advance i — re-check this segment in case it's still long
        else:
            i += 1

    # Merge (one pass; never remove start or goal)
    i = 0
    while i < len(dts) - 1 and len(xs) > p.n_min:
        if dts[i] < lower and (i + 1) < len(xs) - 1:
            # Remove pose i+1; combine its dt with dt_i
            del xs[i + 1]
            del ys[i + 1]
            combined = dts[i] + dts[i + 1]
            del dts[i + 1]
            dts[i] = min(combined, p.dt_max)
        else:
            i += 1

    return xs, ys, dts


# ─────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────

def euclidean(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.position.x - p2.position.x,
                      p1.position.y - p2.position.y)


class PathPlanningNode(Node):

    def __init__(self):
        super().__init__('path_planning_node')

        # ── ROS-side parameters ──────────────────────────────────────────
        self.planning_freq          = 10
        self.lookahead_dist         = 4.0
        self.min_waypoints_required = 2
        self.goal_tolerance         = 0.4

        # Costmap → points
        self.obstacle_threshold = 60
        self.search_margin      = 4.0

        # ── TEB parameters ───────────────────────────────────────────────
        self.teb_params = TebParams(
            w_time    = 5.0,
            w_path    = 1.0,
            w_obs     = 50.0,
            w_inf     = 10.0,
            d_min     = 0.30,
            d_inflate = 0.80,
            dt_min    = 0.05,
            dt_max    = 1.50,
            dt_ref    = 0.30,
            dt_hyst   = 0.10,
            n_min     = 4,
            n_max     = 60,
            max_iter  = 8,
            ftol      = 1e-3,
            xtol      = 1e-3,
            v_max     = 1.5,
        )
        self.optimizer = TebOptimizer(self.teb_params)

        # ── ROS interfaces ───────────────────────────────────────────────
        self.create_subscription(Odometry,      '/odom',          self._odom_cb,     10)
        self.create_subscription(Path,          '/global_path',   self._global_cb,   10)
        self.create_subscription(OccupancyGrid, '/local_costmap', self._costmap_cb,  10)

        self.cmd_vel_pub  = self.create_publisher(TwistStamped, '/cmd_vel',         10)
        self.path_pub     = self.create_publisher(Path,         '/planned_path',    10)
        self.obstacle_pub = self.create_publisher(MarkerArray,  '/obstacle_points', 10)

        # ── State ────────────────────────────────────────────────────────
        self._robot_pose: Pose | None             = None
        self._global_path: Path | None            = None
        self._local_costmap: OccupancyGrid | None = None
        self._prune_index: int                    = 0

        self._warm_xs:  list[float] | None = None
        self._warm_ys:  list[float] | None = None
        self._warm_dts: list[float] | None = None

        self._obstacle_points: np.ndarray = np.empty((0, 2))

        self.create_timer(1.0 / self.planning_freq, self._replan)
        self.get_logger().info('PathPlanningNode (TEB scipy-LM, holonomic fixed-yaw) started')

    # ── Callbacks ────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._robot_pose = msg.pose.pose

    def _costmap_cb(self, msg: OccupancyGrid):
        self._local_costmap = msg

    def _global_cb(self, msg: Path):
        if self._global_path is not None and len(msg.poses) > 0:
            same_stamp = (msg.header.stamp.sec == self._global_path.header.stamp.sec and
                          msg.header.stamp.nanosec == self._global_path.header.stamp.nanosec)
            if same_stamp:
                return
            new_goal = msg.poses[-1].pose.position
            old_goal = self._global_path.poses[-1].pose.position
            same_goal = (abs(new_goal.x - old_goal.x) < 0.01 and
                         abs(new_goal.y - old_goal.y) < 0.01)
            if same_goal:
                self._global_path = msg
                self._prune_index = 0
                return

        self._global_path = msg
        self._prune_index = 0
        self._warm_xs = self._warm_ys = self._warm_dts = None
        self.get_logger().info(f'New goal received ({len(msg.poses)} waypoints)')

    # ── Planning loop ────────────────────────────────────────────────────
    def _replan(self):
        if self._robot_pose is None or self._global_path is None or self._local_costmap is None:
            return
        if len(self._global_path.poses) < self.min_waypoints_required:
            return
        if euclidean(self._robot_pose, self._global_path.poses[-1].pose) < self.goal_tolerance:
            self._pub_stop()
            return

        # 1. Costmap → point obstacles
        self._obstacle_points = self._costmap_to_points()
        self._publish_points()

        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        self._prune_index = self._find_closest_index()

        # 2. Stable lookahead goal (arc-interpolated — not waypoint-hopping)
        lx, ly = self._find_lookahead_goal()

        # 3. Build initial band
        #    Warm path: use previous solution directly, update only endpoints.
        #    Cold path: build from reference waypoints.
        if self._warm_xs and len(self._warm_xs) >= 2:
            xs  = self._warm_xs[:]
            ys  = self._warm_ys[:]
            dts = self._warm_dts[:]
            # Move start to current robot pose (corrects small odom drift)
            xs[0], ys[0] = rx, ry
            # Update goal to current lookahead point
            xs[-1], ys[-1] = lx, ly
        else:
            segment_pts = self._extract_local_segment(lx, ly)
            xs, ys, dts = init_band_from_segment(
                (rx, ry), segment_pts,
                v_max=self.teb_params.v_max,
                dt_ref=self.teb_params.dt_ref,
                min_step=0.4,
            )
            if len(xs) < 2:
                return

        # 4. Auto-resize to keep dt_i near dt_ref
        xs, ys, dts = auto_resize(xs, ys, dts, self.teb_params)

        # 5. Optimize
        opt_xs, opt_ys, opt_dts = self.optimizer.optimize(
            xs, ys, dts, self._obstacle_points,
        )

        # 6. Cache for next cycle
        self._warm_xs  = opt_xs[:]
        self._warm_ys  = opt_ys[:]
        self._warm_dts = opt_dts[:]

        # 7. Publish
        frame = self._global_path.header.frame_id or 'odom'
        self._publish_path(opt_xs, opt_ys, frame)
        self.cmd_vel_pub.publish(self._compute_cmd_vel(opt_xs, opt_ys, opt_dts))

    # ── Costmap → points ─────────────────────────────────────────────────
    def _costmap_to_points(self) -> np.ndarray:
        cm = self._local_costmap
        info = cm.info
        w, h = info.width, info.height
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        data = np.asarray(cm.data, dtype=np.int16).reshape(h, w)

        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        margin_cells = int((self.lookahead_dist + self.search_margin) / res)
        cx_robot = int((rx - ox) / res)
        cy_robot = int((ry - oy) / res)
        i_lo = max(0, cx_robot - margin_cells)
        i_hi = min(w, cx_robot + margin_cells + 1)
        j_lo = max(0, cy_robot - margin_cells)
        j_hi = min(h, cy_robot + margin_cells + 1)
        if i_hi <= i_lo or j_hi <= j_lo:
            return np.empty((0, 2))

        sub = data[j_lo:j_hi, i_lo:i_hi]
        js, is_ = np.where(sub >= self.obstacle_threshold)
        if js.size == 0:
            return np.empty((0, 2))
        wx = ox + (is_ + i_lo + 0.5) * res
        wy = oy + (js + j_lo + 0.5) * res
        return np.column_stack((wx, wy)).astype(float)

    # ── cmd_vel ──────────────────────────────────────────────────────────
    def _compute_cmd_vel(self, xs, ys, dts) -> TwistStamped:
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        if len(xs) < 2 or not dts:
            return cmd

        qz  = self._robot_pose.orientation.z
        qw  = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        cos_y = math.cos(yaw); sin_y = math.sin(yaw)

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        dt = max(dts[0], 1e-3)
        vx_w = dx / dt
        vy_w = dy / dt

        # World → body
        vx_b =  vx_w * cos_y + vy_w * sin_y
        vy_b = -vx_w * sin_y + vy_w * cos_y

        # Cap by v_max and decelerate near goal
        d_goal = euclidean(self._robot_pose, self._global_path.poses[-1].pose)
        v_lim = min(self.teb_params.v_max, max(0.05, d_goal * 1.5))
        v_mag = math.hypot(vx_b, vy_b)
        if v_mag > v_lim:
            vx_b *= v_lim / v_mag
            vy_b *= v_lim / v_mag

        cmd.twist.linear.x  = vx_b
        cmd.twist.linear.y  = vy_b
        cmd.twist.angular.z = 0.0
        return cmd

    def _pub_stop(self):
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(cmd)

    # ── Visualization ────────────────────────────────────────────────────
    def _publish_points(self):
        ma = MarkerArray()
        frame = self._local_costmap.header.frame_id
        stamp = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        if self._obstacle_points.size > 0:
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
            for px, py in self._obstacle_points:
                pt = Point()
                pt.x, pt.y, pt.z = float(px), float(py), 0.0
                m.points.append(pt)
            ma.markers.append(m)

        self.obstacle_pub.publish(ma)

    def _publish_path(self, xs, ys, frame):
        n = len(xs)
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = frame
        for i in range(n):
            theta = math.atan2(
                ys[min(i + 1, n - 1)] - ys[max(i - 1, 0)],
                xs[min(i + 1, n - 1)] - xs[max(i - 1, 0)],
            )
            ps = PoseStamped()
            ps.header.frame_id    = frame
            ps.header.stamp       = stamp
            ps.pose.position.x    = float(xs[i])
            ps.pose.position.y    = float(ys[i])
            ps.pose.orientation.z = math.sin(theta / 2.0)
            ps.pose.orientation.w = math.cos(theta / 2.0)
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    # ── Path helpers ─────────────────────────────────────────────────────
    def _find_closest_index(self) -> int:
        wp = self._global_path.poses
        idx, best = self._prune_index, float('inf')
        for i in range(self._prune_index, len(wp)):
            d = euclidean(self._robot_pose, wp[i].pose)
            if d < best:
                best, idx = d, i
        return idx

    def _find_lookahead_goal(self) -> tuple[float, float]:
        """Return a point at exactly lookahead_dist ahead along the global
        path arc, interpolated smoothly.  This prevents the goal from
        jumping when the robot crosses a waypoint boundary."""
        wp = self._global_path.poses
        if not wp:
            return (self._robot_pose.position.x, self._robot_pose.position.y)

        # Start accumulating arc distance from the prune-index waypoint
        idx = self._prune_index
        prev_x = wp[idx].pose.position.x
        prev_y = wp[idx].pose.position.y
        # Arc offset: distance from robot to the starting waypoint
        arc = math.hypot(self._robot_pose.position.x - prev_x,
                         self._robot_pose.position.y - prev_y)

        for i in range(idx + 1, len(wp)):
            px = wp[i].pose.position.x
            py = wp[i].pose.position.y
            seg_len = math.hypot(px - prev_x, py - prev_y)
            if arc + seg_len >= self.lookahead_dist:
                # Interpolate to land exactly at lookahead_dist
                t = (self.lookahead_dist - arc) / max(seg_len, 1e-9)
                t = max(0.0, min(1.0, t))
                return (prev_x + t * (px - prev_x),
                        prev_y + t * (py - prev_y))
            arc += seg_len
            prev_x, prev_y = px, py

        # Path shorter than lookahead — use the final goal
        return (wp[-1].pose.position.x, wp[-1].pose.position.y)

    def _extract_local_segment(self,
                               goal_x: float,
                               goal_y: float) -> list[tuple[float, float]]:
        """Waypoints from prune_index up to (but not past) the lookahead goal.
        Used only on cold start (no warm band available).
        Waypoints geometrically behind the robot are skipped.
        """
        wp = self._global_path.poses
        rx = self._robot_pose.position.x
        ry = self._robot_pose.position.y
        qz = self._robot_pose.orientation.z
        qw = self._robot_pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        fwd_x = math.cos(yaw)
        fwd_y = math.sin(yaw)

        seg: list[tuple[float, float]] = []
        for i in range(self._prune_index, len(wp)):
            px = wp[i].pose.position.x
            py = wp[i].pose.position.y
            d = math.hypot(px - rx, py - ry)
            if d > self.lookahead_dist:
                break
            # Skip waypoints clearly behind the robot (dot product < 0)
            dot = (px - rx) * fwd_x + (py - ry) * fwd_y
            if dot < -0.2:
                continue
            seg.append((px, py))

        # Ensure the lookahead goal is included as the final anchor
        if not seg or math.hypot(seg[-1][0] - goal_x, seg[-1][1] - goal_y) > 0.2:
            seg.append((goal_x, goal_y))
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
