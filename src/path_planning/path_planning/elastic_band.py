"""
elastic_band.py
===============
TEB-style elastic-band local planner — pure Python, no ROS dependencies.

Imported by `path_planning_node`. Keeps the optimisation logic decoupled
from ROS plumbing so it can be tested or reused in isolation.

Public API
----------
    convex_hull(points)
    mcch_decompose(cluster_pts, split_threshold)
    point_polygon_signed_dist(x, y, poly)
    ElasticBandConfig
    ElasticBandPlanner
"""

import math
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────
# Polygon helpers
# ──────────────────────────────────────────────────────────────────────────

def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
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


def mcch_decompose(
        cluster_pts: list[tuple[float, float]],
        split_threshold: float,
        max_recursion: int = 6,
) -> list[list[tuple[float, float]]]:
    """
    Multi-Convex-Hull decomposition (nav2 costmap_converter MCCH style).

    A plain convex hull is a poor approximation for concave clusters
    (L-, U-, T-shapes): the hull's diagonal edges cut across free space,
    creating fake polygons that swallow the concave region. MCCH detects
    "fictitious" edges — those whose midpoint lies far from any cluster
    point — and recursively splits the cluster perpendicular to that
    edge until every remaining sub-cluster's hull tracks the real
    occupancy.

    Args:
        cluster_pts: world-frame (x, y) cluster points (cell centres).
        split_threshold: an edge is fictitious if its midpoint is farther
            than this from any cluster point. Smaller -> finer split.
        max_recursion: safety cap on recursion depth.

    Returns:
        list of convex hulls (each as CCW open polyline, no repeated first
        point). Sub-hulls with < 2 points are dropped.
    """
    if len(cluster_pts) < 2:
        return []
    return _mcch_recurse(cluster_pts, split_threshold, max_recursion)


def _mcch_recurse(cluster_pts, threshold, depth):
    hull = convex_hull(cluster_pts)
    if len(hull) < 3 or depth <= 0:
        return [hull] if len(hull) >= 2 else []

    # Step 1: identify the worst (most fictitious) hull edge by the
    # distance from its midpoint to the nearest cluster point. This metric
    # stays small for convex blobs (every midpoint sits near real cells)
    # and grows large for diagonal edges that cut across the concave gap
    # of an L/U-shape.
    worst_dist = 0.0
    worst_idx = -1
    for i in range(len(hull)):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % len(hull)]
        mx = 0.5 * (ax + bx)
        my = 0.5 * (ay + by)
        d = min(math.hypot(mx - p[0], my - p[1]) for p in cluster_pts)
        if d > worst_dist:
            worst_dist = d
            worst_idx = i

    if worst_dist <= threshold or worst_idx < 0:
        return [hull]

    # Step 2: split along the edge direction at the edge midpoint.
    # Perpendicular splits fail for L-shapes (every cluster point lies on
    # the same perpendicular side of the hypotenuse). Splitting by
    # along-edge projection at the midpoint is robust: for an L, the two
    # arms project onto opposite halves of the hypotenuse; for a U, the
    # split bisects the open mouth and each half becomes an L that the
    # recursion splits further.
    ax, ay = hull[worst_idx]
    bx, by = hull[(worst_idx + 1) % len(hull)]
    ex, ey = bx - ax, by - ay
    edge_len = math.hypot(ex, ey)
    if edge_len < 1e-9:
        return [hull]
    ux, uy = ex / edge_len, ey / edge_len
    t_mid = 0.5 * edge_len

    side1, side2 = [], []
    for px, py in cluster_pts:
        t = (px - ax) * ux + (py - ay) * uy
        if t <= t_mid:
            side1.append((px, py))
        else:
            side2.append((px, py))

    if not side1 or not side2:
        return [hull]

    return (_mcch_recurse(side1, threshold, depth - 1) +
            _mcch_recurse(side2, threshold, depth - 1))


def point_polygon_signed_dist(
        x: float, y: float,
        poly: list[tuple[float, float]],
) -> tuple[float, float, float, bool]:
    """
    Returns (distance_to_boundary, closest_x, closest_y, inside).
    Distance is unsigned distance to the polygon boundary.
    `inside` is True when (x, y) is strictly inside the polygon (n >= 3).
    Degenerate polygons (n=1 point, n=2 segment) are handled.
    """
    n = len(poly)
    if n == 0:
        return float('inf'), x, y, False
    if n == 1:
        return math.hypot(x - poly[0][0], y - poly[0][1]), poly[0][0], poly[0][1], False

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
            t = max(0.0, min(1.0, t))
            cx, cy = x1 + t * dx, y1 + t * dy
        d = math.hypot(x - cx, y - cy)
        if d < best_d:
            best_d, best_x, best_y = d, cx, cy

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


# ──────────────────────────────────────────────────────────────────────────
# Elastic-band planner
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ElasticBandConfig:
    n_iter: int      = 60     # gradient descent iterations / cycle
    w_obs: float     = 50.0   # obstacle repulsion weight
    w_smooth: float  = 2.0    # smoothness (elastic band) weight
    inflation: float = 1.0    # m — repulsion radius around polygons
    step: float      = 0.05   # gradient step
    max_delta: float = 0.10   # m — clamp per-iteration node motion
    skip_dist: float = 0.4    # m — drop segment waypoints closer than this


class ElasticBandPlanner:
    """
    TEB-style elastic-band optimiser against polygon obstacles.

    One cycle (`optimize`):
      1. Initial band = [robot pose, ...filtered global-path waypoints]
      2. Warm start: arc-length re-project the previous solution onto the
         new band so the avoidance shape persists across cycles
      3. Pre-route: push any band node still inside a polygon to one
         consistent side (single homotopy class) to break the symmetric
         force lock-in
      4. Gradient descent against
           - polygon obstacle edges (outward force inside an inflation zone,
             escape force when inside a polygon)
           - smoothness (pull toward midpoint of neighbours)
         Endpoints fixed; per-iteration motion is clamped so the band
         cannot teleport across narrow gaps.

    Anchoring:
      The band is anchored at the *actual* robot pose (xs[0]) — not the
      closest global-path waypoint. This matters when the robot has
      deviated laterally to avoid an obstacle: anchoring at the global
      path would put xs[0] inside the inflation zone and the smoothness
      force would keep dragging the band back into the obstacle.
    """

    def __init__(self, config: ElasticBandConfig | None = None):
        self.cfg = config or ElasticBandConfig()
        self._warm_xs: list[float] | None = None
        self._warm_ys: list[float] | None = None

    def reset_warm_start(self) -> None:
        """Drop the cached previous solution. Call when the goal changes."""
        self._warm_xs = None
        self._warm_ys = None

    def optimize(
            self,
            robot_xy: tuple[float, float],
            segment_xys: list[tuple[float, float]],
            polygons: list[list[tuple[float, float]]],
    ) -> tuple[list[float], list[float]]:
        """
        Optimise an elastic band from `robot_xy` along `segment_xys`,
        deforming around `polygons`. Endpoints fixed. Returns (xs, ys)
        and caches it as the warm seed for the next call.
        """
        rx, ry = robot_xy

        # 1. Initial band: robot pose + filtered global-path waypoints.
        xs, ys = [rx], [ry]
        for px, py in segment_xys:
            if math.hypot(px - xs[-1], py - ys[-1]) > self.cfg.skip_dist:
                xs.append(px)
                ys.append(py)

        n = len(xs)
        if n < 2:
            return xs, ys

        self._apply_warm_start(xs, ys)
        self._preroute(xs, ys, polygons)
        self._gradient_descent(xs, ys, polygons)

        self._warm_xs = xs[:]
        self._warm_ys = ys[:]
        return xs, ys

    # ── warm start ──────────────────────────────────────────────────────

    def _apply_warm_start(self, xs: list[float], ys: list[float]) -> None:
        """
        Re-project the previous band onto the new band by arc length.
        Nearest-neighbour re-projection collapses laterally-displaced nodes
        back onto the global path; arc-length re-projection preserves the
        avoidance shape even when the lateral deviation exceeds the
        global-path waypoint spacing.
        """
        if not (self._warm_xs and self._warm_ys and len(self._warm_xs) >= 2):
            return
        warm_xs, warm_ys = self._warm_xs, self._warm_ys
        n = len(xs)

        warm_arc = [0.0]
        for k in range(1, len(warm_xs)):
            warm_arc.append(warm_arc[-1] + math.hypot(
                warm_xs[k] - warm_xs[k - 1], warm_ys[k] - warm_ys[k - 1]))
        L_warm = warm_arc[-1]

        new_arc = [0.0]
        for k in range(1, n):
            new_arc.append(new_arc[-1] + math.hypot(
                xs[k] - xs[k - 1], ys[k] - ys[k - 1]))
        L_new = new_arc[-1]
        if L_warm < 1e-3 or L_new < 1e-3:
            return

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

    # ── obstacle force ──────────────────────────────────────────────────

    def _obstacle_force(
            self, x: float, y: float,
            polygons: list[list[tuple[float, float]]],
    ) -> tuple[float, float]:
        """
        Repulsive force on a waypoint from the closest polygon.
          - 0 outside the inflation radius
          - linear in penetration depth, away from nearest boundary point
          - inside a polygon: strong push toward nearest boundary (escape)
        """
        if not polygons:
            return 0.0, 0.0

        best_d = float('inf')
        best_cx = best_cy = 0.0
        best_inside = False

        for poly in polygons:
            d, cx, cy, inside = point_polygon_signed_dist(x, y, poly)
            if inside:
                if not best_inside or d < best_d:
                    best_d, best_cx, best_cy, best_inside = d, cx, cy, True
            elif not best_inside and d < best_d:
                best_d, best_cx, best_cy = d, cx, cy

        infl = self.cfg.inflation
        if best_inside:
            vx, vy = best_cx - x, best_cy - y
            m = math.hypot(vx, vy)
            if m < 1e-9:
                return infl, 0.0
            return infl * vx / m, infl * vy / m

        if best_d >= infl:
            return 0.0, 0.0

        vx, vy = x - best_cx, y - best_cy
        m = math.hypot(vx, vy)
        if m < 1e-9:
            return 0.0, 0.0
        mag = infl - best_d
        return mag * vx / m, mag * vy / m

    # ── pre-routing ─────────────────────────────────────────────────────

    def _preroute(
            self,
            xs: list[float], ys: list[float],
            polygons: list[list[tuple[float, float]]],
    ) -> None:
        """
        Local optimisation cannot escape a polygon that fully contains the
        band — symmetric obstacle forces pin it. Push any node still inside
        a polygon to one consistent side (whichever has the shorter detour)
        before optimising. nav2 TEB does this implicitly by exploring
        multiple homotopy classes in parallel; here we approximate it with
        a single, stable choice per polygon per cycle.
        """
        n = len(xs)
        if n < 3:
            return

        for poly in polygons:
            if len(poly) < 3:
                continue

            inside_idx = []
            for i in range(1, n - 1):
                _, _, _, inside = point_polygon_signed_dist(xs[i], ys[i], poly)
                if inside:
                    inside_idx.append(i)
            if not inside_idx:
                continue

            i_first = inside_idx[0]
            i_last  = inside_idx[-1]
            i_pre   = max(0,     i_first - 1)
            i_post  = min(n - 1, i_last  + 1)

            bx = xs[i_post] - xs[i_pre]
            by = ys[i_post] - ys[i_pre]
            bm = math.hypot(bx, by)
            if bm < 1e-6:
                continue
            bx, by = bx / bm, by / bm

            cxr = (xs[i_first] + xs[i_last]) / 2.0
            cyr = (ys[i_first] + ys[i_last]) / 2.0

            max_left  = 0.0
            max_right = 0.0
            for vx, vy in poly:
                d_left  = (vx - cxr) * (-by) + (vy - cyr) * bx
                d_right = (vx - cxr) * by   + (vy - cyr) * (-bx)
                if d_left  > max_left:  max_left  = d_left
                if d_right > max_right: max_right = d_right

            if max_left <= max_right:
                perp_x, perp_y, extent = -by, bx, max_left
            else:
                perp_x, perp_y, extent =  by, -bx, max_right

            offset = extent + self.cfg.inflation + 0.1
            for i in inside_idx:
                xs[i] += offset * perp_x
                ys[i] += offset * perp_y

    # ── gradient descent ────────────────────────────────────────────────

    def _gradient_descent(
            self,
            xs: list[float], ys: list[float],
            polygons: list[list[tuple[float, float]]],
    ) -> None:
        n = len(xs)
        step      = self.cfg.step
        max_delta = self.cfg.max_delta
        w_obs     = self.cfg.w_obs
        w_smooth  = self.cfg.w_smooth

        for _ in range(self.cfg.n_iter):
            for i in range(1, n - 1):
                fx_obs, fy_obs = self._obstacle_force(xs[i], ys[i], polygons)

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
