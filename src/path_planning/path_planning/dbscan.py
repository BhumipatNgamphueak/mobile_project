"""
dbscan.py
=========
Self-contained DBSCAN clustering for 2-D costmap obstacle points.

Why a separate module:
  The Elastic Band optimiser only needs *where* to push away from. Raw
  costmap cells give one point per occupied cell, so a single physical
  obstacle becomes dozens of densely packed points. Clustering folds those
  back into a small number of obstacle entities (centroid + bounding
  radius), which is useful for:
    - dropping isolated noise cells (sensor flicker),
    - logging / visualising obstacles per-entity,
    - any future cost shaping that wants per-obstacle data
      (size, density, velocity tracking across cycles).

Public surface:
    DBSCAN(eps, min_samples).fit(points) -> labels[int]
    cluster_obstacles(points, eps, min_samples) -> list[Cluster]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


NOISE = -1


@dataclass
class Cluster:
    """One obstacle entity recovered from DBSCAN."""
    points:   list[tuple[float, float]] = field(default_factory=list)
    centroid: tuple[float, float]       = (0.0, 0.0)
    radius:   float                     = 0.0   # max(|p - centroid|) over members
    hull:     list[tuple[float, float]] = field(default_factory=list)
    # ``hull`` is the convex hull of ``points`` in CCW order (Andrew's
    # monotone chain). Used as the obstacle primitive by the EB optimiser:
    # repulsion is computed against polygon edges, not against raw cells.

    solidity: float                     = 1.0
    reliable: bool                      = True
    # ``solidity`` = (n_points * cell_size^2) / hull_area
    #   - ~1.0 for solid / line-shaped clusters whose hull tightly hugs
    #     the cells (these are safe to use as polygons),
    #   - much smaller (<<0.5) for concave shapes like L, U, rings whose
    #     hull bridges across empty space.
    # ``reliable`` is the boolean used by callers: when False the
    # cluster's hull encloses free space and must NOT be used as a
    # polygon obstacle (the robot would be falsely classified as
    # "inside"). Caller should fall back to point-based repulsion
    # against ``points`` instead.


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float
        Neighbourhood radius (metres). Two points are neighbours iff their
        Euclidean distance is <= eps.
    min_samples : int
        A point is a *core* point when it has >= min_samples neighbours
        (including itself). Clusters grow from core points; non-core points
        reachable from a core point become *border* points; everything else
        is noise.

    Notes
    -----
    Neighbour queries use a uniform grid hash with cell size = eps, so each
    query inspects only the 9 surrounding buckets. For the few-hundred
    points produced by a local costmap this is well below 1 ms.
    """

    def __init__(self, eps: float = 0.2, min_samples: int = 3):
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self.eps         = eps
        self.min_samples = min_samples

    # ------------------------------------------------------------------
    def fit(self, points: list[tuple[float, float]]) -> list[int]:
        """
        Returns a list of cluster labels, one per input point.
        Labels are 0..K-1 for clusters and NOISE (-1) for outliers.
        """
        n = len(points)
        labels  = [NOISE] * n
        visited = [False]  * n
        if n == 0:
            return labels

        eps  = self.eps
        eps2 = eps * eps
        cell = eps

        # Build spatial hash: bucket index -> list of point indices.
        grid: dict[tuple[int, int], list[int]] = {}
        for i, (x, y) in enumerate(points):
            grid.setdefault((int(x / cell), int(y / cell)), []).append(i)

        def region_query(idx: int) -> list[int]:
            x, y   = points[idx]
            cx, cy = int(x / cell), int(y / cell)
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = grid.get((cx + dx, cy + dy))
                    if not bucket:
                        continue
                    for j in bucket:
                        ox, oy = points[j]
                        if (ox - x) * (ox - x) + (oy - y) * (oy - y) <= eps2:
                            out.append(j)
            return out

        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neigh = region_query(i)
            if len(neigh) < self.min_samples:
                # Stays NOISE for now; may later be promoted to border.
                continue

            # Seed a new cluster and BFS-expand it.
            labels[i] = cluster_id
            seeds = list(neigh)
            k = 0
            while k < len(seeds):
                j = seeds[k]
                k += 1
                if not visited[j]:
                    visited[j] = True
                    j_neigh = region_query(j)
                    if len(j_neigh) >= self.min_samples:
                        seeds.extend(j_neigh)
                if labels[j] == NOISE:
                    labels[j] = cluster_id
            cluster_id += 1

        return labels


# ----------------------------------------------------------------------
# Convenience wrapper used by the path-planning node.
# ----------------------------------------------------------------------
def cluster_obstacles(
    points:             list[tuple[float, float]],
    eps:                float = 0.2,
    min_samples:        int   = 3,
    cell_size:          float | None = None,
    solidity_threshold: float = 0.5,
) -> list[Cluster]:
    """
    Cluster 2-D obstacle points and summarise each group.

    Parameters
    ----------
    cell_size : float, optional
        Source-grid cell size (e.g. costmap resolution, m). When given,
        each cluster's solidity is computed as
        ``(n_points * cell_size**2) / hull_area`` and clusters with
        solidity below ``solidity_threshold`` are flagged
        ``reliable=False`` — their convex hull bridges over empty space
        (typical for L / U / ring shapes) and must NOT be used as a
        polygon obstacle. When ``None``, the concavity check is skipped
        and every non-degenerate hull is marked reliable.
    solidity_threshold : float
        Cut-off below which a hull is judged unreliable. Defaults to 0.5.

    Returns
    -------
    list[Cluster]
        Noise points are dropped. Each Cluster carries points, centroid,
        radius, hull, solidity and the ``reliable`` flag.
    """
    if not points:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    grouped: dict[int, list[tuple[float, float]]] = {}
    for lbl, p in zip(labels, points):
        if lbl == NOISE:
            continue
        grouped.setdefault(lbl, []).append(p)

    clusters: list[Cluster] = []
    for pts in grouped.values():
        sx = sum(p[0] for p in pts) / len(pts)
        sy = sum(p[1] for p in pts) / len(pts)
        rad  = max(math.hypot(p[0] - sx, p[1] - sy) for p in pts)
        hull = convex_hull(pts)

        # Hulls with 0 or 1 vertices are degenerate (no usable geometry).
        # Hulls with 2 vertices are line segments — no "interior" to bridge,
        # so they are always reliable. Only polygons (>=3 vertices, finite
        # area) need the concavity check.
        solidity = 1.0
        reliable = len(hull) >= 2
        if reliable and len(hull) >= 3 and cell_size is not None:
            harea = polygon_area(hull)
            if harea > 1e-6:
                covered  = len(pts) * cell_size * cell_size
                solidity = covered / harea
                reliable = solidity >= solidity_threshold
            # else: collinear polygon (zero area) — treat as line, reliable.

        clusters.append(Cluster(
            points=pts, centroid=(sx, sy), radius=rad,
            hull=hull, solidity=solidity, reliable=reliable,
        ))
    return clusters


# ----------------------------------------------------------------------
# Polygon area — shoelace formula. Used for solidity / concavity check.
# ----------------------------------------------------------------------
def polygon_area(hull: list[tuple[float, float]]) -> float:
    """Absolute area of a simple polygon (vertices in any consistent order)."""
    n = len(hull)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


# ----------------------------------------------------------------------
# Convex hull — Andrew's monotone chain.
# ----------------------------------------------------------------------
def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Compute the convex hull of a 2-D point set in CCW order.

    Implements Andrew's monotone chain algorithm — O(n log n), pure-Python,
    numerically robust enough for costmap-cell coordinates.

    Returns
    -------
    list[tuple[float, float]]
        CCW-ordered hull vertices (no duplicate closing vertex). Returns
        a copy of the unique input points if there are fewer than three
        distinct points (cannot form a polygon).
    """
    pts = sorted(set(points))
    n = len(pts)
    if n <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate; drop the last point of each list (it is the start of
    # the other list).
    return lower[:-1] + upper[:-1]


# ----------------------------------------------------------------------
# Point-to-polygon distance (signed) and outward push direction.
# ----------------------------------------------------------------------
def point_to_convex_polygon(
    px: float, py: float,
    hull: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """
    Signed distance from a 2-D point to a CCW convex polygon, plus the
    unit "push outward" direction.

    Returns
    -------
    (distance, nx, ny)
        ``distance`` is positive when the point is outside the polygon
        (Euclidean distance to the nearest boundary point) and negative
        when inside (negated penetration depth, useful as a strong
        repulsion signal). ``(nx, ny)`` is a unit vector that, when
        applied to ``(px, py)``, moves it away from the polygon
        boundary; for a point on the boundary it is the (zero) zero
        vector.

    Degenerate hulls:
      - empty       -> (+inf, 0, 0)
      - single pt   -> distance to that point, unit radial direction
      - two pts     -> distance to the line segment
      - three or more -> general convex polygon
    """
    n = len(hull)
    if n == 0:
        return float('inf'), 0.0, 0.0
    if n == 1:
        return _point_to_point(px, py, hull[0])
    if n == 2:
        return _point_to_segment(px, py, hull[0], hull[1])

    # Inside test: for a CCW polygon, the point is interior iff it lies
    # on the LEFT of every directed edge.
    inside = True
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < 0.0:
            inside = False
            break

    # Closest point on polygon boundary (clamp-to-segment per edge).
    best_d  = float('inf')
    best_qx = 0.0
    best_qy = 0.0
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        l2 = ex * ex + ey * ey
        if l2 < 1e-12:
            qx, qy = x1, y1
        else:
            t = ((px - x1) * ex + (py - y1) * ey) / l2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            qx = x1 + t * ex
            qy = y1 + t * ey
        d = math.hypot(px - qx, py - qy)
        if d < best_d:
            best_d, best_qx, best_qy = d, qx, qy

    dx, dy = px - best_qx, py - best_qy
    if best_d < 1e-9:
        # On the boundary: zero-vector direction; caller handles.
        return (-best_d if inside else best_d), 0.0, 0.0

    nx, ny = dx / best_d, dy / best_d
    if inside:
        # Push outward = opposite of (boundary -> query) since query is
        # inside; the boundary point is on the *outside* of the query.
        return -best_d, -nx, -ny
    return best_d, nx, ny


def _point_to_point(px, py, p):
    x0, y0 = p
    dx, dy = px - x0, py - y0
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return 0.0, 0.0, 0.0
    return d, dx / d, dy / d


# ----------------------------------------------------------------------
# Segment vs. convex polygon intersection — used for homotopy checks.
# ----------------------------------------------------------------------
def segment_intersects_polygon(
    p1:   tuple[float, float],
    p2:   tuple[float, float],
    hull: list[tuple[float, float]],
) -> bool:
    """
    True iff the 2-D segment ``p1`` -> ``p2`` intersects (or lies inside) a
    convex polygon given as CCW-ordered hull vertices.

    Used by the EB optimiser to detect homotopy violations: a band edge
    that crosses an obstacle polygon is on the wrong side of that
    obstacle and cannot be repaired by local gradient descent. Reports
    True if either endpoint is inside the hull, or if the segment crosses
    any hull edge.
    """
    n = len(hull)
    if n < 3:
        return False
    if (_point_in_convex_polygon(p1, hull) or
        _point_in_convex_polygon(p2, hull)):
        return True
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        if _segments_intersect(p1, p2, a, b):
            return True
    return False


def _point_in_convex_polygon(p, hull) -> bool:
    """Strict-or-on-edge test for CCW convex polygons."""
    px, py = p
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        if (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) < 0.0:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Proper segment-segment intersection (open endpoints)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if v >  1e-12: return 1
        if v < -1e-12: return -1
        return 0
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False


def _point_to_segment(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    ex, ey = x2 - x1, y2 - y1
    l2 = ex * ex + ey * ey
    if l2 < 1e-12:
        return _point_to_point(px, py, p1)
    t = ((px - x1) * ex + (py - y1) * ey) / l2
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    qx, qy = x1 + t * ex, y1 + t * ey
    dx, dy = px - qx, py - qy
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return 0.0, 0.0, 0.0
    return d, dx / d, dy / d
