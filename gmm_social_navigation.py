"""
3D Gaussian Mixture Model Visualization for Social Navigation
Based on: "Sensor Fusion for Social Navigation on a Mobile Robot Based on
Fast Marching Square and Gaussian Mixture Model" (Mora et al., Sensors 2022)

Two cases:
  1. Single person  — asymmetric two-Gaussian mixture (front > rear)
  2. Group of people — single orientation-aware 2D Gaussian (O-Space)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ──────────────────────────────────────────────
# Helper: 2-D Gaussian kernel
# ──────────────────────────────────────────────
def gaussian_2d(X, Y, cx, cy, sigma_x, sigma_y, theta=0.0):
    """
    Evaluate a 2-D anisotropic Gaussian at grid points (X, Y).

    Parameters
    ----------
    cx, cy    : centre of the Gaussian
    sigma_x   : std along the (rotated) x-axis
    sigma_y   : std along the (rotated) y-axis
    theta     : rotation angle (rad) – person heading
    """
    # Translate
    Xc = X - cx
    Yc = Y - cy

    # Rotate into person frame
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xr =  cos_t * Xc + sin_t * Yc
    Yr = -sin_t * Xc + cos_t * Yc

    return np.exp(-0.5 * ((Xr / sigma_x) ** 2 + (Yr / sigma_y) ** 2))


# ──────────────────────────────────────────────
# Eq. (2) – Single-person GMM
# Φi(q) = δ(yq)·ΦF(q) + (1 - δ(yq))·ΦR(q)
#
# Covariances (Eq. 3):
#   Front : Σ_F = diag(σx², 4σx²)  → elongated forward
#   Rear  : Σ_R = diag(σx², σx²)   → symmetric
#   σx = 0.46/2 = 0.23 m
# ──────────────────────────────────────────────
SIGMA_X = 0.46 / 2          # 0.23 m  (individual personal distance / 2)
SIGMA_Y_FRONT = 2 * SIGMA_X # front lobe is twice as wide
SIGMA_Y_REAR  = SIGMA_X     # rear lobe is symmetric


def person_gmm(X, Y, cx, cy, theta):
    """
    Two-Gaussian mixture for one person facing direction `theta`.
    The person's *forward* direction is +Y in their local frame.
    """
    # Local y-coordinate (positive = in front of person)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xc = X - cx;  Yc = Y - cy
    local_y = -sin_t * Xc + cos_t * Yc   # forward component

    phi_front = gaussian_2d(X, Y, cx, cy, SIGMA_X, SIGMA_Y_FRONT, theta)
    phi_rear  = gaussian_2d(X, Y, cx, cy, SIGMA_X, SIGMA_Y_REAR,  theta)

    delta = (local_y >= 0).astype(float)
    return delta * phi_front + (1 - delta) * phi_rear


# ──────────────────────────────────────────────
# Eqs. (4-8) – Group O-Space Gaussian
# σx = DH/4,  σy = Di/2
# ──────────────────────────────────────────────
def group_gmm(X, Y, people):
    """
    Single Gaussian representing the O-Space of a group.

    Parameters
    ----------
    people : list of (x, y) tuples – positions of group members
    """
    if len(people) < 2:
        raise ValueError("Need at least 2 people for a group.")

    pts = np.array(people)

    # Centroid C  (Eq. generalisation)
    if len(people) == 2:
        cx, cy = pts.mean(axis=0)
    else:
        cx, cy = pts.mean(axis=0)

    # Polygon edges: connect consecutive vertices (simple convex order)
    # DH = mean edge length (Eq. 6)
    edges = []
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        d = np.linalg.norm(pts[j] - pts[i])
        if d <= 1.22:           # max social distance
            edges.append(d)
    DH = np.mean(edges) if edges else np.linalg.norm(pts[1] - pts[0])

    # Di = distance from centre to farthest person (Eq. 7)
    dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    far_idx = np.argmax(dists)
    Di = dists[far_idx]
    Hfar = pts[far_idx]

    # Θ = orientation angle (Eq. 8)
    theta = np.arctan2(Hfar[0] - cx, Hfar[1] - cy)

    sigma_x = DH / 4
    sigma_y = Di / 2

    return gaussian_2d(X, Y, cx, cy, sigma_x, sigma_y, theta), cx, cy, theta


# ──────────────────────────────────────────────
# Build grid
# ──────────────────────────────────────────────
EXTENT = 1.6          # metres each side
RES    = 300
x_lin  = np.linspace(-EXTENT, EXTENT, RES)
y_lin  = np.linspace(-EXTENT, EXTENT, RES)
X, Y   = np.meshgrid(x_lin, y_lin)


# ══════════════════════════════════════════════
# FIGURE 1 – Single-person GMM at 5 orientations
# ══════════════════════════════════════════════
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
orient_labels = ["0°  (facing +Y)", "45°", "90°  (right)", "135°", "180° (facing -Y)"]

fig1 = plt.figure(figsize=(18, 5))
fig1.suptitle("Single-Person Two-Gaussian Mixture Model  (Eq. 2–3)\n"
              r"$\sigma_x = 0.23$ m,  front lobe: $\sigma_y = 2\sigma_x$,  "
              r"rear lobe: $\sigma_y = \sigma_x$",
              fontsize=13, y=1.01)

for idx, (theta, label) in enumerate(zip(orientations, orient_labels)):
    Z = person_gmm(X, Y, 0, 0, theta)

    ax = fig1.add_subplot(1, 5, idx + 1, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0,
                           antialiased=True, alpha=0.85)
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap="plasma", alpha=0.4, levels=15)

    # Arrow showing person facing direction
    dx =  np.sin(theta) * 0.5
    dy =  np.cos(theta) * 0.5
    ax.quiver(0, 0, 1.05, dx, dy, 0, color="cyan", linewidth=2,
              arrow_length_ratio=0.35)

    ax.set_title(f"θ = {label}", fontsize=9)
    ax.set_xlabel("x (m)", fontsize=7, labelpad=2)
    ax.set_ylabel("y (m)", fontsize=7, labelpad=2)
    ax.set_zlabel("Φ", fontsize=7, labelpad=2)
    ax.set_zlim(0, 1.1)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=30, azim=-55)

fig1.tight_layout()
plt.show()
# fig1.savefig("/mnt/user-data/outputs/gmm_single_person.png", dpi=160, bbox_inches="tight")
# print("Saved gmm_single_person.png")
