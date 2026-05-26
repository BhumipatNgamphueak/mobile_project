#!/usr/bin/env python3
"""
analyze_angle_error.py
======================
Position error vs angle from camera boresight for Case B (walking human).
Hypothesis: error is lowest at 0° (human directly ahead) and rises toward FOV edges.
"""

import glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = os.path.expanduser('~/robot_logs')
OUT_PNG = os.path.join(os.path.dirname(__file__), 'angle_error.png')
FOV_HALF = 30.0
RANGES   = [1, 3, 5, 7]
CMAP     = plt.cm.plasma
COLORS   = {r: CMAP(i / (len(RANGES) - 1)) for i, r in enumerate(RANGES)}

# ── Load ───────────────────────────────────────────────────────────────────────
frames = []
for path in sorted(glob.glob(os.path.join(LOG_DIR, 'perception_log_world_perc_walk_*.csv'))):
    df = pd.read_csv(path)
    if len(df) < 2:
        continue
    med_x = df['gt_x'].median()
    if med_x < 0:
        continue
    rng = round(med_x - 0.4)
    if rng not in RANGES:
        continue
    df['range_m'] = rng
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
det    = all_df[all_df['det_id'].notna()].copy()
det['angle'] = np.degrees(np.arctan2(det['gt_y'], det['gt_x']))

# ── Figure layout: 1 wide + 4 per-range ───────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Position Error vs Viewing Angle from Camera Boresight\n'
             'Hypothesis: lowest error at 0° (front-on), highest near ±30° (FOV edge)',
             fontsize=13, fontweight='bold')

# gs: left column = combined all-range; right 2x2 = per-range
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
ax_all  = fig.add_subplot(gs[:, 0])
ax_grid = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(1, 3)]

BIN_EDGES  = np.linspace(-32, 32, 14)
BIN_CENTS  = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# ── Combined plot ──────────────────────────────────────────────────────────────
for rng in RANGES:
    sub = det[det['range_m'] == rng]
    ax_all.scatter(sub['angle'], sub['pos_error_m'],
                   s=10, alpha=0.25, color=COLORS[rng], zorder=2)

    # binned mean ± std
    means, stds, valid_cents = [], [], []
    for lo, hi, c in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_CENTS):
        b = sub[(sub['angle'] >= lo) & (sub['angle'] < hi)]
        if len(b) >= 3:
            means.append(b['pos_error_m'].mean())
            stds.append(b['pos_error_m'].std())
            valid_cents.append(c)
    if means:
        ax_all.errorbar(valid_cents, means, yerr=stds,
                        color=COLORS[rng], lw=2, capsize=3,
                        label=f'{rng} m (n={len(sub)})', zorder=4)

    # quadratic fit to visualise trend
    if len(sub) >= 10:
        try:
            sub_clean = sub[['angle', 'pos_error_m']].dropna()
            coeffs = np.polyfit(sub_clean['angle'].values, sub_clean['pos_error_m'].values, 2)
            xs = np.linspace(sub_clean['angle'].min(), sub_clean['angle'].max(), 100)
            ax_all.plot(xs, np.polyval(coeffs, xs),
                        '--', color=COLORS[rng], lw=1.2, alpha=0.7, zorder=3)
        except np.linalg.LinAlgError:
            pass

ax_all.axvline( FOV_HALF, color='grey', ls=':', lw=1.2, alpha=0.7)
ax_all.axvline(-FOV_HALF, color='grey', ls=':', lw=1.2, alpha=0.7,
               label='FOV edge ±30°')
ax_all.axvline(0, color='black', ls='-', lw=0.6, alpha=0.4)
ax_all.annotate('Front-on\n(0°)', xy=(0, ax_all.get_ylim()[0] if False else 0.1),
                ha='center', fontsize=8, color='black', alpha=0.6)
ax_all.set_xlabel('Angle from camera boresight (°)', fontsize=10)
ax_all.set_ylabel('Position error (m)', fontsize=10)
ax_all.set_title('All ranges combined', fontweight='bold')
ax_all.set_xlim(-35, 35)
ax_all.set_ylim(0, 1.7)
ax_all.legend(fontsize=7)
ax_all.grid(True, alpha=0.25)

# ── Per-range plots ────────────────────────────────────────────────────────────
for ax, rng in zip(ax_grid, RANGES):
    sub = det[det['range_m'] == rng]
    c   = COLORS[rng]

    # scatter
    ax.scatter(sub['angle'], sub['pos_error_m'],
               s=14, alpha=0.4, color=c, zorder=2, label='samples')

    # binned mean ± std
    means, stds, valid_cents = [], [], []
    for lo, hi, cent in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_CENTS):
        b = sub[(sub['angle'] >= lo) & (sub['angle'] < hi)]
        if len(b) >= 2:
            means.append(b['pos_error_m'].mean())
            stds.append(b['pos_error_m'].std())
            valid_cents.append(cent)
    if means:
        ax.errorbar(valid_cents, means, yerr=stds,
                    color='black', lw=2, capsize=3, zorder=5, label='mean ± std')

    # quadratic fit
    if len(sub) >= 8:
        try:
            sub_clean = sub[['angle', 'pos_error_m']].dropna()
            coeffs = np.polyfit(sub_clean['angle'].values, sub_clean['pos_error_m'].values, 2)
            xs = np.linspace(-30, 30, 120)
            ys = np.polyval(coeffs, xs)
            ax.plot(xs, ys, '--', color=c, lw=2, zorder=4, label='quadratic fit')
            # vertex of parabola = angle of minimum error
            if coeffs[0] != 0:
                vertex_x = -coeffs[1] / (2 * coeffs[0])
                vertex_y = np.polyval(coeffs, vertex_x)
                ax.annotate(f'min @ {vertex_x:.1f}°',
                            xy=(vertex_x, vertex_y), xytext=(vertex_x + 6, vertex_y + 0.15),
                            arrowprops=dict(arrowstyle='->', color='dimgrey', lw=0.8),
                            fontsize=7, color='dimgrey')
        except np.linalg.LinAlgError:
            pass

    # FOV limits
    fov_limit = min(FOV_HALF, sub['angle'].abs().max() + 2)
    ax.axvspan(-FOV_HALF, -fov_limit, alpha=0.08, color='red')
    ax.axvspan( fov_limit,  FOV_HALF, alpha=0.08, color='red')
    ax.axvline( FOV_HALF, color='red', ls=':', lw=1.0, alpha=0.6)
    ax.axvline(-FOV_HALF, color='red', ls=':', lw=1.0, alpha=0.6)
    ax.axvline(0, color='black', ls='-', lw=0.5, alpha=0.3)

    in_fov_rows = all_df[(all_df['range_m'] == rng) &
                          (all_df['gt_y'].abs() < rng * np.tan(np.radians(FOV_HALF)))]
    det_rate_in = in_fov_rows['det_id'].notna().mean() * 100 if len(in_fov_rows) else 0.0
    ax.set_title(f'{rng} m  (n={len(sub)}, det-in-FOV={det_rate_in:.0f}%)',
                 fontweight='bold', fontsize=9)
    ax.set_xlabel('Angle (°)', fontsize=9)
    ax.set_ylabel('Pos error (m)', fontsize=9)
    ax.set_xlim(-35, 35)
    ax.set_ylim(0, 1.7)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)

plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT_PNG}")

# ── Console summary ────────────────────────────────────────────────────────────
print(f"\n{'Range':>6}  {'Centre ±5°':>12}  {'Edge ±25-30°':>14}  {'Δ error':>9}  {'Hypothesis':>12}")
print('-' * 65)
for rng in RANGES:
    sub = det[det['range_m'] == rng]
    centre = sub[sub['angle'].abs() <= 5]['pos_error_m']
    edge   = sub[sub['angle'].abs() >= 20]['pos_error_m']
    c_mean = centre.mean() if len(centre) >= 2 else float('nan')
    e_mean = edge.mean()   if len(edge)   >= 2 else float('nan')
    delta  = e_mean - c_mean
    verdict = 'CONFIRMED' if delta > 0.05 else ('weak' if delta > 0 else 'NOT confirmed')
    print(f"{rng:>5}m  {c_mean:>10.3f}m  {e_mean:>13.3f}m  {delta:>+8.3f}m  {verdict:>12}")
