#!/usr/bin/env python3
"""
analyze_walk.py
===============
Deep-dive visualisation of Case B (walking human, perpendicular) perception data.

Six subplots:
  1. Top-down FOV map — bird-eye of all ranges with hit/miss markers
  2. Detection rate vs angle from camera boresight (FOV sensitivity curve)
  3. Expected-in-FOV vs actually-detected (shows two separate failure modes)
  4. Position-error bias vectors (where the GMM places humans relative to GT)
  5. Position error vs lateral y-position (edge-of-FOV degradation)
  6. Speed estimation quality along the crossing (GMM warm-up & lag)
"""

import glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

LOG_DIR = os.path.expanduser('~/robot_logs')
OUT_PNG = os.path.join(os.path.dirname(__file__), 'walk_analysis.png')

FOV_HALF_DEG = 30.0          # camera half-FOV
HUMAN_TRAVEL = 12.0          # y: -6 → +6
RANGES       = [1, 3, 5, 7, 9]
CMAP         = plt.cm.viridis
RANGE_COLORS = {r: CMAP(i / (len(RANGES) - 1)) for i, r in enumerate(RANGES)}


# ── Load walk data ─────────────────────────────────────────────────────────────
def load_walk() -> dict[int, pd.DataFrame]:
    """Return {range_m: DataFrame} for walk case only."""
    data: dict[int, list] = {}
    for path in sorted(glob.glob(os.path.join(LOG_DIR, 'perception_log_world_perc_walk_*.csv'))):
        df = pd.read_csv(path)
        if len(df) < 2:
            continue
        med_x = df['gt_x'].median()
        if med_x < 0:
            continue
        rng = round(med_x - 0.4)
        data.setdefault(rng, []).append(df)
    # concat duplicates
    return {r: pd.concat(dfs, ignore_index=True) for r, dfs in data.items()}


# ── Helper ─────────────────────────────────────────────────────────────────────
def angle_deg(row) -> float:
    """Horizontal angle of the GT human from camera boresight (+x axis)."""
    return np.degrees(np.arctan2(row['gt_y'], row['gt_x']))


# ── Plot 1: Top-down FOV map ───────────────────────────────────────────────────
def plot_fov_map(ax, datasets: dict):
    ax.set_title('1. Top-down FOV Map', fontweight='bold')

    max_range = max(RANGES) + 1.5
    fov_rad   = np.radians(FOV_HALF_DEG)

    # FOV cone
    fov_x = [0, max_range * np.cos(fov_rad), max_range * np.cos(-fov_rad), 0]
    fov_y = [0, max_range * np.sin(fov_rad), max_range * np.sin(-fov_rad), 0]
    ax.fill(fov_x, fov_y, alpha=0.08, color='steelblue', zorder=0)
    ax.plot([0, max_range * np.cos(fov_rad)],
            [0, max_range * np.sin(fov_rad)], 'b--', lw=0.8, alpha=0.5)
    ax.plot([0, max_range * np.cos(-fov_rad)],
            [0, max_range * np.sin(-fov_rad)], 'b--', lw=0.8, alpha=0.5, label='60° FOV')

    for rng, df in datasets.items():
        c = RANGE_COLORS[rng]
        # human walking path line
        ax.axvline(rng, color=c, lw=0.6, alpha=0.3)

        missed = df[df['det_id'].isna()]
        hit    = df[df['det_id'].notna()]

        ax.scatter(missed['gt_x'], missed['gt_y'], color='red',  s=4,  alpha=0.25, zorder=2)
        ax.scatter(hit['gt_x'],    hit['gt_y'],    color='lime', s=8,  alpha=0.6,  zorder=3)

    # robot marker
    ax.plot(0, 0, 's', color='black', markersize=10, zorder=5)
    ax.annotate('Robot\n(camera)', (0, 0), textcoords='offset points',
                xytext=(6, -18), fontsize=7)

    ax.set_xlabel('x — range (m)')
    ax.set_ylabel('y — lateral (m)')
    ax.set_xlim(-0.5, max_range)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    legend_elements = [
        mpatches.Patch(color='lime', label='Detected'),
        mpatches.Patch(color='red',  label='Missed'),
        mpatches.Patch(color='steelblue', alpha=0.2, label='60° FOV'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')


# ── Plot 2: Detection rate vs FOV angle ───────────────────────────────────────
def plot_angle_sensitivity(ax, datasets: dict):
    ax.set_title('2. Detection Rate vs Angle from Boresight', fontweight='bold')
    bins = np.linspace(-35, 35, 15)
    centres = 0.5 * (bins[:-1] + bins[1:])

    for rng, df in datasets.items():
        df = df.copy()
        df['angle'] = df.apply(angle_deg, axis=1)
        df['detected'] = df['det_id'].notna().astype(int)

        rates = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (df['angle'] >= lo) & (df['angle'] < hi)
            subset = df[mask]
            rates.append(subset['detected'].mean() * 100 if len(subset) >= 3 else np.nan)

        ax.plot(centres, rates, marker='o', markersize=4,
                color=RANGE_COLORS[rng], label=f'{rng} m')

    ax.axvline(-FOV_HALF_DEG, color='grey', ls=':', lw=1.2)
    ax.axvline( FOV_HALF_DEG, color='grey', ls=':', lw=1.2, label='FOV boundary ±30°')
    ax.set_xlabel('Angle from camera axis (°)')
    ax.set_ylabel('Detection rate (%)')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)


# ── Plot 3: Expected-in-FOV vs actually detected ───────────────────────────────
def plot_fov_vs_det(ax, datasets: dict):
    ax.set_title('3. FOV Coverage vs Actual Detection Rate', fontweight='bold')

    ranges_sorted = sorted(datasets.keys())
    x = np.arange(len(ranges_sorted))
    w = 0.3

    fov_pcts, det_pcts, in_fov_det_pcts = [], [], []
    for rng in ranges_sorted:
        df = datasets[rng]
        in_fov_mask = np.abs(df['gt_y']) < rng * np.tan(np.radians(FOV_HALF_DEG))
        fov_pct     = in_fov_mask.mean() * 100

        det_pct     = df['det_id'].notna().mean() * 100

        in_fov_df   = df[in_fov_mask]
        in_fov_det  = in_fov_df['det_id'].notna().mean() * 100 if len(in_fov_df) else 0.0

        fov_pcts.append(fov_pct)
        det_pcts.append(det_pct)
        in_fov_det_pcts.append(in_fov_det)

    ax.bar(x - w, fov_pcts,       w, label='% time in FOV (geometry)', color='steelblue', alpha=0.8)
    ax.bar(x,     in_fov_det_pcts, w, label='Det% within FOV (YOLO)',   color='limegreen', alpha=0.8)
    ax.bar(x + w, det_pcts,        w, label='Det% overall',             color='tomato',    alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{r} m' for r in ranges_sorted])
    ax.set_xlabel('Range')
    ax.set_ylabel('%')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis='y')


# ── Plot 4: Position-error bias vectors ───────────────────────────────────────
def plot_bias_vectors(ax, datasets: dict):
    ax.set_title('4. Position Error Bias (GT → Detected)', fontweight='bold')

    for rng, df in datasets.items():
        det = df[df['det_id'].notna()].copy()
        if det.empty:
            continue
        dx = det['det_x'] - det['gt_x']
        dy = det['det_y'] - det['gt_y']
        ax.scatter(dx, dy, s=8, alpha=0.3, color=RANGE_COLORS[rng], label=f'{rng} m')

    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Error in x / range direction (m)')
    ax.set_ylabel('Error in y / lateral direction (m)')
    circle = plt.Circle((0, 0), 0.5, color='grey', fill=False, ls='--', lw=1, alpha=0.5)
    ax.add_patch(circle)
    ax.annotate('0.5 m circle', xy=(0.36, 0.37), fontsize=6, color='grey')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)


# ── Plot 5: Error vs lateral y-position ───────────────────────────────────────
def plot_error_vs_y(ax, datasets: dict):
    ax.set_title('5. Position Error vs Lateral y-position', fontweight='bold')

    for rng, df in datasets.items():
        det = df[df['det_id'].notna()].copy()
        if det.empty:
            continue
        # bin by y
        bins  = np.linspace(-6, 6, 13)
        cents = 0.5 * (bins[:-1] + bins[1:])
        means, stds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sub = det[(det['gt_y'] >= lo) & (det['gt_y'] < hi)]
            means.append(sub['pos_error_m'].mean() if len(sub) >= 2 else np.nan)
            stds.append(sub['pos_error_m'].std()   if len(sub) >= 2 else np.nan)
        ax.plot(cents, means, marker='o', markersize=4,
                color=RANGE_COLORS[rng], label=f'{rng} m')

    # mark FOV edges for each range (different per range, use shaded zone)
    ax.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Human lateral y-position (m)')
    ax.set_ylabel('Mean position error (m)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)


# ── Plot 6: Speed estimation along the crossing ───────────────────────────────
def plot_speed_timeline(ax, datasets: dict):
    ax.set_title('6. Speed Estimation Along Crossing Trajectory', fontweight='bold')

    for rng in [3, 5, 7]:   # most informative ranges
        df = datasets.get(rng)
        if df is None:
            continue
        det = df[df['det_id'].notna()].sort_values('gt_y').copy()
        if det.empty:
            continue
        ax.scatter(det['gt_y'], det['det_speed_mps'],
                   s=12, alpha=0.5, color=RANGE_COLORS[rng],
                   label=f'{rng} m detected speed')

    # GT ground truth
    ax.axhline(1.2, color='black', ls='--', lw=1.5, label='GT speed 1.2 m/s')
    ax.axhline(0.0, color='grey',  ls=':',  lw=0.8)

    ax.set_xlabel('Human y-position during crossing (m)')
    ax.set_ylabel('Detected speed (m/s)')
    ax.set_ylim(-0.1, 2.2)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading walk data …")
    datasets = load_walk()
    for r, df in sorted(datasets.items()):
        print(f"  range={r}m  rows={len(df)}  detected={df['det_id'].notna().sum()}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Case B — Walking Human (Perpendicular): YOLO+GMM Capability & Limitations',
                 fontsize=13, fontweight='bold')

    plot_fov_map(          axes[0, 0], datasets)
    plot_angle_sensitivity(axes[0, 1], datasets)
    plot_fov_vs_det(       axes[0, 2], datasets)
    plot_bias_vectors(     axes[1, 0], datasets)
    plot_error_vs_y(       axes[1, 1], datasets)
    plot_speed_timeline(   axes[1, 2], datasets)

    # shared range colour legend on the right
    handles = [mpatches.Patch(color=RANGE_COLORS[r], label=f'{r} m') for r in RANGES]
    fig.legend(handles=handles, title='Range', loc='lower center',
               ncol=len(RANGES), fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {OUT_PNG}")

    # ── Brief textual summary ──────────────────────────────────────────────────
    print("\n── Key findings ──────────────────────────────────────────────────────")
    for rng in sorted(datasets.keys()):
        df   = datasets[rng]
        det  = df[df['det_id'].notna()]
        in_fov = df[np.abs(df['gt_y']) < rng * np.tan(np.radians(FOV_HALF_DEG))]
        in_fov_det_rate = in_fov['det_id'].notna().mean() * 100 if len(in_fov) else 0
        fov_coverage    = len(in_fov) / len(df) * 100
        pos_err  = det['pos_error_m'].mean() if len(det) else float('nan')
        spd_err  = (det['det_speed_mps'] - det['gt_speed_mps']).abs().mean() if len(det) else float('nan')
        print(f"  {rng}m | in-FOV {fov_coverage:4.1f}% | YOLO-within-FOV {in_fov_det_rate:5.1f}% "
              f"| pos_err {pos_err:.3f}m | spd_err {spd_err:.3f} m/s")


if __name__ == '__main__':
    main()
