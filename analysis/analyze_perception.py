#!/usr/bin/env python3
"""
analyze_perception.py
=====================
Summarise all perception-test CSVs in ~/robot_logs/ and produce:
  - Console table: detection rate, mean/std position error, speed error per case+range
  - perception_summary.png: 4 subplots (det-rate, pos-error, speed-error, FOV analysis)

CSV columns:
  time_s, gt_id, gt_x, gt_y, gt_speed_mps,
  det_id, det_x, det_y, pos_error_m, det_speed_mps,
  n_gt, n_detected
"""

import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR   = os.path.expanduser('~/robot_logs')
OUT_PNG   = os.path.join(os.path.dirname(__file__), 'perception_summary.png')

# ── Load all CSVs ──────────────────────────────────────────────────────────────

def load_files(log_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(log_dir, 'perception_log_*.csv'))):
        fname = os.path.basename(path)
        world = 'stand' if 'stand' in fname else 'walk'
        df = pd.read_csv(path)
        if df.empty or len(df) < 2:
            continue
        # infer range from median gt_x (camera range = gt_x - 0.4 m)
        median_gtx = df['gt_x'].median()
        camera_range = round(median_gtx - 0.4)   # rounds to nearest 1 m
        if camera_range < 0:                       # skip misconfigured run
            print(f"  [skip] {fname}  (gt_x median {median_gtx:.2f} → range {camera_range} m)")
            continue
        df['world']  = world
        df['range_m'] = camera_range
        df['file']   = fname
        frames.append(df)
        print(f"  loaded {fname}  world={world}  range={camera_range} m  rows={len(df)}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Per-file stats ─────────────────────────────────────────────────────────────

def per_run_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (world, range_m, fname), grp in df.groupby(['world', 'range_m', 'file']):
        n_gt_rows   = len(grp)
        n_detected  = grp['det_id'].notna().sum()
        det_rate    = n_detected / n_gt_rows if n_gt_rows else 0.0

        detected    = grp[grp['det_id'].notna()]
        pos_mean    = detected['pos_error_m'].mean() if len(detected) else np.nan
        pos_std     = detected['pos_error_m'].std()  if len(detected) else np.nan

        # speed error (only when human is actually moving)
        moving      = detected[detected['gt_speed_mps'] > 0.1]
        spd_err_mean = (moving['det_speed_mps'] - moving['gt_speed_mps']).abs().mean() \
                       if len(moving) else np.nan

        records.append(dict(
            world=world, range_m=range_m, file=fname,
            n_rows=n_gt_rows, n_detected=n_detected,
            det_rate=det_rate,
            pos_mean_m=pos_mean, pos_std_m=pos_std,
            spd_err_mean=spd_err_mean,
        ))
    return pd.DataFrame(records)


# ── Aggregate across runs at the same (world, range) ──────────────────────────

def aggregate(stats: pd.DataFrame) -> pd.DataFrame:
    agg = (stats
           .groupby(['world', 'range_m'])
           .agg(
               runs           = ('file', 'count'),
               det_rate_mean  = ('det_rate',    'mean'),
               det_rate_std   = ('det_rate',    'std'),
               pos_mean_m     = ('pos_mean_m',  'mean'),
               pos_std_m      = ('pos_std_m',   'mean'),
               spd_err_mean   = ('spd_err_mean','mean'),
           )
           .reset_index()
           .sort_values(['world', 'range_m']))
    return agg


# ── Print console table ────────────────────────────────────────────────────────

def print_table(agg: pd.DataFrame) -> None:
    sep = '-' * 78
    hdr = f"{'Case':<8} {'Range':>6}  {'Runs':>4}  {'Det%':>6}  {'±':>5}  " \
          f"{'PosErr(m)':>9}  {'±':>5}  {'SpdErr(m/s)':>11}"
    print('\n' + sep)
    print(hdr)
    print(sep)
    for _, r in agg.iterrows():
        det_std  = f"{r.det_rate_std*100:4.1f}" if not np.isnan(r.det_rate_std)  else "  — "
        pos_std  = f"{r.pos_std_m:5.3f}"        if not np.isnan(r.pos_std_m)     else "  —  "
        spd_err  = f"{r.spd_err_mean:7.3f}"     if not np.isnan(r.spd_err_mean)  else "     —    "
        print(f"{r.world:<8} {r.range_m:>5.0f}m  {int(r.runs):>4}  "
              f"{r.det_rate_mean*100:5.1f}%  {det_std}  "
              f"{r.pos_mean_m:8.3f}m  {pos_std}  {spd_err}")
    print(sep + '\n')


# ── Plots ─────────────────────────────────────────────────────────────────────

COLOURS = {'stand': '#1f77b4', 'walk': '#ff7f0e'}
LABELS  = {'stand': 'Case A – Standing', 'walk': 'Case B – Walking'}

def plot_summary(agg: pd.DataFrame, raw: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('GMM Perception Accuracy vs Detection Range', fontsize=14, fontweight='bold')

    ax_det, ax_pos, ax_spd, ax_fov = axes.flat

    for world, grp in agg.groupby('world'):
        c = COLOURS[world]
        lbl = LABELS[world]
        x = grp['range_m'].values

        # ── Detection rate ────────────────────────────────────────────
        yerr = grp['det_rate_std'].fillna(0).values
        ax_det.errorbar(x, grp['det_rate_mean']*100, yerr=yerr*100,
                        marker='o', color=c, label=lbl, capsize=4)

        # ── Position error ─────────────────────────────────────────────
        ax_pos.errorbar(x, grp['pos_mean_m'], yerr=grp['pos_std_m'].fillna(0),
                        marker='s', color=c, label=lbl, capsize=4)

        # ── Speed error (walk only meaningful) ─────────────────────────
        valid = grp[grp['spd_err_mean'].notna()]
        if len(valid):
            ax_spd.plot(valid['range_m'].values, valid['spd_err_mean'].values,
                        marker='^', color=c, label=lbl)

    # ── FOV visible-y analysis (walk case) ───────────────────────────────────
    walk_raw = raw[raw['world'] == 'walk'].copy()
    if not walk_raw.empty:
        walk_raw['in_fov'] = walk_raw.apply(
            lambda r: abs(r['gt_y']) < r['range_m'] * np.tan(np.radians(30)), axis=1)
        for range_m, grp in walk_raw.groupby('range_m'):
            in_fov = grp[grp['in_fov']]
            if in_fov.empty:
                continue
            det = in_fov['det_id'].notna().mean() * 100
            ax_fov.scatter(range_m, det, color=COLOURS['walk'], zorder=5, s=60)
        walk_agg = agg[agg['world']=='walk']
        ax_fov.plot(walk_agg['range_m'].values,
                    walk_agg['det_rate_mean'].values*100,
                    '--', color=COLOURS['walk'], alpha=0.5, label='overall det%')
        ax_fov.set_title('Case B: Detection rate (within 60° FOV only)')
        ax_fov.set_xlabel('Range (m)')
        ax_fov.set_ylabel('Detection rate (%)')
        ax_fov.set_ylim(0, 105)
        ax_fov.legend()
        ax_fov.grid(True, alpha=0.3)
    else:
        ax_fov.set_visible(False)

    # ── Format axes ───────────────────────────────────────────────────────────
    ax_det.set_title('Detection Rate vs Range')
    ax_det.set_xlabel('Range (m)')
    ax_det.set_ylabel('Detection rate (%)')
    ax_det.set_ylim(0, 105)
    ax_det.legend()
    ax_det.grid(True, alpha=0.3)

    ax_pos.set_title('Position Error vs Range')
    ax_pos.set_xlabel('Range (m)')
    ax_pos.set_ylabel('Mean position error (m)')
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)

    ax_spd.set_title('Speed Estimation Error vs Range')
    ax_spd.set_xlabel('Range (m)')
    ax_spd.set_ylabel('Mean |det_speed − gt_speed| (m/s)')
    ax_spd.legend()
    ax_spd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nLoading CSVs from {LOG_DIR} …")
    raw = load_files(LOG_DIR)
    if raw.empty:
        print("No valid CSV files found.")
        return

    stats = per_run_stats(raw)
    agg   = aggregate(stats)

    print_table(agg)
    plot_summary(agg, raw, OUT_PNG)


if __name__ == '__main__':
    main()
