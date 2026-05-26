#!/usr/bin/env python3
"""
make_video.py
=============
Produce experiment_summary.mp4 — an animated overview of all 3 experiment parts.

Part 1  (0–Ns):   Elastic band lookahead tuning — robot paths side-by-side
Part 2A (Ns–Ms):  Perception Case A — standing human, range sweep
Part 2B (Ms–Ks):  Perception Case B — walking human, detection animation
Part 3  (Ks–end): Integration — robot navigating past humans with social costmap

Requires: ffmpeg  (sudo apt install ffmpeg)
"""

import glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

BASE    = '/home/prime/mobile_project_new/src/robot_logs'
OUT     = os.path.join(os.path.dirname(__file__), 'experiment_summary.mp4')
FPS     = 24
FOV_HALF = 30.0

# ── colour palette ─────────────────────────────────────────────────────────────
C_ROBOT   = '#1565C0'
C_HUMAN   = '#E53935'
C_DET     = '#43A047'
C_MISS    = '#EF9A9A'
C_PATH    = '#90CAF9'
C_EGG     = '#FF8F00'
C_FOV     = '#B3E5FC'

# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def fov_patch(x, y, yaw, half_deg=FOV_HALF, length=8.0):
    """Return a Wedge patch for the camera FOV centred at (x,y)."""
    angle_start = np.degrees(yaw) - half_deg
    angle_end   = np.degrees(yaw) + half_deg
    return mpatches.Wedge((x, y), length, angle_start, angle_end,
                          color=C_FOV, alpha=0.25, zorder=1)


def load_walk_csv(log_dir, range_m):
    frames = []
    for path in sorted(glob.glob(os.path.join(log_dir,
                                              'perception_log_world_perc_walk_*.csv'))):
        df = pd.read_csv(path)
        if len(df) < 2:
            continue
        if round(df['gt_x'].median() - 0.4) == range_m:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_best_nav_path():
    """Return the largest crossing_humans path (most complete run)."""
    best, best_len = None, 0
    for f in glob.glob(os.path.join(BASE, 'robot_path_crossing_humans_*.csv')):
        df = pd.read_csv(f)
        if len(df) > best_len:
            best, best_len = df, len(df)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Lookahead tuning animation
# ══════════════════════════════════════════════════════════════════════════════

def make_part1_frames(n_frames=180):
    """Return list of (fig, drawn) — one figure per animation frame."""
    look_files = {
        2:  os.path.join(BASE, 'world2', 'world2_look_2.csv'),
        5:  os.path.join(BASE, 'world2', 'world2_look_5.csv'),
        10: os.path.join(BASE, 'world2', 'world2_look_10.csv'),
        15: os.path.join(BASE, 'world2', 'world2_look_15.csv'),
    }
    dfs = {}
    for look, f in look_files.items():
        if os.path.exists(f):
            dfs[look] = pd.read_csv(f)

    # resample all to same length
    target = n_frames
    resampled = {}
    for look, df in dfs.items():
        idx = np.round(np.linspace(0, len(df)-1, target)).astype(int)
        resampled[look] = df.iloc[idx].reset_index(drop=True)

    LOOK_CMAP = plt.cm.plasma
    norm      = plt.Normalize(2, 15)
    colors    = {l: LOOK_CMAP(norm(l)) for l in dfs}

    frames = []
    for frame_i in range(target):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('#0d1117')
        ax.set_title('Part 1: Elastic Band — Lookahead Tuning (world2)',
                     color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('x (m)', color='white')
        ax.set_ylabel('y (m)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        fig.patch.set_facecolor('#0d1117')

        for look, df in resampled.items():
            c = colors[look]
            # full ghost path
            ax.plot(df['x'].values, df['y'].values,
                    color=c, alpha=0.15, lw=1.0, zorder=2)
            # progress up to this frame
            sub = df.iloc[:frame_i+1]
            ax.plot(sub['x'].values, sub['y'].values,
                    color=c, alpha=0.9, lw=1.8, zorder=3,
                    label=f'look={look:.0f}m')
            if frame_i > 0:
                ax.plot(sub['x'].iloc[-1], sub['y'].iloc[-1],
                        'o', color=c, ms=6, zorder=5)

        ax.set_xlim(-1, 16)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=8,
                  facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
        ax.grid(True, alpha=0.1, color='white')

        # progress bar
        pct = (frame_i + 1) / target
        ax.axhline(-2.7, xmin=0, xmax=pct, color='white', lw=3, alpha=0.5)
        ax.text(0, -2.5, f'{pct*100:.0f}%', color='white', fontsize=7, alpha=0.7)

        plt.tight_layout()
        frames.append(fig)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Perception animation (walking human)
# ══════════════════════════════════════════════════════════════════════════════

def make_part2_frames(n_frames=200):
    # Use 3m and 7m walking data
    data = {}
    for r in [3, 7]:
        df = load_walk_csv(BASE, r)
        if df is not None:
            data[r] = df

    if not data:
        return []

    frames = []
    for r, df in data.items():
        # subsample to n_frames
        idx = np.round(np.linspace(0, len(df)-1, n_frames)).astype(int)
        sub = df.iloc[idx].reset_index(drop=True)
        robot_x = r + 0.4   # camera x in robot-relative coords = range from GT

        for i in range(n_frames):
            row = sub.iloc[i]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor('#0d1117')
            fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

            ax.set_title(f'Part 2B: Perception — Walking Human @ {r} m range',
                         color='white', fontsize=11, fontweight='bold')
            ax.set_xlabel('x (m)', color='white')
            ax.set_ylabel('y (m)', color='white')

            # FOV cone — robot at x=0, facing +x
            wedge = fov_patch(0, 0, 0, length=r + 2)
            ax.add_patch(wedge)

            # Robot
            ax.plot(0, 0, 's', color=C_ROBOT, ms=12, zorder=5, label='Robot')
            ax.annotate('Camera', (0, 0), (0.3, 0.5),
                        color='white', fontsize=7, alpha=0.7)

            # GT human path (ghost)
            ax.axvline(row['gt_x'], color='grey', ls='--', lw=0.8, alpha=0.3)
            ax.plot(row['gt_x'], row['gt_y'], 'o',
                    color=C_HUMAN, ms=10, zorder=6, label='GT position')

            # detected or not
            detected = not pd.isna(row['det_id'])
            if detected:
                ax.plot(row['det_x'], row['det_y'], 'D',
                        color=C_DET, ms=8, zorder=7, label='GMM detected')
                ax.plot([row['gt_x'], row['det_x']],
                        [row['gt_y'], row['det_y']],
                        color='yellow', lw=1.2, alpha=0.6, zorder=4)
                err_txt = f'err={row["pos_error_m"]:.2f}m'
                spd_txt = f'spd={row["det_speed_mps"]:.2f}m/s'
                ax.text(row['det_x'] + 0.2, row['det_y'],
                        f'{err_txt}\n{spd_txt}', color=C_DET, fontsize=7)
            else:
                ax.plot(row['gt_x'], row['gt_y'], 'x',
                        color=C_MISS, ms=12, mew=2, zorder=7, label='Missed')

            # FOV boundary indicators
            fov_y = r * np.tan(np.radians(FOV_HALF))
            ax.axhline( fov_y, xmin=0.05, xmax=0.95,
                        color='cyan', ls=':', lw=0.8, alpha=0.5)
            ax.axhline(-fov_y, xmin=0.05, xmax=0.95,
                        color='cyan', ls=':', lw=0.8, alpha=0.5, label=f'±FOV @{r}m')

            status = 'DETECTED ✓' if detected else 'MISSED ✗'
            col    = C_DET if detected else C_MISS
            ax.text(0.02, 0.95, status, transform=ax.transAxes,
                    color=col, fontsize=13, fontweight='bold', va='top')
            ax.text(0.02, 0.88,
                    f't={row["time_s"]:.1f}s  gt_y={row["gt_y"]:.2f}m',
                    transform=ax.transAxes, color='white', fontsize=8, va='top')

            ax.set_xlim(-1, r + 3)
            ax.set_ylim(-7, 7)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=7,
                      facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
            ax.grid(True, alpha=0.1, color='white')
            plt.tight_layout()
            frames.append(fig)

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Integration navigation animation
# ══════════════════════════════════════════════════════════════════════════════

def make_part3_frames(n_frames=200):
    df = load_best_nav_path()
    if df is None:
        return []

    # subsample
    idx = np.round(np.linspace(0, len(df)-1, n_frames)).astype(int)
    sub = df.iloc[idx].reset_index(drop=True)

    OBSTACLES = [(11, 1), (3, -9), (9, -8)]

    frames = []
    trail_x, trail_y = [], []

    for i in range(n_frames):
        row = sub.iloc[i]
        trail_x.append(row['x'])
        trail_y.append(row['y'])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#0d1117')

        # ── left: top-down trajectory ─────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        ax.set_title('Part 3: Integration — Robot Navigating\nwith Social Costmap',
                     color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

        # obstacles
        for cx, cy in OBSTACLES:
            ax.add_patch(plt.Rectangle((cx-1, cy-1), 2, 2,
                         color='#455A64', alpha=0.8, zorder=2))

        # trail
        if len(trail_x) > 1:
            ax.plot(trail_x, trail_y, color=C_PATH, lw=1.5, alpha=0.7, zorder=3)

        # robot + FOV
        yaw = row['yaw_rad']
        wedge = fov_patch(row['x'], row['y'], yaw, length=5)
        ax.add_patch(wedge)
        ax.plot(row['x'], row['y'], 's', color=C_ROBOT, ms=10, zorder=6)

        # goal
        ax.plot(row['goal_x'], row['goal_y'], '*',
                color='gold', ms=14, zorder=6, label='Goal')

        ax.set_xlim(-4, 18)
        ax.set_ylim(-15, 7)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', color='white')
        ax.set_ylabel('y (m)', color='white')
        ax.grid(True, alpha=0.1, color='white')
        ax.legend(loc='upper right', fontsize=7,
                  facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')

        # ── right: metrics over time ──────────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor('#0d1117')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        ax2.set_title('Min Obstacle Distance & Speed',
                      color='white', fontsize=10, fontweight='bold')

        t_window = sub.iloc[max(0, i-60):i+1]
        if len(t_window) > 1:
            ax2.plot(t_window['time_s'].values, t_window['min_obs_dist_m'].values,
                     color='#64B5F6', lw=1.5, label='Min obs dist (m)')
            speed = np.sqrt(t_window['vx'].values**2 + t_window['vy'].values**2)
            ax2.plot(t_window['time_s'].values, speed,
                     color='#A5D6A7', lw=1.5, label='Speed (m/s)')
        ax2.axhline(0.35, color='red',    ls='--', lw=1, alpha=0.7, label='Inflation r')
        ax2.axhline(1.35, color='orange', ls=':',  lw=1, alpha=0.5, label='Safe 1.35m')
        ax2.set_xlim(sub['time_s'].iloc[max(0, i-60)], sub['time_s'].iloc[i] + 1)
        ax2.set_ylim(0, 4)
        ax2.set_xlabel('Time (s)', color='white')
        ax2.legend(loc='upper left', fontsize=7,
                   facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
        ax2.grid(True, alpha=0.1, color='white')

        # status text
        ax2.text(0.98, 0.95,
                 f'dist={row["min_obs_dist_m"]:.2f}m\npolys={int(row["n_polygons"])}',
                 transform=ax2.transAxes, color='white', fontsize=8,
                 ha='right', va='top')

        plt.tight_layout()
        frames.append(fig)

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Title cards
# ══════════════════════════════════════════════════════════════════════════════

def make_title_card(title, subtitle='', n_frames=36):
    frames = []
    for fi in range(n_frames):
        alpha = min(1.0, fi / 10)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.axis('off')
        ax.text(0.5, 0.55, title, transform=ax.transAxes,
                color='white', fontsize=18, fontweight='bold',
                ha='center', va='center', alpha=alpha)
        ax.text(0.5, 0.38, subtitle, transform=ax.transAxes,
                color='#90CAF9', fontsize=11,
                ha='center', va='center', alpha=alpha)
        plt.tight_layout()
        frames.append(fig)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Assemble & encode
# ══════════════════════════════════════════════════════════════════════════════

def figs_to_mp4(all_figs, out_path, fps=FPS):
    """Convert a list of matplotlib figures to MP4 via ffmpeg pipe."""
    import subprocess, io
    from PIL import Image

    print(f'Encoding {len(all_figs)} frames at {fps} fps …')

    # Get frame size from first figure
    buf = io.BytesIO()
    all_figs[0].savefig(buf, format='png', dpi=100, facecolor=all_figs[0].get_facecolor())
    buf.seek(0)
    img0  = Image.open(buf)
    W, H  = img0.size

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}',
        '-r', str(fps),
        '-i', '-',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i, fig in enumerate(all_figs):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        img = Image.open(buf).convert('RGB').resize((W, H))
        proc.stdin.write(np.array(img).tobytes())
        plt.close(fig)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(all_figs)} frames written')
    proc.stdin.close()
    proc.wait()
    print(f'Saved → {out_path}')


def main():
    print('Building title card …')
    frames  = make_title_card(
        'Socially-Aware Mecanum Robot Navigation',
        'Elastic Band + YOLO/GMM Social Costmap  |  ROS 2 Humble + Ignition Gazebo',
        n_frames=48,
    )

    print('Building Part 1 (lookahead tuning) …')
    frames += make_title_card('Part 1: Elastic Band — Lookahead Tuning',
                               'Vary lookahead_dist: 2 / 5 / 10 / 15 m  |  world2', n_frames=24)
    frames += make_part1_frames(n_frames=180)

    print('Building Part 2 (perception) …')
    frames += make_title_card('Part 2: Perception Accuracy (GMM)',
                               'Case B — Walking human perpendicular  |  3 m & 7 m range', n_frames=24)
    frames += make_part2_frames(n_frames=150)

    print('Building Part 3 (integration) …')
    frames += make_title_card('Part 3: Integration',
                               'Elastic Band + GMM social costmap  |  crossing_humans world', n_frames=24)
    frames += make_part3_frames(n_frames=180)

    frames += make_title_card('Experiment Complete', '', n_frames=36)

    figs_to_mp4(frames, OUT, fps=FPS)


if __name__ == '__main__':
    main()
