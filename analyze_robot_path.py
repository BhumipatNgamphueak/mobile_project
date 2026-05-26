#!/usr/bin/env python3
"""
analyze_robot_path.py
=====================
Visualise all robot navigation data from src/robot_logs/.

Three figures:
  nav_overview.png       — crossing_humans: trajectories, goal rate, safety, efficiency
  lookahead_tuning.png   — effect of lookahead distance on path quality
  social_costmap_effect.png — with vs without GMM egg social costmap
"""

import glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE    = '/home/prime/mobile_project_new/src/robot_logs'
OUT_DIR = os.path.dirname(__file__)

# ── Obstacle geometry (odom frame, from CLAUDE.md) ─────────────────────────────
OBSTACLES = [(11, 1), (3, -9), (9, -8)]   # centres, 2×2 m squares
WALL_X    = (-3, 17)
WALL_Y    = (-14, 6)

def draw_world(ax, walls=True, obstacles=True):
    if obstacles:
        for cx, cy in OBSTACLES:
            ax.add_patch(plt.Rectangle((cx-1, cy-1), 2, 2,
                         color='dimgrey', alpha=0.5, zorder=2))
    if walls:
        ax.set_xlim(WALL_X[0]-1, WALL_X[1]+1)
        ax.set_ylim(WALL_Y[0]-1, WALL_Y[1]+1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Navigation overview (crossing_humans)
# ══════════════════════════════════════════════════════════════════════════════

def fig_nav_overview():
    # ── load robot paths ──────────────────────────────────────────────────────
    paths = []
    for f in sorted(glob.glob(os.path.join(BASE, 'robot_path_crossing_humans_*.csv'))):
        df = pd.read_csv(f).reset_index(drop=True)
        if len(df) < 20:
            continue
        df['file'] = os.path.basename(f)
        paths.append(df)

    # ── load run summaries ────────────────────────────────────────────────────
    summaries = []
    for f in sorted(glob.glob(os.path.join(BASE, 'run_summary_*.csv'))):
        s = pd.read_csv(f)
        summaries.append(s)
    all_s = pd.concat(summaries, ignore_index=True)
    all_s = all_s[all_s['path_length_m'] > 0.1].copy()   # remove zero-path rows
    all_s['efficiency'] = all_s['straight_line_m'] / all_s['path_length_m'].clip(lower=0.01)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Robot Navigation Overview — crossing_humans & related worlds',
                 fontsize=13, fontweight='bold')

    # ── [0,0] Trajectories ────────────────────────────────────────────────────
    ax = axes[0, 0]
    cmap = plt.cm.tab10
    valid_paths = [df for df in paths if
                   abs(df['x'].iloc[-1]) + abs(df['y'].iloc[-1]) > 0.5]
    for i, df in enumerate(valid_paths[:12]):
        ax.plot(df['x'].values, df['y'].values, lw=0.9, alpha=0.7,
                color=cmap(i % 10))
        ax.plot(df['x'].iloc[0],  df['y'].iloc[0],  'o',
                color=cmap(i % 10), ms=4, zorder=4)
        ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], '^',
                color=cmap(i % 10), ms=4, zorder=4)
    draw_world(ax)
    ax.set_title('Robot Trajectories (crossing_humans)', fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.2)
    ax.legend(handles=[mpatches.Patch(color='dimgrey', alpha=0.5, label='Obstacle')],
              loc='lower right', fontsize=7)

    # ── [0,1] Goal success rate by world ──────────────────────────────────────
    ax = axes[0, 1]
    world_order = ['crossing_humans', 'world3_cross_opposite', 'world4_cross_same']
    world_labels = ['crossing\nhumans', 'world3\ncross-opposite', 'world4\ncross-same']
    rates, counts = [], []
    for w in world_order:
        sub = all_s[all_s['world'] == w]
        rates.append(sub['goal_reached'].mean() * 100 if len(sub) else 0)
        counts.append(len(sub))
    colours = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(world_labels, rates, color=colours, alpha=0.85, edgecolor='white')
    for bar, n, r in zip(bars, counts, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{r:.0f}%\n(n={n})', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Goal success rate (%)')
    ax.set_title('Goal Success Rate by World', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # ── [0,2] Path efficiency ─────────────────────────────────────────────────
    ax = axes[0, 2]
    for i, w in enumerate(world_order):
        sub = all_s[all_s['world'] == w]['efficiency'].dropna()
        if sub.empty:
            continue
        ax.scatter([world_labels[i]] * len(sub), sub.values, alpha=0.5,
                   color=colours[i], s=30, zorder=3)
        ax.plot(world_labels[i], sub.mean(), 'D', color=colours[i],
                ms=10, zorder=5, label=f'{world_labels[i].replace(chr(10)," ")} μ={sub.mean():.2f}')
    ax.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5, label='straight-line ideal')
    ax.set_ylabel('Efficiency (straight/actual)')
    ax.set_title('Path Efficiency\n(1.0 = straight line)', fontweight='bold')
    ax.legend(fontsize=6)
    ax.grid(True, axis='y', alpha=0.3)

    # ── [1,0] Min obstacle distance distribution ───────────────────────────────
    ax = axes[1, 0]
    all_min = pd.concat([df.assign(run=i) for i, df in enumerate(paths[:15])])
    runs = all_min['run'].unique()
    data_per_run = [all_min[all_min['run'] == r]['min_obs_dist_m'].values for r in runs]
    data_per_run = [d for d in data_per_run if len(d) > 5]
    if data_per_run:
        bp = ax.boxplot(data_per_run, patch_artist=True, showfliers=False,
                        medianprops=dict(color='red', lw=2))
        for patch in bp['boxes']:
            patch.set_facecolor('#90CAF9')
            patch.set_alpha(0.7)
    ax.axhline(0.35, color='red', ls='--', lw=1.2, label='inflation radius 0.35 m')
    ax.axhline(1.35, color='orange', ls='--', lw=1.0, label='safe clearance 1.35 m')
    ax.set_xlabel('Run index')
    ax.set_ylabel('Min obstacle distance (m)')
    ax.set_title('Safety: Min Obstacle Distance\n(per run)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # ── [1,1] Travel time vs straight-line distance ────────────────────────────
    ax = axes[1, 1]
    for i, w in enumerate(world_order):
        sub = all_s[all_s['world'] == w]
        if sub.empty:
            continue
        reached = sub[sub['goal_reached'] == 1]
        failed  = sub[sub['goal_reached'] == 0]
        ax.scatter(reached['straight_line_m'].values, reached['travel_time_s'].values,
                   color=colours[i], marker='o', s=50, alpha=0.8, zorder=3,
                   label=f'{world_labels[i].replace(chr(10)," ")} ✓')
        ax.scatter(failed['straight_line_m'].values,  failed['travel_time_s'].values,
                   color=colours[i], marker='x', s=60, alpha=0.8, zorder=3,
                   linewidths=2)
    ax.set_xlabel('Straight-line distance to goal (m)')
    ax.set_ylabel('Travel time (s)')
    ax.set_title('Travel Time vs Distance\n(○ reached, × failed)', fontweight='bold')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)

    # ── [1,2] Final position error ─────────────────────────────────────────────
    ax = axes[1, 2]
    reached_all = all_s[all_s['goal_reached'] == 1]
    for i, w in enumerate(world_order):
        sub = reached_all[reached_all['world'] == w]['final_pos_error_m']
        if sub.empty:
            continue
        ax.scatter([world_labels[i]] * len(sub), sub.values, alpha=0.6,
                   color=colours[i], s=35, zorder=3)
        ax.plot(world_labels[i], sub.mean(), 'D', color=colours[i],
                ms=10, zorder=5)
    ax.axhline(0.5, color='grey', ls='--', lw=1, alpha=0.6, label='0.5 m threshold')
    ax.set_ylabel('Final position error (m) — reached only')
    ax.set_title('Goal Accuracy\n(successfully reached runs)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'nav_overview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Lookahead distance tuning
# ══════════════════════════════════════════════════════════════════════════════

def load_lookahead_data():
    """Return dict: {world_tag: {lookahead: DataFrame}}"""
    groups = {}

    # world2
    for f in glob.glob(os.path.join(BASE, 'world2', '*.csv')):
        name = os.path.basename(f)
        if 'look_' not in name:
            continue
        look = float(name.split('look_')[1].replace('.csv', ''))
        groups.setdefault('world2', {})[look] = pd.read_csv(f)

    # world3_delay6
    for f in glob.glob(os.path.join(BASE, 'world3_delay6', '*.csv')):
        name = os.path.basename(f)
        look = float(name.split('look')[1].replace('.csv', ''))
        groups.setdefault('world3', {})[look] = pd.read_csv(f)

    # world4
    for f in glob.glob(os.path.join(BASE, 'world4', '*.csv')):
        name = os.path.basename(f)
        look = float(name.split('look')[1].replace('.csv', ''))
        groups.setdefault('world4', {})[look] = pd.read_csv(f)

    return groups


def fig_lookahead_tuning():
    groups = load_lookahead_data()
    if not groups:
        print('No lookahead data found, skipping.')
        return

    world_tags = sorted(groups.keys())
    n_worlds   = len(world_tags)
    fig, axes  = plt.subplots(3, n_worlds, figsize=(6 * n_worlds, 13))
    if n_worlds == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle('Effect of Lookahead Distance on Navigation Quality',
                 fontsize=13, fontweight='bold')

    LOOK_CMAP = plt.cm.plasma

    for col, tag in enumerate(world_tags):
        lookaheads = sorted(groups[tag].keys())
        norm       = plt.Normalize(min(lookaheads), max(lookaheads))

        # row 0: trajectories
        ax_traj = axes[0, col]
        for look in lookaheads:
            df  = groups[tag][look]
            c   = LOOK_CMAP(norm(look))
            ax_traj.plot(df['x'].values, df['y'].values, lw=1.2, color=c, alpha=0.85,
                         label=f'look={look:.0f}m')
        ax_traj.set_title(f'{tag} — Trajectories', fontweight='bold')
        ax_traj.set_xlabel('x (m)')
        ax_traj.set_ylabel('y (m)')
        ax_traj.legend(fontsize=7)
        ax_traj.grid(True, alpha=0.2)
        ax_traj.set_aspect('equal')

        # row 1: min obstacle distance over travelled distance
        ax_safe = axes[1, col]
        for look in lookaheads:
            df   = groups[tag][look]
            dist = np.cumsum(np.sqrt(np.diff(df['x'].values, prepend=df['x'].iloc[0])**2 +
                                     np.diff(df['y'].values, prepend=df['y'].iloc[0])**2))
            c    = LOOK_CMAP(norm(look))
            ax_safe.plot(dist, df['min_obs_dist_m'].values, lw=1.0, color=c,
                         alpha=0.7, label=f'{look:.0f}m')
        ax_safe.axhline(0.35, color='red', ls='--', lw=1, alpha=0.7,
                         label='inflation r')
        ax_safe.set_title(f'{tag} — Min Obstacle Distance', fontweight='bold')
        ax_safe.set_xlabel('Distance travelled (m)')
        ax_safe.set_ylabel('Min obs dist (m)')
        ax_safe.legend(fontsize=6)
        ax_safe.grid(True, alpha=0.25)

        # row 2: replan time over travel distance
        ax_replan = axes[2, col]
        for look in lookaheads:
            df   = groups[tag][look]
            dist = np.cumsum(np.sqrt(np.diff(df['x'].values, prepend=df['x'].iloc[0])**2 +
                                     np.diff(df['y'].values, prepend=df['y'].iloc[0])**2))
            c    = LOOK_CMAP(norm(look))
            # smooth replan_ms with rolling window
            smoothed = df['replan_ms'].rolling(10, min_periods=1).mean().values
            ax_replan.plot(dist, smoothed, lw=1.0, color=c, alpha=0.7,
                           label=f'{look:.0f}m')
        ax_replan.set_title(f'{tag} — Replan Time (smoothed)', fontweight='bold')
        ax_replan.set_xlabel('Distance travelled (m)')
        ax_replan.set_ylabel('Replan time (ms)')
        ax_replan.legend(fontsize=6)
        ax_replan.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'lookahead_tuning.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Social costmap (egg) effect
# ══════════════════════════════════════════════════════════════════════════════

def fig_social_costmap_effect():
    pairs = {
        'world3': (
            os.path.join(BASE, 'world3_back',  'world3_look5.csv'),
            os.path.join(BASE, 'world3_egg',   'look5.csv'),
        ),
        'world4': (
            os.path.join(BASE, 'world4',       'world4_look5.csv'),
            os.path.join(BASE, 'world4_egg',   'look5.csv'),
        ),
    }

    valid = {k: v for k, v in pairs.items()
             if os.path.exists(v[0]) and os.path.exists(v[1])}
    if not valid:
        print('No egg/no-egg pairs found, skipping.')
        return

    n = len(valid)
    fig, axes = plt.subplots(2, n * 2, figsize=(7 * n, 10))
    fig.suptitle('Social Costmap (GMM Egg) Effect: With vs Without',
                 fontsize=13, fontweight='bold')

    COLORS = {'no_egg': '#FF7043', 'egg': '#42A5F5'}

    for i, (tag, (path_no, path_egg)) in enumerate(valid.items()):
        df_no  = pd.read_csv(path_no)
        df_egg = pd.read_csv(path_egg)
        col_base = i * 2

        # ── trajectory ──────────────────────────────────────────────────────
        ax_traj = axes[0, col_base]
        ax_traj.plot(df_no['x'].values,  df_no['y'].values,  lw=1.8, color=COLORS['no_egg'],
                     label='Without egg (LiDAR only)', zorder=3)
        ax_traj.plot(df_egg['x'].values, df_egg['y'].values, lw=1.8, color=COLORS['egg'],
                     label='With egg (LiDAR + GMM)', zorder=3)
        ax_traj.plot(df_no['x'].iloc[0],   df_no['y'].iloc[0],   's',
                     color='black', ms=8, zorder=5, label='Start')
        ax_traj.plot(df_no['goal_x'].iloc[-1], df_no['goal_y'].iloc[-1], '*',
                     color='gold', ms=12, zorder=5, label='Goal')
        ax_traj.set_title(f'{tag}: Trajectory', fontweight='bold')
        ax_traj.set_xlabel('x (m)')
        ax_traj.set_ylabel('y (m)')
        ax_traj.legend(fontsize=7)
        ax_traj.grid(True, alpha=0.2)
        ax_traj.set_aspect('equal')

        # ── safety + speed comparison ───────────────────────────────────────
        ax_comp = axes[0, col_base + 1]
        metrics_no  = {
            'min_dist_m':    df_no['min_obs_dist_m'].min(),
            'mean_dist_m':   df_no['min_obs_dist_m'].mean(),
            'path_len_m':    df_no['path_len_m'].iloc[-1] if 'path_len_m' in df_no else 0,
            'travel_t_s':    df_no['time_s'].max() - df_no['time_s'].min(),
            'replan_ms_avg': df_no['replan_ms'].mean(),
        }
        metrics_egg = {
            'min_dist_m':    df_egg['min_obs_dist_m'].min(),
            'mean_dist_m':   df_egg['min_obs_dist_m'].mean(),
            'path_len_m':    df_egg['path_len_m'].iloc[-1] if 'path_len_m' in df_egg else 0,
            'travel_t_s':    df_egg['time_s'].max() - df_egg['time_s'].min(),
            'replan_ms_avg': df_egg['replan_ms'].mean(),
        }
        labels = ['Min\ndist (m)', 'Mean\ndist (m)', 'Travel\ntime (s)', 'Replan\ntime (ms)']
        keys   = ['min_dist_m', 'mean_dist_m', 'travel_t_s', 'replan_ms_avg']
        x = np.arange(len(labels))
        w = 0.35
        ax_comp.bar(x - w/2, [metrics_no[k]  for k in keys], w,
                    color=COLORS['no_egg'], alpha=0.85, label='Without egg')
        ax_comp.bar(x + w/2, [metrics_egg[k] for k in keys], w,
                    color=COLORS['egg'],    alpha=0.85, label='With egg')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(labels, fontsize=8)
        ax_comp.set_title(f'{tag}: Metrics Comparison', fontweight='bold')
        ax_comp.legend(fontsize=7)
        ax_comp.grid(True, axis='y', alpha=0.3)

        # ── min obs dist over time ───────────────────────────────────────────
        ax_dist = axes[1, col_base]
        ax_dist.plot(df_no['time_s'].values,  df_no['min_obs_dist_m'].values,
                     color=COLORS['no_egg'], lw=1.0, alpha=0.8, label='Without egg')
        ax_dist.plot(df_egg['time_s'].values, df_egg['min_obs_dist_m'].values,
                     color=COLORS['egg'],    lw=1.0, alpha=0.8, label='With egg')
        ax_dist.axhline(0.35, color='red', ls='--', lw=1, alpha=0.7, label='inflation r')
        ax_dist.axhline(1.35, color='orange', ls=':', lw=1, alpha=0.6, label='safe 1.35m')
        ax_dist.set_xlabel('Time (s)')
        ax_dist.set_ylabel('Min obstacle distance (m)')
        ax_dist.set_title(f'{tag}: Safety Over Time', fontweight='bold')
        ax_dist.legend(fontsize=7)
        ax_dist.grid(True, alpha=0.25)

        # ── speed profile ────────────────────────────────────────────────────
        ax_vel = axes[1, col_base + 1]
        speed_no  = np.sqrt(df_no['vx'].values**2  + df_no['vy'].values**2)
        speed_egg = np.sqrt(df_egg['vx'].values**2 + df_egg['vy'].values**2)
        ax_vel.plot(df_no['time_s'].values,  speed_no,
                    color=COLORS['no_egg'], lw=1.0, alpha=0.8, label='Without egg')
        ax_vel.plot(df_egg['time_s'].values, speed_egg,
                    color=COLORS['egg'],    lw=1.0, alpha=0.8, label='With egg')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Speed (m/s)')
        ax_vel.set_title(f'{tag}: Speed Profile', fontweight='bold')
        ax_vel.legend(fontsize=7)
        ax_vel.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'social_costmap_effect.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Figure 1: Navigation overview …')
    fig_nav_overview()

    print('Figure 2: Lookahead tuning …')
    fig_lookahead_tuning()

    print('Figure 3: Social costmap effect …')
    fig_social_costmap_effect()

    print('\nDone. Files saved to:', OUT_DIR)
