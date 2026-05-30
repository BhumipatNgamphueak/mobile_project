"""
analyze_integration.py
======================
Plot integrated GMM + EB vs planner-only baseline for World C (cross-opposite)
and World D (cross-same-direction) at lookahead L = 5 m. The video recordings
for these runs were unreliable so we substitute static overhead plots:

  - Robot trajectory (baseline vs integrated, colour-coded)
  - Both humans' ground-truth trajectories, interpolated from the SDF actor
    waypoints against the run's sim-time (SDF actor poses are not published
    on /pose/info in Ignition Fortress).
  - Snapshot poses of each human at the *moment the robot reaches the
    human's lane* — this is what determines whether a collision occurs.
  - Min robot-human distance per run, annotated on each subplot.

Inputs (CSV columns documented in path_planning_node.py / social_costmap_node.py):
  - /home/prime/robot_logs/world3_delay6/w3__delay6_look5.csv  (W3 baseline)
  - /home/prime/robot_logs/world3_egg/look5.csv                (W3 integrated)
  - /home/prime/robot_logs/world4/world4_look5.csv             (W4 baseline)
  - /home/prime/robot_logs/world4_egg/look5.csv                (W4 integrated)

Outputs:
  - figures/integration_world3.png
  - figures/integration_world4.png
  - figures/integration_summary.png  (combined 1x2 grid for the README)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = '/home/prime/mobile_project/figures'
LOG = {
    'w3_base': '/home/prime/robot_logs/world3_delay6/w3__delay6_look5.csv',
    'w3_egg':  '/home/prime/robot_logs/world3_egg/look5.csv',
    'w4_base': '/home/prime/robot_logs/world4/world4_look5.csv',
    'w4_egg':  '/home/prime/robot_logs/world4_egg/look5.csv',
}

# SDF actor waypoints (t_sim, x, y).  Extracted verbatim from
# world3_cross_opposite.sdf and world4_cross_same.sdf.
WAYPOINTS = {
    'w3_h1': [(0.0, 2.0, -8.0), (2.0, 2.0, -5.6), (4.0, 2.0, -3.2),
              (6.0, 2.0, -0.8), (8.0, 2.0, 1.6),  (10.0, 2.0, 4.0),
              (12.0, 2.0, 6.4), (13.3, 2.0, 8.0),
              (14.8, 2.0, 8.0), (16.8, 2.0, 5.6), (18.8, 2.0, 3.2),
              (20.8, 2.0, 0.8), (22.8, 2.0,-1.6), (24.8, 2.0,-4.0),
              (26.8, 2.0,-6.4), (28.1, 2.0,-8.0)],
    'w3_h2': [(0.0,-2.0, 8.0), (2.0,-2.0, 5.6), (4.0,-2.0, 3.2),
              (6.0,-2.0, 0.8), (8.0,-2.0,-1.6), (10.0,-2.0,-4.0),
              (12.0,-2.0,-6.4), (13.3,-2.0,-8.0),
              (14.8,-2.0,-8.0), (16.8,-2.0,-5.6), (18.8,-2.0,-3.2),
              (20.8,-2.0,-0.8), (22.8,-2.0, 1.6), (24.8,-2.0, 4.0),
              (26.8,-2.0, 6.4), (28.1,-2.0, 8.0)],
    'w4_h1': [(0.0, 1.0, -8.0), (2.0, 1.0, -5.6), (4.0, 1.0, -3.2),
              (6.0, 1.0, -0.8), (8.0, 1.0, 1.6),  (10.0, 1.0, 4.0),
              (12.0, 1.0, 6.4), (13.3, 1.0, 8.0),
              (14.8, 1.0, 8.0), (16.8, 1.0, 5.6), (18.8, 1.0, 3.2),
              (20.8, 1.0, 0.8), (22.8, 1.0,-1.6), (24.8, 1.0,-4.0),
              (26.8, 1.0,-6.4), (28.1, 1.0,-8.0)],
    'w4_h2': [(0.0,-1.0, -8.0), (2.0,-1.0, -5.6), (4.0,-1.0, -3.2),
              (6.0,-1.0, -0.8), (8.0,-1.0, 1.6),  (10.0,-1.0, 4.0),
              (12.0,-1.0, 6.4), (13.3,-1.0, 8.0),
              (14.8,-1.0, 8.0), (16.8,-1.0, 5.6), (18.8,-1.0, 3.2),
              (20.8,-1.0, 0.8), (22.8,-1.0,-1.6), (24.8,-1.0,-4.0),
              (26.8,-1.0,-6.4), (28.1,-1.0,-8.0)],
}


def interp_actor(name, t_query):
    """Piecewise-linear interpolation of an SDF actor at given sim times."""
    wps = WAYPOINTS[name]
    ts  = np.array([w[0] for w in wps])
    xs  = np.array([w[1] for w in wps])
    ys  = np.array([w[2] for w in wps])
    t_clip = np.clip(t_query, ts[0], ts[-1])
    return np.interp(t_clip, ts, xs), np.interp(t_clip, ts, ys)


def load_robot(path):
    df = pd.read_csv(path)
    # Strip the long stationary tail (robot has reached goal and is idle).
    moving = (df['vx'].abs() + df['vy'].abs() + df['omega'].abs()) > 0.01
    if moving.any():
        last_move = moving[::-1].idxmax()
        df = df.loc[:last_move + 5]
    return df


def min_dist_robot_human(robot_df, human_name):
    hx, hy = interp_actor(human_name, robot_df['time_s'].values)
    d = np.hypot(robot_df['x'].values - hx, robot_df['y'].values - hy)
    i = int(np.argmin(d))
    return float(d[i]), float(robot_df['x'].iloc[i]), float(robot_df['y'].iloc[i]), float(hx[i]), float(hy[i])


def plot_one_world(ax, base_path, egg_path, humans, title, lane_x):
    base = load_robot(base_path)
    egg  = load_robot(egg_path)

    # Backdrop: arena bounds.
    ax.axvline(0, color='#dddddd', lw=0.6, zorder=0)
    ax.axhline(0, color='#dddddd', lw=0.6, zorder=0)

    # Human GT trajectories over each run's time span (faint, full motion).
    t_base = base['time_s'].values
    t_egg  = egg['time_s'].values
    t_all  = np.linspace(min(t_base.min(), t_egg.min()),
                          max(t_base.max(), t_egg.max()), 200)
    human_colors = ['#888888', '#bbbbbb']
    for hname, color in zip(humans, human_colors):
        hx, hy = interp_actor(hname, t_all)
        ax.plot(hx, hy, color=color, lw=1.4, ls=':', alpha=0.8, zorder=1)
        # Human start
        sx, sy = interp_actor(hname, [t_all[0]])
        ax.plot(sx, sy, marker='o', color=color, ms=7,
                 markeredgecolor='k', markeredgewidth=0.5, zorder=4)
        ax.annotate(hname.split('_')[1].upper(), (sx[0], sy[0]),
                     textcoords='offset points', xytext=(8, 0),
                     fontsize=8, color=color)

    # Robot trajectories.
    ax.plot(base['x'].to_numpy(), base['y'].to_numpy(), color='#1f77b4', lw=2.2,
            label='Baseline (LiDAR only)', zorder=3)
    ax.plot(egg ['x'].to_numpy(), egg ['y'].to_numpy(), color='#d62728', lw=2.2, ls='--',
            label='Integrated (LiDAR + GMM)', zorder=3)

    # Start (cross) + goal (star).
    ax.plot(0, 0, marker='X', color='k', ms=11, zorder=6, label='Start')
    ax.plot(15, 0, marker='*', color='gold', ms=18,
             markeredgecolor='k', markeredgewidth=0.8, zorder=6, label='Goal')

    # Snapshot: position of each human at the moment of closest approach
    # to the robot (i.e. the instant that decides whether a collision occurred).
    for hname, color in zip(humans, human_colors):
        for run_label, df, run_color in [
            ('base', base, '#1f77b4'),
            ('egg',  egg,  '#d62728'),
        ]:
            d_min, rx, ry, hx, hy = min_dist_robot_human(df, hname)
            ax.plot(hx, hy, marker='o', color=color, ms=11,
                    markeredgecolor=run_color, markeredgewidth=2.0, zorder=5)
            ax.plot([rx, hx], [ry, hy],
                    color=run_color, lw=1.0, ls='-', alpha=0.55, zorder=2)
            # Label the closest-approach distance at the midpoint.
            mx, my = (rx + hx) / 2, (ry + hy) / 2
            ax.annotate(f'{d_min:.2f} m', (mx, my), fontsize=8,
                         color=run_color, ha='center',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                   ec=run_color, alpha=0.85, lw=0.5))

    # Min-distance annotation.
    txt_lines = []
    for run_label, df in [('Baseline', base), ('Integrated', egg)]:
        per_human_min = []
        for hname in humans:
            d, _, _, _, _ = min_dist_robot_human(df, hname)
            per_human_min.append(d)
        overall_min = min(per_human_min)
        path_len = df['path_len_m'].iloc[-1]
        flag = '⚠ collision' if overall_min < 0.5 else 'OK'
        txt_lines.append(
            f'{run_label}: path = {path_len:.2f} m,  min dist = {overall_min:.2f} m  ({flag})')
    ax.text(0.02, 0.02, '\n'.join(txt_lines), transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='#cccccc', alpha=0.95))

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('x  (m)')
    ax.set_ylabel('y  (m)')
    ax.set_xlim(-2.0, 17.0)
    ax.set_ylim(-9.0, 9.0)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, (ax_w3, ax_w4) = plt.subplots(1, 2, figsize=(15, 7))

    plot_one_world(
        ax_w3,
        LOG['w3_base'], LOG['w3_egg'],
        humans=['w3_h1', 'w3_h2'],
        title='World C — Cross Opposite (L = 5 m)',
        lane_x={'w3_h1': 2.0, 'w3_h2': -2.0},
    )

    plot_one_world(
        ax_w4,
        LOG['w4_base'], LOG['w4_egg'],
        humans=['w4_h1', 'w4_h2'],
        title='World D — Cross Same Direction (L = 5 m)',
        lane_x={'w4_h1': 1.0, 'w4_h2': -1.0},
    )

    # Common legend at the bottom for the snapshot marker meaning.
    extra_handles = [
        Line2D([0], [0], color='#888888', lw=1.4, ls=':',
               label='Human GT trajectory (over run)'),
        Line2D([0], [0], marker='o', color='#888888', ms=10,
               markeredgecolor='#1f77b4', markeredgewidth=2.0, lw=0,
               label='Human at moment robot crosses its lane (baseline)'),
        Line2D([0], [0], marker='o', color='#888888', ms=10,
               markeredgecolor='#d62728', markeredgewidth=2.0, lw=0,
               label='Human at moment robot crosses its lane (integrated)'),
    ]
    fig.legend(handles=extra_handles, loc='lower center', ncol=3,
                bbox_to_anchor=(0.5, -0.02), frameon=True,
                edgecolor='#cccccc')

    fig.suptitle('Dynamic Obstacle (Integrated System) — Robot vs Human GT trajectories',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'integration_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)

    # Individual high-res plots (for the per-world subsections).
    for tag, (base_k, egg_k, humans, title, lane) in {
        'w3': (LOG['w3_base'], LOG['w3_egg'], ['w3_h1', 'w3_h2'],
               'World C — Cross Opposite (L = 5 m)',
               {'w3_h1': 2.0, 'w3_h2': -2.0}),
        'w4': (LOG['w4_base'], LOG['w4_egg'], ['w4_h1', 'w4_h2'],
               'World D — Cross Same Direction (L = 5 m)',
               {'w4_h1': 1.0, 'w4_h2': -1.0}),
    }.items():
        f2, a2 = plt.subplots(figsize=(9, 7))
        plot_one_world(a2, base_k, egg_k, humans, title, lane)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f'integration_world{tag[-1]}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
        plt.close(f2)


if __name__ == '__main__':
    main()
