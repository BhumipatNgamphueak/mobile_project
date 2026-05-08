#!/usr/bin/env python3
"""
scale_human_speed.py
====================
Adjust the walking speed of one or all humans in a world.  Both the
animated <actor> and the kinematic <model><plugin filename="__waypoints__">
share the same waypoint table, so this single edit affects what you see
in Gazebo *and* the ground-truth pose published to /gz_dynamic_poses.

Modes
-----
  --show                       Print current walking speed per human
                               (no file edits)
  --speed K                    Multiply the speed of selected human(s) by K
                               (K > 1 faster, K < 1 slower).  Round-trip
                               safe: scale by 0.5 then 2.0 = identity.
  --target-mps V               Rescale so the selected human(s) walk at V
                               metres per second (computed from current
                               waypoint geometry).

Scope
-----
  --world W                    Target one world (e.g. "crossing_humans")
                               or "ALL" for every world SDF.
  --human NAME                 Target one human (e.g. "human_3").  Default:
                               every human in the chosen world(s).

Examples
--------
  # See current pace for every human in every world
  ros2 run mecanum_robot_sim scale_human_speed --show --world ALL

  # Halve human_1 in crossing_humans (1.2 → 0.6 m/s)
  ros2 run mecanum_robot_sim scale_human_speed --world crossing_humans \\
      --human human_1 --speed 0.5

  # Set human_3 in world7 to exactly 0.8 m/s
  ros2 run mecanum_robot_sim scale_human_speed --world world7_crowd_vertical \\
      --human human_3 --target-mps 0.8

  # Slow every human in every world to 80 % of current pace
  ros2 run mecanum_robot_sim scale_human_speed --world ALL --speed 0.8
"""
import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _find_worlds_dir() -> Path:
    """
    Locate the worlds/ directory in either layout:
      • running directly:  src/mecanum_robot_sim/worlds/  (parent.parent of script)
      • running via `ros2 run`: install/mecanum_robot_sim/share/mecanum_robot_sim/worlds/
        (resolved via ament).  With --symlink-install, that path is a symlink
        back to the source tree, so edits go to src/ and persist across builds.
    """
    here = Path(__file__).resolve().parent.parent / 'worlds'
    if here.is_dir():
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('mecanum_robot_sim')) / 'worlds'
        if share.is_dir():
            return share
    except Exception:
        pass
    raise FileNotFoundError(
        'worlds/ directory not found.  Build the package and source the '
        'install setup, or run the script directly from src/.')


WORLDS_DIR = _find_worlds_dir()

# A <waypoint><time>...</time><pose>...</pose></waypoint> block.
# Captures: prefix, time, suffix.
WP_TIME = re.compile(
    r'(<waypoint>\s*<time>)([^<]+)(</time>)',
)

# Block boundaries — to scope the edit to one human at a time.
ACTOR_RE = re.compile(
    r'(<actor name="([^"]+)">)(.*?)(</actor>)', re.DOTALL)
MODEL_RE = re.compile(
    r'(<model name="([^"]+)">)(.*?)(</model>)', re.DOTALL)


def scale_block(block: str, factor: float) -> tuple[str, int]:
    """Multiply every <time> inside `block` by `factor`. Returns (new, n)."""
    n = [0]
    def repl(m):
        try:
            t = float(m.group(2).strip())
        except ValueError:
            return m.group(0)
        new_t = t * factor
        n[0] += 1
        # Keep formatting tidy: 5 decimals trimmed, never negative
        return f'{m.group(1)}{max(0.0, new_t):>6.4f}{m.group(3)}'
    new = WP_TIME.sub(repl, block)
    return new, n[0]


def matches_actor_for(name: str, target: str) -> bool:
    """The <actor> sibling of model `target` is named `target_anim`."""
    return name == target or name == f'{target}_anim'


def process_world(path: Path, speed: float, only: str | None) -> int:
    """Apply 1/speed scaling to every human waypoint in `path`."""
    if speed <= 0.0:
        raise ValueError(f'speed must be > 0, got {speed}')
    factor = 1.0 / speed
    text = path.read_text()
    total = 0

    def model_repl(m):
        nonlocal total
        name, body = m.group(2), m.group(3)
        if not name.startswith('human'):
            return m.group(0)
        if only and name != only:
            return m.group(0)
        new_body, n = scale_block(body, factor)
        total += n
        return f'{m.group(1)}{new_body}{m.group(4)}'

    def actor_repl(m):
        nonlocal total
        name, body = m.group(2), m.group(3)
        if not name.startswith('human'):
            return m.group(0)
        if only and not matches_actor_for(name, only):
            return m.group(0)
        new_body, n = scale_block(body, factor)
        total += n
        return f'{m.group(1)}{new_body}{m.group(4)}'

    text = MODEL_RE.sub(model_repl, text)
    text = ACTOR_RE.sub(actor_repl, text)
    path.write_text(text)
    return total


def _human_waypoints(model_or_actor) -> list:
    """Return [(t, x, y), ...] sorted by time, from a model or actor element."""
    out = []
    for wp in model_or_actor.iter('waypoint'):
        t = wp.findtext('time')
        p = wp.findtext('pose')
        if t is None or p is None:
            continue
        try:
            t = float(t)
            xs = [float(s) for s in p.split()]
            if len(xs) >= 2:
                out.append((t, xs[0], xs[1]))
        except ValueError:
            pass
    out.sort(key=lambda w: w[0])
    return out


def _avg_speed(wps: list) -> float:
    """Mean walking speed over consecutive waypoint segments (m/s)."""
    if len(wps) < 2:
        return 0.0
    dist = 0.0
    dur  = 0.0
    for i in range(len(wps) - 1):
        t0, x0, y0 = wps[i]
        t1, x1, y1 = wps[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-3:    # skip pivot-in-place segments (yaw turns)
            continue
        dist += d
        dur  += dt
    return dist / dur if dur > 0 else 0.0


def show_speeds(files, only):
    print(f'{"world":35s} {"human":12s} {"waypoints":>10} '
          f'{"period(s)":>10} {"avg(m/s)":>10}')
    for f in files:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        world = root.find('world')
        if world is None:
            continue
        for model in world.findall('model'):
            name = model.get('name', '')
            if not name.startswith('human'):
                continue
            if only and name != only:
                continue
            # Prefer the controller's plugin waypoints — those drive the
            # ground-truth /gz_dynamic_poses motion.
            wp_plug = next((p for p in model.findall('plugin')
                            if p.get('filename') == '__waypoints__'), None)
            wps = _human_waypoints(wp_plug) if wp_plug is not None else []
            if not wps:
                continue
            period = wps[-1][0] - wps[0][0]
            print(f'  {f.name:33s} {name:12s} {len(wps):>10} '
                  f'{period:>10.2f} {_avg_speed(wps):>10.3f}')


def main():
    ap = argparse.ArgumentParser(
        description='Inspect or rescale walking speed of human entities in '
                    'mecanum_robot_sim worlds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('--world', required=True,
                    help='World SDF stem (e.g. "crossing_humans") or "ALL"')
    ap.add_argument('--human', default=None,
                    help='Limit to one human (e.g. "human_1"); default: all humans')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--show', action='store_true',
                      help='Print current speeds; no file edits')
    mode.add_argument('--speed', type=float,
                      help='Multiply speed by this factor (>1 faster, <1 slower)')
    mode.add_argument('--target-mps', type=float, dest='target_mps',
                      help='Rescale to this absolute walking speed (m/s)')
    args = ap.parse_args()

    if args.world.upper() == 'ALL':
        files = sorted(WORLDS_DIR.glob('*.sdf'))
    else:
        p = WORLDS_DIR / f'{args.world}.sdf'
        if not p.exists():
            print(f'error: {p} not found', file=sys.stderr)
            return 1
        files = [p]

    if args.show:
        show_speeds(files, args.human)
        return 0

    # Determine the multiplier per file/human.
    if args.speed is not None:
        if args.speed <= 0:
            print('error: --speed must be > 0', file=sys.stderr)
            return 1
        for f in files:
            n = process_world(f, args.speed, args.human)
            print(f'  {f.name}: scaled {n} <time> field(s) by ×{1.0/args.speed:.4f}')
        return 0

    # --target-mps: compute per-human factor (current_mps / target_mps).
    if args.target_mps <= 0:
        print('error: --target-mps must be > 0', file=sys.stderr)
        return 1
    for f in files:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        world = root.find('world')
        if world is None:
            continue
        for model in world.findall('model'):
            name = model.get('name', '')
            if not name.startswith('human'):
                continue
            if args.human and name != args.human:
                continue
            plug = next((p for p in model.findall('plugin')
                         if p.get('filename') == '__waypoints__'), None)
            if plug is None:
                continue
            current = _avg_speed(_human_waypoints(plug))
            if current <= 0:
                continue
            speed_mult = args.target_mps / current
            n = process_world(f, speed_mult, name)
            print(f'  {f.name}: {name:12s} {current:.3f} → {args.target_mps:.3f} m/s '
                  f'(×{speed_mult:.4f}, scaled {n} time field(s))')
    return 0


if __name__ == '__main__':
    sys.exit(main())
