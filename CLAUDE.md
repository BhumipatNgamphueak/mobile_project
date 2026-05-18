# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

ROS 2 Humble + Ignition Gazebo Fortress workspace. Python ROS nodes only (no C++ packages despite `mecanum_robot_sim` having a CMakeLists.txt — it just installs scripts/urdf/worlds). Always source ROS before any colcon/ros2 command:

```bash
source /opt/ros/humble/setup.bash
source ~/mobile_project/install/setup.bash   # after first build
```

## Build & run

```bash
# Full build from workspace root
colcon build
# Iterate on one package
colcon build --packages-select path_planning && source install/setup.bash
# Clean build (when launch/setup.py changes don't seem to take effect)
rm -rf build install log && colcon build
```

There are no unit tests configured; verification is done in simulation.

Most-used launches (see `run.md` for the canonical list):

```bash
# Full sim with pure_pursuit as controller (mecanum_robot_sim default)
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py
# Sim + local costmap only (debug perception)
ros2 launch path_planning costmap_sim.launch.py
# Sim + TEB stack as the sole velocity controller (NO pure_pursuit)
ros2 launch path_planning teb_direct.launch.py
# Send a goal without RViz's 2D Nav Goal tool
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'odom'}, pose: {position: {x: 5.0, y: 3.0}, orientation: {w: 1.0}}}"
```

Useful `spawn_mecanum.launch.py` args: `test_path:=true path:=scurve|straight|uturn|diagonal|slalom|loop` bypasses `path_planning` and feeds a canned trajectory to `pure_pursuit`; `evaluate:=true` writes a tracking-error PNG.

## Two control stacks live in this repo

The system can be driven by **either** of two mutually exclusive controller pipelines — picking the wrong launch file silently activates the other one.

1. **Pure-pursuit pipeline** (launched by `mecanum_robot_sim/spawn_mecanum.launch.py`)
   `path_planning_node` (originally a stub) → `/planned_path` → `pure_pursuit_node` (RPP) → `/cmd_vel` (TwistStamped) → `cmd_vel_relay` → `/cmd_vel_gz` (Twist).
   `pure_pursuit` is treated as **read-only / complete** per the README.

2. **TEB direct pipeline** (launched by `path_planning/launch/teb_direct.launch.py`, the active branch's focus)
   `local_costmap_node` (LiDAR → `/local_costmap`) + `global_path_node` (RViz goal → straight-line `/global_path`) → `path_planning_node` (TEB elastic-band optimiser) → `/cmd_vel` directly. `pure_pursuit` is **not** launched.

When editing `path_planning_node.py`, know which pipeline you're in: under TEB it owns velocity, under pure-pursuit it only emits `/planned_path`. The current `path_planning_node.py` is the TEB version (subscribes to `/global_path` + `/local_costmap`, publishes `/cmd_vel` + `/planned_path`).

## Frames and the y-axis quirk

- Robot spawns at world (−7, 4), which becomes **odom origin (0, 0)**. `gz_pose_odom.py` produces `/odom` and the `odom→base_link` TF from Gazebo ground-truth pose. All planning / paths must be in the `odom` frame.
- `cmd_vel_relay.py` negates `linear.y` to compensate for an Ignition `VelocityControl` y-axis flip. If a fix to mecanum strafing seems to need a sign change, change it in the relay — don't add a second negation in pure_pursuit or path_planning.

## Static world geometry (hardcoded in path_planning_node)

- Obstacles: A (11, 1), B (3, −9), C (9, −8), all 2×2 m squares (odom frame). Walls bound x∈[−3, 17], y∈[−14, 6].
- Robot inflation radius 0.35 m → safe clearance 1.35 m from obstacle centre edges.
- The static `/real_map` (200×200, 0.1 m, origin (−3, −14)) used by the pure-pursuit pipeline is published by `path_planning_node`'s pure-pursuit-era code; the TEB pipeline instead uses the dynamic `/local_costmap` from LiDAR.

## Current focus: Elastic Band local planner

Active development is on the **Elastic Band local planner** in `src/path_planning/`. A basic version of the algorithm is already implemented and runs end-to-end via `teb_direct.launch.py` — assume the pipeline works and treat new work as iteration on top of it (tuning, obstacle handling, social-cost integration), not a from-scratch build.

Where the algorithm lives:

- `src/path_planning/path_planning/path_planning_node.py` — the elastic-band optimiser itself: costmap → point-obstacle extraction, local-segment pruning + warm-start, gradient descent against obstacle / smoothness terms, then holonomic velocity extraction to `/cmd_vel`. Tunables (`teb_n_iter`, `teb_w_obs`, `teb_w_smooth`, `teb_inflation`, `teb_step`, `teb_max_delta`, `lookahead_dist`) are class attributes near the top of `__init__`.
- `src/path_planning/path_planning/local_costmap_node.py` — LiDAR → inflated occupancy grid feeding the planner.
- `src/path_planning/path_planning/global_path_node.py` — straight-line reference path from `/goal_pose`; this is the *reference* the elastic band deforms, not a real global planner.

`src/pure_pursuit/` and `src/human_detection/` are out of scope for the current work — pure_pursuit stays read-only, and the human-detection stub doesn't need to be filled in for the local-planner pipeline to run.


## Standalone scripts

`gmm_social_navigation.py` at the repo root is a Matplotlib-only research script (paper reproduction for social-navigation Gaussians) and is unrelated to the ROS build. It runs as `python3 gmm_social_navigation.py`.

## Topic cheatsheet

| Topic | Type | Direction |
|---|---|---|
| `/scan` | LaserScan | gz bridge → human_detection / local_costmap |
| `/odom` | Odometry | gz_pose_odom → planners + controller |
| `/goal_pose` | PoseStamped | RViz / CLI → global_path_node / path_planning |
| `/global_path` | Path | global_path_node → path_planning (TEB) |
| `/local_costmap` | OccupancyGrid | local_costmap_node → path_planning (TEB) |
| `/planned_path` | Path | path_planning → pure_pursuit (or RViz under TEB) |
| `/real_map` | OccupancyGrid | path_planning → pure_pursuit, RViz |
| `/cmd_vel` (TwistStamped) | path_planning(TEB) **or** pure_pursuit → cmd_vel_relay |
| `/cmd_vel_gz` | Twist | cmd_vel_relay → Ignition VelocityControl |

## Prerequisites that aren't obvious from setup.py

- `ign fuel download -u https://fuel.gazebosim.org/1.0/mingfei/models/actor` is required once for the walking actors in `crossing_humans.sdf` to render.
- The Gazebo world depends on `<plugin filename="libignition-gazebo-sensors-system.so">` and the `ogre2` render engine — sensor topics will be silent if either is missing.
