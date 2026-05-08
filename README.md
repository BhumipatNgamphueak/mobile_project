# Mecanum Robot Simulation — ROS 2 Humble + Ignition Gazebo Fortress

A research simulation platform for developing and testing **human detection**, **path planning**, and **path following** algorithms on a mecanum-wheeled mobile robot.

Three skeleton packages ship with full ROS 2 infrastructure wired up. Each developer only needs to fill in their algorithm inside the clearly marked `TODO` stubs — everything else (simulation, bridges, odometry, visualisation) is already running.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Ubuntu | 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Gazebo | Ignition Fortress (gz-sim 6) |
| Python | 3.10+ |
| colcon | latest |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BhumipatNgamphueak/mobile_project.git ~/mobile_project
```

### 2. Install ROS dependencies

```bash
sudo apt update && sudo apt install -y \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-image \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-xacro \
  ros-humble-rviz2 \
  ros-humble-teleop-twist-keyboard \
  ros-humble-nav-msgs \
  ros-humble-sensor-msgs \
  ros-humble-visualization-msgs \
  ros-humble-tf2-ros
```

### 3. Download the walking actor mesh

```bash
ign fuel download -u https://fuel.gazebosim.org/1.0/mingfei/models/actor
```

### 4. Build

```bash
cd ~/mobile_project
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### 5. Add to `.bashrc`

```bash
echo "source ~/mobile_project/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Running the Simulation

### Launch everything (default world)

```bash
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py
```

Gazebo and RViz open together. The robot spawns at world (−8, 0) after ~8 seconds. Humans (if present in the world) start walking after a 1 s delay.

### Choose a world

Nine worlds ship out of the box — pass the `world` argument to switch:

```bash
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py world:=world3_cross_opposite
```

| World | Walls | Obstacles | Humans | Notes |
|---|---|---|---|---|
| `crossing_humans` *(default)* | 4 | 3 | 2 | Two humans cross at centre, three static boxes |
| `world1_static_large` | 4 | 1 | 0 | Single 6×6 m block in the centre |
| `world2_static_two` | 4 | 2 | 0 | Two long walls forming a chicane |
| `world3_cross_opposite` | 4 | 0 | 2 | Two humans walking perpendicular, opposite directions |
| `world4_cross_same` | 4 | 0 | 2 | Two humans walking perpendicular, same direction |
| `world5_human_oncoming` | 4 | 0 | 2 | One oncoming human + one perpendicular |
| `world6_human_ahead` | 4 | 0 | 1 | Single human directly in robot's path |
| `world7_crowd_vertical` | 4 | 0 | 5 | Vertical crowd of five humans |
| `world8_humans_crossing` | 4 | 0 | 2 | Two humans crossing |

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `world` | `crossing_humans` | World SDF stem under `worlds/` |
| `rviz` | `true` | Open RViz2 |
| `test_path` | `false` | Use a built-in test path instead of `path_planning_node` |
| `path` | `scurve` | Test path to use (see table below) |
| `evaluate` | `false` | Save a tracking-error PNG when goal is reached |
| `spawn_x` | `-8.0` | Robot spawn x in world frame |
| `spawn_y` | `0.0` | Robot spawn y in world frame |
| `time_scale` | `1.0` | Scale factor applied to the *kinematic-model controller* only — useful for debugging marker drift; for normal use, edit waypoint speeds via `scale_human_speed` instead |

### Test paths (bypass path_planning)

```bash
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py test_path:=true path:=uturn
```

| Name | Tests |
|------|-------|
| `scurve` | Straight + curves + proximity slowdown (default) |
| `straight` | Pure straight line — baseline speed and heading |
| `uturn` | Rectangular U-turn — curvature regulation |
| `diagonal` | 45° diagonal — mecanum lateral correction (`k_lat`) |
| `slalom` | Weave between obstacles — combined curvature + lateral |
| `loop` | Full rectangular circuit — checks heading drift |

### Teleoperation

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p stamped:=true
```

---

## Adjusting Human Walking Speed

Each human follows a list of `<waypoint><time>` entries in the world SDF. Both the visible animated `<actor>` and the invisible kinematic `<model>` (which drives `/gz_dynamic_poses`) read the same waypoint table, so a single edit changes the speed everywhere.

The `scale_human_speed` tool inspects and rewrites those times in place:

```bash
# Show current pace (m/s) for every human in every world
ros2 run mecanum_robot_sim scale_human_speed --show --world ALL

# Set a specific human to an exact speed in m/s
ros2 run mecanum_robot_sim scale_human_speed \
    --world crossing_humans --human human_1 --target-mps 0.8

# Multiplier instead of absolute speed (×1.5 = 50% faster)
ros2 run mecanum_robot_sim scale_human_speed \
    --world world7_crowd_vertical --human human_3 --speed 1.5

# Slow every human in every world to 80%
ros2 run mecanum_robot_sim scale_human_speed --world ALL --speed 0.8
```

Re-launch Gazebo after editing speeds. Round-trip is safe: scaling by 0.5 then 2.0 returns to the original timing. Per-human edits don't touch siblings.

---

## Architecture

```
                ┌────────────────────────── Ignition Gazebo ──────────────────────────┐
                │                                                                     │
  /cmd_vel_gz ─►│ VelocityControl ─► robot motion                                     │
                │                                                                     │
                │ kinematic <model name="human_*">  ◄── /world/<w>/set_pose service   │
                │       └─ pose published to       ─► dynamic_pose/info               │
                │                                                                     │
                │ visible <actor name="human_*_anim"> (mesh + walk cycle)             │
                │                                                                     │
                │ /lidar  ─►   /front_camera/image_raw  ─►   /imu_raw  ─►             │
                └─────────────────────────────────────────────────────────────────────┘
                       ▲                          │                       │
                       │                          ▼                       ▼
            human_controller             /gz_dynamic_poses             /scan etc.
            (set_pose @ 20 Hz)                  │
                       │                        ├──► gz_pose_odom ─► /odom + TF
                       │                        │
                       │                        └──► human_marker_publisher
                       │                                    │
                       │                                    ▼
                       │                       /visualization/world (RViz markers)
                       │
                       └─ reads SDF waypoints + sim time, drives every kinematic human

  ┌──────────────────┐  /detected_human_poses  ┌──────────────────────┐
  │ human_detection  │ ───────────────────────► │   path_planning      │
  │  [IMPLEMENT]     │ ◄── /scan                │   [IMPLEMENT]        │
  │                  │ ◄── /front_camera/…       │   ◄── /odom         │
  └──────────────────┘                          │   ◄── /goal_pose    │
                                                │   ──► /planned_path │
                                                └──────┬──────────────┘
                                                       │
  ┌─────────────────────────────────────────────────────▼─────────────┐
  │  pure_pursuit  (Regulated Pure Pursuit — COMPLETE)                │
  │   ◄── /planned_path                                                │
  │   ◄── /odom                                                        │
  │   ──► /cmd_vel  (TwistStamped)                                     │
  └─────────────────────────────────────────────────────────────────┘
                │
         cmd_vel_relay
                │ (strips header, fixes VelocityControl y-axis)
                ▼
          /cmd_vel_gz  ──► Ignition VelocityControl
```

**Why kinematic models for humans:** Ignition Fortress does *not* publish `<actor>` poses to any topic, so actors alone can't be used as ground truth. Each human is split in two: a `<static>true</static>` `<model>` (no visual, just collision + a `<plugin filename="__waypoints__">` block carrying the trajectory) gets teleported by `human_controller` via `/world/<w>/set_pose`, which makes its pose appear in `/gz_dynamic_poses`. A sibling `<actor>` named `<name>_anim` shares the same waypoints with `<interpolate_x>false</interpolate_x>`, so Gazebo plays the walk animation at exactly the same pace the controller drives the model — visible mesh and ground-truth pose stay in sync by construction.

---

## Topic Interface Map

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/scan` | `sensor_msgs/LaserScan` | Gazebo bridge | `human_detection` |
| `/front_camera/image_raw` | `sensor_msgs/Image` | Gazebo bridge | `human_detection` |
| `/imu` | `sensor_msgs/Imu` | Gazebo bridge | *(available)* |
| `/odom` | `nav_msgs/Odometry` | `gz_pose_odom` | `path_planning`, `pure_pursuit` |
| `/gz_dynamic_poses` | `tf2_msgs/TFMessage` | Gazebo bridge | `gz_pose_odom`, `human_marker_publisher` |
| `/world/<w>/set_pose` | `ros_gz_interfaces/srv/SetEntityPose` | bridge → Gazebo | `human_controller` |
| `/goal_pose` | `geometry_msgs/PoseStamped` | operator | `path_planning` |
| `/detected_human_poses` | `geometry_msgs/PoseArray` | `human_detection` | `path_planning` |
| `/detected_humans` | `visualization_msgs/MarkerArray` | `human_detection` | RViz |
| `/visualization/world` | `visualization_msgs/MarkerArray` | `human_marker_publisher` | RViz |
| `/visualization/human_paths` | `visualization_msgs/MarkerArray` | `human_marker_publisher` | RViz |
| `/visualization/robot_trail` | `nav_msgs/Path` | `human_marker_publisher` | RViz |
| `/planned_path` | `nav_msgs/Path` | `path_planning` | `pure_pursuit` |
| `/real_map` | `nav_msgs/OccupancyGrid` | `path_planning` | `pure_pursuit`, RViz |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | `pure_pursuit` | `cmd_vel_relay` |

---

## World Geometry

Every world is a 20 × 20 m arena with 0.2 m thick walls along x = ±10 and y = ±10.
The robot spawns at world (−8, 0) by default, which becomes the **odom-frame origin (0, 0)** seen by ROS:

```
odom_x = world_x − spawn_x   (= world_x + 8 by default)
odom_y = world_y − spawn_y   (= world_y − 0 by default)
```

Reference layout for the **default world** `crossing_humans`:

```
  world y
   +10 ─── wall_north ────────────────────────────────────
          │                                               │
          │                                               │
          │      obstacle_A (4, 5)  2×2×2 m  ■           │
          │                                               │
          │      human_1 → walks east  along y = +1       │
    0     spawn (-8, 0)            ─────────►             │
          │      human_2 ← walks west  along y = -1       │
          │                                               │
          │                       obstacle_C (2, -4) ■   │
          │  obstacle_B (-4, -5) ■                        │
   -10 ─── wall_south ────────────────────────────────────
       -10                                               +10  → world x
```

Each human's collision is a 0.35 m radius × 1.7 m tall cylinder, so the lidar at z ≈ 0.32 m sees them and the robot is physically blocked from passing through. Worlds 3–8 share the same wall + spawn layout but vary the obstacles and human trajectories — see the *Choose a world* table above.

---

## Developer Guide

### Package ownership

| Package | Owner | Status |
|---------|-------|--------|
| `mecanum_robot_sim` | Sim infra | Complete |
| `pure_pursuit` | Path following | **Complete — do not modify** |
| `path_planning` | Path planning team | **Implement here** |
| `human_detection` | Perception team | **Implement here** |

---

### Implementing `human_detection`

**File:** [src/human_detection/human_detection/human_detection_node.py](src/human_detection/human_detection/human_detection_node.py)

Find `_stub_detect()` and replace it with your algorithm. Return a list of `(x, y)` positions in the **odom frame**. The node handles all publishing automatically.

```python
def _stub_detect(self) -> list[tuple[float, float]]:
    # TODO: implement your detection logic
    # Return [(x1, y1), (x2, y2), ...]  in odom frame
    return []
```

**Inputs available:**

| Attribute | Type | Updated |
|-----------|------|---------|
| `self._latest_scan` | `LaserScan` | 10 Hz |
| `self._latest_image` | `Image` | 30 Hz |

**Helper:** `_laser_to_odom(range, angle_rad)` — currently a stub. Implement it properly using `tf2_ros.Buffer` / `TransformListener` to transform from `lidar_link` → `odom`.

**Test without Gazebo:**

```bash
# Fake a scan
ros2 topic pub /scan sensor_msgs/LaserScan '{header: {frame_id: lidar_link}}'

ros2 run human_detection human_detection_node

# Check output
ros2 topic echo /detected_human_poses
```

---

### Implementing `path_planning`

**File:** [src/path_planning/path_planning/path_planning_node.py](src/path_planning/path_planning/path_planning_node.py)

Find `_stub_plan()` and replace it with your algorithm. Return a `nav_msgs/Path` in the **odom frame**.

```python
def _stub_plan(self) -> Path | None:
    # TODO: A*, RRT*, TEB, etc.
    # Use self._robot_pose, self._goal_pose, self._human_poses
    # Return nav_msgs/Path with header.frame_id = 'odom'
    ...
```

**Inputs available:**

| Attribute | Type | Source |
|-----------|------|--------|
| `self._robot_pose` | `geometry_msgs/Pose` | from `/odom` |
| `self._goal_pose` | `geometry_msgs/Pose` | from `/goal_pose` |
| `self._human_poses` | `list[Pose]` | from `/detected_human_poses` |
| `self._static_obstacles` | `list[(cx, cy, half_size)]` | hardcoded from SDF |

**Static map** is already built and published on `/real_map` (200×200 cells, 0.1 m resolution, origin at odom (−3, −14)).
Use it as the grid for A* or similar — static obstacles and walls are already marked.
Inflate human positions by `self.human_radius` (default 0.6 m) before planning.

**Path contract with pure_pursuit:**

- `path.header.frame_id = 'odom'`
- Waypoint spacing ≤ 0.05 m (sparse points cause jumpy tracking)
- Re-publish at ~1–2 Hz — pure_pursuit always needs a fresh copy
- All points must be outside inflated obstacle zones

**Send a goal pose:**

```bash
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: odom}, pose: {position: {x: 8.0, y: 2.0}, orientation: {w: 1.0}}}'
```

**Test without Gazebo:**

```bash
# Terminal 1 — fake odometry
ros2 topic pub /odom nav_msgs/Odometry \
  '{pose: {pose: {position: {x: 0.0, y: 0.0}, orientation: {w: 1.0}}}}'

# Terminal 2
ros2 run path_planning path_planning_node

# Terminal 3 — send a goal
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: odom}, pose: {position: {x: 8.0, y: 2.0}}}'

ros2 topic echo /planned_path
```

---

### Pure Pursuit (complete — read-only)

Implements **Regulated Pure Pursuit (RPP)** adapted for mecanum kinematics.

**Subscriptions:**

| Topic | Required |
|-------|----------|
| `/planned_path` | Yes |
| `/odom` | Yes |
| `/real_map` | No (proximity slowdown only) |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_max` | 0.8 m/s | Maximum forward speed |
| `v_min` | 0.05 m/s | Minimum speed |
| `lt` | 1.0 s | Lookahead time gain |
| `lt_min` | 0.25 m | Minimum lookahead distance |
| `lt_max` | 1.2 m | Maximum lookahead distance |
| `omega_max` | 3.2 rad/s | Maximum angular speed |
| `r_min` | 0.9 m | Min turn radius for curvature heuristic |
| `d_prox` | 0.7 m | Proximity slowdown trigger distance |
| `alpha` | 0.7 | Proximity gain (≤ 1.0) |
| `k_lat` | 0.5 | Lateral correction gain (mecanum strafe) |
| `v_y_max` | 0.3 m/s | Max lateral speed |
| `goal_tolerance` | 0.15 m | Stop when within this distance |
| `t_collision` | 1.5 s | Predictive collision horizon |
| `control_hz` | 50.0 Hz | Control loop rate |

---

## Project Structure

```
mobile_project/
├── README.md
└── src/
    ├── mecanum_robot_sim/          Simulation infrastructure
    │   ├── urdf/
    │   │   └── mecanum_robot.urdf.xacro   Robot + sensors + VelocityControl plugin
    │   ├── worlds/
    │   │   ├── crossing_humans.sdf        Default world (3 obstacles, 2 humans)
    │   │   ├── world1_static_large.sdf    1 large obstacle, no humans
    │   │   ├── world2_static_two.sdf      2 walls forming a chicane, no humans
    │   │   ├── world3_cross_opposite.sdf  2 humans crossing opposite
    │   │   ├── world4_cross_same.sdf      2 humans crossing same direction
    │   │   ├── world5_human_oncoming.sdf  oncoming + perpendicular
    │   │   ├── world6_human_ahead.sdf     1 human directly in path
    │   │   ├── world7_crowd_vertical.sdf  crowd of 5 humans
    │   │   └── world8_humans_crossing.sdf 2 humans crossing
    │   ├── config/
    │   │   └── robot_viz.rviz             RViz config
    │   ├── launch/
    │   │   └── spawn_mecanum.launch.py    Main launch file (also bridges set_pose)
    │   └── scripts/
    │       ├── gz_pose_odom.py            Gazebo ground-truth → /odom + TF
    │       ├── cmd_vel_relay.py           TwistStamped→Twist, fixes y-axis flip
    │       ├── human_controller.py        Drives every kinematic human via set_pose
    │       ├── human_marker_publisher.py  RViz markers for walls/obstacles/humans
    │       └── scale_human_speed.py       CLI: inspect & rescale human walking speed
    │
    ├── human_detection/            ← IMPLEMENT HERE
    │   └── human_detection/
    │       └── human_detection_node.py
    │
    ├── path_planning/              ← IMPLEMENT HERE
    │   └── path_planning/
    │       └── path_planning_node.py
    │
    └── pure_pursuit/               Complete — do not modify
        └── pure_pursuit/
            ├── pure_pursuit_node.py        RPP controller
            ├── test_path_publisher.py      Built-in test paths (bypass path_planning)
            └── path_evaluator.py           Records tracking error, saves PNG plot
```

---

## Troubleshooting

**Robot does not appear in RViz immediately**
> Normal — the robot spawns in Gazebo after ~8 s. RViz shows it as soon as `gz_pose_odom` receives the first ground-truth pose from Gazebo.

**Robot moves wrong direction laterally**
> `cmd_vel_relay.py` negates `linear.y` to correct VelocityControl's y-axis flip. If the robot strafes backwards, this is already handled.

**Humans not appearing in RViz / not moving**
> Check `ros2 topic echo /gz_dynamic_poses --once | grep child_frame_id | grep human` — you should see `human_1`, `human_2`, … If absent, `human_controller` isn't successfully calling `set_pose`. Verify the `set_pose_bridge` node is running: `ros2 node list | grep set_pose`. The bridge needs Gazebo's world to be fully loaded before it connects.

**Visible actor and RViz marker drift apart**
> Both must read the same waypoint timing. Check that the `<actor>` has `<interpolate_x>false</interpolate_x>` (waypoint-time pacing, not animation-driven) and that you haven't set `time_scale` to anything other than `1.0`. To change human speed cleanly, use `ros2 run mecanum_robot_sim scale_human_speed` instead of `time_scale`.

**Robot drives through humans**
> Each kinematic human has a 0.35 m × 1.7 m collision cylinder, but if your `path_planning_node` ignores obstacles in `/scan` it will plan straight through. Confirm with `ros2 topic echo /scan` that points show up at human positions.

**Sensor topics empty**
> The `<plugin filename="libignition-gazebo-sensors-system.so">` block must be present in the world SDF. Requires the `ogre2` render engine.

**Build errors after editing**
```bash
colcon build --packages-select <package_name>
source install/setup.bash
```

**Full clean build**
```bash
rm -rf build install log
colcon build
source install/setup.bash
```

---

## License

Apache 2.0

The mecanum robot URDF is adapted from [gz_ros2_control](https://github.com/ros-controls/gz_ros2_control) by ros-controls contributors (Apache 2.0).
Human actor mesh: [mingfei/actor](https://app.gazebosim.org/mingfei/fuels/models/actor) on Ignition Fuel.
