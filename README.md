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

### Launch everything

```bash
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py
```

Gazebo and RViz open together. Two humans walk across the world. The robot spawns at world (−7, 4) after ~8 seconds.

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `rviz` | `true` | Open RViz2 |
| `test_path` | `false` | Use a built-in test path instead of `path_planning_node` |
| `path` | `scurve` | Test path to use (see table below) |
| `evaluate` | `false` | Save a tracking-error PNG when goal is reached |
| `spawn_x` | `-7.0` | Robot spawn x in world frame |
| `spawn_y` | `4.0` | Robot spawn y in world frame |

### Test paths (bypass path_planning)

```bash
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py test_path:=true path:=uturn
```

| Name | Tests |
|------|-------|
| `scurve` | Straight + curves + proximity slowdown near obstacle A (default) |
| `straight` | Pure straight line — baseline speed and heading |
| `uturn` | Rectangular U-turn — curvature regulation |
| `diagonal` | 45° diagonal — mecanum lateral correction (`k_lat`) |
| `slalom` | Weave between obstacles B & C — combined curvature + lateral |
| `loop` | Full rectangular circuit — checks heading drift |

### Teleoperation

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p stamped:=true
```

---

## Architecture

```
Ignition Gazebo
  │
  ├─ VelocityControl plugin ◄────── /cmd_vel_gz  (Twist)
  ├─ ground-truth pose ───────────► /gz_dynamic_poses
  ├─ /lidar ──────────────────────► /scan          (LaserScan)
  ├─ /front_camera ───────────────► /front_camera/image_raw
  └─ /imu_raw ────────────────────► /imu

gz_pose_odom ──────────────────────► /odom  +  odom→base_link TF

  ┌──────────────────┐  /detected_human_poses  ┌──────────────────────┐
  │ human_detection  │ ───────────────────────► │   path_planning      │
  │  [IMPLEMENT]     │ ◄── /scan                │   [IMPLEMENT]        │
  │                  │ ◄── /front_camera/...     │                      │
  └──────────────────┘                          │ ◄── /odom            │
                                                │ ◄── /goal_pose       │
                                                │ ──► /planned_path ─┐ │
                                                │ ──► /real_map      │ │
                                                └────────────────────┘ │
                                                                        │
  ┌─────────────────────────────────────────────────────────────────┐  │
  │  pure_pursuit  (Regulated Pure Pursuit — COMPLETE)               │  │
  │   ◄── /planned_path ◄──────────────────────────────────────────┘  │
  │   ◄── /odom                                                        │
  │   ◄── /real_map                                                    │
  │   ──► /cmd_vel  (TwistStamped)                                     │
  └─────────────────────────────────────────────────────────────────┘
                │
         cmd_vel_relay
                │ (strips header, fixes VelocityControl y-axis)
                ▼
          /cmd_vel_gz  ──► Ignition VelocityControl
```

---

## Topic Interface Map

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/scan` | `sensor_msgs/LaserScan` | Gazebo bridge | `human_detection` |
| `/front_camera/image_raw` | `sensor_msgs/Image` | Gazebo bridge | `human_detection` |
| `/imu` | `sensor_msgs/Imu` | Gazebo bridge | *(available)* |
| `/odom` | `nav_msgs/Odometry` | `gz_pose_odom` | `path_planning`, `pure_pursuit` |
| `/goal_pose` | `geometry_msgs/PoseStamped` | operator | `path_planning` |
| `/detected_human_poses` | `geometry_msgs/PoseArray` | `human_detection` | `path_planning` |
| `/detected_humans` | `visualization_msgs/MarkerArray` | `human_detection` | RViz |
| `/planned_path` | `nav_msgs/Path` | `path_planning` | `pure_pursuit` |
| `/real_map` | `nav_msgs/OccupancyGrid` | `path_planning` | `pure_pursuit`, RViz |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | `pure_pursuit` | `cmd_vel_relay` |

---

## World Geometry (odom frame)

Robot spawns at world (−7, 4) which becomes odom origin (0, 0).
Conversion: `odom_x = world_x + 7`,  `odom_y = world_y − 4`

```
  odom y
    6 ─── north wall ────────────────────────────────────
          │                                              │
          │        obstacle_A (11, 1)  2×2 m            │
          │              ■                               │
    0  spawn(0,0)                                        │
          │                                              │
          │                     obstacle_C (9, −8) ■    │
          │  obstacle_B (3, −9) ■                        │
  −14 ─── south wall ───────────────────────────────────
       −3                                              17  → odom x
```

Obstacles are 2×2 m squares. With `robot_radius = 0.35 m` inflation, the safe clearance border is **1.35 m** from each obstacle centre edge.

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
    │   │   └── crossing_humans.sdf        Gazebo world with walking humans
    │   ├── config/
    │   │   └── robot_viz.rviz             RViz config
    │   ├── launch/
    │   │   └── spawn_mecanum.launch.py    Main launch file
    │   └── scripts/
    │       ├── gz_pose_odom.py            Gazebo ground-truth → /odom + TF
    │       ├── cmd_vel_relay.py           TwistStamped→Twist, fixes y-axis flip
    │       └── human_marker_publisher.py  Analytic human markers for RViz
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
> Normal — the robot spawns in Gazebo after ~8 s. RViz will show it as soon as `gz_pose_odom` receives the first ground-truth pose from Gazebo.

**Robot moves wrong direction laterally**
> `cmd_vel_relay.py` negates `linear.y` to correct VelocityControl's y-axis flip. If the robot strafes backwards, this is already handled.

**Human poses not visible in RViz**
> Ignition actors are not in `dynamic_pose/info`. `human_marker_publisher` computes positions analytically from `/clock` + SDF waypoints. Check the `/human_markers` topic.

**Sensor topics empty**
> The `<plugin filename="libignition-gazebo-sensors-system.so">` block must be present in `crossing_humans.sdf`. Requires the `ogre2` render engine.

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
