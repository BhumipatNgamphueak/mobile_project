# Mecanum Robot Simulation — ROS 2 Humble + Gazebo Ignition Fortress

A research simulation platform for developing and testing **human detection**, **path planning**, and **pure pursuit control** algorithms on a mecanum-wheeled mobile robot.

The project ships three skeleton packages with full ROS 2 infrastructure wired up — developers only need to fill in their algorithm inside the clearly marked `TODO` stubs.

## Prerequisites

- **ROS 2 Humble Hawksbill**
- **Ubuntu 22.04 LTS**
- **Python 3.10+**
- **Gazebo Ignition Fortress (gz-sim 6)**
- **colcon** build tool

If you don't have ROS 2 installed, follow the official guide: https://docs.ros.org/en/humble/Installation.html

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BhumipatNgamphueak/mobile_project.git ~/mobile_project
```

### 2. Install Dependencies
```bash
sudo apt update && sudo apt install -y \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-image \
  ros-humble-gz-ros2-control \
  ros-humble-mecanum-drive-controller \
  ros-humble-joint-state-broadcaster \
  ros-humble-controller-manager \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  ros-humble-rviz2 \
  ros-humble-teleop-twist-keyboard \
  ros-humble-nav-msgs \
  ros-humble-sensor-msgs \
  ros-humble-visualization-msgs
```

Download the actor mesh (human walking animation) used in the Gazebo world:

```bash
mkdir -p ~/.ignition/fuel/fuel.gazebosim.org/mingfei/models/actor/1/meshes
ign fuel download -u https://fuel.gazebosim.org/1.0/mingfei/models/actor
```

### 3. Build the Project
```bash
cd ~/mobile_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### 4. Source the Environment
```bash
source ~/mobile_project/install/setup.bash
```

### 5. Add to bashrc for Automatic Sourcing
```bash
echo "source ~/mobile_project/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Running the Project

You need **TWO terminals** to run this project.

### Terminal 1: Launch Simulation
```bash
cd ~/mobile_project
source install/setup.bash
ros2 launch mecanum_robot_sim spawn_mecanum.launch.py
```

This opens Gazebo + RViz together. Two humans walk across the world, the mecanum robot spawns at the origin.

### Terminal 2: Teleoperation (optional)
```bash
cd ~/mobile_project
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p stamped:=true -r /cmd_vel:=/cmd_vel
```

> **Important**: the `-p stamped:=true` flag is required.
> The `mecanum_drive_controller` expects `geometry_msgs/TwistStamped`, not plain `Twist`.

### Launch Options

| Option | Command |
|--------|---------|
| Without RViz | `ros2 launch mecanum_robot_sim spawn_mecanum.launch.py rviz:=false` |
| Custom spawn position | `ros2 launch mecanum_robot_sim spawn_mecanum.launch.py spawn_x:=-5.0 spawn_y:=2.0` |

## Sending a Goal Pose

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: odom}, pose: {position: {x: 5.0, y: -2.0}, orientation: {w: 1.0}}}'
```

The `path_planning_node` will receive the goal and publish a path on `/planned_path`.
The `pure_pursuit_node` will follow it and publish velocity commands on `/cmd_vel`.

## Implementing the Algorithms

All three skeleton nodes are in `src/<package>/<package>/<package>_node.py`.
Each file has a `_stub_*()` method to replace and helper utilities already written.

### 1. `human_detection` — [src/human_detection/human_detection/human_detection_node.py](src/human_detection/human_detection/human_detection_node.py)

**Where to implement**: `_run_detection()` and `_stub_detect()`

```
Subscribes:
  /scan                     sensor_msgs/LaserScan    LiDAR (10 Hz)
  /front_camera/image_raw   sensor_msgs/Image        Camera (30 Hz)

Publishes:
  /detected_humans          visualization_msgs/MarkerArray
  /detected_human_poses     geometry_msgs/PoseArray
```

**Suggested approaches:**

| Method | Description |
|--------|-------------|
| LiDAR leg detector | Cluster scan returns into ~0.3 m diameter blobs; pairs ≈ 0.5 m apart are legs |
| YOLOv8 (camera) | Run inference on `/front_camera/image_raw`; estimate depth from bounding box size |
| Fusion | Match LiDAR clusters to YOLO bounding boxes for robust 3D pose estimation |

**Key note**: `_laser_to_odom()` is a stub — implement it with `tf2_ros.Buffer` to properly transform laser scan hits from `lidar_link` frame to `odom` frame.

### 2. `path_planning` — [src/path_planning/path_planning/path_planning_node.py](src/path_planning/path_planning/path_planning_node.py)

**Where to implement**: `_replan()` and `_stub_plan()`

```
Subscribes:
  /odom                     nav_msgs/Odometry
  /goal_pose                geometry_msgs/PoseStamped
  /detected_human_poses     geometry_msgs/PoseArray

Publishes:
  /planned_path             nav_msgs/Path              (replanned at 2 Hz)
  /real_map                 nav_msgs/OccupancyGrid     (latched, published once)
```

**Suggested approaches:**

| Algorithm | Notes |
|-----------|-------|
| A\* | Discretise the `/real_map` grid; inflate static + dynamic obstacles; search |
| RRT / RRT\* | Sample-based; good for cluttered environments |
| DWA | Dynamic Window Approach; handles moving obstacles natively |
| TEB | Timed Elastic Band; good for narrow passages |

**Map details:**
- Resolution: 0.1 m/cell, 200×200 cells covering ±10 m world
- Origin in odom: `(-3, -14)` — accounts for robot spawn offset `(-7, 4)` from world centre
- Static obstacles (A, B, C) pre-inflated by `robot_radius` (default 0.35 m)
- Human positions are received as `PoseArray` — inflate by `human_radius` (default 0.6 m) before planning

### 3. `pure_pursuit` — [src/pure_pursuit/pure_pursuit/pure_pursuit_node.py](src/pure_pursuit/pure_pursuit/pure_pursuit_node.py)

**Where to implement**: `_control_loop()` and `_stub_control()`

```
Subscribes:
  /planned_path   nav_msgs/Path
  /odom           nav_msgs/Odometry

Publishes:
  /cmd_vel        geometry_msgs/TwistStamped   (20 Hz)
```

**Classic Pure Pursuit algorithm:**
1. Find the lookahead point: first path pose ≥ `lookahead_dist` metres from robot
2. Transform it to robot body frame using `_world_to_robot()`
3. Compute curvature `κ = 2·y_local / L²` where `L = lookahead_dist`
4. Set `angular_vel = max_linear_vel × κ`
5. Clamp and publish via `_make_twist(vx, vy, wz)`

**Mecanum extension:** Set `linear.y` proportional to the cross-track error so the robot slides sideways to correct path deviation without excessive yawing.

Helper methods already implemented:
- `_find_lookahead_point()` — walks path array, returns (x, y) in odom frame
- `_world_to_robot(wx, wy)` — odom → robot body frame rotation
- `_make_twist(vx, vy, wz)` — clamps velocities, builds TwistStamped
- `_yaw_from_quaternion(q)` — extracts yaw from geometry_msgs Quaternion

> **Important**: `/cmd_vel` must be `geometry_msgs/TwistStamped`.
> Publishing plain `Twist` will be silently ignored by `mecanum_drive_controller`.

## Configuration Parameters

### Available Parameters by Node

#### 1. Human Detection Node (`/human_detection_node`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cluster_tolerance` | 0.3 | LiDAR cluster diameter threshold (m) |
| `leg_pair_dist` | 0.5 | Expected distance between leg clusters (m) |
| `human_radius` | 0.3 | Radius of detected human marker (m) |

#### 2. Path Planning Node (`/path_planning_node`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replan_rate` | 2.0 | Replanning frequency (Hz) |
| `robot_radius` | 0.35 | Inflation radius for static obstacles (m) |
| `human_radius` | 0.6 | Inflation radius for dynamic obstacles (m) |
| `map_resolution` | 0.1 | Grid cell size (m/cell) |

#### 3. Pure Pursuit Node (`/pure_pursuit_node`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookahead_dist` | 1.0 | Lookahead distance (m) |
| `max_linear_vel` | 1.0 | Maximum linear velocity (m/s) |
| `max_angular_vel` | 1.5 | Maximum angular velocity (rad/s) |
| `control_rate` | 20.0 | Control loop frequency (Hz) |

**Example usage:**
```bash
# Get all node names
ros2 node list

# List parameters for a node
ros2 param list /pure_pursuit_node

# Get a parameter value
ros2 param get /pure_pursuit_node lookahead_dist

# Set a parameter value
ros2 param set /pure_pursuit_node lookahead_dist 1.5
```

## Project Structure

```
src/
├── mecanum_robot_sim/      Simulation environment (ament_cmake)
│   ├── urdf/               Robot URDF/xacro files
│   ├── worlds/             Gazebo SDF world files
│   ├── rviz/               RViz configuration
│   └── launch/             Launch files
├── human_detection/        [SKELETON] Perception node (ament_python)
├── path_planning/          [SKELETON] Planning node  (ament_python)
└── pure_pursuit/           [SKELETON] Control node   (ament_python)
```

### Data Flow

```
Gazebo Ignition
  ├── /scan           LiDAR 360°, 12 m range, 10 Hz
  ├── /imu            Accelerometer + gyro, 100 Hz
  ├── /front_camera   RGB 640×480, 30 Hz
  └── /odom           Odometry from mecanum_drive_controller

/scan + /front_camera  →  [human_detection_node]  →  /detected_human_poses
/odom + /goal_pose + /detected_human_poses  →  [path_planning_node]  →  /planned_path
/planned_path + /odom  →  [pure_pursuit_node]  →  /cmd_vel (TwistStamped)
/cmd_vel  →  mecanum_drive_controller  →  Gazebo wheel velocities
```

## Troubleshooting

**Robot does not move with teleop**
> Use `-p stamped:=true` — the controller requires `TwistStamped`, not `Twist`.

**Humans blink or disappear in Gazebo**
> Ensure the world physics `max_step_size` is `0.01` (not `0.001`).
> The `Sensors` plugin must be present in the SDF world plugins block.

**Sensor topics are empty**
> The `<plugin filename="libignition-gazebo-sensors-system.so">` block must be in `crossing_humans.sdf`. Check that `ogre2` render engine is available.

**`robot_description` YAML parse error**
> The xacro command must be wrapped: `ParameterValue(Command([...]), value_type=str)`.

**Human poses not appearing in RViz**
> Ignition actors are NOT published via `pose/info` or `dynamic_pose/info`.
> `human_marker_publisher` computes positions analytically from `/clock` + SDF waypoints.

**`mecanum_drive_controller` fails to spawn**
> It must be spawned *after* `joint_state_broadcaster`. The launch file uses chained `RegisterEventHandler(OnProcessExit(...))` for this.

**Build errors after changes**
```bash
rm -rf build install log
colcon build --symlink-install
```

## Citation

The mecanum robot URDF kinematics is adapted from:

> **gz_ros2_control** by ros-controls contributors
> https://github.com/ros-controls/gz_ros2_control
> License: Apache 2.0

Human actor mesh: **mingfei/actor** on Ignition Fuel
https://app.gazebosim.org/mingfei/fuels/models/actor

## License

Apache 2.0
