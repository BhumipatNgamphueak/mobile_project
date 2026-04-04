"""
Launch the crossing_humans world, spawn the mecanum robot, and open RViz2.

Robot URDF adapted from gz_ros2_control_demos (Apache 2.0):
  https://github.com/ros-controls/gz_ros2_control
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_robot  = get_package_share_directory('mecanum_robot_sim')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ── Arguments ────────────────────────────────────────────────────────
    world_arg   = DeclareLaunchArgument('world',   default_value='crossing_humans')
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='-7.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value=' 4.0')
    spawn_z_arg = DeclareLaunchArgument('spawn_z', default_value='0.0')
    rviz_arg    = DeclareLaunchArgument('rviz',    default_value='true',
                                        description='Launch RViz2')

    # ── Gazebo world ──────────────────────────────────────────────────────
    gz_args = PythonExpression(
        ["'-r -v2 ' + '", pkg_robot, "/worlds/' + '",
         LaunchConfiguration('world'), "' + '.sdf'"]
    )
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': gz_args,
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # ── Robot description ─────────────────────────────────────────────────
    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'), ' ',
            os.path.join(pkg_robot, 'urdf', 'mecanum_robot.urdf.xacro'),
        ]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    # ── Spawn robot ───────────────────────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name',  'mecanum_robot',
            '-x', LaunchConfiguration('spawn_x'),
            '-y', LaunchConfiguration('spawn_y'),
            '-z', LaunchConfiguration('spawn_z'),
        ],
    )

    # ── ros2_control controllers ──────────────────────────────────────────
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )
    mecanum_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'mecanum_drive_controller',
            '--param-file',
            os.path.join(pkg_robot, 'config', 'mecanum_drive_controller.yaml'),
        ],
    )

    # ── Bridges ───────────────────────────────────────────────────────────
    # /clock
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )

    # LiDAR: Ignition /lidar → ROS /scan
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='lidar_bridge',
        arguments=[
            '/lidar@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
        ],
        remappings=[('/lidar', '/scan')],
        output='screen',
    )

    # IMU: Ignition /imu_raw → ROS /imu
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=[
            '/imu_raw@sensor_msgs/msg/Imu[ignition.msgs.IMU',
        ],
        remappings=[('/imu_raw', '/imu')],
        output='screen',
    )

    # Camera image: ros_gz_image bridge (Image is not supported by parameter_bridge)
    camera_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='camera_image_bridge',
        arguments=['/front_camera'],
        output='screen',
    )

    # Camera info: Ignition /front_camera/camera_info → ROS (same topic)
    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_info_bridge',
        arguments=[
            '/front_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        output='screen',
    )

    # ── Human marker publisher ────────────────────────────────────────────
    human_marker_node = Node(
        package='mecanum_robot_sim',
        executable='human_marker_publisher',
        name='human_marker_publisher',
        output='screen',
        parameters=[{
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
            'use_sim_time': True,
        }],
    )

    # ── Skeleton algorithm nodes ──────────────────────────────────────────
    human_detection_node = Node(
        package='human_detection',
        executable='human_detection_node',
        name='human_detection_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    path_planning_node = Node(
        package='path_planning',
        executable='path_planning_node',
        name='path_planning_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    pure_pursuit_node = Node(
        package='pure_pursuit',
        executable='pure_pursuit_node',
        name='pure_pursuit_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_robot, 'config', 'robot_viz.rviz')],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        world_arg, spawn_x_arg, spawn_y_arg, spawn_z_arg, rviz_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        clock_bridge,
        lidar_bridge,
        imu_bridge,
        camera_image_bridge,
        camera_info_bridge,
        human_marker_node,
        human_detection_node,
        path_planning_node,
        pure_pursuit_node,
        rviz_node,
        # controllers chained after spawn
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[mecanum_controller],
        )),
    ])
