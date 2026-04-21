"""
teb_direct.launch.py
====================
Full simulation stack with TEB local planner driving /cmd_vel directly.
Pure pursuit is NOT launched — path_planning_node is the sole velocity controller.

Adds:
  - local_costmap_node   (LiDAR → occupancy grid)
  - global_path_node     (RViz goal → straight-line reference path)
  - path_planning_node   (TEB optimiser → /cmd_vel)

Does NOT launch:
  - pure_pursuit / regulated_pure_pursuit_node
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_robot  = get_package_share_directory('mecanum_robot_sim')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ── Arguments ─────────────────────────────────────────────────────────
    world_arg   = DeclareLaunchArgument('world',   default_value='crossing_humans')
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='-7.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value=' 4.0')
    spawn_z_arg = DeclareLaunchArgument('spawn_z', default_value='0.0')
    rviz_arg    = DeclareLaunchArgument('rviz',    default_value='true',
                                        description='Launch RViz2')

    # ── Gazebo ────────────────────────────────────────────────────────────
    from launch.substitutions import PythonExpression
    gz_args = PythonExpression(
        ["'-r -v2 ' + '", pkg_robot, "/worlds/' + '",
         LaunchConfiguration('world'), "' + '.sdf'"]
    )
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items(),
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

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

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

    # ── Bridges ───────────────────────────────────────────────────────────
    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )
    cmd_vel_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='cmd_vel_bridge',
        arguments=['/cmd_vel_gz@geometry_msgs/msg/Twist]ignition.msgs.Twist'],
        output='screen',
    )
    lidar_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='lidar_bridge',
        arguments=['/lidar@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan'],
        remappings=[('/lidar', '/scan')],
        output='screen',
    )
    imu_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='imu_bridge',
        arguments=['/imu_raw@sensor_msgs/msg/Imu[ignition.msgs.IMU'],
        remappings=[('/imu_raw', '/imu')],
        output='screen',
    )
    dynamic_pose_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='dynamic_pose_bridge',
        arguments=[
            '/world/crossing_humans/dynamic_pose/info'
            '@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
        ],
        remappings=[('/world/crossing_humans/dynamic_pose/info', '/gz_dynamic_poses')],
        output='screen',
    )

    # ── Motion + Odometry ─────────────────────────────────────────────────
    cmd_vel_relay = Node(
        package='mecanum_robot_sim', executable='cmd_vel_relay',
        name='cmd_vel_relay', output='screen',
        parameters=[{'use_sim_time': True}],
    )
    gz_pose_odom = Node(
        package='mecanum_robot_sim', executable='gz_pose_odom',
        name='gz_pose_odom', output='screen',
        parameters=[{
            'use_sim_time': True,
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
        }],
    )

    # ── Planning stack ────────────────────────────────────────────────────
    local_costmap_node = Node(
        package='path_planning', executable='local_costmap_node',
        name='local_costmap_node', output='screen',
        parameters=[{'use_sim_time': True}],
    )
    global_path_node = Node(
        package='path_planning', executable='global_path_node',
        name='global_path_node', output='screen',
        parameters=[{'use_sim_time': True}],
    )
    path_planning_node = Node(
        package='path_planning', executable='path_planning_node',
        name='path_planning_node', output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # ── RViz ──────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', os.path.join(pkg_robot, 'config', 'robot_viz.rviz')],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        world_arg, spawn_x_arg, spawn_y_arg, spawn_z_arg, rviz_arg,
        gz_sim,
        robot_state_publisher,
        joint_state_publisher,
        TimerAction(period=8.0, actions=[spawn_robot]),
        clock_bridge,
        cmd_vel_bridge,
        lidar_bridge,
        imu_bridge,
        dynamic_pose_bridge,
        cmd_vel_relay,
        gz_pose_odom,
        local_costmap_node,
        global_path_node,
        path_planning_node,
        rviz_node,
    ])
