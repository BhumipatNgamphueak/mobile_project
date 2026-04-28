"""
Launch the crossing_humans world, spawn the mecanum robot, and open RViz2.

Motion model: Ignition VelocityControl (direct body-frame velocity) instead
of gz_ros2_control + sphere-collision wheel physics.  This correctly
simulates mecanum lateral motion.

Odometry: kinematic integration of /cmd_vel (exact because VelocityControl
applies the commanded velocity with no wheel-slip error).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction)
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
    rviz_arg      = DeclareLaunchArgument('rviz',      default_value='true',
                                          description='Launch RViz2')
    test_path_arg = DeclareLaunchArgument('test_path', default_value='false',
                                          description='Publish test path instead of path_planning_node')
    path_arg      = DeclareLaunchArgument('path',      default_value='scurve',
                                          description='Test path name: scurve|straight|uturn|diagonal|slalom|loop')
    evaluate_arg  = DeclareLaunchArgument('evaluate',  default_value='false',
                                          description='Run path evaluator (saves PNG plot)')

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

    # ── Joint state publisher (wheel joints — visual only, not controlled) ──
    # Without gz_ros2_control there is no joint_state_broadcaster.
    # joint_state_publisher reads the URDF and publishes zeros so RViz can
    # show the robot model with wheels in their default position.
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
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

    # ── Bridges ───────────────────────────────────────────────────────────
    # /clock
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )

    # cmd_vel: ROS Twist → Ignition Twist → VelocityControl plugin
    # (cmd_vel_relay strips the TwistStamped header before bridging)
    cmd_vel_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='cmd_vel_bridge',
        arguments=['/cmd_vel_gz@geometry_msgs/msg/Twist]ignition.msgs.Twist'],
        output='screen',
    )

    # LiDAR: Ignition /lidar → ROS /scan
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='lidar_bridge',
        arguments=['/lidar@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan'],
        remappings=[('/lidar', '/scan')],
        output='screen',
    )

    # IMU: Ignition /imu_raw → ROS /imu
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=['/imu_raw@sensor_msgs/msg/Imu[ignition.msgs.IMU'],
        remappings=[('/imu_raw', '/imu')],
        output='screen',
    )

    # Camera image
    camera_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='camera_image_bridge',
        arguments=['/front_camera'],
        remappings=[('/front_camera', '/front_camera/image_raw')],
        output='screen',
    )

    # Dynamic pose: Ignition world poses → /gz_dynamic_poses (TFMessage)
    # Used by gz_pose_odom to get ground-truth robot position.
    dynamic_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='dynamic_pose_bridge',
        arguments=[
            '/world/crossing_humans/dynamic_pose/info'
            '@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
        ],
        remappings=[
            ('/world/crossing_humans/dynamic_pose/info', '/gz_dynamic_poses'),
        ],
        output='screen',
    )

    # Camera info
    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_info_bridge',
        arguments=[
            '/front_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        output='screen',
    )

    # ── Motion + Odometry ─────────────────────────────────────────────────
    # cmd_vel_relay: /cmd_vel (TwistStamped) → /cmd_vel_gz (Twist)
    cmd_vel_relay = Node(
        package='mecanum_robot_sim',
        executable='cmd_vel_relay',
        name='cmd_vel_relay',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # gz_pose_odom: converts Ignition ground-truth pose → /odom + odom→base_link TF
    # Only starts publishing after the robot actually spawns (natural sync).
    gz_pose_odom = Node(
        package='mecanum_robot_sim',
        executable='gz_pose_odom',
        name='gz_pose_odom',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
        }],
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
        executable='regulated_pure_pursuit_node',
        name='Regulated_Pure_Pursuit',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    test_path_node = Node(
        package='pure_pursuit',
        executable='test_path_publisher',
        name='test_path_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'path': LaunchConfiguration('path'),
        }],
        condition=IfCondition(LaunchConfiguration('test_path')),
    )

    path_evaluator_node = Node(
        package='pure_pursuit',
        executable='path_evaluator_node',
        name='path_evaluator',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('evaluate')),
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

    # Delay spawn — gives Gazebo time to load the world before the create
    # node contacts it (avoids repeated retries in the log).
    spawn_robot_delayed = TimerAction(period=8.0, actions=[spawn_robot])

    return LaunchDescription([
        world_arg, spawn_x_arg, spawn_y_arg, spawn_z_arg,
        rviz_arg, test_path_arg, path_arg, evaluate_arg,
        gz_sim,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot_delayed,
        clock_bridge,
        cmd_vel_bridge,
        lidar_bridge,
        imu_bridge,
        camera_image_bridge,
        camera_info_bridge,
        dynamic_pose_bridge,
        cmd_vel_relay,
        gz_pose_odom,
        human_marker_node,
        human_detection_node,
        path_planning_node,
        pure_pursuit_node,
        test_path_node,
        path_evaluator_node,
        rviz_node,
    ])
