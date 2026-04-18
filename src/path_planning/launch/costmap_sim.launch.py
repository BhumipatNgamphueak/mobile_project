import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 1. Include the spawn_mecanum.launch.py from mecanum_robot_sim package
    spawn_mecanum_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('mecanum_robot_sim'),
            '/launch/',
            'spawn_mecanum.launch.py'
        ])
    )

    # 2. Add the local_costmap_node from path_planning package
    local_costmap_node = Node(
        package='path_planning',
        executable='local_costmap_node',
        name='local_costmap_node',
        output='screen'
    )

    return LaunchDescription([
        spawn_mecanum_launch,
        local_costmap_node
    ])
