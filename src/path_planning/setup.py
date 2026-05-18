from setuptools import setup

package_name = 'path_planning'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/costmap_sim.launch.py',
            'launch/teb_direct.launch.py',
            'launch/teb_human_detection.launch.py'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Skeleton path planning node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'path_planning_node = path_planning.path_planning_node:main',
            'local_costmap_node = path_planning.local_costmap_node:main',
            'global_path_node = path_planning.global_path_node:main',
            'test_cmd_vel_node = path_planning.test_cmd_vel_node:main',
        ],
    },
)
