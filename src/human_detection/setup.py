from setuptools import setup

package_name = 'human_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Skeleton human detection node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'human_detection_node = human_detection.human_detection_node:main',
            'social_costmap_node = human_detection.social_costmap_node:main',
        ],
    },
)
