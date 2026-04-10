cmake_minimum_required(VERSION 3.8)
project(mecanum_robot_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/human_marker_publisher.py
  DESTINATION lib/${PROJECT_NAME} RENAME human_marker_publisher)
install(PROGRAMS scripts/odom_tf_broadcaster.py
  DESTINATION lib/${PROJECT_NAME} RENAME odom_tf_broadcaster)
install(PROGRAMS scripts/cmd_vel_relay.py
  DESTINATION lib/${PROJECT_NAME} RENAME cmd_vel_relay)
install(PROGRAMS scripts/kinematic_odom.py
  DESTINATION lib/${PROJECT_NAME} RENAME kinematic_odom)
install(PROGRAMS scripts/gz_pose_odom.py
  DESTINATION lib/${PROJECT_NAME} RENAME gz_pose_odom)

ament_package()
