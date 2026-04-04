cmake_minimum_required(VERSION 3.8)
project(mecanum_robot_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/human_marker_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME human_marker_publisher
)

ament_package()
