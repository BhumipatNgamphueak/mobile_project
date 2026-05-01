cmake_minimum_required(VERSION 3.8)
project(teb_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

# Same g2o include workaround used in teb_cpp_pkg
include_directories(/opt/ros/humble/include)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(teb_planner_node
  src/teb_planner_node.cpp
  src/timed_elastic_band.cpp
  src/optimal_planner.cpp
)

target_include_directories(teb_planner_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(teb_planner_node PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(teb_planner_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

target_link_libraries(teb_planner_node
  g2o::core
  g2o::stuff
  g2o::types_slam2d
  g2o::solver_eigen
  glog::glog
)

install(TARGETS teb_planner_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
