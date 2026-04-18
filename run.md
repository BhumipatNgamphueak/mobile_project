

# Run local cost map
ros2 launch path_planning costmap_sim.launch.py

# Run global path
ros2 run path_planning global_path_node

# Test goal -> alternative way is push the 2D Nav Goal in RViz
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {
    stamp: {sec: 0, nanosec: 0}, 
    frame_id: 'odom'
  }, 
  pose: {
    position: {x: 5.0, y: 3.0, z: 0.0}, 
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}"