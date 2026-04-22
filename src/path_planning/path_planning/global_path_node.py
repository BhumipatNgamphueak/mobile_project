import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry

class GlobalPathNode(Node):
    def __init__(self):
        super().__init__('global_path_node')

        # Subscribe to RViz's "2D Goal Pose" tool
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            10
        )

        # Timer callback
        self.path_dt    = 1
        self.global_path_timer = self.create_timer(self.path_dt, self.global_path_timer_callback)

        # State
        self.goal_msg  = None
        self._robot_x  = 0.0
        self._robot_y  = 0.0

        # Publisher for the global path
        self.path_pub = self.create_publisher(Path, '/global_path', 10)
        self.get_logger().info("Global Path Node initialized. Waiting for /goal_pose from RViz...")

    def _odom_callback(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def goal_callback(self, msg: PoseStamped):
        # self.get_logger().info(f"Received new goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        # Adopt the frame_id from the goal pose (usually 'map' or 'odom')
        path_msg.header.frame_id = msg.header.frame_id
        # Straight line path from current robot position to the goal.
        num_points = 50
        start_x = self._robot_x
        start_y = self._robot_y

        dx = msg.pose.position.x - start_x
        dy = msg.pose.position.y - start_y

        for i in range(num_points + 1):
            pose = PoseStamped()
            pose.header.frame_id = path_msg.header.frame_id
            pose.header.stamp = path_msg.header.stamp

            # Interpolate position
            ratio = i / num_points
            pose.pose.position.x = start_x + dx * ratio
            pose.pose.position.y = start_y + dy * ratio

            # Default orientation
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        # Give the final pose the exact target orientation
        path_msg.poses[-1].pose.orientation = msg.pose.orientation
        self.goal_msg = path_msg
        self.path_pub.publish(self.goal_msg)  # publish immediately — don't wait for timer
            

    def global_path_timer_callback(self):
        if self.goal_msg is None: # if no msg
            return

        self.path_pub.publish(self.goal_msg)
        self.get_logger().info(f"Published straight-line global path with {len(self.goal_msg.poses)} waypoints.")
        
def main(args=None):
    rclpy.init(args=args)
    node = GlobalPathNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
