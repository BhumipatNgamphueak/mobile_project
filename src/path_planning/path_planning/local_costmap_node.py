import array
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
from tf2_ros import TransformException

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan


class LocalCostMapNode(Node):
    def __init__(self):
        super().__init__('local_costmap')

        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('width', 15.0)           # meters
        self.declare_parameter('height', 15.0)          # meters
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('inflation_radius', 0.12)
        self.declare_parameter('cost_scaling_factor', 10.0)

        self.resolution = self.get_parameter('resolution').value
        width_m = self.get_parameter('width').value
        height_m = self.get_parameter('height').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.cost_scaling_factor = self.get_parameter('cost_scaling_factor').value

        self.width_cells = int(width_m / self.resolution)
        self.height_cells = int(height_m / self.resolution)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap', 10)

        self._build_inflation_kernel()

    def _build_inflation_kernel(self):
        self.inflation_kernel = []
        inflation_cells = int(self.inflation_radius / self.resolution)
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                dist = math.hypot(dx, dy) * self.resolution
                if dist <= self.inflation_radius:
                    if dist == 0.0:
                        cost = 100
                    else:
                        # Nav2-style exponential decay
                        factor = math.exp(-self.cost_scaling_factor * dist)
                        cost = max(1, int(100 * factor))
                    self.inflation_kernel.append((dx, dy, cost))

    def _scan_callback(self, msg: LaserScan):
        # Lookup transform: scan frame -> global frame (odom)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Rolling window: origin tracks robot position in odom frame
        grid = OccupancyGrid()
        grid.header.stamp = msg.header.stamp
        grid.header.frame_id = self.global_frame
        grid.info.resolution = self.resolution
        grid.info.width = self.width_cells
        grid.info.height = self.height_cells
        grid.info.origin.position.x = tx - (self.width_cells * self.resolution) / 2.0
        grid.info.origin.position.y = ty - (self.height_cells * self.resolution) / 2.0
        grid.info.origin.position.z = 0.0

        grid_data = [0] * (self.width_cells * self.height_cells)
        obstacle_cells = set()

        # Find obstacle
        current_angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Rotate scan point from scan frame into global frame
                lx = r * math.cos(current_angle)
                ly = r * math.sin(current_angle)
                gx = tx + lx * math.cos(yaw) - ly * math.sin(yaw)
                gy = ty + lx * math.sin(yaw) + ly * math.cos(yaw)

                cx = int((gx - grid.info.origin.position.x) / self.resolution)
                cy = int((gy - grid.info.origin.position.y) / self.resolution)

                if 0 <= cx < self.width_cells and 0 <= cy < self.height_cells:
                    obstacle_cells.add((cx, cy))
            current_angle += msg.angle_increment

        # Add cost to cost map
        for ox, oy in obstacle_cells: # Search all
            for dx, dy, cost in self.inflation_kernel:
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < self.width_cells and 0 <= ny < self.height_cells:
                    idx = ny * self.width_cells + nx
                    if grid_data[idx] < cost:
                        grid_data[idx] = cost

        grid.data = array.array('b', grid_data)
        self.costmap_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
