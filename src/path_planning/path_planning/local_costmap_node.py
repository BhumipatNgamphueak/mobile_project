import array
import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from sensor_msgs.msg import LaserScan

class LocalCostMapNode(Node):
    def __init__(self):
        super().__init__('local_costmap_node')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self._scan_callback, 10)
        self.cost_map_pub = self.create_publisher(OccupancyGrid, '/local_costmap', 10)

        # Cost map param
        self.costmap_resolution = 0.1
        self.width_cells = 250
        self.height_cells = 250
        
        # Max cost clamped to 100  because ROS OccupancyGrid max value is 127 (values >100 are ignored or crash)
        self.obstacle_cost = min(100, 200)
        self.inflation_radius = 1.0  # meters
        
        # Precompute inflation kernel
        self.inflation_kernel = []
        inflation_cells = int(self.inflation_radius / self.costmap_resolution)
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                dist = math.hypot(dx, dy) * self.costmap_resolution
                if dist <= self.inflation_radius:
                    if dist == 0.0:
                        cost = self.obstacle_cost
                    else:
                        cost = max(1, int(self.obstacle_cost * (1.0 - (dist / self.inflation_radius))))
                    self.inflation_kernel.append((dx, dy, cost))
        
    def _scan_callback(self, msg: LaserScan):
        self.get_logger().info(f'Scan received: {len(msg.ranges)} ranges')
        # Set header
        grid = OccupancyGrid()
        grid.header = msg.header
        grid.info.resolution = self.costmap_resolution
        grid.info.width = self.width_cells
        grid.info.height = self.height_cells

        # Set center of costmap to robot position
        grid.info.origin.position.x = -(self.width_cells * self.costmap_resolution) / 2.0
        grid.info.origin.position.y = -(self.height_cells * self.costmap_resolution) / 2.0
        grid.info.origin.position.z = 0.0

        # Initialize the grid with 0 (Free Space). -1 is Unknown, 100 is Lethal Obstacle.
        grid_data = [0] * (self.width_cells * self.height_cells)
        
        current_angle = msg.angle_min # first scan
        
        # Keep track of obstacle cells to avoid redundancy
        obstacle_cells = set()

        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Convert to catesian
                x = r*math.cos(current_angle)
                y = r*math.sin(current_angle)

                # offset grid to robot center
                grid_x = int((x - grid.info.origin.position.x) / self.costmap_resolution)
                grid_y = int((y - grid.info.origin.position.y) / self.costmap_resolution)
                
                if 0 <= grid_x < self.width_cells and 0 <= grid_y < self.height_cells:
                    obstacle_cells.add((grid_x, grid_y))
            current_angle += msg.angle_increment

        # Inflate obstacles using precomputed kernel
        for ox, oy in obstacle_cells:
            for dx, dy, cost in self.inflation_kernel:
                nx = ox + dx
                ny = oy + dy
                if 0 <= nx < self.width_cells and 0 <= ny < self.height_cells:
                    index = (ny * self.width_cells) + nx
                    if grid_data[index] < cost:
                        grid_data[index] = cost

        # Convert python list to C-type array to bypass slow rclpy bounds checking
        grid.data = array.array('b', grid_data)
        
        self.cost_map_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()