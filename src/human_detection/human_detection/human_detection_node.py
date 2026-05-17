#!/usr/bin/env python3
"""
human_detection_node.py
=======================
Detect humans from LiDAR scan and/or camera image and publish
their estimated poses and asymmetric Gaussian social costmap for navigation.
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from nav_msgs.msg import Odometry, OccupancyGrid

import math
import array
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import mediapipe as mp

class HumanDetectionNode(Node):

    def __init__(self):
        super().__init__('human_detection_node')

        # ----------------- Parameters -----------------
        self.declare_parameter('scan_topic',      '/scan')
        self.declare_parameter('image_topic',     '/front_camera/image_raw')
        self.declare_parameter('odom_frame',      'odom')
        self.declare_parameter('detection_range', 8.0)
        
        # Costmap Parameters
        self.declare_parameter('costmap_resolution', 0.05) # 5 cm per cell
        self.declare_parameter('costmap_width', 15.0)       # 15 meters
        self.declare_parameter('costmap_height', 15.0)      # 15 meters

        self.scan_topic      = self.get_parameter('scan_topic').value
        self.image_topic     = self.get_parameter('image_topic').value
        self.odom_frame      = self.get_parameter('odom_frame').value
        self.detection_range = self.get_parameter('detection_range').value
        
        self.resolution   = self.get_parameter('costmap_resolution').value
        self.width_cells  = int(self.get_parameter('costmap_width').value / self.resolution)
        self.height_cells = int(self.get_parameter('costmap_height').value / self.resolution)

        # ----------------- Subscribers -----------------
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_callback, 10)

        # ----------------- Publishers -----------------
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_humans', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/detected_human_poses', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/dynamic_local_costmap', 10)

        # ----------------- Internal state -----------------
        self._latest_scan:  LaserScan | None = None
        self._latest_image: Image     | None = None

        # Robot Odometry
        self.current_robot_x = 0.0
        self.current_robot_y = 0.0
        self.current_robot_yaw = 0.0

        # ----------------- OpenCV Bridge and Detection Models -----------------
        self.bridge = CvBridge()
        self.human_detect_model = YOLO('yolov8n.pt')

        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=0, 
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Dynamic Scale Configurations
        self.human_history = {} 
        self.min_front_scale = 1.2 
        self.velocity_factor = 0.8
        self.alpha = 0.8
        self.min_vel_threshold = 0.1

        # ----------------- Camera Intrinsics -----------------
        self.img_w = 640
        self.img_h = 480
        self.fov = 1.047  # 60 degrees
        
        self.fx = (self.img_w / 2.0) / math.tan(self.fov / 2.0)
        self.fy = self.fx
        self.uc = self.img_w / 2.0
        self.vc = self.img_h / 2.0

        self.get_logger().info('HumanDetectionNode started with Integrated Gaussian Costmap.')

    # ---------------------------------------------------
    # Callbacks
    # ---------------------------------------------------

    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg
        
    def _image_callback(self, msg: Image):
        self._latest_image = msg
        self._run_detection()
            
    def _odom_callback(self, msg: Odometry):
        self.current_robot_x = msg.pose.pose.position.x
        self.current_robot_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ──────────────────────────────────────────────────────────────────────
    # Detection pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _run_detection(self):
        if self._latest_scan is None or self._latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
            results = self.human_detect_model.predict(source=cv_image, classes=[0], conf=0.5, verbose=False)
            
            all_markers = MarkerArray()
            now = self.get_clock().now()
            current_frame_ids = set()

            # OccupancyGrid Structure for Costmap
            costmap = OccupancyGrid()
            costmap.header.stamp = self._latest_image.header.stamp
            costmap.header.frame_id = self.odom_frame
            costmap.info.resolution = self.resolution
            costmap.info.width = self.width_cells
            costmap.info.height = self.height_cells
            costmap.info.origin.position.x = self.current_robot_x - (self.width_cells * self.resolution) / 2.0
            costmap.info.origin.position.y = self.current_robot_y - (self.height_cells * self.resolution) / 2.0
            costmap.info.origin.position.z = 0.0

            # create empty Grid data (starting with 0 occupancy scores)
            grid_data = [0] * (self.width_cells * self.height_cells)

            for i, r in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, r.xyxy[0])

                # --- 3D Position ---
                u_center = (x1 + x2) / 2
                v_center = (y1 + y2) / 2
                dist_m = self._get_distance_from_scan(u_center)

                if dist_m > 0 and dist_m <= self.detection_range:
                    wx, wy = self._get_human_world_pose(u_center, v_center, dist_m)
                    
                    smooth_velocity = 0.0
                    person_yaw = 0.0  
                    person_id = i 
                    current_frame_ids.add(person_id)
                    
                    if person_id in self.human_history:
                        prev_data = self.human_history[person_id]
                        prev_pos = prev_data['pos']
                        prev_time = prev_data['time']
                        prev_smooth_vel = prev_data.get('vel', 0.0)
                        prev_yaw = prev_data.get('yaw', 0.0)
                        
                        # 1. calculate velocity (EMA Low-pass filter)
                        dist_diff = math.sqrt((wx - prev_pos[0])**2 + (wy - prev_pos[1])**2)
                        time_diff = (now - prev_time).nanoseconds / 1e9
                        
                        if 0.0 < time_diff < 1.0:
                            instant_velocity = dist_diff / time_diff
                            instant_velocity = min(instant_velocity, 2.0) 
                            smooth_velocity = (self.alpha * prev_smooth_vel) + ((1.0 - self.alpha) * instant_velocity)
                        else:
                            smooth_velocity = prev_smooth_vel

                        # 2. calculate movement orientation from world coordinates
                        self.get_logger().info(f"Person {person_id} - Pos: ({wx:.2f}, {wy:.2f})")
                        calculated_yaw = self._get_orientation(wx, wy, prev_pos[0], prev_pos[1])
                        if calculated_yaw is not None:
                            person_yaw = calculated_yaw
                        else:
                            person_yaw = prev_yaw
                    
                    if smooth_velocity < self.min_vel_threshold:
                        smooth_velocity = 0.0

                    # record History
                    self.human_history[person_id] = {
                        'pos': (wx, wy), 
                        'time': now, 
                        'vel': smooth_velocity,
                        'yaw': person_yaw
                    }

                    # 3. calculate and add asymmetric Gaussian cost to the costmap grid
                    self._add_human_gaussian_to_grid(
                        grid_data,
                        costmap.info.origin.position.x,
                        costmap.info.origin.position.y,
                        wx, wy, person_yaw, smooth_velocity
                    )

                    # 4. create Egg Mesh for visualization on RViz
                    egg_marker = self.create_egg_marker(wx, wy, person_yaw, person_id, smooth_velocity)
                    all_markers.markers.append(egg_marker)

                    # show on OpenCV
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"ID:{person_id} V: {smooth_velocity:.1f} m/s", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- clear outdated human history ---
            from visualization_msgs.msg import Marker
            for old_id in list(self.human_history.keys()):
                if old_id not in current_frame_ids:
                    delete_marker = Marker()
                    delete_marker.header.frame_id = self.odom_frame
                    delete_marker.ns = "human_social_zones"
                    delete_marker.id = old_id
                    delete_marker.action = Marker.DELETE
                    all_markers.markers.append(delete_marker)
                    del self.human_history[old_id]

            # Publish on RViz
            if len(all_markers.markers) > 0:
                self.marker_pub.publish(all_markers)

            # Publish occupancy grid costmap
            costmap.data = array.array('b', grid_data)
            self.costmap_pub.publish(costmap)

            # cv2.imshow("YOLO Multi-Human Detection", cv_image)
            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Detection Error: {e}')

    def _add_human_gaussian_to_grid(self, grid_data, origin_x, origin_y, h_x, h_y, h_yaw, h_vel):

        # Adjust the slope radius of the slope mode (m)
        sigma_front = self.min_front_scale + (h_vel * self.velocity_factor)
        sigma_back  = 0.4
        sigma_y     = 0.5 

        max_influence_range = max(sigma_front, sigma_back, sigma_y) * 3.0
        cells_range = int(max_influence_range / self.resolution)

        # convert human position to cell coordinates
        human_cx = int((h_x - origin_x) / self.resolution)
        human_cy = int((h_y - origin_y) / self.resolution)

        for dx in range(-cells_range, cells_range + 1):
            for dy in range(-cells_range, cells_range + 1):
                nx = human_cx + dx
                ny = human_cy + dy

                if 0 <= nx < self.width_cells and 0 <= ny < self.height_cells:

                    local_x_m = dx * self.resolution
                    local_y_m = dy * self.resolution

                    rx = local_x_m * math.cos(h_yaw) + local_y_m * math.sin(h_yaw)
                    ry = -local_x_m * math.sin(h_yaw) + local_y_m * math.cos(h_yaw)

                    sig_x = sigma_front if rx > 0 else sigma_back

                    # 2D Gaussian
                    gaussian_value = math.exp(-((rx**2) / (sig_x**2) + (ry**2) / (sigma_y**2)))

                    human_cost = int(85 * gaussian_value)

                    if human_cost > 0:
                        idx = ny * self.width_cells + nx
                        if grid_data[idx] < human_cost:
                            grid_data[idx] = human_cost

    def _get_human_world_pose(self, u_pixel, v_pixel, distance_m):
        z_cam = distance_m
        x_cam = (u_pixel - self.uc) * z_cam / self.fx

        x_robot = z_cam 
        y_robot = -x_cam

        rx, ry = self.current_robot_x, self.current_robot_y
        ryaw = self.current_robot_yaw

        world_x = rx + (x_robot * math.cos(ryaw) - y_robot * math.sin(ryaw))
        world_y = ry + (x_robot * math.sin(ryaw) + y_robot * math.cos(ryaw))

        return world_x, world_y

    def _get_orientation(self, current_x, current_y, prev_x, prev_y):
        dx = current_x - prev_x
        dy = current_y - prev_y
        movement_dist = math.sqrt(dx**2 + dy**2)

        if movement_dist < 0.05: 
            return None 

        yaw = math.atan2(dy, dx)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return yaw

    def _get_distance_from_scan(self, u_center):
        if self._latest_scan is None:
            return 0.0

        angle_from_center = - (u_center - self.uc) * (self.fov / self.img_w)
        target_angle = angle_from_center 

        scan = self._latest_scan
        idx = int((target_angle - scan.angle_min) / scan.angle_increment)

        if idx < 0 or idx >= len(scan.ranges):
            return 0.0

        window_size = 2
        valid_ranges = []
        for i in range(idx - window_size, idx + window_size + 1):
            if 0 <= i < len(scan.ranges):
                r = scan.ranges[i]
                if scan.range_min < r < scan.range_max:
                    valid_ranges.append(r)

        if not valid_ranges:
            return 1000000.0

        return sorted(valid_ranges)[len(valid_ranges) // 2]

    def create_egg_marker(self, x, y, yaw, person_id, velocity):
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "human_social_zones"
        marker.id = person_id  
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)
        
        dynamic_front_scale = self.min_front_scale + (velocity * self.velocity_factor)
        back_scale = 0.4
        side_scale = 0.5

        num_points = 100
        for i in range(num_points):
            theta1 = 2.0 * math.pi * i / num_points
            theta2 = 2.0 * math.pi * (i + 1) / num_points
            
            def get_asymmetric_point(theta):
                r_x = dynamic_front_scale if math.cos(theta) > 0 else back_scale
                r_y = side_scale
                
                px = r_x * math.cos(theta)
                py = r_y * math.sin(theta)
                
                rx = px * math.cos(yaw) - py * math.sin(yaw)
                ry = px * math.sin(yaw) + py * math.cos(yaw)
                return Point(x=float(rx + x), y=float(ry + y), z=0.02)

            marker.points.append(Point(x=float(x), y=float(y), z=0.02))
            marker.points.append(get_asymmetric_point(theta1))
            marker.points.append(get_asymmetric_point(theta2))

        return marker
    
def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()