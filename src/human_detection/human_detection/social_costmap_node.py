#!/usr/bin/env python3
"""
social_costmap_node.py
======================
Fuses the LiDAR-based local costmap from `local_costmap_node` (`/local_costmap`)
with asymmetric Gaussian "social" cost around humans detected via YOLO on the
front camera. Publishes the merged grid on `/local_costmap_social`.

Pipeline:
    /scan ──▶ local_costmap_node ──▶ /local_costmap ─┐
                                                      │
    /front_camera/image_raw ─┐                        ▼
    /scan ───────────────────┼──▶ social_costmap_node ──▶ /local_costmap_social
    /odom ───────────────────┘
"""

import array
import math

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import ColorRGBA
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


class SocialCostmapNode(Node):

    def __init__(self):
        super().__init__('social_costmap_node')

        # ----------------- Parameters -----------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('image_topic', '/front_camera/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('input_costmap_topic', '/local_costmap')
        self.declare_parameter('output_costmap_topic', '/local_costmap_social')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('detection_range', 8.0)

        # Asymmetric-Gaussian shape parameters (same defaults as
        # human_detection_node so visual behavior matches).
        self.declare_parameter('min_front_scale', 1.2)
        self.declare_parameter('velocity_factor', 0.6)
        self.declare_parameter('sigma_back', 0.4)
        self.declare_parameter('sigma_side', 0.5)
        self.declare_parameter('peak_cost', 85)
        self.declare_parameter('alpha', 0.8)
        self.declare_parameter('min_vel_threshold', 0.1)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.input_costmap_topic = self.get_parameter('input_costmap_topic').value
        self.output_costmap_topic = self.get_parameter('output_costmap_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.detection_range = self.get_parameter('detection_range').value

        self.min_front_scale = self.get_parameter('min_front_scale').value
        self.velocity_factor = self.get_parameter('velocity_factor').value
        self.sigma_back = self.get_parameter('sigma_back').value
        self.sigma_side = self.get_parameter('sigma_side').value
        self.peak_cost = int(self.get_parameter('peak_cost').value)
        self.alpha = self.get_parameter('alpha').value
        self.min_vel_threshold = self.get_parameter('min_vel_threshold').value

        # ----------------- Subscribers -----------------
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, self.input_costmap_topic, self._costmap_callback, 10)

        # ----------------- Publishers -----------------
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, self.output_costmap_topic, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/detected_humans', 10)

        # ----------------- Internal state -----------------
        self._latest_scan: LaserScan | None = None
        self._latest_image: Image | None = None
        self._latest_costmap: OccupancyGrid | None = None

        self.current_robot_x = 0.0
        self.current_robot_y = 0.0
        self.current_robot_yaw = 0.0

        self.human_history: dict = {}

        # ----------------- Vision -----------------
        self.bridge = CvBridge()
        self.human_detect_model = YOLO('yolov8n.pt')

        self.img_w = 640
        self.img_h = 480
        self.fov = 1.047  # 60 deg
        self.fx = (self.img_w / 2.0) / math.tan(self.fov / 2.0)
        self.fy = self.fx
        self.uc = self.img_w / 2.0
        self.vc = self.img_h / 2.0

        self.get_logger().info(
            f"SocialCostmapNode started. "
            f"In: {self.input_costmap_topic}  Out: {self.output_costmap_topic}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _odom_callback(self, msg: Odometry):
        self.current_robot_x = msg.pose.pose.position.x
        self.current_robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _costmap_callback(self, msg: OccupancyGrid):
        # Cache the latest LiDAR costmap; the image callback drives publishing
        # so the fused grid stays aligned with each detection frame.
        self._latest_costmap = msg

    def _image_callback(self, msg: Image):
        self._latest_image = msg
        self._run_detection_and_publish()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def _run_detection_and_publish(self):
        if (self._latest_costmap is None
                or self._latest_scan is None
                or self._latest_image is None):
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                self._latest_image, desired_encoding='bgr8')
            results = self.human_detect_model.predict(
                source=cv_image, classes=[0], conf=0.5, verbose=False)

            # Start from a copy of the latest LiDAR costmap.
            base = self._latest_costmap
            width = base.info.width
            height = base.info.height
            origin_x = base.info.origin.position.x
            origin_y = base.info.origin.position.y
            resolution = base.info.resolution

            # OccupancyGrid.data is signed bytes; copy into a mutable list so
            # we can overlay GMM cost without mutating the cached message.
            grid_data = list(base.data)

            all_markers = MarkerArray()
            now = self.get_clock().now()
            current_frame_ids = set()

            for i, r in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                u_center = (x1 + x2) / 2
                v_center = (y1 + y2) / 2
                dist_m = self._get_distance_from_scan(u_center)

                if not (0 < dist_m <= self.detection_range):
                    continue

                wx, wy = self._get_human_world_pose(u_center, v_center, dist_m)

                person_id = i
                current_frame_ids.add(person_id)

                smooth_velocity = 0.0
                person_yaw = 0.0

                if person_id in self.human_history:
                    prev = self.human_history[person_id]
                    px, py = prev['pos']
                    prev_time = prev['time']
                    prev_vel = prev.get('vel', 0.0)
                    prev_yaw = prev.get('yaw', 0.0)

                    dist_diff = math.hypot(wx - px, wy - py)
                    time_diff = (now - prev_time).nanoseconds / 1e9
                    if 0.0 < time_diff < 1.0:
                        instant_vel = min(dist_diff / time_diff, 2.0)
                        smooth_velocity = (self.alpha * prev_vel
                                           + (1.0 - self.alpha) * instant_vel)
                    else:
                        smooth_velocity = prev_vel

                    calc_yaw = self._get_orientation(wx, wy, px, py)
                    person_yaw = calc_yaw if calc_yaw is not None else prev_yaw

                if smooth_velocity < self.min_vel_threshold:
                    smooth_velocity = 0.0

                self.human_history[person_id] = {
                    'pos': (wx, wy),
                    'time': now,
                    'vel': smooth_velocity,
                    'yaw': person_yaw,
                }

                # Overlay asymmetric Gaussian onto the received grid.
                self._add_human_gaussian_to_grid(
                    grid_data, width, height,
                    origin_x, origin_y, resolution,
                    wx, wy, person_yaw, smooth_velocity)

                all_markers.markers.append(
                    self._create_egg_marker(wx, wy, person_yaw,
                                            person_id, smooth_velocity))

            # Delete markers for humans that left the frame.
            for old_id in list(self.human_history.keys()):
                if old_id not in current_frame_ids:
                    dm = Marker()
                    dm.header.frame_id = self.odom_frame
                    dm.ns = "human_social_zones"
                    dm.id = old_id
                    dm.action = Marker.DELETE
                    all_markers.markers.append(dm)
                    del self.human_history[old_id]

            if all_markers.markers:
                self.marker_pub.publish(all_markers)

            # Publish fused costmap reusing the input grid's geometry/stamp.
            fused = OccupancyGrid()
            fused.header.stamp = base.header.stamp
            fused.header.frame_id = base.header.frame_id
            fused.info = base.info
            fused.data = array.array('b', [self._clip_cost(v) for v in grid_data])
            self.costmap_pub.publish(fused)

        except Exception as e:
            self.get_logger().error(f'SocialCostmap error: {e}')

    # ------------------------------------------------------------------
    # GMM overlay
    # ------------------------------------------------------------------
    def _add_human_gaussian_to_grid(self, grid_data, width, height,
                                    origin_x, origin_y, resolution,
                                    h_x, h_y, h_yaw, h_vel):
        sigma_front = self.min_front_scale + h_vel * self.velocity_factor
        sigma_back = self.sigma_back
        sigma_y = self.sigma_side

        max_influence = max(sigma_front, sigma_back, sigma_y) * 3.0
        cells_range = int(max_influence / resolution)

        human_cx = int((h_x - origin_x) / resolution)
        human_cy = int((h_y - origin_y) / resolution)

        cos_y = math.cos(h_yaw)
        sin_y = math.sin(h_yaw)

        for dx in range(-cells_range, cells_range + 1):
            for dy in range(-cells_range, cells_range + 1):
                nx = human_cx + dx
                ny = human_cy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                local_x = dx * resolution
                local_y = dy * resolution
                rx = local_x * cos_y + local_y * sin_y
                ry = -local_x * sin_y + local_y * cos_y

                sig_x = sigma_front if rx > 0 else sigma_back
                g = math.exp(-((rx * rx) / (sig_x * sig_x)
                               + (ry * ry) / (sigma_y * sigma_y)))
                human_cost = int(self.peak_cost * g)
                if human_cost <= 0:
                    continue

                idx = ny * width + nx
                # OccupancyGrid.data is signed int8; treat unknown (-1) as 0
                # before maxing so unknown cells get populated by social cost.
                current = grid_data[idx]
                if current < 0:
                    current = 0
                if current < human_cost:
                    grid_data[idx] = human_cost

    @staticmethod
    def _clip_cost(v: int) -> int:
        # Preserve -1 (unknown) if it survived; otherwise clamp 0..100.
        if v < 0:
            return -1
        if v > 100:
            return 100
        return v

    # ------------------------------------------------------------------
    # Geometry helpers (mirrors human_detection_node)
    # ------------------------------------------------------------------
    def _get_human_world_pose(self, u_pixel, v_pixel, distance_m):
        z_cam = distance_m
        x_cam = (u_pixel - self.uc) * z_cam / self.fx
        x_robot = z_cam
        y_robot = -x_cam
        rx, ry = self.current_robot_x, self.current_robot_y
        ryaw = self.current_robot_yaw
        wx = rx + (x_robot * math.cos(ryaw) - y_robot * math.sin(ryaw))
        wy = ry + (x_robot * math.sin(ryaw) + y_robot * math.cos(ryaw))
        return wx, wy

    def _get_orientation(self, cx, cy, px, py):
        dx = cx - px
        dy = cy - py
        if math.hypot(dx, dy) < 0.05:
            return None
        yaw = math.atan2(dy, dx)
        return math.atan2(math.sin(yaw), math.cos(yaw))

    def _get_distance_from_scan(self, u_center):
        scan = self._latest_scan
        if scan is None:
            return 0.0
        angle_from_center = -(u_center - self.uc) * (self.fov / self.img_w)
        idx = int((angle_from_center - scan.angle_min) / scan.angle_increment)
        if idx < 0 or idx >= len(scan.ranges):
            return 0.0

        window = 2
        valid = []
        for i in range(idx - window, idx + window + 1):
            if 0 <= i < len(scan.ranges):
                r = scan.ranges[i]
                if scan.range_min < r < scan.range_max:
                    valid.append(r)
        if not valid:
            return 1e6
        return sorted(valid)[len(valid) // 2]

    def _create_egg_marker(self, x, y, yaw, person_id, velocity):
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "human_social_zones"
        marker.id = person_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)

        front_scale = self.min_front_scale + velocity * self.velocity_factor
        back_scale = self.sigma_back
        side_scale = self.sigma_side

        num = 100
        for i in range(num):
            t1 = 2.0 * math.pi * i / num
            t2 = 2.0 * math.pi * (i + 1) / num

            def asym_pt(theta):
                r_x = front_scale if math.cos(theta) > 0 else back_scale
                r_y = side_scale
                px = r_x * math.cos(theta)
                py = r_y * math.sin(theta)
                rx = px * math.cos(yaw) - py * math.sin(yaw)
                ry = px * math.sin(yaw) + py * math.cos(yaw)
                return Point(x=float(rx + x), y=float(ry + y), z=0.02)

            marker.points.append(Point(x=float(x), y=float(y), z=0.02))
            marker.points.append(asym_pt(t1))
            marker.points.append(asym_pt(t2))
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = SocialCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
