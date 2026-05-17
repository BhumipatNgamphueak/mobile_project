#!/usr/bin/env python3
"""
human_detection_node.py
=======================
Detect humans from LiDAR scan and/or camera image and publish
their estimated poses for path planning and visualisation.

Inputs
------
/scan                   sensor_msgs/LaserScan      2-D LiDAR (10 Hz)
/front_camera/image_raw sensor_msgs/Image          RGB camera 640×480 30 Hz

Outputs
-------
/detected_humans        visualization_msgs/MarkerArray   one cylinder per detected human
/detected_human_poses   geometry_msgs/PoseArray          pose of each detected human (odom frame)

Parameters
----------
scan_topic       (str,   default '/scan')
image_topic      (str,   default '/front_camera/image_raw')
odom_frame       (str,   default 'odom')
detection_range  (float, default 8.0)   max range [m] to report a detection
use_sim_time     (bool,  default True)
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs import msg
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid

import math
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

        self.scan_topic      = self.get_parameter('scan_topic').value
        self.image_topic     = self.get_parameter('image_topic').value
        self.odom_frame      = self.get_parameter('odom_frame').value
        self.detection_range = self.get_parameter('detection_range').value

        # ----------------- Subscribers -----------------
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_callback, 10)

        # ---------------------------------------------------

        # ----------------- Publishers -----------------
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_humans', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/detected_human_poses', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/dynamic_local_costmap', 10)

        # ---------------------------------------------------

        # ----------------- Internal state -----------------
        self._latest_scan:  LaserScan | None = None
        self._latest_image: Image     | None = None

        # Robot Odometry
        self.current_robot_x = 0.0
        self.current_robot_y = 0.0
        self.current_robot_yaw = 0.0

        # ---------------------------------------------------

        # ----------------- OpenCV Bridge and Detection Models -----------------
        self.bridge = CvBridge()
        self.human_detect_model = YOLO('yolov8n.pt')

        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=0, # 0 for saving CPU, 1 for accuracy
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.human_history = {} # เก็บ {id: {'pos': (x, y), 'time': timestamp}}
        self.min_front_scale = 1.2 # scale พื้นฐานเมื่อหยุดนิ่ง
        self.velocity_factor = 0.8  # ตัวคูณความเร็ว ยิ่งเยอะไข่ยิ่งยืด
        self.alpha = 0.8
        self.min_vel_threshold = 0.1

        # ---------------------------------------------------

        # ----------------- Camera Intrinsics -----------------
        self.img_w = 640
        self.img_h = 480
        self.fov = 1.047  # 60 degrees
        
        # Focal Length (f)
        self.fx = (self.img_w / 2.0) / math.tan(self.fov / 2.0)
        self.fy = self.fx
        
        # Optical Center
        self.uc = self.img_w / 2.0
        self.vc = self.img_h / 2.0

        # ---------------------------------------------------

        self.get_logger().info('HumanDetectionNode started')

    # ---------------------------------------------------
    # Callbacks
    # ---------------------------------------------------

    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg
        
    def _image_callback(self, msg: Image):
        # self.get_logger().info("Receive Image!!")
        self._latest_image = msg
        self._run_detection()
            
    def _odom_callback(self, msg: Odometry):
        self.current_robot_x = msg.pose.pose.position.x
        self.current_robot_y = msg.pose.pose.position.y
        
        # convert Quaternion to Yaw (Euler)
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
            h, w, _ = cv_image.shape
            results = self.human_detect_model.predict(source=cv_image, classes=[0], conf=0.5, verbose=False)
            
            all_markers = MarkerArray()
            now = self.get_clock().now()
            
            # เก็บ ID ของคนที่เราตรวจเจอในเฟรมนี้ เพื่อเอาไว้เช็คเคลียร์คนที่หายไปจากกล้อง
            current_frame_ids = set()

            # YOLO results[0].boxes ให้ข้อมูล ID มาด้วย (ถ้าใช้ mode track)
            for i, r in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, r.xyxy[0])

                # --- 3D Position ---
                u_center = (x1 + x2) / 2
                v_center = (y1 + y2) / 2
                dist_m = self._get_distance_from_scan(u_center)

                if dist_m > 0:
                    wx, wy = self._get_human_world_pose(u_center, v_center, dist_m)
                    
                    # ─── คำนวณความเร็วและทิศทางจากประวัติการเคลื่อนที่ ───
                    smooth_velocity = 0.0
                    person_yaw = 0.0  # ค่าเริ่มต้น
                    person_id = i 
                    current_frame_ids.add(person_id)
                    
                    if person_id in self.human_history:
                        prev_data = self.human_history[person_id]
                        prev_pos = prev_data['pos']
                        prev_time = prev_data['time']
                        prev_smooth_vel = prev_data.get('vel', 0.0)
                        prev_yaw = prev_data.get('yaw', 0.0)
                        
                        # 1. คำนวณความเร็ว
                        dist_diff = math.sqrt((wx - prev_pos[0])**2 + (wy - prev_pos[1])**2)
                        time_diff = (now - prev_time).nanoseconds / 1e9
                        
                        if 0.0 < time_diff < 1.0:
                            instant_velocity = dist_diff / time_diff
                            instant_velocity = min(instant_velocity, 2.0) 
                            smooth_velocity = (self.alpha * prev_smooth_vel) + ((1.0 - self.alpha) * instant_velocity)
                        else:
                            smooth_velocity = prev_smooth_vel

                        # 2. คำนวณ Orientation จากการเคลื่อนที่
                        self.get_logger().info(f"Person ID {person_id}: "f"Pos=({wx:.2f}, {wy:.2f})")  
                        calculated_yaw = self._get_orientation(wx, wy, prev_pos[0], prev_pos[1])
                        
                        if calculated_yaw is not None:
                            person_yaw = calculated_yaw
                        else:
                            person_yaw = prev_yaw
                    
                    if smooth_velocity < self.min_vel_threshold:
                        smooth_velocity = 0.0

                    # อัปเดตประวัติรอบนี้
                    self.human_history[person_id] = {
                        'pos': (wx, wy), 
                        'time': now, 
                        'vel': smooth_velocity,
                        'yaw': person_yaw
                    }

                    # สร้างไข่และแพ็กลงกระเป๋า MarkerArray
                    egg_marker = self.create_egg_marker(wx, wy, person_yaw, person_id, smooth_velocity)
                    all_markers.markers.append(egg_marker)

                    # วาดรายละเอียดลงบนภาพของ OpenCV
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"ID:{person_id} V: {smooth_velocity:.1f} m/s", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ─── จัดการลบ Marker ของคนที่เดินหลุดเฟรมไปแล้ว ───
            # หากคนไหนที่เฟรมนี้หาไม่เจอแล้ว ให้ส่งสัญลักษณ์ DELETE ไปเคลียร์ใน RViz ทิ้ง
            from visualization_msgs.msg import Marker
            for old_id in list(self.human_history.keys()):
                if old_id not in current_frame_ids:
                    delete_marker = Marker()
                    delete_marker.header.frame_id = "odom"
                    delete_marker.ns = "human_social_zones"
                    delete_marker.id = old_id
                    delete_marker.action = Marker.DELETE
                    all_markers.markers.append(delete_marker)
                    # ลบออกจากหน่วยความจำด้วยเพื่อไม่ให้หน่วงเครื่อง
                    del self.human_history[old_id]

            # ✅ แก้ไข: Publish และโชว์หน้าต่างเพียงรอบเดียวตอนท้ายสุดให้ถูกต้อง
            if len(all_markers.markers) > 0:
                self.marker_pub.publish(all_markers)

            cv2.imshow("YOLO Multi-Human Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Detection Error: {e}')

    def pub_costmap(self):

        msg = OccupancyGrid()

        pass

    def _get_human_world_pose(self, u_pixel, v_pixel, distance_m):
        """
        คำนวณพิกัดคนใน Odom Frame (World Frame)
        u_pixel: จุดกึ่งกลางคนในแกน x (pixel)
        v_pixel: จุดกึ่งกลางคนในแกน y (pixel)
        distance_m: ระยะห่างจริงจาก LiDAR
        """
        # --- 1. คำนวณใน Camera Frame (Pinhole Model) ---
        z_cam = distance_m
        x_cam = (u_pixel - self.uc) * z_cam / self.fx
        # y_cam ไม่ค่อยได้ใช้ในระนาบ 2D planning แต่คำนวณไว้ได้
        # y_cam = (v_pixel - self.vc) * z_cam / self.fy

        # --- 2. Transform เป็น Robot Frame (X-forward, Y-left) ---
        # สมมติกล้องติดอยู่หน้าหุ่นยนต์พอดี (Offset = 0)
        # ถ้ากล้องติดเยื้อง ให้บวกค่า Offset เมตรเข้าไปด้วย
        x_robot = z_cam 
        y_robot = -x_cam

        # --- 3. Transform เป็น World Frame (Odom) ---
        # เราต้องใช้ตำแหน่งหุ่นยนต์ปัจจุบัน (Robot Pose ใน Odom Frame)
        # สมมติคุณมีตัวแปรเก็บตำแหน่งหุ่นยนต์ self.robot_x, self.robot_y, self.robot_yaw
        # (ซึ่งควรดึงมาจาก Topic /odom หรือ TF)
        
        # ดึงค่าจากตัวแปรที่คุณต้อง Update ใน Odom Callback
        # ตัวอย่างสมมติค่า:
        rx, ry = self.current_robot_x, self.current_robot_y
        ryaw = self.current_robot_yaw

        # Rotation Matrix 2D เพื่อหมุนพิกัดคนเข้ากับมุมหันของหุ่นยนต์
        world_x = rx + (x_robot * math.cos(ryaw) - y_robot * math.sin(ryaw))
        world_y = ry + (x_robot * math.sin(ryaw) + y_robot * math.cos(ryaw))

        return world_x, world_y

    # ──────────────────────────────────────────────────────────────────────
    # Helper — coordinate conversion
    # ──────────────────────────────────────────────────────────────────────
    def _get_orientation(self, current_x, current_y, prev_x, prev_y):
        """
        คำนวณมุมการหันหน้า (Heading) จากทิศทางการเคลื่อนที่พิกัดโลก (odom frame)
        current_x, current_y: พิกัดเฟรม odom ปัจจุบัน
        prev_x, prev_y: พิกัดเฟรม odom เฟรมที่แล้ว
        """
        # คำนวณความต่างของตำแหน่ง (เวกเตอร์การเคลื่อนที่)
        dx = current_x - prev_x
        dy = current_y - prev_y

        # คำนวณระยะทางที่เคลื่อนที่ในเฟรมนี้
        movement_dist = math.sqrt(dx**2 + dy**2)

        # ตั้งค่า Threshold: ถ้าขยับน้อยมาก (เช่น ยืนนิ่งๆ แล้วพิกัดแกว่งจาก Noise) 
        # ไม่ควรคำนวณมุมใหม่ ให้ใช้มุมเดิมไปก่อนเพื่อป้องกันมุมหมุนสุ่ม (Jitter)
        if movement_dist < 0.05: # ขยับน้อยกว่า 5 เซนติเมตร
            return None # ส่ง None กลับไป เพื่อให้ระบบรู้ว่าไม่ต้องอัปเดตทิศทาง

        # คำนวณมุมในเฟรมโลก (odom) โดยตรง 
        # ใน odom frame: 0 Radian คือทิศหน้าหุ่นยนต์พุ่งไป (ตามมาตรฐาน ROS)
        yaw = math.atan2(dy, dx)

        # Normalize มุมให้อยู่ในช่วง [-PI, PI] เสมอเพื่อความปลอดภัย
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        return yaw

    def _get_distance_from_scan(self, u_center):
        """
        Extract the distance from LiDAR based on the pixel coordinates in the image.
        u_center: The center point of the person on the x-axis (pixel).
        """
        if self._latest_scan is None:
            return 0.0

        # 1. Convert pixels to relative angle (Relative Angle)
        # u_center = 320 (center of image 640) will result in angle_from_center = 0
        # FOV is 1.047 rad (60 degrees)
        angle_from_center = - (u_center - self.uc) * (self.fov / self.img_w)

        # 2. Convert relative angles to angles in the LiDAR frame.
        # Typically, LiDAR 0 rad represents the robot's face. 
        # If the camera is mounted directly in front of the robot, the angles will match.
        target_angle = angle_from_center 

        # 3. หา Index ใน LaserScan.ranges
        scan = self._latest_scan
        idx = int((target_angle - scan.angle_min) / scan.angle_increment)

        # 4. Check the correctness of the Index.
        if idx < 0 or idx >= len(scan.ranges):
            return 0.0

        # 5. Robust distance measurement (uses the average value around 5 detection points to reduce noise).
        window_size = 2
        valid_ranges = []
        for i in range(idx - window_size, idx + window_size + 1):
            if 0 <= i < len(scan.ranges):
                r = scan.ranges[i]
                if scan.range_min < r < scan.range_max:
                    valid_ranges.append(r)

        if not valid_ranges:
            return 1000000.0

        # Use the median to eliminate outliers.
        return sorted(valid_ranges)[len(valid_ranges) // 2]

    # ──────────────────────────────────────────────────────────────────────
    # Publisher helpers
    # ──────────────────────────────────────────────────────────────────────

    def create_egg_marker(self, x, y, yaw, person_id):

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "human_social_zones"
        marker.id = person_id  # สำคัญมาก: ID ต้องต่างกันเพื่อให้ RViz แสดงผลแยกกัน
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Scale ของ Triangle List ต้องเป็น 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        
        # สีแดงโปร่งแสง
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)

        # สมการรูปไข่ (Egg Shape Parametric)
        front_scale = 3.0  # ด้านหน้ายืด 3 เท่า (+x)
        back_scale = 1.0   # ด้านหลังยืด 1 เท่า (-x)
        side_scale = 1.0   # ด้านซ้าย/ขวา ยืด 1 เท่า (แกน y)

        num_points = 40  # ความละเอียดของขอบรูปทรง
        
        for i in range(num_points):
            # มุมที่ 1 และ 2 ของสามเหลี่ยมแผ่นนี้
            theta1 = 2.0 * math.pi * i / num_points
            theta2 = 2.0 * math.pi * (i + 1) / num_points

            def get_asymmetric_point(theta):
                # ตรรกะแยกฝั่ง: ด้านหน้ายืดมาก ด้านหลังยืดน้อย
                # ใช้ cos(theta) มาเป็นตัวตัดสินการยืดหด
                if math.cos(theta) > 0:
                    r_x = front_scale
                else:
                    r_x = back_scale
                
                r_y = side_scale

                # คำนวณพิกัด Ellipse พื้นฐาน
                px = r_x * math.cos(theta)
                py = r_y * math.sin(theta)

                # หมุนจุดตามทิศทาง Yaw ของคน (Rotation Matrix)
                rotated_x = px * math.cos(yaw) - py * math.sin(yaw)
                rotated_y = px * math.sin(yaw) + py * math.cos(yaw)

                return Point(x=float(rotated_x + x), y=float(rotated_y + y), z=0.02)

            # สร้างสามเหลี่ยม 1 แผ่น (จุดศูนย์กลางคน -> จุดขอบ 1 -> จุดขอบ 2)
            marker.points.append(Point(x=float(x), y=float(y), z=0.02)) # จุดกึ่งกลางคน
            marker.points.append(get_asymmetric_point(theta1))
            marker.points.append(get_asymmetric_point(theta2))

        return marker

    def create_egg_marker(self, x, y, yaw, person_id, velocity):

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "human_social_zones"
        marker.id = person_id  # สำคัญมาก: ID ต้องต่างกันเพื่อให้ RViz แสดงผลแยกกัน
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Scale ของ Triangle List ต้องเป็น 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        
        # สีแดงโปร่งแสง
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)
        
        # ─── Dynamic Scale Calculation ───
        # ด้านหน้า = ค่าเริ่มต้น + (ความเร็ว * ตัวคูณ)
        dynamic_front_scale = self.min_front_scale + (velocity * self.velocity_factor)
        back_scale = 0.8
        side_scale = 1.0

        num_points = 32
        for i in range(num_points):
            theta1 = 2.0 * math.pi * i / num_points
            theta2 = 2.0 * math.pi * (i + 1) / num_points
            
            def get_asymmetric_point(theta):
                # ถ้ามุมอยู่ด้านหน้า (cos > 0) ให้ใช้ dynamic_front_scale
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
