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

        # ----------------- Human Trajectory Script (Ground Truth) -----------------
        self.start_sim_time = None
        self.trajectory_duration = 28.6

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

    # def _run_detection(self):
    #     """
    #     Called whenever a new scan arrives.
    #     Populate detections with (x, y) positions in the odom frame,
    #     then call self._publish(detections).
    #     """
    #     if self._latest_scan is None:
    #         return

    #     detections: list[tuple[float, float]] = []

    #     try:
    #         # 1. แปลงภาพเป็น OpenCV
    #         cv_image = self.bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
    #         h, w, _ = cv_image.shape
            
    #         # 2. รัน YOLO Detection (กรองเอาเฉพาะ 'person' class คือ ID 0)
    #         # stream=True ช่วยลด memory usage
    #         results = self.human_detect_model.predict(source=cv_image, classes=[0], conf=0.5, verbose=False)
            
    #         detections: list[tuple[float, float]] = []

    #         # 3. จัดการผลลัพธ์
    #         for r in results:
    #             boxes = r.boxes
    #             for box in boxes:
    #                 # ตีกรอบลงบนภาพ
    #                 x1, y1, x2, y2 = box.xyxy[0]
    #                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #                 cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 # cv2.putText(cv_image, "Human", (x1, y1 - 10), 
    #                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #                 # ขั้นตอนที่ 2: Crop ภาพเฉพาะส่วนที่ YOLO เจอคน
    #                 # เพิ่ม Padding เล็กน้อยเพื่อให้ MediaPipe เห็นหัวและเท้าครบ
    #                 pad = 20
    #                 crop_y1, crop_y2 = max(0, y1-pad), min(h, y2+pad)
    #                 crop_x1, crop_x2 = max(0, x1-pad), min(w, x2+pad)
    #                 person_crop = cv_image[crop_y1:crop_y2, crop_x1:crop_x2]

    #                 if person_crop.size == 0: continue

    #                 # ขั้นตอนที่ 3: ส่ง Crop ไปให้ MediaPipe (ประหยัด CPU เพราะภาพเล็ก)
    #                 person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    #                 pose_results = self.pose_detector.process(person_rgb)

    #                 # self.get_logger().info(f'{pose_results.pose_landmarks}')
    #                 if pose_results.pose_landmarks:

    #                     self.mp_draw.draw_landmarks(
    #                         person_crop, 
    #                         pose_results.pose_landmarks, 
    #                         self.mp_pose.POSE_CONNECTIONS,
    #                         self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # dot
    #                         self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)  # line
    #                     )

    #                     # 2. แสดงหน้าต่างแยกเฉพาะคน (จะเห็น Skeleton ชัดเจน)
    #                     # cv2.imshow("Mediapipe Skeleton (Crop)", person_crop)

    #                     # คำนวณทิศทางใบหน้า (Orientation)
    #                     orientation = self._get_orientation(pose_results.pose_landmarks.landmark)
                        
    #                     cv2.putText(cv_image, f"Dir: {orientation}", (x1, y1-10), 
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                           
    #                 # ── ส่วนคำนวณตำแหน่ง 3D (Logic เบื้องต้น) ──────────────────
    #                 # ในระบบจริง คุณต้องนำพิกัดกลางภาพ (u, v) ไปเทียบกับ LiDAR scan
    #                 # หรือใช้ Depth Camera เพื่อหา x, y ใน odom frame
                    
    #                 # หาจุดกึ่งกลางพิกเซลของคน
    #                 u_center = (x1 + x2) / 2
    #                 v_center = (y1 + y2) / 2
                    
    #                 dist_m = self._get_distance_from_scan(u_center)

    #                 if dist_m > 0:
    #                     # คำนวณพิกัดโลก
    #                     wx, wy = self._get_human_world_pose(u_center, v_center, dist_m)

    #                     # 2. เรียกใช้ฟังก์ชันรูปไข่ตรงนี้!
    #                     # มันจะสร้าง Marker ใน RViz ที่ตำแหน่ง wx, wy และหมุนตาม person_yaw
    #                     self.publish_social_zone_ellipse(wx, wy, orientation)
    #                 else:
    #                     wx, wy = 0.0, 0.0
                    
    #                 # self.get_logger().info(f"Human at: x={wx:.2f}m, y={wy:.2f}m")
    #                 detections.append((wx, wy))

    #         # 4. แสดงผลภาพที่ตีกรอบแล้ว
    #         cv2.imshow("YOLO Human Detection", cv_image)
    #         cv2.waitKey(1)

        # except Exception as e:
        #     self.get_logger().error(f'Detection Error: {e}')

    def _run_detection(self):
        if self._latest_scan is None or self._latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
            h, w, _ = cv_image.shape
            results = self.human_detect_model.predict(source=cv_image, classes=[0], conf=0.5, verbose=False)
            
            # สร้าง MarkerArray ว่างไว้รวบรวม Marker ของทุกคน
            all_markers = MarkerArray()

            # วนลูปจัดการคนทุกคนที่ YOLO เจอ
            for i, r in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                
                # --- MediaPipe & Orientation ---
                person_crop = cv_image[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                person_yaw = 0.0
                
                # self.get_logger().info(f'Processing person {i} with crop size: {person_crop.shape}')
                if person_crop.size > 0:
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose_detector.process(person_rgb)
                    if pose_results.pose_landmarks:
                        # ใช้ฟังก์ชันคำนวณมุมแบบละเอียด 360 องศาที่เราทำกันก่อนหน้า
                        # person_yaw = self._get_orientation(pose_results.pose_landmarks.landmark)
                        
                        # วาด Skeleton ลงบนภาพหลักเพื่อดูหลายคนพร้อมกัน
                        self.mp_draw.draw_landmarks(person_crop, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # self.get_logger().info(f'Person {i} orientation (yaw): {math.degrees(person_yaw):.1f} degrees')
                # --- 3D Position ---
                u_center = (x1 + x2) / 2
                v_center = (y1 + y2) / 2
                dist_m = self._get_distance_from_scan(u_center)

                if dist_m > 0:
                    wx, wy = self._get_human_world_pose(u_center, v_center, dist_m)
                    
                    # 🔥 เรียกใช้ฟังก์ชันรูปไข่ โดยส่ง i (Index) เข้าไปด้วย
                    egg_marker = self.create_egg_marker(wx, wy, person_yaw, person_id=i)
                    all_markers.markers.append(egg_marker)

                    # ใส่ Label บนภาพ
                    cv2.putText(cv_image, f"ID:{i}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # self.get_logger().info(f"Person {i} at: x={wx:.2f}m, y={wy:.2f}m, yaw={math.degrees(person_yaw):.1f} deg")

            # 5. Publish Marker ของทุกคนออกไปพร้อมกันในครั้งเดียว
            self.marker_pub.publish(all_markers)

            cv2.imshow("YOLO Multi-Human Detection", cv_image)
            cv2.waitKey(1)

            # self.get_logger().info(f'Published {len(all_markers.markers)} human markers.')

        except Exception as e:
            self.get_logger().error(f'Detection Error: {e}')

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

    def _get_orientation(self, landmarks):
        """
        คำนวณมุมการหันของตัวคนแบบ 360 องศา เทียบกับเฟรมของกล้อง/LiDAR
        0 องศา = หันหน้าไปทางเดียวกับหุ่นยนต์ (หันหลังให้หุ่น)
        180 หรือ -180 องศา = หันหน้าเข้าหาหุ่นยนต์
        """
        # 1. ดึงพิกัด (MediaPipe ให้ค่า x, z โดย z คือความลึก)
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

        # 2. สร้างเวกเตอร์ไหล่ (Right -> Left)
        # ใน MediaPipe x ไปทางขวาภาพ, z พุ่งเข้าหาภาพ
        dx = l_shoulder.x - r_shoulder.x
        dz = l_shoulder.z - r_shoulder.z

        # 3. คำนวณมุมของไหล่ และหาเวกเตอร์ตั้งฉาก (Heading Vector)
        # มุมตั้งฉากคือ atan2(-dz, dx) สำหรับทิศทางพุ่งออก
        person_yaw = math.atan2(dx, -dz) 

        # 4. ใช้จมูกเช็ค Front/Back (แก้ปัญหาความกำกวมของมุม)
        # ถ้าจมูกมีค่า z น้อยกว่าค่าเฉลี่ยของไหล่ (z น้อยคือเข้าใกล้กล้องมากขึ้นในบางระบบ)
        # หรือใช้ความสัมพันธ์ของตำแหน่ง x ในการ Re-check
        shoulder_center_z = (l_shoulder.z + r_shoulder.z) / 2
        
        # ถ้าจมูกอยู่ "หลัง" ไหล่ในมิติความลึก (ในพิกัด MediaPipe)
        if nose.z > shoulder_center_z:
            # กรณีนี้อาจเป็นการหันหลัง ให้กลับทิศทางเวกเตอร์ถ้าจำเป็น 
            # (ขึ้นอยู่กับว่าคุณต้องการมุมสัมบูรณ์เทียบกับตัวหุ่นยนต์อย่างไร)
            pass

        # 5. ปรับให้เข้ากับมาตรฐาน LiDAR (0 = หน้าหุ่น)
        # ปกติถ้าคนหันหน้าเข้าหาหุ่นยนต์ มุมควรจะเป็น PI (180 deg)
        # ถ้าคนหันหลังให้หุ่นยนต์ (เดินไปทางเดียวกับหุ่น) มุมควรจะเป็น 0
        final_yaw = person_yaw + math.pi / 2 # ปรับ Offset ให้ตรงกับทิศทาง Robot
        
        # Normalize มุมให้อยู่ในช่วง -PI ถึง PI
        final_yaw = math.atan2(math.sin(final_yaw), math.cos(final_yaw))

        return final_yaw # หน่วยเป็น Radian

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
            return 0.0

        # Use the median to eliminate outliers.
        return sorted(valid_ranges)[len(valid_ranges) // 2]

    def get_scripted_ground_truth(self):
        """Calculate a person's location in real time. (Ground Truth from Script)"""
        if self.start_sim_time is None:
            self.start_sim_time = self.get_clock().now()
            return -8.0, 1.0, 0.0 # start point (x, y, yaw)

        # 1. Calculate the time that has elapsed since the start (in seconds).
        now = self.get_clock().now()
        elapsed = (now - self.start_sim_time).nanoseconds / 1e9
        
        # 2. Handle the loop (loop lasting 28.6 seconds).
        t = elapsed % self.trajectory_duration

        # 3. Segments
        if 0.0 <= t < 13.3: # Move from (-8, 1) to (8, 1)
            ratio = t / 13.3
            x = -8.0 + (8.0 - (-8.0)) * ratio
            y = 1.0
            yaw = 0.0
        elif 13.3 <= t < 14.3: # Stop and turn around
            x = 8.0
            y = 1.0
            # rotate from 0 ไป 3.1416
            ratio = (t - 13.3) / 1.0
            yaw = 0.0 + (3.1416 - 0.0) * ratio
        elif 14.3 <= t < 27.6: # Move from (8, 1) to (-8, 1)
            ratio = (t - 14.3) / (27.6 - 14.3)
            x = 8.0 + (-8.0 - 8.0) * ratio
            y = 1.0
            yaw = 3.1416
        else: # 27.6 <= t < 28.6 Stop and turn around
            x = -8.0
            y = 1.0
            ratio = (t - 27.6) / (28.6 - 27.6)
            yaw = 3.1416 + (0.0 - 3.1416) * ratio

        return x, y, yaw

    # ──────────────────────────────────────────────────────────────────────
    # Publisher helpers
    # ──────────────────────────────────────────────────────────────────────

    def publish_social_zone_ellipse(self, x, y, yaw, index):
        """
        สร้างพื้นที่รูปไข่ (Social Zone) รอบตัวคน
        x, y: พิกัดในเฟรม Odom
        yaw: มุมการหันหน้าในหน่วย Radian
        index: ลำดับของคนที่ตรวจพบ (ใช้เป็น Marker ID)
        """
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "social_navigation"
        
        # 1. ใช้ index ที่รับมาเป็น id เพื่อให้ Marker ของแต่ละคนแยกกัน
        marker.id = index 
        
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # 2. กำหนดตำแหน่งและมุม (Pose)
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.02
        
        # แปลง Yaw เป็น Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        marker.pose.orientation = q

        # 3. กำหนดขนาด (Scale)
        marker.scale.x = 2.4  # ด้านหน้า-หลัง
        marker.scale.y = 1.6  # ด้านข้าง
        marker.scale.z = 0.05

        # 4. กำหนดสี
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)

        # 5. ตั้งค่า Lifetime
        marker.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

        # 6. ส่งข้อมูลเป็น MarkerArray
        marker_array = MarkerArray()
        marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

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
