#!/usr/bin/env python3
"""
human_detection_node.py  —  SKELETON
=====================================
Detect humans from LiDAR scan and/or camera image and publish
their estimated poses for path planning and visualisation.

Inputs
------
/scan                   sensor_msgs/LaserScan      2-D LiDAR (bridged from /lidar)
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
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


class HumanDetectionNode(Node):

    def __init__(self):
        super().__init__('human_detection_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('scan_topic',      '/scan')
        self.declare_parameter('image_topic',     '/front_camera/image_raw')
        self.declare_parameter('odom_frame',      'odom')
        self.declare_parameter('detection_range', 8.0)

        self.scan_topic      = self.get_parameter('scan_topic').value
        self.image_topic     = self.get_parameter('image_topic').value
        self.odom_frame      = self.get_parameter('odom_frame').value
        self.detection_range = self.get_parameter('detection_range').value

        # ── Subscribers ───────────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic,
            self._scan_callback, 10)

        self.image_sub = self.create_subscription(
            Image, self.image_topic,
            self._image_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.marker_pub = self.create_publisher(
            MarkerArray, '/detected_humans', 10)

        self.pose_pub = self.create_publisher(
            PoseArray, '/detected_human_poses', 10)

        # ── Internal state ────────────────────────────────────────────────
        self._latest_scan:  LaserScan | None = None
        self._latest_image: Image     | None = None

        self.get_logger().info('HumanDetectionNode started (skeleton)')

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _scan_callback(self, msg: LaserScan):
        """Called at ~10 Hz with every new LiDAR scan."""
        self._latest_scan = msg
        self._run_detection()

    def _image_callback(self, msg: Image):
        """Called at ~30 Hz with every new camera frame."""
        self._latest_image = msg
        # TODO (optional): run YOLO here; results can be fused in _run_detection()

    # ──────────────────────────────────────────────────────────────────────
    # Detection pipeline  ← IMPLEMENT THIS
    # ──────────────────────────────────────────────────────────────────────

    def _run_detection(self):
        """
        Main detection entry point, called whenever a new scan arrives.

        Suggested approaches
        --------------------
        LiDAR-only  : leg detector (two circular blobs ~0.3 m apart in scan)
        Camera-only : YOLOv8 bounding box → depth estimate via known geometry
        Fusion      : match LiDAR clusters to YOLO detections for robust 3-D pose

        Expected output
        ---------------
        A list of (x, y) positions in the *odom* frame.
        Use self._laser_to_odom(range, angle) to convert a scan hit to odom.
        Then call self._publish(detections).
        """
        if self._latest_scan is None:
            return

        # TODO: replace stub with real detection algorithm
        detections: list[tuple[float, float]] = self._stub_detect()

        self._publish(detections)

    def _stub_detect(self) -> list[tuple[float, float]]:
        """
        Stub — returns an empty list.
        Replace with your detection logic; return [(x1,y1), (x2,y2), …].
        """
        return []

    # ──────────────────────────────────────────────────────────────────────
    # Helper — coordinate conversion
    # ──────────────────────────────────────────────────────────────────────

    def _laser_to_odom(self, distance: float, angle_rad: float) -> tuple[float, float]:
        """
        Convert a single LiDAR return (range, angle in lidar_link frame) to
        the odom frame.

        NOTE: This stub does NOT apply the lidar_link → odom transform.
              Implement properly using tf2_ros.Buffer / TransformListener,
              or use the robot's current odometry pose directly.

        Parameters
        ----------
        distance  : range measurement [m]
        angle_rad : beam angle in lidar_link frame [rad]

        Returns
        -------
        (x, y) in odom frame [m]
        """
        import math
        # TODO: apply actual lidar_link → odom TF lookup
        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)
        return x, y

    # ──────────────────────────────────────────────────────────────────────
    # Publisher helpers
    # ──────────────────────────────────────────────────────────────────────

    def _publish(self, detections: list[tuple[float, float]]):
        """Publish MarkerArray + PoseArray from a list of (x, y) positions."""
        stamp = self.get_clock().now().to_msg()
        header = Header(frame_id=self.odom_frame, stamp=stamp)

        # --- PoseArray ---
        pose_array = PoseArray(header=header)
        for x, y in detections:
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = 0.9          # mid-body height
            p.orientation.w = 1.0
            pose_array.poses.append(p)
        self.pose_pub.publish(pose_array)

        # --- MarkerArray (cylinders) ---
        marker_array = MarkerArray()

        # Delete old markers first
        delete_all = Marker()
        delete_all.header = header
        delete_all.ns = 'detected_humans'
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i, (x, y) in enumerate(detections):
            m = Marker()
            m.header = header
            m.ns = 'detected_humans'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.9
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 1.8
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 0.8
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)


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
