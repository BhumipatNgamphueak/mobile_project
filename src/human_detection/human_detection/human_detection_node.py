#!/usr/bin/env python3
"""
human_detection_node.py
=======================
Detect humans from camera image (YOLOv8) + LiDAR range and publish
their estimated poses and an asymmetric Gaussian social costmap.

Inputs
------
/scan                   sensor_msgs/LaserScan      LiDAR (10 Hz)
/front_camera/image_raw sensor_msgs/Image          RGB camera 640×480 (30 Hz)
/odom                   nav_msgs/Odometry          robot pose

Outputs
-------
/detected_humans        visualization_msgs/MarkerArray   egg-shaped social zones
/detected_human_poses   geometry_msgs/PoseArray          odom-frame human poses
/local_costmap          nav_msgs/OccupancyGrid           Gaussian social costmap
"""

import array
import math
from collections import deque

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import ColorRGBA, Header
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

# ── Tracking constants (mirrors social_costmap_node) ──────────────────────
_MATCH_DIST = 1.5   # m
# A track survives this long without a fresh detection, coasting on its
# fitted velocity.  YOLO at conf=0.5 routinely drops a person for a frame
# or two (motion blur, partial FOV, occlusion); deleting on the first miss
# made the egg blink out and respawn with a new ID + zeroed velocity —
# exactly the "jumps / disappears" behaviour.
_TRACK_TIMEOUT = 0.8   # s


class PersonTracker:
    """Per-person trajectory buffer with least-squares velocity estimation.

    See social_costmap_node.PersonTracker for full rationale — duplicated
    here so both nodes can run standalone without a shared module.
    """

    # Window widened to ~3 walking-animation cycles.  The actor mesh has
    # no collision, so distance falls back to bbox-height monocular depth
    # (see _run_detection); that estimate oscillates ~0.4 m at the ~2 Hz
    # walk cadence.  A sub-cycle window would alias that periodic noise
    # into a fake radial velocity → the egg's heading flips ("spinning").
    HISTORY_SECONDS = 1.5
    MAX_HISTORY     = 45
    MIN_FIT_POINTS  = 5
    MIN_FIT_DISP    = 0.20
    MIN_FIT_SPEED   = 0.15
    VEL_ALPHA       = 0.5
    POS_ALPHA       = 0.5    # EMA on raw position before the fit

    def __init__(self, t_sec: float, x: float, y: float):
        self.history: deque[tuple[float, float, float]] = deque()
        self.history.append((t_sec, x, y))
        self.x  = x          # smoothed position (egg anchor + fit input)
        self.y  = y
        self.x_raw = x       # latest raw detection (used for ID matching)
        self.y_raw = y
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.last_seen = t_sec

    def update(self, t_sec: float, x: float, y: float) -> None:
        # EMA-smooth the raw position to suppress the periodic bbox-height
        # depth noise.  EMA is linear so the fitted velocity (slope) is
        # preserved; only high-frequency jitter is removed.
        self.x_raw = x
        self.y_raw = y
        self.x = self.POS_ALPHA * self.x + (1.0 - self.POS_ALPHA) * x
        self.y = self.POS_ALPHA * self.y + (1.0 - self.POS_ALPHA) * y
        self.last_seen = t_sec
        x, y = self.x, self.y

        cutoff = t_sec - self.HISTORY_SECONDS
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        self.history.append((t_sec, x, y))
        while len(self.history) > self.MAX_HISTORY:
            self.history.popleft()

        if len(self.history) < self.MIN_FIT_POINTS:
            return

        x0, y0 = self.history[0][1], self.history[0][2]
        xN, yN = self.history[-1][1], self.history[-1][2]
        if math.hypot(xN - x0, yN - y0) < self.MIN_FIT_DISP:
            return

        n = len(self.history)
        t_mean = sum(p[0] for p in self.history) / n
        x_mean = sum(p[1] for p in self.history) / n
        y_mean = sum(p[2] for p in self.history) / n
        sxt = syt = stt = 0.0
        for t, xp, yp in self.history:
            dt   = t - t_mean
            sxt += dt * (xp - x_mean)
            syt += dt * (yp - y_mean)
            stt += dt * dt
        if stt < 1e-9:
            return

        new_vx = sxt / stt
        new_vy = syt / stt
        self.vx = self.VEL_ALPHA * self.vx + (1.0 - self.VEL_ALPHA) * new_vx
        self.vy = self.VEL_ALPHA * self.vy + (1.0 - self.VEL_ALPHA) * new_vy

        if self.speed >= self.MIN_FIT_SPEED:
            self.yaw = math.atan2(self.vy, self.vx)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def predict(self, t_sec: float) -> tuple[float, float]:
        """Where this track is expected to be at t_sec (constant-velocity).

        Matching against the prediction rather than the last seen point
        keeps IDs stable when two people pass close together — each
        track's expected position separates along its own heading even
        while the raw detections briefly overlap.
        """
        dt = t_sec - self.last_seen
        return self.x + self.vx * dt, self.y + self.vy * dt


class HumanDetectionNode(Node):

    def __init__(self):
        super().__init__('human_detection_node')

        # ----------------- Parameters -----------------
        self.declare_parameter('scan_topic',        '/scan')
        self.declare_parameter('image_topic',       '/front_camera/image_raw')
        self.declare_parameter('odom_frame',        'odom')
        self.declare_parameter('detection_range',   8.0)
        self.declare_parameter('costmap_resolution', 0.05)
        self.declare_parameter('costmap_width',      15.0)
        self.declare_parameter('costmap_height',     15.0)

        self.scan_topic      = self.get_parameter('scan_topic').value
        self.image_topic     = self.get_parameter('image_topic').value
        self.odom_frame      = self.get_parameter('odom_frame').value
        self.detection_range = self.get_parameter('detection_range').value

        resolution    = self.get_parameter('costmap_resolution').value
        self.resolution   = resolution
        self.width_cells  = int(self.get_parameter('costmap_width').value  / resolution)
        self.height_cells = int(self.get_parameter('costmap_height').value / resolution)

        # Gaussian shape parameters
        self.min_front_scale   = 1.2
        self.velocity_factor   = 0.6
        self.sigma_back        = 0.4
        self.sigma_side        = 0.5
        self.peak_cost         = 85
        self.vel_alpha         = 0.8    # velocity EMA weight on prev
        self.min_vel_threshold = 0.1

        # Camera mount: urdf camera_joint origin x=0.40 (forward of
        # base_link).  Detections are projected from base_link, so this is
        # added back in _pixel_range_to_world.
        self.CAMERA_FWD_OFFSET = 0.40
        # A LiDAR cluster within this depth of the chosen return is taken
        # to be the same (human) surface; farther = background wall.
        self.CLUSTER_DEPTH = 0.5
        # Boxes whose left/right edge is within this many px of the image
        # border are clipped (person half out of frame) → bearing unsafe.
        self.EDGE_CLIP_PX = 8

        # ----------------- Subscribers -----------------
        # Image queue depth 1: YOLO-on-CPU is slower than the 30 Hz camera,
        # so a deeper queue just feeds the detector stale frames (adds
        # latency on top of the prediction we already compensate for).
        self.create_subscription(LaserScan, self.scan_topic,  self._scan_callback,  10)
        self.create_subscription(Image,     self.image_topic, self._image_callback,  1)
        self.create_subscription(Odometry,  '/odom',          self._odom_callback,  10)

        # ----------------- Publishers -----------------
        self.marker_pub  = self.create_publisher(MarkerArray,  '/detected_humans',       10)
        self.pose_pub    = self.create_publisher(PoseArray,    '/detected_human_poses',  10)
        self.costmap_pub = self.create_publisher(OccupancyGrid,'/local_costmap',         10)

        # ----------------- Internal state -----------------
        self._latest_scan:  LaserScan | None = None
        self._latest_image: Image     | None = None

        self.current_robot_x   = 0.0
        self.current_robot_y   = 0.0
        self.current_robot_yaw = 0.0

        # Per-person robust trackers (least-squares velocity estimation).
        self.trackers: dict[int, PersonTracker] = {}

        # ----------------- Vision -----------------
        self.bridge             = CvBridge()
        self.human_detect_model = YOLO('yolov8n.pt')

        self.img_w = 640
        self.img_h = 480
        self.fov   = 1.047   # 60 deg
        self.fx    = (self.img_w / 2.0) / math.tan(self.fov / 2.0)
        self.uc    = self.img_w / 2.0

        self.get_logger().info('HumanDetectionNode started.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _image_callback(self, msg: Image):
        self._latest_image = msg
        self._run_detection()

    def _odom_callback(self, msg: Odometry):
        self.current_robot_x = msg.pose.pose.position.x
        self.current_robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------
    def _run_detection(self):
        if self._latest_scan is None or self._latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                self._latest_image, desired_encoding='bgr8')
            results = self.human_detect_model.predict(
                source=cv_image, classes=[0], conf=0.5, verbose=False, device='cpu')

            now   = self.get_clock().now()
            t_now = now.nanoseconds / 1e9

            # The detection describes where the human was when the *image*
            # was captured, not now — YOLO-on-CPU + queue latency is
            # 100-300 ms.  Timestamp the track with the image stamp and
            # anchor the egg at the velocity-predicted pose at t_now, so
            # the blob sits on the human instead of trailing it.
            istamp = self._latest_image.header.stamp
            t_img  = istamp.sec + istamp.nanosec * 1e-9
            if not (0.0 < t_img <= t_now):      # bad/zero stamp → no extrap
                t_img = t_now
            t_sec  = t_img

            # Costmap geometry follows the robot (rolling window)
            origin_x = self.current_robot_x - (self.width_cells  * self.resolution) / 2.0
            origin_y = self.current_robot_y - (self.height_cells * self.resolution) / 2.0
            grid_data = [0] * (self.width_cells * self.height_cells)

            all_markers  = MarkerArray()
            pose_array   = PoseArray()
            pose_array.header = Header(
                frame_id=self.odom_frame,
                stamp=now.to_msg())

            # ── Step 1: collect valid detections ──────────────────────────
            detections: list[tuple] = []
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])

                # Reject boxes clipped at the image border: the person is
                # only partly in frame, so the box centre is pulled inward
                # and the derived bearing is wrong.  Skipping lets the
                # track coast on its velocity instead of snapping to a
                # false position at the FOV edge.
                if x1 <= self.EDGE_CLIP_PX or x2 >= self.img_w - self.EDGE_CLIP_PX:
                    continue

                u_center = (x1 + x2) / 2
                bbox_h   = max(1, y2 - y1)
                # Monocular height prior (no wall confusion, ~biased but
                # bounded).  LiDAR then refines it; if LiDAR only sees the
                # side wall, we keep the prior instead of the wall range.
                mono = self.fx * 1.7 / bbox_h if bbox_h > 30 else 0.0
                lidar = self._get_distance_from_scan(u_center, mono)
                dist_m = lidar if lidar > 0.0 else mono
                if not (0 < dist_m <= self.detection_range):
                    continue
                wx, wy = self._pixel_range_to_world(u_center, dist_m)
                detections.append((wx, wy, x1, y1, x2, y2))

            # ── Step 2: stable ID assignment ──────────────────────────────
            person_ids        = self._assign_ids(
                [(wx, wy) for wx, wy, *_ in detections], t_now)
            current_frame_ids = set(person_ids)

            # ── Step 3: update trackers + build outputs ───────────────────
            for (wx, wy, x1, y1, x2, y2), person_id in zip(detections, person_ids):
                tracker = self.trackers.get(person_id)
                if tracker is None:
                    tracker = PersonTracker(t_sec, wx, wy)
                    self.trackers[person_id] = tracker
                else:
                    tracker.update(t_sec, wx, wy)

                speed = tracker.speed
                if speed < self.min_vel_threshold:
                    speed = 0.0

                # Latency-compensated anchor: smoothed pose extrapolated
                # along the fitted velocity from the image-capture time up
                # to now.  Removes both the EMA lag and the YOLO pipeline
                # delay so the egg overlays the live human.
                ex, ey = tracker.predict(t_now)

                self._add_gaussian_to_grid(
                    grid_data, origin_x, origin_y,
                    ex, ey, tracker.yaw, speed)

                all_markers.markers.append(
                    self._create_egg_marker(
                        ex, ey, tracker.yaw, person_id, speed))

                p = Pose()
                p.position.x    = ex
                p.position.y    = ey
                p.position.z    = 0.9
                p.orientation.w = 1.0
                pose_array.poses.append(p)

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image,
                            f'ID:{person_id} V:{speed:.1f}m/s',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── Step 4: coast undetected tracks, expire only timed-out ────
            for old_id in list(self.trackers.keys()):
                if old_id in current_frame_ids:
                    continue
                tracker = self.trackers[old_id]
                if t_now - tracker.last_seen <= _TRACK_TIMEOUT:
                    # Brief YOLO dropout — keep the egg alive at the
                    # velocity-predicted pose so it doesn't blink/respawn.
                    cx, cy = tracker.predict(t_now)
                    cspeed = tracker.speed
                    if cspeed < self.min_vel_threshold:
                        cspeed = 0.0
                    self._add_gaussian_to_grid(
                        grid_data, origin_x, origin_y,
                        cx, cy, tracker.yaw, cspeed)
                    all_markers.markers.append(
                        self._create_egg_marker(
                            cx, cy, tracker.yaw, old_id, cspeed))
                    cp = Pose()
                    cp.position.x    = cx
                    cp.position.y    = cy
                    cp.position.z    = 0.9
                    cp.orientation.w = 1.0
                    pose_array.poses.append(cp)
                else:
                    dm = Marker()
                    dm.header.frame_id = self.odom_frame
                    dm.ns     = 'human_social_zones'
                    dm.id     = old_id
                    dm.action = Marker.DELETE
                    all_markers.markers.append(dm)
                    del self.trackers[old_id]

            if all_markers.markers:
                self.marker_pub.publish(all_markers)
            self.pose_pub.publish(pose_array)

            costmap                  = OccupancyGrid()
            costmap.header.stamp     = self._latest_image.header.stamp
            costmap.header.frame_id  = self.odom_frame
            costmap.info.resolution  = self.resolution
            costmap.info.width       = self.width_cells
            costmap.info.height      = self.height_cells
            costmap.info.origin.position.x = origin_x
            costmap.info.origin.position.y = origin_y
            costmap.data             = array.array('b', grid_data)
            self.costmap_pub.publish(costmap)

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

    # ------------------------------------------------------------------
    # Stable person ID assignment
    # ------------------------------------------------------------------
    def _assign_ids(self, positions: list[tuple[float, float]],
                    t_now: float) -> list[int]:
        """Globally-greedy matching of detections to predicted track poses.

        Two fixes over the previous per-detection greedy loop:
          * Match against each track's *predicted* position at t_now
            (constant-velocity), not its last seen point, so crossing
            people don't steal each other's IDs.
          * Resolve all candidate pairs in ascending-distance order
            instead of detection order — the result no longer depends on
            YOLO's (arbitrary) box ordering, which was itself a source of
            frame-to-frame ID swaps.
        """
        available = set(self.trackers.keys())
        next_id   = max(self.trackers.keys(), default=-1) + 1
        ids: list[int | None] = [None] * len(positions)

        # All (distance, det_idx, pid) pairs within the gate, closest first.
        pairs = []
        for di, (wx, wy) in enumerate(positions):
            for pid in available:
                sx, sy = self.trackers[pid].predict(t_now)
                d = math.hypot(wx - sx, wy - sy)
                if d < _MATCH_DIST:
                    pairs.append((d, di, pid))
        pairs.sort(key=lambda p: p[0])

        used_pid: set[int] = set()
        for _d, di, pid in pairs:
            if ids[di] is not None or pid in used_pid:
                continue
            ids[di] = pid
            used_pid.add(pid)

        for di in range(len(positions)):
            if ids[di] is None:
                ids[di] = next_id
                next_id += 1

        return ids  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Gaussian overlay
    # ------------------------------------------------------------------
    def _add_gaussian_to_grid(self, grid_data,
                               origin_x, origin_y,
                               h_x, h_y, h_yaw, h_vel):
        sigma_front = self.min_front_scale + h_vel * self.velocity_factor
        sigma_back  = self.sigma_back
        sigma_y     = self.sigma_side

        max_influence = max(sigma_front, sigma_back, sigma_y) * 3.0
        cells_range   = int(max_influence / self.resolution)

        human_cx = int((h_x - origin_x) / self.resolution)
        human_cy = int((h_y - origin_y) / self.resolution)
        cos_y    = math.cos(h_yaw)
        sin_y    = math.sin(h_yaw)

        for dx in range(-cells_range, cells_range + 1):
            for dy in range(-cells_range, cells_range + 1):
                nx = human_cx + dx
                ny = human_cy + dy
                if not (0 <= nx < self.width_cells and 0 <= ny < self.height_cells):
                    continue

                local_x =  dx * self.resolution
                local_y =  dy * self.resolution
                rx =  local_x * cos_y + local_y * sin_y
                ry = -local_x * sin_y + local_y * cos_y

                sig_x      = sigma_front if rx > 0 else sigma_back
                g          = math.exp(-((rx * rx) / (sig_x * sig_x)
                                        + (ry * ry) / (sigma_y * sigma_y)))
                human_cost = int(self.peak_cost * g)
                if human_cost <= 0:
                    continue

                idx = ny * self.width_cells + nx
                if grid_data[idx] < human_cost:
                    grid_data[idx] = human_cost

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _pixel_range_to_world(self, u_pixel: float,
                               range_m: float) -> tuple[float, float]:
        """Correct pinhole projection: depth = range * cos(θ), not range.

        The camera is mounted CAMERA_FWD_OFFSET ahead of base_link
        (urdf camera_joint origin x=0.40); without this term every human
        is reported 0.4 m short, which on a crossing target reads as the
        egg sitting off the person.
        """
        theta   = math.atan2(u_pixel - self.uc, self.fx)
        x_robot =  range_m * math.cos(theta) + self.CAMERA_FWD_OFFSET
        y_robot = -range_m * math.sin(theta)

        ryaw    = self.current_robot_yaw
        world_x = (self.current_robot_x
                   + x_robot * math.cos(ryaw) - y_robot * math.sin(ryaw))
        world_y = (self.current_robot_y
                   + x_robot * math.sin(ryaw) + y_robot * math.cos(ryaw))
        return world_x, world_y

    def _get_distance_from_scan(self, u_center: float,
                                mono_hint: float) -> float:
        """LiDAR range at the camera bearing, refined toward mono_hint.

        Returns 0.0 (→ caller falls back to the monocular estimate) when
        the LiDAR sees nothing consistent with the monocular prior — at
        far range / FOV edges the wide window straddles the arena wall
        and a plain nearest-return would lock the egg onto the wall.
        """
        scan = self._latest_scan
        if scan is None:
            return 0.0

        angle = -math.atan2(u_center - self.uc, self.fx)
        idx   = int((angle - scan.angle_min) / scan.angle_increment)
        if not (0 <= idx < len(scan.ranges)):
            return 0.0

        # Wider window (~17 samples ≈ 17° at 1°/ray) so a person 0.5 m
        # wide at several metres is fully captured even with bearing error.
        window = 8
        valid  = [
            scan.ranges[i]
            for i in range(idx - window, idx + window + 1)
            if 0 <= i < len(scan.ranges)
            and scan.range_min <= scan.ranges[i] <= scan.range_max
        ]
        if not valid:
            return 0.0

        if mono_hint > 0.0:
            # Trust the height prior for *which* object: pick the return
            # closest to it, but only if something is actually near it.
            # Tolerance grows with range (monocular depth error ∝ range).
            tol      = max(1.0, 0.30 * mono_hint)
            near     = [r for r in valid if abs(r - mono_hint) <= tol]
            if not near:
                return 0.0          # only wall/other in view → use mono
            center = min(near, key=lambda r: abs(r - mono_hint))
        else:
            center = min(valid)     # no prior: nearest coherent object

        cluster = [r for r in valid if abs(r - center) <= self.CLUSTER_DEPTH]
        return sum(cluster) / len(cluster)

    def _create_egg_marker(self, x: float, y: float, yaw: float,
                            person_id: int, velocity: float) -> Marker:
        marker              = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = 'human_social_zones'
        marker.id              = person_id
        marker.type            = Marker.TRIANGLE_LIST
        marker.action          = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color           = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4)

        front_scale = self.min_front_scale + velocity * self.velocity_factor
        cos_y       = math.cos(yaw)
        sin_y       = math.sin(yaw)

        def asym_pt(theta: float) -> Point:
            r_x = front_scale if math.cos(theta) > 0 else self.sigma_back
            px  = r_x * math.cos(theta)
            py  = self.sigma_side * math.sin(theta)
            return Point(
                x=float(px * cos_y - py * sin_y + x),
                y=float(px * sin_y + py * cos_y + y),
                z=0.02)

        centre = Point(x=float(x), y=float(y), z=0.02)
        for i in range(100):
            t1 = 2.0 * math.pi * i / 100
            t2 = 2.0 * math.pi * (i + 1) / 100
            marker.points.append(centre)
            marker.points.append(asym_pt(t1))
            marker.points.append(asym_pt(t2))

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
