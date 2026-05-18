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
from collections import deque

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

# ── Tracking constants ─────────────────────────────────────────────────────
# Max world-frame distance to associate a detection to an existing track.
_MATCH_DIST = 1.5   # m
# A track coasts on its fitted velocity this long without a fresh
# detection before being deleted — bridges YOLO dropouts so the egg
# doesn't blink out and respawn ("jumps / disappears").
_TRACK_TIMEOUT = 0.8   # s


class PersonTracker:
    """Per-person trajectory buffer with least-squares velocity estimation.

    Why not atan2 of per-frame deltas?
        At 30 fps a 1.2 m/s human moves ~0.04 m/frame.  LiDAR range jitter
        and YOLO bbox-center wobble inject ~±0.02 m of position noise, so
        the direction of any single frame-to-frame delta has 20–30° of
        random error → the egg spins.

    Approach
        Keep the last ~1 s of (t, x, y) samples.  Fit a line via ordinary
        least squares — `vx = Σ(t−t̄)(x−x̄) / Σ(t−t̄)²`, same for vy.
        Heading = atan2(vy, vx).  Every sample contributes, so a single
        bad frame is averaged out instead of flipping the direction.
    """

    # Window widened to ~3 walking-animation cycles.  The actor mesh has
    # no collision, so distance falls back to bbox-height monocular depth;
    # that estimate oscillates ~0.4 m at the ~2 Hz walk cadence.  A
    # sub-cycle window aliases that periodic noise into a fake radial
    # velocity → the egg's heading flips ("spinning").
    HISTORY_SECONDS = 1.5    # window for line fit
    MAX_HISTORY     = 45     # cap buffer length
    MIN_FIT_POINTS  = 5      # need ≥ this many samples
    MIN_FIT_DISP    = 0.20   # first→last sample must travel ≥ this (m)
    MIN_FIT_SPEED   = 0.15   # fitted speed must exceed this (m/s)
    VEL_ALPHA       = 0.5    # EMA on fitted (vx, vy) for extra smoothness
    POS_ALPHA       = 0.5    # EMA on raw position before the fit

    def __init__(self, t_sec: float, x: float, y: float):
        self.history: deque[tuple[float, float, float]] = deque()
        self.history.append((t_sec, x, y))
        self.x        = x       # smoothed position (egg anchor + fit input)
        self.y        = y
        self.x_raw    = x       # latest raw detection (used for ID matching)
        self.y_raw    = y
        self.vx       = 0.0
        self.vy       = 0.0
        self.yaw      = 0.0
        self.last_seen = t_sec

    def update(self, t_sec: float, x: float, y: float) -> None:
        # EMA-smooth the raw position to suppress the periodic bbox-height
        # depth noise.  EMA is linear so the fitted velocity (slope) is
        # preserved; only high-frequency jitter is removed.
        self.x_raw     = x
        self.y_raw     = y
        self.x         = self.POS_ALPHA * self.x + (1.0 - self.POS_ALPHA) * x
        self.y         = self.POS_ALPHA * self.y + (1.0 - self.POS_ALPHA) * y
        self.last_seen = t_sec
        x, y = self.x, self.y

        # Drop samples older than HISTORY_SECONDS and cap buffer length.
        cutoff = t_sec - self.HISTORY_SECONDS
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        self.history.append((t_sec, x, y))
        while len(self.history) > self.MAX_HISTORY:
            self.history.popleft()

        if len(self.history) < self.MIN_FIT_POINTS:
            return

        # Require enough span between first and last sample so the fit
        # has real signal, not just noise around a stationary person.
        x0, y0 = self.history[0][1], self.history[0][2]
        xN, yN = self.history[-1][1], self.history[-1][2]
        if math.hypot(xN - x0, yN - y0) < self.MIN_FIT_DISP:
            return

        # Ordinary least-squares slope of x vs t and y vs t.
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

        # Light EMA on the velocity vector for visual stability.  This is
        # safe because the regression already averages out high-frequency
        # noise; the EMA only smooths slow drift across windows.
        self.vx = self.VEL_ALPHA * self.vx + (1.0 - self.VEL_ALPHA) * new_vx
        self.vy = self.VEL_ALPHA * self.vy + (1.0 - self.VEL_ALPHA) * new_vy

        if self.speed >= self.MIN_FIT_SPEED:
            self.yaw = math.atan2(self.vy, self.vx)
        # Otherwise leave self.yaw unchanged — stationary person keeps heading.

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def predict(self, t_sec: float) -> tuple[float, float]:
        """Where this track is expected to be at t_sec (constant-velocity).

        Matching against the prediction rather than the last seen point
        keeps IDs stable when two people pass close together.
        """
        dt = t_sec - self.last_seen
        return self.x + self.vx * dt, self.y + self.vy * dt


class SocialCostmapNode(Node):

    def __init__(self):
        super().__init__('social_costmap_node')

        # ----------------- Parameters -----------------
        self.declare_parameter('scan_topic',            '/scan')
        self.declare_parameter('image_topic',           '/front_camera/image_raw')
        self.declare_parameter('odom_topic',            '/odom')
        self.declare_parameter('input_costmap_topic',   '/local_costmap')
        self.declare_parameter('output_costmap_topic',  '/local_costmap_social')
        self.declare_parameter('odom_frame',            'odom')
        self.declare_parameter('detection_range',       8.0)
        self.declare_parameter('min_front_scale',       1.2)
        self.declare_parameter('velocity_factor',       0.6)
        self.declare_parameter('sigma_back',            0.4)
        self.declare_parameter('sigma_side',            0.5)
        self.declare_parameter('peak_cost',             85)
        self.declare_parameter('alpha',                 0.8)   # velocity EMA
        self.declare_parameter('min_vel_threshold',     0.1)

        self.scan_topic           = self.get_parameter('scan_topic').value
        self.image_topic          = self.get_parameter('image_topic').value
        self.odom_topic           = self.get_parameter('odom_topic').value
        self.input_costmap_topic  = self.get_parameter('input_costmap_topic').value
        self.output_costmap_topic = self.get_parameter('output_costmap_topic').value
        self.odom_frame           = self.get_parameter('odom_frame').value
        self.detection_range      = self.get_parameter('detection_range').value
        self.min_front_scale      = self.get_parameter('min_front_scale').value
        self.velocity_factor      = self.get_parameter('velocity_factor').value
        self.sigma_back           = self.get_parameter('sigma_back').value
        self.sigma_side           = self.get_parameter('sigma_side').value
        self.peak_cost            = int(self.get_parameter('peak_cost').value)
        self.vel_alpha            = self.get_parameter('alpha').value
        self.min_vel_threshold    = self.get_parameter('min_vel_threshold').value

        # Camera mount: urdf camera_joint origin x=0.40 forward of
        # base_link; added back in _pixel_range_to_world.
        self.CAMERA_FWD_OFFSET = 0.40
        # LiDAR returns within this depth of the chosen one are the same
        # (human) surface; farther = background wall, discarded.
        self.CLUSTER_DEPTH = 0.5
        # Boxes with an edge within this many px of the image border are
        # clipped (person half out of frame) → bearing unreliable.
        self.EDGE_CLIP_PX = 8

        # ----------------- Subscribers -----------------
        # Image queue depth 1: YOLO-on-CPU is slower than the camera, a
        # deeper queue just feeds stale frames and adds latency.
        self.create_subscription(LaserScan,    self.scan_topic,          self._scan_callback,    10)
        self.create_subscription(Image,        self.image_topic,         self._image_callback,    1)
        self.create_subscription(Odometry,     self.odom_topic,          self._odom_callback,    10)
        self.create_subscription(OccupancyGrid,self.input_costmap_topic, self._costmap_callback, 10)

        # ----------------- Publishers -----------------
        self.costmap_pub = self.create_publisher(OccupancyGrid, self.output_costmap_topic, 10)
        self.marker_pub  = self.create_publisher(MarkerArray,   '/detected_humans',        10)

        # ----------------- Internal state -----------------
        self._latest_scan:    LaserScan     | None = None
        self._latest_image:   Image         | None = None
        self._latest_costmap: OccupancyGrid | None = None

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
        self.fov   = 1.047   # 60 deg horizontal FOV
        # Focal length from pinhole model: fx = (W/2) / tan(fov/2)
        self.fx    = (self.img_w / 2.0) / math.tan(self.fov / 2.0)
        self.uc    = self.img_w / 2.0

        self.get_logger().info(
            f'SocialCostmapNode started. '
            f'In: {self.input_costmap_topic}  Out: {self.output_costmap_topic}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _odom_callback(self, msg: Odometry):
        self.current_robot_x = msg.pose.pose.position.x
        self.current_robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _costmap_callback(self, msg: OccupancyGrid):
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
                source=cv_image, classes=[0], conf=0.5, verbose=False, device='cpu')

            base       = self._latest_costmap
            width      = base.info.width
            height     = base.info.height
            origin_x   = base.info.origin.position.x
            origin_y   = base.info.origin.position.y
            resolution = base.info.resolution
            # Copy into a mutable list; values are signed int8 (−1=unknown, 0-100=cost)
            grid_data  = list(base.data)

            all_markers = MarkerArray()
            now = self.get_clock().now()
            t_now = now.nanoseconds / 1e9

            # Detection reflects the image-capture instant, not now
            # (YOLO-on-CPU + queue latency ≈ 100-300 ms).  Stamp tracks
            # with the image time and draw at the velocity-predicted pose
            # at t_now so the Gaussian overlays the live human.
            istamp = self._latest_image.header.stamp
            t_img  = istamp.sec + istamp.nanosec * 1e-9
            if not (0.0 < t_img <= t_now):
                t_img = t_now
            t_sec  = t_img

            # ── Step 1: collect valid detections ──────────────────────────
            # Each entry: (wx, wy, x1, y1, x2, y2)
            detections: list[tuple] = []
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])

                # Skip border-clipped boxes: person half out of frame,
                # box centre biased inward → bearing wrong.  Let the
                # track coast instead of jumping to a false FOV-edge pose.
                if x1 <= self.EDGE_CLIP_PX or x2 >= self.img_w - self.EDGE_CLIP_PX:
                    continue

                u_center = (x1 + x2) / 2
                bbox_h   = max(1, y2 - y1)
                mono = self.fx * 1.7 / bbox_h if bbox_h > 30 else 0.0
                lidar = self._get_distance_from_scan(u_center, mono)
                dist_m = lidar if lidar > 0.0 else mono
                if not (0 < dist_m <= self.detection_range):
                    continue
                wx, wy = self._pixel_range_to_world(u_center, dist_m)
                detections.append((wx, wy, x1, y1, x2, y2))

            # ── Step 2: stable ID assignment ──────────────────────────────
            # Match detections to history by nearest smoothed world position
            # instead of using YOLO's frame-index (which can swap between frames).
            person_ids = self._assign_ids(
                [(wx, wy) for wx, wy, *_ in detections], t_now)
            current_frame_ids = set(person_ids)

            # ── Step 3: update trackers + overlay Gaussians ───────────────
            # Each tracker maintains its own position history and does an
            # OLS line fit to derive a noise-resistant velocity vector.
            # The egg is drawn at the velocity-predicted (latency-
            # compensated) pose; orientation comes from the same fit.
            for (wx, wy, x1, y1, x2, y2), person_id in zip(detections, person_ids):
                tracker = self.trackers.get(person_id)
                if tracker is None:
                    tracker = PersonTracker(t_sec, wx, wy)
                    self.trackers[person_id] = tracker
                else:
                    tracker.update(t_sec, wx, wy)

                speed_for_costmap = tracker.speed
                if speed_for_costmap < self.min_vel_threshold:
                    speed_for_costmap = 0.0

                # Latency-compensated anchor: smoothed pose extrapolated
                # along the fitted velocity from image-capture time to now.
                ex, ey = tracker.predict(t_now)

                self._add_human_gaussian_to_grid(
                    grid_data, width, height,
                    origin_x, origin_y, resolution,
                    ex, ey, tracker.yaw, speed_for_costmap)

                all_markers.markers.append(
                    self._create_egg_marker(
                        ex, ey, tracker.yaw, person_id, speed_for_costmap))

            # ── Step 4: coast undetected tracks, expire only timed-out ────
            for old_id in list(self.trackers.keys()):
                if old_id in current_frame_ids:
                    continue
                tracker = self.trackers[old_id]
                if t_now - tracker.last_seen <= _TRACK_TIMEOUT:
                    cx, cy = tracker.predict(t_now)
                    cspeed = tracker.speed
                    if cspeed < self.min_vel_threshold:
                        cspeed = 0.0
                    self._add_human_gaussian_to_grid(
                        grid_data, width, height,
                        origin_x, origin_y, resolution,
                        cx, cy, tracker.yaw, cspeed)
                    all_markers.markers.append(
                        self._create_egg_marker(
                            cx, cy, tracker.yaw, old_id, cspeed))
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

            fused                 = OccupancyGrid()
            fused.header.stamp    = base.header.stamp
            fused.header.frame_id = base.header.frame_id
            fused.info            = base.info
            fused.data            = array.array(
                'b', [self._clip_cost(v) for v in grid_data])
            self.costmap_pub.publish(fused)

        except Exception as e:
            self.get_logger().error(f'SocialCostmap error: {e}')

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
            YOLO's (arbitrary) box ordering, itself a source of ID swaps.
        """
        available = set(self.trackers.keys())
        next_id   = max(self.trackers.keys(), default=-1) + 1
        ids: list[int | None] = [None] * len(positions)

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
    def _add_human_gaussian_to_grid(self, grid_data, width, height,
                                    origin_x, origin_y, resolution,
                                    h_x, h_y, h_yaw, h_vel):
        sigma_front = self.min_front_scale + h_vel * self.velocity_factor
        sigma_back  = self.sigma_back
        sigma_y     = self.sigma_side

        max_influence = max(sigma_front, sigma_back, sigma_y) * 3.0
        cells_range   = int(max_influence / resolution)

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

                # Rotate world-frame offset into human-local frame
                local_x =  dx * resolution
                local_y =  dy * resolution
                rx =  local_x * cos_y + local_y * sin_y
                ry = -local_x * sin_y + local_y * cos_y

                sig_x      = sigma_front if rx > 0 else sigma_back
                g          = math.exp(-((rx * rx) / (sig_x * sig_x)
                                        + (ry * ry) / (sigma_y * sigma_y)))
                human_cost = int(self.peak_cost * g)
                if human_cost <= 0:
                    continue

                idx     = ny * width + nx
                current = max(0, grid_data[idx])   # treat −1 (unknown) as 0
                if current < human_cost:
                    grid_data[idx] = human_cost

    @staticmethod
    def _clip_cost(v: int) -> int:
        if v < 0:
            return -1          # preserve unknown cells
        return min(v, 100)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _pixel_range_to_world(self, u_pixel: float,
                               range_m: float) -> tuple[float, float]:
        """Convert a camera pixel column + LiDAR range to odom-frame (x, y).

        Previous version used `z_cam = range_m` (range as depth), which
        over-estimates forward distance by 1/cos(θ) for off-centre detections.
        Correct formula: depth = range * cos(θ), lateral = range * sin(θ).
        """
        # +CAMERA_FWD_OFFSET: camera is mounted 0.40 m ahead of base_link
        # (urdf camera_joint); detections are projected from base_link.
        theta   = math.atan2(u_pixel - self.uc, self.fx)  # + = right in image
        x_robot =  range_m * math.cos(theta) + self.CAMERA_FWD_OFFSET
        y_robot = -range_m * math.sin(theta)   # left (camera +x = robot −y)

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

        # atan2 pinhole angle; negate because LiDAR CCW = camera left = small u
        angle = -math.atan2(u_center - self.uc, self.fx)
        idx   = int((angle - scan.angle_min) / scan.angle_increment)
        if not (0 <= idx < len(scan.ranges)):
            return 0.0

        # Wider window (~17 samples ≈ 17°) so a person several metres out
        # is fully captured even with some bearing error.
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
            # Trust the height prior for *which* object: nearest return to
            # it, but only if something actually lies near it.  Tolerance
            # grows with range (monocular depth error ∝ range).
            tol  = max(1.0, 0.30 * mono_hint)
            near = [r for r in valid if abs(r - mono_hint) <= tol]
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
        back_scale  = self.sigma_back
        side_scale  = self.sigma_side
        # Precompute rotation — avoids recomputing sin/cos 100× per marker
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        def asym_pt(theta: float) -> Point:
            r_x = front_scale if math.cos(theta) > 0 else back_scale
            px  = r_x * math.cos(theta)
            py  = side_scale * math.sin(theta)
            return Point(
                x=float(px * cos_y - py * sin_y + x),
                y=float(px * sin_y + py * cos_y + y),
                z=0.02)

        centre = Point(x=float(x), y=float(y), z=0.02)
        num    = 100
        for i in range(num):
            t1 = 2.0 * math.pi * i / num
            t2 = 2.0 * math.pi * (i + 1) / num
            marker.points.append(centre)
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
