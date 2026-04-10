#!/usr/bin/env python3
"""
test_path_publisher.py
======================
Publishes a pre-defined test path directly to /planned_path for isolated
RPP algorithm testing — bypasses path_planning_node entirely.

Each path is designed to exercise a specific set of RPP regulation features.

World geometry (odom frame, spawn at world -7,4):
  obstacle_A : (11,  1)   half_size 1.0 → inflated box x∈[9.65,12.35] y∈[-0.35,2.35]
  obstacle_B : ( 3, -9)   half_size 1.0 → inflated box x∈[1.65,4.35]  y∈[-10.35,-7.65]
  obstacle_C : ( 9, -8)   half_size 1.0 → inflated box x∈[7.65,10.35] y∈[-9.35,-6.65]
  Playfield  : x∈[-3,17]  y∈[-14,6]  (odom frame, wall inflation ≈ 0.45 m)

Available paths (select with --ros-args -p path:=<name>)
─────────────────────────────────────────────────────────
  scurve     Original S-curve — straight + curves + near obstacle_A proximity [default]
  straight   Pure straight-line east — baseline speed / heading tracking
  uturn      Rectangular U-turn — tight curvature regulation
  diagonal   NE diagonal then east — tests mecanum lateral correction
  slalom     Zigzag between pairs of obstacles B & C then north — full workout
  loop       Wide clockwise rectangular circuit — sustained path following

Usage
-----
  ros2 launch mecanum_robot_sim spawn_mecanum.launch.py test_path:=true

  # choose path at runtime:
  ros2 launch mecanum_robot_sim spawn_mecanum.launch.py \
      test_path:=true ros_args:="-p path:=uturn"

Parameters
----------
  path         (str,   'scurve')  which test path to publish
  frame_id     (str,   'odom')    frame for the path header
  publish_hz   (float,  1.0)      how often to re-publish (keeps path fresh)
  use_sim_time (bool,   true)
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


# ═══════════════════════════════════════════════════════════════════════════════
# PATH DEFINITIONS  (odom frame, all waypoints verified obstacle-free)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. S-CURVE (original) ─────────────────────────────────────────────────────
# Tests: straight sections, gentle + tight curves, proximity slowdown near A.
# obstacle_A (11,1): path end at (10.5, 0.5) → distance ≈ 1.12 m → inside d_prox ✓
PATH_SCURVE = [
    (0.0,  0.0),
    (3.0,  0.0),   # straight east
    (4.0,  0.5),
    (5.0,  1.5),
    (6.0,  2.0),   # gentle left curve
    (8.0,  2.0),   # straight at y=2
    (9.0,  1.0),
    (9.8,  0.5),   # tighter right curve
    (10.5, 0.5),   # goal — near obstacle_A (proximity heuristic activates)
]

# ── 2. STRAIGHT LINE ──────────────────────────────────────────────────────────
# Tests: pure heading tracking at full speed, no lateral correction needed.
# Verifies the robot drives in a straight line and stops cleanly at the goal.
PATH_STRAIGHT = [
    (0.0,  0.0),
    (3.0,  0.0),
    (6.0,  0.0),
    (9.0, -0.5),   # slight jog south so goal is clear of obstacle_A inflation
]

# ── 3. U-TURN ─────────────────────────────────────────────────────────────────
# Tests: curvature heuristic — two 90° turns back-to-back slow the robot.
# The turn radius at each corner is ≈ 1 m, well inside r_min → strong slowdown.
PATH_UTURN = [
    (0.0,  0.0),
    (5.0,  0.0),   # straight east
    (6.0,  1.0),
    (6.0,  3.0),   # 90° left turn north
    (5.0,  4.0),
    (2.0,  4.0),   # 90° left turn west (now heading back)
    (0.0,  4.0),   # goal — well clear of all obstacles
]

# ── 4. DIAGONAL (mecanum lateral test) ────────────────────────────────────────
# Tests: oblique motion — robot must use vy to track a diagonal path,
# exercising the mecanum lateral correction term (k_lat) in RPP.
# Then straightens east to approach the far side of the field.
PATH_DIAGONAL = [
    (0.0,  0.0),
    (2.0,  2.0),   # NE diagonal (45°)
    (4.0,  4.0),
    (6.0,  4.5),   # flatten out
    (8.0,  4.5),
    (10.0, 3.5),   # SE diagonal — drops below obstacle_A's y=2.35 inflation
    (10.0, 3.0),   # goal — x=10 < 9.65 east edge of A's inflation? wait…
    # obstacle_A inflated x∈[9.65,12.35], y∈[-0.35,2.35].
    # (10.0, 3.0): x=10 is inside x range but y=3.0 > 2.35 → clear ✓
]

# ── 5. SLALOM ─────────────────────────────────────────────────────────────────
# Tests: combined curvature + lateral correction.
# Weaves through the southern half of the field past obstacles B and C,
# then returns north — the hardest path for both regulation heuristics.
#
# Route:  spawn (0,0) → south to y≈-5 → east between B and C → east clear
#         → north up the east corridor → goal near (12,-4)
#
# Clearance verification:
#   obstacle_B (3,-9): inflated x∈[1.65,4.35] y∈[-10.35,-7.65]
#     → stay y > -7.65 when x∈[1.65,4.35]  (path at y=-5 or y=-6 → ✓)
#   obstacle_C (9,-8): inflated x∈[7.65,10.35] y∈[-9.35,-6.65]
#     → stay y > -6.65 or x < 7.65 or x > 10.35  (path at y=-5 → ✓)
PATH_SLALOM = [
    (0.0,  0.0),
    (1.0, -2.0),   # head south-east
    (1.0, -5.0),   # pass WEST of obstacle_B (x=1 < 1.65 inflation edge)
    (3.0, -6.5),   # jog east, stay above B's y=-7.65 south edge
    (6.0, -6.0),   # between B and C — both clear at this y
    (7.0, -5.5),   # approach C from west, staying north of y=-6.65
    (10.0,-5.0),   # pass NORTH of obstacle_C inflated zone (y=-5 > -6.65 ✓)
    (12.0,-4.0),   # east corridor
    (12.0,-1.0),   # turn north
    (12.0, 2.0),   # goal — east side of field, clear of obstacle_A (x=12 < 12.35, y=2=2.35 borderline)
    # safer: back off slightly
    (11.5, 2.5),   # final goal clear of A (y=2.5 > 2.35 ✓)
]

# ── 6. RECTANGULAR LOOP ───────────────────────────────────────────────────────
# Tests: sustained path following for 4 straight legs + 4 corners.
# Verifies the robot completes a full circuit without accumulated heading drift.
# Stays well inside the playfield, well clear of all obstacles.
PATH_LOOP = [
    (0.0,  0.0),
    (7.0,  0.0),   # east leg
    (7.0,  4.5),   # north leg
    (0.0,  4.5),   # west leg
    (0.0,  0.5),   # south leg back — goal just above start (avoids re-triggering)
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════
PATHS: dict[str, list[tuple[float, float]]] = {
    'scurve':   PATH_SCURVE,
    'straight': PATH_STRAIGHT,
    'uturn':    PATH_UTURN,
    'diagonal': PATH_DIAGONAL,
    'slalom':   PATH_SLALOM,
    'loop':     PATH_LOOP,
}

INTERP_STEP = 0.05   # metres between interpolated poses


def _interpolate(waypoints: list[tuple[float, float]],
                 step: float) -> list[tuple[float, float]]:
    """Linear interpolation between waypoints at fixed step distance."""
    pts: list[tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        n    = max(1, int(dist / step))
        for j in range(n):
            t = j / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    pts.append(waypoints[-1])
    return pts


class TestPathPublisher(Node):

    def __init__(self):
        super().__init__('test_path_publisher')

        self.declare_parameter('path',       'scurve')
        self.declare_parameter('frame_id',   'odom')
        self.declare_parameter('publish_hz', 1.0)

        path_name  = self.get_parameter('path').value
        frame_id   = self.get_parameter('frame_id').value
        publish_hz = self.get_parameter('publish_hz').value

        if path_name not in PATHS:
            self.get_logger().error(
                f"Unknown path '{path_name}'. Valid options: {list(PATHS.keys())}")
            raise ValueError(f"Unknown path: {path_name}")

        self._frame_id = frame_id
        self._pts = _interpolate(PATHS[path_name], INTERP_STEP)

        self.get_logger().info(
            f"Test path '{path_name}': {len(self._pts)} poses, "
            f"goal={PATHS[path_name][-1]}, re-publishing at {publish_hz} Hz")

        self._pub = self.create_publisher(Path, '/planned_path', 10)
        self.create_timer(1.0 / publish_hz, self._publish)

    def _publish(self):
        stamp  = self.get_clock().now().to_msg()
        header = Header(frame_id=self._frame_id, stamp=stamp)
        path   = Path(header=header)

        for x, y in self._pts:
            ps = PoseStamped(header=header)
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self._pub.publish(path)
        self.get_logger().debug(f'Published path ({len(path.poses)} poses)')


def main(args=None):
    rclpy.init(args=args)
    node = TestPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
