# **Elastic Band Planner with Object Detection**

### Team Members and Roles

| Name | Roles |
|-------------|---------|
| Pavaris Asawakijtananont | Planner, evaluation |
| Anuwit Intet | Object Detection |
| Bhumipat Ngamphueak | Experiment setup, evaluation |

### Table of Contents
1. [Introduction](#1-introduction)
2. [Project Scope](#2-project-scope)
   - [Methodology](#methodology)
   - [GMM With Elastic Band](#gmm-with-elastic-band)
3. [Elastic Band Planner](#3-elastic-band-planner)
   - [Multiple Convex Hull for Object Representation (MCCH)](#multiple-convex-hull-for-object-representation-mcch)
   - [Pre Routing](#pre-routing)
4. [Human Detection](#4-human-detection)
   - [Gaussian Mixture Model](#gaussian-mixture-model)
   - [Per-Person Velocity Estimation](#per-person-velocity-estimation)
6. [Experiment Design](#6-experiment-design)
   - [Perception](#perception)
   - [Planner and Integration](#planner-and-integration)
7. [Results](#7-results)
   - [Perception](#perception-1)
   - [Planner & System Integration](#planner--system-integration)
     - [Static Obstacle (Planner Only)](#static-obstacle-planner-only)
     - [Dynamic Obstacle (Planner Only)](#dynamic-obstacle-planner-only)
     - [Dynamic Obstacle (Integrated System)](#dynamic-obstacle-integrated-system)
8. [Discussion](#8-discussion)
  
## 1. Introduction

In real environments, a mobile robot constantly faces uncertainty from both its control system and its surroundings, which can deform the originally planned path into an inefficient (or unsafe) one. This problem becomes more severe in social settings, where moving people are themselves a source of uncertainty. Traditional local planners treat every obstacle as a static occupied cell — they do not exploit higher-level social information such as a pedestrian's position, heading, or velocity. As a consequence, the robot tends to react to a human only at the moment of collision rather than anticipating where the human will be a few seconds later.

This project addresses that gap by combining a **Elastic Band local planner** with a perception module that detects pedestrians, estimates their velocity, and projects an **asymmetric Gaussian "social zone"** in front of each person. The social zones are fused into the same local costmap that the planner already uses for static obstacles, so the elastic band naturally deforms around both the person and the space they are about to occupy.

<div align="center">
  <img src="./figures/cpteb_2.png" alt="CPTEB reference trajectories" width="400">
  <p><em>Reference result from the CPTEB paper. The top row shows trajectories produced by the baseline TEB planner; the bottom row shows the Collision-Prediction TEB (CPTEB). The robot is the blue circle with two wheels and the two pedestrians (P1, P2) are the cyan/magenta circles. Red circles mark collision zones, and blue arrows indicate the heading of the robot and each person.</em></p>
</div>

*Reference: The Collision Prediction Time Elastic Band (CPTEB) model.*



## 2. Project Scope
The goal of this project is to extend a traditional Elastic Band local planner so that it can use camera-based perception of pedestrians as additional planning information. The scope is:

1. Implement an Elastic Band local planner from scratch.
2. Detect humans and estimate their velocity with YOLOv8 + LiDAR depth refinement.
3. Use the estimated velocity to project an asymmetric (egg-shaped) Gaussian social cost so the planner can react to **where the human is going**, not only where the human currently is.
4. The platform is a **holonomic mecanum robot** (3-DoF base: `vx`, `vy`, `ωz`).

### Methodology

Perception and planning communicate through **one shared local costmap**: every detected human is written as elevated cost into the same grid that already holds the LiDAR obstacles. From the planner's point of view a moving person is just another (soft, oriented) obstacle, so the EB code does not need a separate "human list".

<!-- ```
LiDAR  ─► local_costmap_node ──► /local_costmap ─┐
                                                 ▼
Camera ─┐                                       social_costmap_node ──► /local_costmap_social
LiDAR  ─┼─► YOLO + tracker  ──► Gaussian per human ▲
Odom   ─┘                                          │
                                                 (fused)
                                                   │
Global path ─► path_planning_node (MCCH + EB optimiser) ──► /cmd_vel
``` -->

<tr>
  <td align="center">
    <img src="./figures/system_arch.png" width="800" alt="World 3 path comparison across lookaheads">
    <p><b>Figure W3.</b> World 3 (Cross Opposite, planner only) — recorded robot trajectories for L = 2, 5, 10 m. Larger lookaheads commit to a wider swerve (peak lateral offset grows from ~0.5 m at L = 2 to ~3.7 m at L = 10) and produce longer overall paths (15.1 → 17.0 → 18.2 m), because the band reacts to the second human earlier on the long-horizon plan.</p>
  </td>
</tr>

### GMM With Elastic Band

The EB planner has no concept of "human" — it only knows polygons. The integration therefore happens entirely inside the costmap:

1. For each tracked person, `social_costmap_node` paints the asymmetric Gaussian (forward lobe length ∝ `|v|`, heading = `atan2(v_y, v_x)`) into the same grid that already carries the LiDAR walls, using `max(existing, gaussian)` so static obstacles are never erased.
2. `path_planning_node` BFS-clusters every cell above the lethal threshold and runs MCCH on each cluster. The Gaussian's elevated cells become **one or two convex polygons that point along the human's velocity**.
3. From there the EB optimiser treats those polygons exactly like any other obstacle — the repulsion force `F^{obs}` (see §3) deforms the band around the human's *future* position, not just their current one.

<!-- Because the social channel is just an alternate cost layer, switching it on or off is a launch-file parameter (`cost_map_topic: /local_costmap` vs. `/local_costmap_social`) — this is how we separate the *planner-only* and *integrated* experiments later. -->

## 3. Elastic Band Planner

The band is a chain of waypoints `x_i` connected by virtual springs. Each interior node is updated by gradient descent under two forces:

$$
\Delta x_i \;=\; \eta\,\bigl(\,w_{\text{obs}}\,F^{\text{obs}}_i \;+\; w_{\text{smooth}}\,F^{\text{smooth}}_i\,\bigr)
$$

- **Smoothness / contraction** — pulls each node toward the midpoint of its neighbours:

$$
F^{\text{smooth}}_i \;=\; \tfrac{1}{2}\bigl(x_{i-1}+x_{i+1}\bigr) \;-\; x_i
$$

- **Obstacle repulsion** — non-zero only inside an inflation radius `d_inf`, pointing away from the nearest polygon boundary point `p^*`:

$$
F^{\text{obs}}_i \;=\;
\begin{cases}
\bigl(d_{\text{inf}} - d_i\bigr)\,\dfrac{x_i - p^*}{\lVert x_i - p^*\rVert} & \text{if } 0 < d_i < d_{\text{inf}} \\[4pt]
0 & \text{if } d_i \ge d_{\text{inf}}
\end{cases}
$$

Endpoints are fixed and `|Δx_i|` is clamped to `max_delta = 0.10 m/iter` so the band cannot teleport across narrow gaps. The band is anchored at the **actual robot pose** (not the closest global-path waypoint), otherwise the smoothness force would constantly drag the first node back into the obstacle whenever the robot deviates laterally.

<!-- Code: [`elastic_band.py`](src/path_planning/path_planning/elastic_band.py) (optimiser) and [`path_planning_node.py`](src/path_planning/path_planning/path_planning_node.py) (ROS plumbing). -->

### Multiple Convex Hull for Object Representation (MCCH)

To make obstacle distance cheap, occupied costmap cells are clustered (BFS, 8-connected) and each cluster is approximated by **convex polygons**. A single hull fails for concave clusters (L, U, T) because its diagonal edge swallows free space and the robot gets pinned inside a fake polygon. MCCH detects and fixes this by recursive splitting:

1. Compute the cluster's convex hull.
2. For each edge, let `e_mid` be its midpoint and measure `d = min_p ‖e_mid − p‖` over cluster points. Real edges → small `d`; fictitious edges → large `d`.
3. If `max(d) > split_threshold` (0.3 m), split the cluster along that edge direction at its midpoint and recurse on the two halves. Splitting along (not perpendicular to) the edge is robust: for an L-shape the two arms project onto opposite halves of the diagonal.
4. Stop when no edge exceeds the threshold.

Result: a small set of convex hulls that follows the real occupied geometry, so the repulsion force sees true free space.

### Pre Routing

Gradient descent cannot escape a polygon that fully contains a band node — the obstacle forces on the two sides cancel and the band stays pinned (symmetric force lock-in). nav2's TEB sidesteps this by exploring multiple homotopy classes; we instead deterministically push trapped nodes to **whichever side has the shorter detour**:

1. For each polygon, find the contiguous run of band nodes inside it.
2. Build a local "band direction" `b̂` from the nodes just before and after that run.
3. Project the polygon vertices onto `b̂⊥` and measure how far the polygon extends to each side (`max_left`, `max_right`).
4. Pick the side with the smaller extent and shift all trapped nodes by `extent + d_inf + 0.1 m` along that perpendicular.


## 4. Human Detection

<!-- [`social_costmap_node.py`](src/human_detection/human_detection/social_costmap_node.py) turns camera + LiDAR into an asymmetric Gaussian social cost that is fused into the local costmap. Pipeline per frame:

1. **Detect** people with YOLOv8n on the front RGB camera (640 × 480, 60° HFOV). Boxes within 8 px of the image border are rejected (bearing is biased when the person is half out of frame).
2. **Recover depth** by combining a monocular height prior with a LiDAR range:

$$
d_{\text{mono}} \;=\; \frac{f_x \cdot 1.7}{h_{\text{bbox}}}, \qquad
d \;=\; \text{mean}\bigl\{r_k \in \text{LiDAR window} : |r_k - d_{\text{mono}}| \le \text{tol}\bigr\}
$$

with `tol = max(1 m, 0.3 · d_mono)`. The monocular prior anchors *which* return is the human, so the LiDAR cannot accidentally lock onto the back wall.

3. **Project** to the `odom` frame with the pinhole model (`depth = d · cos θ`, `lateral = d · sin θ`), correcting for the camera's 0.40 m forward offset from `base_link`.
4. **Track and estimate velocity** with a per-person `PersonTracker` (see [Per-Person Velocity Estimation](#per-person-velocity-estimation)).
5. **Render** an asymmetric Gaussian around each tracked person and merge it into `/local_costmap_social`. -->

### Gaussian Mixture Model

<!-- Each human is rendered as a single 2-D Gaussian **oriented along its velocity vector** and **asymmetric front-to-back**. In the human-local frame `(r_x, r_y)`:

$$
\text{cost}(r_x, r_y) \;=\; C_{\text{peak}} \cdot \exp\!\left(-\,\frac{r_x^{2}}{\sigma_x^{2}} \;-\; \frac{r_y^{2}}{\sigma_y^{2}}\right)
$$

$$
\sigma_x \;=\;
\begin{cases}
\sigma_{\text{front}} \;=\; \sigma_{\min} + k_v\,\lVert v\rVert & r_x > 0 \;(\text{ahead}) \\
\sigma_{\text{back}} & r_x \le 0 \;(\text{behind})
\end{cases}
\qquad
\sigma_y \;=\; \sigma_{\text{side}}
$$

with `σ_min = 1.2 m`, `k_v = 0.6 s`, `σ_back = 0.4 m`, `σ_side = 0.5 m`, `C_peak = 85`. The forward lobe stretches with speed; the rear lobe stays compact. The planner then sees a longer forbidden zone ahead of a fast walker than ahead of a stationary one, with **zero changes** to the EB repulsion code.

> **Naming note.** We call this module **GMM** following the project's terminology, but strictly it is *one* asymmetric Gaussian per human, not a probabilistic mixture model. -->

### Per-Person Velocity Estimation

<!-- Frame-to-frame `atan2` deltas are too noisy: at 30 fps a 1.2 m/s walker moves ~0.04 m per frame while LiDAR/bbox jitter is ~±0.02 m, giving 20–30° of random direction error → the egg visibly spins. Instead, each tracker keeps the last ~1.5 s of `(t, x, y)` samples and fits an OLS line:

$$
v_x \;=\; \frac{\sum_k (t_k - \bar t)(x_k - \bar x)}{\sum_k (t_k - \bar t)^{2}}, \qquad
v_y \;=\; \frac{\sum_k (t_k - \bar t)(y_k - \bar y)}{\sum_k (t_k - \bar t)^{2}}, \qquad
\psi \;=\; \operatorname{atan2}(v_y, v_x)
$$

Every sample contributes, so a single bad frame is averaged out instead of flipping the heading.

Two extra details that mattered in practice:
- **Stable IDs.** Detections are matched against each track's constant-velocity prediction at `t_now`, with all candidate pairs resolved in **ascending-distance order** (globally greedy). This stops IDs from swapping when two people cross.
- **Latency compensation.** YOLO-on-CPU + queue latency is 100–300 ms, so tracks are stamped with the *image-capture* time and the egg is drawn at the predicted pose at `t_now`. Missed detections coast on velocity for up to 0.8 s instead of being deleted, which prevents the egg from blinking out during brief YOLO dropouts. -->

<!-- ## 5. Environment Setup -->


## 6. Experiment Design
The experiments are organised in three stages: first we validate the **perception module** in isolation, then the **planner module** in isolation, and finally the **integrated** system that uses perception to feed the planner.

**Environment Setup**
- Robot maximum velocity: `v_max = 1.5 m/s`, `ω_max = 1.2 rad/s` (holonomic).
- Human walking speed: ~1.0 m/s (set per scenario in the SDF world file).
- Obstacles: rectangular static blocks (Static cases) and SDF-animated pedestrians (Dynamic cases).
- Goal tolerance: 0.4 m. Lookahead distances swept across {2, 5, 10, 15} m.

### Perception
This experiment validates the accuracy of the velocity-estimation model. The robot is **fixed in place** in both cases, and we vary the **distance between the robot and the human** to probe how range affects the detection.

1. **Standing human in front of the robot.** Used to measure position bias (the egg should sit on the person) and the false-velocity floor (a static person should report ~0 m/s, not jitter).
2. **Human walking past the robot.** Used to measure the fitted speed and heading against the ground-truth `/human_gt_poses` published by the simulator.

<div align="center">
  <img src="./figures/percep_case.png" alt="Perception test scenarios" width="400">
  <p><em>Perception test scenarios. <b>Left (1st situation):</b> the human stands at a fixed distance from the robot — used to measure position bias and the static-noise floor of the velocity estimator. <b>Right (2nd situation):</b> the human walks laterally past the stationary robot — used to measure the accuracy of the fitted speed and heading as a function of range.</em></p>
</div>


### Planner and Integration
We use three scenarios to test the planner: a **Static Obstacle** world for the MCCH + pre-routing logic, and two **Dynamic Obstacle** worlds (humans walking from the same side, and humans walking from the opposite side) for the planner-only and integrated runs.

<div align="center">
  <img src="./figures/planner_case.png" alt="Planner test scenarios B, C, D" width="600">
  <p><em>Planner test scenarios. <b>B — Static:</b> two rectangular blocks block the straight-line path between the robot (red triangle) and the goal (yellow star). <b>C — Dynamic, Opposite:</b> two pedestrians cross the corridor walking toward each other (one downward, one upward) while the robot drives left → right. <b>D — Dynamic, Same:</b> two pedestrians both walk upward, perpendicular to the robot's left → right direction of travel.</em></p>
</div>

1. **Static Obstacle (B)** — This scenario contains two static blocks that fully block the straight-line trajectory between the initial robot pose and the goal. It is used to stress the polygon-extraction + pre-routing pipeline and to measure how lookahead distance affects the detour length.

2. **Dynamic Obstacle, Same Direction (D)** — Two pedestrians walk **in the same direction** (both moving upward in the figure), perpendicular to the robot's path. They enter the corridor at slightly different `x` positions, so the robot has to weave through a moving "queue" rather than a single front. The forward Gaussian lobes of the two humans overlap, which can create a longer combined cost ridge that the band has to round.

3. **Dynamic Obstacle, Opposite Direction (C)** — Two pedestrians cross the corridor walking **toward each other** (one downward, one upward) while the robot drives left → right. This is the hardest case: the two humans intersect the robot's straight-line path at almost the same instant from opposite sides, and the egg of each one points across the corridor, so any avoidance manoeuvre by the robot has to commit early.

## 7. Results 
> [!NOTE]
> All result videos are stored in the `videos/` folder, organised by world (`world2/`, `world3/`, `world4/`).

### Perception

<!-- *(Perception accuracy numbers and plots are produced by `social_costmap_node` into the per-run `perception_log_*.csv` file and analysed offline; see [analysis/](analysis/) for the post-processing scripts.)* -->


### Planner & System Integration

| World | Look ahead | Path distance (m) | Reach Goal |
| :--- | :---: | :---: | :---: |
| **2 Static Obstacles (B)** | 2 | 9.9 | No |
| | 5 | 16.9 | Yes |
| | 10 | 16.8 | Yes |
| | 15 | 17 | Yes |
| **2 Static Obstacles (B) no_preroute** | 5 | 10.2 | No |
| **2 human with opposite directions (C)** | 2 | 15.1 | Yes |
| | 5 | 17 | Yes |
| | 10 | 18.2 | Yes |
| **2 human with same directions (D)** | 2 | 15.7 | Yes |
| | 5 | 16.5 | Yes |
| | 10 | 16.6 | Yes |



#### **Static Obstacle (Planner Only)**
  - Pre-routing is necessary for the band to escape the two blocking polygons. Without it, the band stays trapped and the robot **does not reach the goal** (see the `no_preroute` row in the table). With pre-routing, the band consistently detours around the closer side.
  - At the smallest lookahead (`L = 2 m`) the planner can also not pass the obstacle — the lookahead window is too short to even contain both blocks, so the band has no room to deform around them. From `L = 5 m` upward, the goal is reliably reached.
  - As the lookahead distance grows, the **path length increases monotonically** (16.9 → 16.8 → 17 m for L = 5 / 10 / 15). With a longer horizon the planner commits earlier to a wider detour around the polygons, which produces a smoother but slightly longer trajectory.

<table>
  <tr>
    <td>
      <video src="./videos/world2/world2_look_2.webm" width="100%" controls></video>
      <p align="center">World 2 — Static slalom, lookahead L = 2 m. Band horizon is too short; robot fails to detour past the first block.</p>
    </td>
    <td>
      <video src="./videos/world2/world2_look_5.webm" width="100%" controls></video>
      <p align="center">World 2 — Static slalom, L = 5 m. Pre-routing + EB find a clean S-shape detour; goal reached at 16.9 m.</p>
    </td>
        <td>
      <video src="./videos/world2/world2_look_10.webm" width="100%" controls></video>
      <p align="center">World 2 — Static slalom, L = 10 m. Longer horizon commits to the detour earlier; goal reached at 16.8 m.</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <video src="./videos/world2/world2_look_15.webm" width="100%" controls></video>
      <p align="center">World 2 — Static slalom, L = 15 m. Maximum tested lookahead; widest detour, goal reached at 17 m.</p>
    </td>
    <td>
      <video src="./videos/world2/world2_no_preroute_look5.webm" width="100%" controls></video>
      <p align="center">World 2 — Static slalom, L = 5 m, <b>pre-routing disabled</b>. Band stays pinned between the two polygons; robot fails to reach the goal (stops at ~10 m, see no_preroute row).</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="./figures/w2_p.png" width="800" alt="World 2 path comparison across lookaheads">
      <p><b>Figure W2.</b> World 2 (Static Slalom) — overhead view of the recorded robot trajectories for lookaheads L = 2, 5, 10, 15 m. The dashed grey line is the 15 m straight-line goal; coloured lines are the executed paths. At L = 2 the robot halts at ~4 m (no detour found); from L = 5 onwards the band consistently produces an S-shape detour with path ratio ~1.12–1.13.</p>
    </td>
  </tr>
  
  <tr>
    <td align="center">
      <img src="./figures/w3_p.png" width="800" alt="World 3 path comparison across lookaheads">
      <p><b>Figure W3.</b> World 3 (Cross Opposite, planner only) — recorded robot trajectories for L = 2, 5, 10 m. Larger lookaheads commit to a wider swerve (peak lateral offset grows from ~0.5 m at L = 2 to ~3.7 m at L = 10) and produce longer overall paths (15.1 → 17.0 → 18.2 m), because the band reacts to the second human earlier on the long-horizon plan.</p>
    </td>
  </tr>
  
  <tr>
    <td align="center">
      <img src="./figures/w4_p.png" width="800" alt="World 4 path comparison across lookaheads">
      <p><b>Figure W4.</b> World 4 (Cross Same Direction, planner only) — recorded robot trajectories for L = 2, 5, 10 m. All three lookaheads reach the goal; the path ratio stays close to 1.05–1.11 because both humans share the same heading, so a single sideways detour clears both at once.</p>
    </td>
  </tr>
</table>


#### **Dynamic Obstacle (Planner Only)**
  - In the dynamic-obstacle worlds the robot reaches the goal at **every tested lookahead**, but it does **not avoid the collision with the first human** — the LiDAR only sees the human as a point cloud at the moment of contact, so the planner has no early warning and the band only deforms once the human is already within the inflation radius.
  - Path length **grows monotonically with lookahead** for the same reason as in the static case: with a longer horizon the planner reacts to the upcoming human earlier and commits to a wider, smoother detour (15.1 → 17.0 → 18.2 m for L = 2 / 5 / 10 in World C).
  - After clearing the first human, the band can **shift to the opposite homotopy class**. Once human #1 is behind the robot, its contraction (smoothness) force pulls subsequent nodes back across the corridor faster than the residual repulsion from #1 pushes them away, so the band snaps from one side of the corridor to the other.

##### World C — Cross Opposite, planner only
<table>
  <tr>
    <td>
      <video src="./videos/world3/w3__delay6_look2.webm" width="100%" controls></video>
      <p align="center">World C — Cross Opposite, L = 2 m. Minimal detour; robot brushes past both humans (path 15.1 m, ratio 1.01).</p>
    </td>
    <td>
      <video src="./videos/world3/w3__delay6_look5.webm" width="100%" controls></video>
      <p align="center">World C — Cross Opposite, L = 5 m. Visible swerve to +y; band briefly settles in the upper homotopy class (path 17 m).</p>
    </td>
        <td>
      <video src="./videos/world3/w3__delay6_look10.webm" width="100%" controls></video>
      <p align="center">World C — Cross Opposite, L = 10 m. Largest swerve (~3.7 m lateral); long horizon plans the detour earlier (path 18.2 m).</p>
    </td>
  </tr>
</table>

##### World D — Cross Same Direction, planner only
<table>
  <tr>
    <td>
      <video src="./videos/world4/world4_look2.webm" width="100%" controls></video>
      <p align="center">World D — Cross Same Direction, L = 2 m. Single small detour clears both humans (path 15.7 m, ratio 1.05).</p>
    </td>
    <td>
      <video src="./videos/world4/world4_look5.webm" width="100%" controls></video>
      <p align="center">World D — Cross Same Direction, L = 5 m. Wider swerve to the same side; path 16.5 m.</p>
    </td>
        <td>
      <video src="./videos/world4/world4_look10.webm" width="100%" controls></video>
      <p align="center">World D — Cross Same Direction, L = 10 m. Earliest commitment to detour; path 16.6 m.</p>
    </td>
  </tr>
</table>

##### Free Environment — Integrated System (GMM + EB) sanity check

<table>
  <tr>
    <td>
      <video src="./videos/Toy_Exp1_look2.webm" width="100%" controls></video>
      <p align="center">Integrated GMM + EB planner in a free (obstacle-free) environment, L = 2 m. Short horizon — the path stays close to the straight line and only deforms when the egg enters the local window.</p>
    </td>
    <td>
      <video src="./videos/Toy_Exp1_look_5.webm" width="100%" controls></video>
      <p align="center">Integrated GMM + EB planner, L = 5 m. Smooth single-side detour around the human, driven by the asymmetric Gaussian.</p>
    </td>
        <td>
      <video src="./videos/Toy_Exp1_look10.webm" width="100%" controls></video>
      <p align="center">Integrated GMM + EB planner, L = 10 m. Long horizon makes the detour start much earlier and overshoots laterally — the path becomes visibly less efficient.</p>
    </td>
  </tr>
</table>

In the free-environment runs it is clear that the robot trajectory depends strongly on the lookahead distance. With a small lookahead the band only reacts to whatever is currently nearby, so the path stays tight; with a very large lookahead the far-horizon nodes deform quickly in response to the future Gaussian and that deformation propagates back to the near nodes, producing an over-eager swerve that the robot must then follow.

#### **Dynamic Obstacle (Integrated System)**

##### World C, D
<table>
  <tr>
    <td>
      <video src="./videos/world3/world3_egg_look5.webm" width="100%" controls></video>
      <p align="center">World C — Integrated GMM + EB, L = 5 m. Egg-shaped social zones are rendered around each detected human, but the robot still grazes the first human (see Discussion for FOV/height analysis).</p>
    </td>
    <td>
      <video src="./videos/world3/world3_no_egg_look5.webm" width="100%" controls></video>
      <p align="center">World C — Same scenario, social zones <b>disabled</b> (planner-only baseline) for comparison.</p>
    </td>
    <td>
      <video src="./videos/world4/world4_egg_look5.webm" width="100%" controls></video>
      <p align="center">World D — Integrated GMM + EB, L = 5 m. Both humans share the same heading; the merged Gaussian ridge produces a slightly tighter detour than the planner-only case.</p>
    </td>
  </tr>
</table>




<div align="center">
  <img src="./figures/GMM_res.png" alt="Path comparison: planner-only vs GMM-integrated" width="800">
  <p><em><b>Figure GMM.</b> Path comparison between the planner-only baseline (blue, "GT costmap") and the integrated GMM social costmap (dashed pink, "GMM social"), at lookahead L = 5 m. <b>Left — World 3 (Cross Opposite):</b> the social costmap shortens the path from 17.0 m to 16.0 m (Δ = −1.1 m) while still clearing both humans. <b>Right — World 4 (Cross Same Direction):</b> the social costmap reduces the path from 16.5 m to 15.3 m (Δ = −1.2 m), because the directional egg lets the planner commit to a single early sideways step instead of a late wide detour.</em></p>
</div>

In Worlds C and D the integrated system still **collides with the first human**, even though the egg shortens the path on average. We isolated the failure in the free-environment runs: the main cause is the **FOV of the camera** combined with the **robot's heading not pointing at the human** during the avoidance manoeuvre, which causes the detection to drop out exactly when it is needed most. The detailed root-cause analysis is in [Section 8 — Discussion](#8-discussion).




##  8. Discussion
Our framework reliably avoids **static unseen obstacles**, but it does **not** consistently avoid collisions with dynamic obstacles (pedestrians). The failure is not in the planner — when the social costmap is present, the band correctly deforms around it. The failure is in **keeping the social costmap alive at the moment of avoidance**. Three perception/setup factors compound at exactly the wrong time:

**1) Robot height — low camera loses the human.**
Our camera is mounted at ~1.0 m above the ground. At a 1–2 m range, the human's torso and face leave the camera's vertical field of view and only the legs remain in frame — at which point YOLO drops the detection (the bounding box no longer matches the "person" template). The social costmap then vanishes, the planner thinks the path is clear, and the robot drives into the human. By contrast, the reference paper uses a taller service-robot platform (camera at ~1.4–1.6 m) that still sees the torso/face at close range.

**2) Limited FOV — the human exits the view.**
The camera has a 60° horizontal cone, so a human is only visible within ±30° of the robot's heading. During an avoidance manoeuvre the robot turns away from the human (or the human moves sideways into the corridor), and the person leaves the FOV. The obstacle information is lost precisely when the planner needs it the most.

**3) Situation mismatch — open space vs. hotel hallway.**
The reference paper operates in narrow hotel corridors (~2 m wide). In that geometry, even when the robot moves sideways the hallway walls keep the human within the camera FOV, so detection is maintained and avoidance is smooth. Our simulation uses wide, open environments — the elastic band can swing the robot several metres to the side, the human exits the FOV entirely, the social costmap disappears, and the planner either overcorrects or collides. Counter-intuitively, **narrow space is an advantage** for this perception setup: the geometry constrains both the robot and the human to stay visible to each other.

> **Root cause.** All three issues compound at the same instant — the robot is close to the human, turning away from it, and in open space — creating a *detection blackout* exactly when the planner needs the social costmap most.

**4) Experiment setup — relative velocity matters.**
The relative velocity between the robot and the human directly sets the **interaction time** — the window in which the planner can both see the human and deform the band. If the robot is too slow, it cannot pass in front of the oncoming human and the band gets stuck against the moving Gaussian. If the robot is too fast, the band reacts only briefly to the human before the human is already behind it, so the avoidance is essentially ignored. Tuning the robot/human speed pair to a sensible interaction time is therefore a prerequisite for a fair evaluation of the social planner.

