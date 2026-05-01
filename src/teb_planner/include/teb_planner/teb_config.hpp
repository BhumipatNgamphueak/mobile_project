#ifndef TEB_PLANNER_TEB_CONFIG_HPP
#define TEB_PLANNER_TEB_CONFIG_HPP

namespace teb_planner {

struct TebConfig {
    // ── Trajectory ──────────────────────────────────────────────────────
    double dt_ref          = 0.3;   // reference Δt between poses (s)
    double dt_hysteresis   = 0.1;   // hysteresis when resampling
    int    min_samples     = 3;
    int    max_samples     = 100;

    // ── Robot (holonomic, yaw-locked) ───────────────────────────────────
    double max_vel_x       = 0.4;   // body-frame forward speed limit (m/s)
    double max_vel_y       = 0.4;   // body-frame lateral speed limit (m/s)
    double max_vel_theta   = 0.05;  // small allowance for numerical slack (rad/s)
    double acc_lim_x       = 0.5;
    double acc_lim_y       = 0.5;
    double acc_lim_theta   = 0.1;
    double yaw_locked_ref  = 0.0;   // reference yaw to hold (filled per cycle)
    bool   yaw_locked      = true;

    // ── Goal ────────────────────────────────────────────────────────────
    double xy_goal_tolerance = 0.2;

    // ── Obstacles (point obstacles only) ────────────────────────────────
    double min_obstacle_dist     = 0.30;  // hard clearance distance
    double inflation_dist        = 0.60;  // soft inflation cushion
    double obstacle_search_radius = 5.0;  // include only obstacles within this radius
    int    costmap_threshold     = 60;    // cells with cost ≥ this become obstacles
    int    costmap_downsample    = 2;     // include every Nth cell to limit edge count

    // ── Path attraction (via points) ────────────────────────────────────
    double via_point_separation  = 0.4;   // resample reference path at this spacing

    // ── Optimization ────────────────────────────────────────────────────
    int    no_inner_iterations = 5;
    int    no_outer_iterations = 4;

    double weight_max_vel_x      = 2.0;
    double weight_max_vel_y      = 2.0;
    double weight_max_vel_theta  = 1.0;
    double weight_acc_lim_x      = 1.0;
    double weight_acc_lim_y      = 1.0;
    double weight_acc_lim_theta  = 1.0;
    double weight_kinematics_yaw = 1000.0;  // high → yaw locked
    double weight_optimaltime    = 1.0;
    double weight_shortest_path  = 0.0;
    double weight_obstacle       = 50.0;
    double weight_inflation      = 0.1;
    double weight_viapoint       = 1.0;
    double weight_adapt_factor   = 2.0;     // multiplied into weights each outer iter

    // ── Lookahead ───────────────────────────────────────────────────────
    double max_global_plan_lookahead_dist = 5.0;
};

}  // namespace teb_planner

#endif  // TEB_PLANNER_TEB_CONFIG_HPP
