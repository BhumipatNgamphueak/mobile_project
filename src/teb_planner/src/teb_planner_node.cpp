// teb_planner_node — ROS2 entry point for the TEB local planner.
//
// Subscribes
//   /odom            nav_msgs/Odometry
//   /global_path     nav_msgs/Path
//   /local_costmap   nav_msgs/OccupancyGrid    (lethal cells → point obstacles)
//
// Publishes
//   /cmd_vel             geometry_msgs/TwistStamped
//   /planned_path        nav_msgs/Path                  (optimized band)
//   /obstacle_polygons   visualization_msgs/MarkerArray (point obstacles)

#include "teb_planner/optimal_planner.hpp"
#include "teb_planner/obstacles.hpp"
#include "teb_planner/pose_se2.hpp"
#include "teb_planner/teb_config.hpp"

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Core>

#include <cmath>
#include <memory>
#include <vector>

namespace teb_planner {

class TebPlannerNode : public rclcpp::Node {
public:
    TebPlannerNode()
    : rclcpp::Node("teb_planner_node"),
      cfg_(),
      planner_(std::make_unique<OptimalPlanner>(&cfg_))
    {
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&TebPlannerNode::odomCallback, this, std::placeholders::_1));
        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/global_path", 10,
            std::bind(&TebPlannerNode::pathCallback, this, std::placeholders::_1));
        cm_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/local_costmap", 10,
            std::bind(&TebPlannerNode::costmapCallback, this, std::placeholders::_1));

        cmd_pub_     = create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);
        plan_pub_    = create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
        polygon_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/obstacle_polygons", 10);

        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&TebPlannerNode::replan, this));

        RCLCPP_INFO(get_logger(), "teb_planner_node (g2o, holonomic, yaw-locked) started");
    }

private:
    // ── State ──────────────────────────────────────────────────────────
    TebConfig                       cfg_;
    std::unique_ptr<OptimalPlanner> planner_;

    nav_msgs::msg::Odometry::SharedPtr     odom_msg_;
    nav_msgs::msg::Path::SharedPtr         path_msg_;
    nav_msgs::msg::OccupancyGrid::SharedPtr costmap_msg_;
    bool reached_goal_ = false;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr          path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr cm_sub_;

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr      cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                   plan_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  polygon_pub_;
    rclcpp::TimerBase::SharedPtr                                        timer_;

    // ── Callbacks ──────────────────────────────────────────────────────
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) { odom_msg_ = msg; }
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) { costmap_msg_ = msg; }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (!path_msg_ || msg->poses.empty()) {
            path_msg_ = msg;
            planner_->clearWarmState();
            reached_goal_ = false;
            return;
        }
        // If goal differs, treat as a brand-new plan.
        const auto& new_goal = msg->poses.back().pose.position;
        const auto& old_goal = path_msg_->poses.back().pose.position;
        const bool same_goal = std::abs(new_goal.x - old_goal.x) < 0.01 &&
                               std::abs(new_goal.y - old_goal.y) < 0.01;
        if (!same_goal) {
            planner_->clearWarmState();
            reached_goal_ = false;
        }
        path_msg_ = msg;
    }

    // ── Helpers ────────────────────────────────────────────────────────
    static double extractYaw(const geometry_msgs::msg::Quaternion& q) {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    void publishStop() {
        geometry_msgs::msg::TwistStamped cmd;
        cmd.header.stamp    = now();
        cmd.header.frame_id = "base_link";
        cmd_pub_->publish(cmd);
    }

    // Costmap → point obstacles inside a window around the robot.
    void extractObstacles(const Eigen::Vector2d& robot_pos,
                          ObstacleContainer& out) const {
        out.clear();
        if (!costmap_msg_) return;

        const auto& cm   = *costmap_msg_;
        const int   w    = cm.info.width;
        const int   h    = cm.info.height;
        const double res = cm.info.resolution;
        const double ox  = cm.info.origin.position.x;
        const double oy  = cm.info.origin.position.y;
        if (res <= 0.0 || w <= 0 || h <= 0) return;

        const double radius   = cfg_.obstacle_search_radius;
        const int    margin   = static_cast<int>(radius / res) + 1;
        const int    cx       = static_cast<int>((robot_pos.x() - ox) / res);
        const int    cy       = static_cast<int>((robot_pos.y() - oy) / res);
        const int    i_lo     = std::max(0, cx - margin);
        const int    i_hi     = std::min(w, cx + margin + 1);
        const int    j_lo     = std::max(0, cy - margin);
        const int    j_hi     = std::min(h, cy + margin + 1);
        const int    step     = std::max(1, cfg_.costmap_downsample);
        const int    threshold = cfg_.costmap_threshold;
        const double r2       = radius * radius;

        for (int j = j_lo; j < j_hi; j += step) {
            for (int i = i_lo; i < i_hi; i += step) {
                const int8_t v = cm.data[j * w + i];
                if (v < threshold) continue;
                const double wx = ox + (i + 0.5) * res;
                const double wy = oy + (j + 0.5) * res;
                const double dx = wx - robot_pos.x();
                const double dy = wy - robot_pos.y();
                if (dx * dx + dy * dy > r2) continue;
                out.push_back(std::make_shared<PointObstacle>(wx, wy));
            }
        }
    }

    // Trim global path to the lookahead window starting at the closest pose.
    std::vector<PoseSE2> extractInitialPlan(const Eigen::Vector2d& robot_pos,
                                            std::vector<Eigen::Vector2d>& via_out) const {
        std::vector<PoseSE2> plan;
        via_out.clear();
        if (!path_msg_ || path_msg_->poses.size() < 2) return plan;

        // Find closest waypoint.
        int closest = 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < path_msg_->poses.size(); ++i) {
            const auto& p = path_msg_->poses[i].pose.position;
            const double dx = p.x - robot_pos.x();
            const double dy = p.y - robot_pos.y();
            const double d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; closest = static_cast<int>(i); }
        }

        // Collect waypoints within lookahead.
        const double L = cfg_.max_global_plan_lookahead_dist;
        for (size_t i = closest; i < path_msg_->poses.size(); ++i) {
            const auto& p = path_msg_->poses[i].pose.position;
            const double dx = p.x - robot_pos.x();
            const double dy = p.y - robot_pos.y();
            if (std::hypot(dx, dy) > L) break;
            plan.emplace_back(p.x, p.y, 0.0);
        }
        if (plan.size() < 2) return plan;

        // Build via points by resampling at via_point_separation along the plan.
        double accumulated = 0.0;
        for (size_t i = 1; i < plan.size(); ++i) {
            const double seg = (plan[i].position() - plan[i - 1].position()).norm();
            accumulated += seg;
            if (accumulated >= cfg_.via_point_separation) {
                via_out.push_back(plan[i].position());
                accumulated = 0.0;
            }
        }
        return plan;
    }

    // ── Main loop ──────────────────────────────────────────────────────
    void replan() {
        if (!odom_msg_) return;
        if (!path_msg_ || path_msg_->poses.size() < 2) return;

        const auto& rp  = odom_msg_->pose.pose.position;
        const auto& rq  = odom_msg_->pose.pose.orientation;
        const double yaw = extractYaw(rq);
        const Eigen::Vector2d robot_pos(rp.x, rp.y);

        // Goal check.
        const auto& gp = path_msg_->poses.back().pose.position;
        if (std::hypot(rp.x - gp.x, rp.y - gp.y) < cfg_.xy_goal_tolerance) {
            if (!reached_goal_) {
                RCLCPP_INFO(get_logger(), "Goal reached.");
                reached_goal_ = true;
            }
            publishStop();
            return;
        }
        reached_goal_ = false;

        // Yaw-lock reference is the current robot yaw — set every cycle so
        // it tracks if the user manually rotates the robot between commands.
        cfg_.yaw_locked_ref = yaw;

        std::vector<Eigen::Vector2d> via_points;
        std::vector<PoseSE2> initial_plan = extractInitialPlan(robot_pos, via_points);
        if (initial_plan.size() < 2) {
            publishStop();
            return;
        }

        const PoseSE2 start(robot_pos.x(), robot_pos.y(), yaw);
        const PoseSE2 goal (initial_plan.back().x(), initial_plan.back().y(), yaw);

        ObstacleContainer obstacles;
        extractObstacles(robot_pos, obstacles);
        publishObstacleMarkers(obstacles);

        if (!planner_->plan(start, goal, initial_plan, obstacles, via_points)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "TEB optimization failed");
            publishStop();
            return;
        }

        publishOptimizedPath();
        publishCmdVel(yaw);
    }

    // ── Outputs ────────────────────────────────────────────────────────
    void publishCmdVel(double yaw) {
        double vx_w = 0.0, vy_w = 0.0;
        if (!planner_->getVelocityCommand(vx_w, vy_w)) {
            publishStop();
            return;
        }
        // World → body frame (yaw locked, so this is fixed each call).
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        const double vx_b =  vx_w * c + vy_w * s;
        const double vy_b = -vx_w * s + vy_w * c;

        // Saturate to the body-frame limits.
        const double sx = std::min(1.0, cfg_.max_vel_x / std::max(std::abs(vx_b), 1e-9));
        const double sy = std::min(1.0, cfg_.max_vel_y / std::max(std::abs(vy_b), 1e-9));
        const double sat = std::min(sx, sy);

        geometry_msgs::msg::TwistStamped cmd;
        cmd.header.stamp     = now();
        cmd.header.frame_id  = "base_link";
        cmd.twist.linear.x   = vx_b * sat;
        cmd.twist.linear.y   = vy_b * sat;
        cmd.twist.angular.z  = 0.0;
        cmd_pub_->publish(cmd);
    }

    void publishOptimizedPath() {
        const auto& poses = planner_->getOptimizedPoses();
        if (poses.empty()) return;

        nav_msgs::msg::Path msg;
        msg.header.stamp    = now();
        msg.header.frame_id = path_msg_ ? path_msg_->header.frame_id : "map";
        msg.poses.reserve(poses.size());

        for (const auto& p : poses) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = msg.header;
            ps.pose.position.x    = p.x();
            ps.pose.position.y    = p.y();
            ps.pose.orientation.z = std::sin(p.theta() / 2.0);
            ps.pose.orientation.w = std::cos(p.theta() / 2.0);
            msg.poses.push_back(ps);
        }
        plan_pub_->publish(msg);
    }

    void publishObstacleMarkers(const ObstacleContainer& obstacles) {
        visualization_msgs::msg::MarkerArray ma;
        const std::string frame = costmap_msg_ ? costmap_msg_->header.frame_id : "map";
        const auto stamp = now();

        // Clear previous markers.
        visualization_msgs::msg::Marker clr;
        clr.header.frame_id = frame;
        clr.action = visualization_msgs::msg::Marker::DELETEALL;
        ma.markers.push_back(clr);

        // Single SPHERE_LIST containing all point obstacles.
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame;
        m.header.stamp    = stamp;
        m.ns              = "teb_obstacles";
        m.id              = 0;
        m.type            = visualization_msgs::msg::Marker::SPHERE_LIST;
        m.action          = visualization_msgs::msg::Marker::ADD;
        m.scale.x = m.scale.y = m.scale.z = 0.15;
        m.color.r = 1.0f; m.color.g = 0.2f; m.color.b = 0.2f; m.color.a = 0.9f;
        m.pose.orientation.w = 1.0;
        for (const auto& obs : obstacles) {
            const auto& c = obs->getCentroid();
            geometry_msgs::msg::Point p;
            p.x = c.x(); p.y = c.y(); p.z = 0.0;
            m.points.push_back(p);
        }
        ma.markers.push_back(m);
        polygon_pub_->publish(ma);
    }
};

}  // namespace teb_planner

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<teb_planner::TebPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
