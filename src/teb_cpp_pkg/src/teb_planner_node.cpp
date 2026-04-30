#include <cmath>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <limits>
#include <nav_msgs/msg/path.hpp>
#include <vector>
#include <nav_msgs/msg/odometry.hpp>
#include <list>
#include "rclcpp/rclcpp.hpp"

// g2o includes
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <Eigen/Core>

// --- Define g2o Vertices and Edges ---

class VertexPose : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VertexPose() {}

  virtual void setToOriginImpl() override {
    _estimate.setZero();
  }

  virtual void oplusImpl(const double* update) override {
    _estimate[0] += update[0];
    _estimate[1] += update[1];
    _estimate[2] += update[2];
    _estimate[2] = std::atan2(std::sin(_estimate[2]), std::cos(_estimate[2]));
  }

  virtual bool read(std::istream& /*is*/) override { return false; }
  virtual bool write(std::ostream& /*os*/) const override { return false; }
};

class EdgeKinematics : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexPose, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeKinematics() {}

  virtual void computeError() override {
    const VertexPose* v1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexPose* v2 = static_cast<const VertexPose*>(_vertices[1]);
    _error = v2->estimate() - v1->estimate();
    _error[2] = std::atan2(std::sin(_error[2]), std::cos(_error[2]));
  }

  virtual bool read(std::istream& /*is*/) override { return false; }
  virtual bool write(std::ostream& /*os*/) const override { return false; }
};

// --- TebPlannerNode ---

class TebPlannerNode : public rclcpp::Node {
private:
  // ROS Subscriptions
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _odom_sub;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr _global_path_sub;
  
  // ROS Publishers
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr _cmd_vel_pub;
  
  // Timer
  rclcpp::TimerBase::SharedPtr _timer;

  // State
  nav_msgs::msg::Path _global_path;
  geometry_msgs::msg::Pose _robot_pose;
  bool _has_odom = false;
  bool _has_path = false;

  int _prune_index = 0;
  double lookahead_dist = 15.0;
  double v_max = 1.5;

  double euclidean(const geometry_msgs::msg::Pose &p1,
                   const geometry_msgs::msg::Pose &p2) {
    double dx = p1.position.x - p2.position.x;
    double dy = p1.position.y - p2.position.y;
    return std::sqrt(dx * dx + dy * dy);
  }

  std::unique_ptr<g2o::SparseOptimizer> setupOptimizer() {
    auto optimizer = std::make_unique<g2o::SparseOptimizer>();
    auto linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>>::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>>>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer->setAlgorithm(algorithm);
    return optimizer;
  }

  void _odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    _robot_pose = msg->pose.pose;
    _has_odom = true;
  }

  void _global_path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    _global_path = *msg;
    _prune_index = 0;
    _has_path = true;
  }

  void _replan() {
    if (!_has_odom || !_has_path || _global_path.poses.size() < 2) return;

    // Check if reached goal
    if (euclidean(_robot_pose, _global_path.poses.back().pose) < 0.4) {
      _pub_stop();
      return;
    }

    _prune_index = _find_closest_index();
    auto segment = _extract_local_segment();
    
    if (segment.size() >= 2) {
      optimize_band(segment);
      auto cmd = _compute_cmd_vel(segment);
      _cmd_vel_pub->publish(cmd);
    }
  }

  void _pub_stop() {
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = this->now();
    cmd.header.frame_id = "base_link";
    _cmd_vel_pub->publish(cmd);
  }

  geometry_msgs::msg::TwistStamped _compute_cmd_vel(const std::vector<geometry_msgs::msg::PoseStamped>& segment) {
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = this->now();
    cmd.header.frame_id = "base_link";

    double rx = _robot_pose.position.x;
    double ry = _robot_pose.position.y;
    double qz = _robot_pose.orientation.z;
    double qw = _robot_pose.orientation.w;
    double yaw = 2.0 * std::atan2(qz, qw);
    double cos_y = std::cos(yaw);
    double sin_y = std::sin(yaw);

    // Lookahead point on optimized band
    double tx = segment.back().pose.position.x;
    double ty = segment.back().pose.position.y;
    double lookahead = 0.5;
    
    for (const auto& p : segment) {
      if (std::hypot(p.pose.position.x - rx, p.pose.position.y - ry) >= lookahead) {
        tx = p.pose.position.x;
        ty = p.pose.position.y;
        break;
      }
    }

    double dxw = tx - rx;
    double dyw = ty - ry;
    double xl = dxw * cos_y + dyw * sin_y;
    double yl = -dxw * sin_y + dyw * cos_y;
    double L = std::hypot(xl, yl);

    if (L < 1e-6) return cmd;

    double d_goal = euclidean(_robot_pose, _global_path.poses.back().pose);
    double v_target = std::min(v_max, std::max(0.05, d_goal * 1.5));

    // Holonomic distribution
    cmd.twist.linear.x = v_target * (xl / L);
    cmd.twist.linear.y = v_target * (yl / L);
    cmd.twist.angular.z = 0.0;
    
    return cmd;
  }

public:
  TebPlannerNode() : Node("teb_cpp_node") {
    _odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&TebPlannerNode::_odom_callback, this, std::placeholders::_1));
    _global_path_sub = this->create_subscription<nav_msgs::msg::Path>(
      "/global_path", 10, std::bind(&TebPlannerNode::_global_path_callback, this, std::placeholders::_1));
    
    _cmd_vel_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);
    
    _timer = this->create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&TebPlannerNode::_replan, this)); // 10 Hz

    RCLCPP_INFO(this->get_logger(), "TEB C++ Node started!");
  }

  int _find_closest_index() {
    const auto &wp = _global_path.poses;
    int idx = _prune_index;
    double best = std::numeric_limits<double>::infinity();

    for (size_t i = _prune_index; i < wp.size(); ++i) {
      double d = euclidean(_robot_pose, wp[i].pose);
      if (d < best) {
        best = d;
        idx = i;
      }
    }
    return idx;
  }

  std::vector<geometry_msgs::msg::PoseStamped> _extract_local_segment() {
    const auto &wp = _global_path.poses;
    std::vector<geometry_msgs::msg::PoseStamped> seg;

    for (size_t i = _prune_index; i < wp.size(); ++i) {
      double d = euclidean(_robot_pose, wp[i].pose);
      if (d <= lookahead_dist) {
        seg.push_back(wp[i]);
      } else {
        break;
      }
    }
    return seg;
  }

  void optimize_band(std::vector<geometry_msgs::msg::PoseStamped>& segment) {
    if (segment.size() < 2) return;

    auto optimizer = setupOptimizer();
    std::vector<VertexPose*> vertices;

    for (size_t i = 0; i < segment.size(); ++i) {
      VertexPose* v = new VertexPose();
      v->setId(i);
      Eigen::Vector3d pose(segment[i].pose.position.x, 
                           segment[i].pose.position.y, 
                           0.0);
      v->setEstimate(pose);
      if (i == 0 || i == segment.size() - 1) v->setFixed(true);
      
      optimizer->addVertex(v);
      vertices.push_back(v);
    }

    for (size_t i = 0; i < vertices.size() - 1; ++i) {
      EdgeKinematics* e = new EdgeKinematics();
      e->setVertex(0, vertices[i]);
      e->setVertex(1, vertices[i+1]);
      Eigen::Matrix3d info = Eigen::Matrix3d::Identity();
      info(0,0) = 10.0; 
      info(1,1) = 10.0;
      e->setInformation(info);
      optimizer->addEdge(e);
    }

    optimizer->initializeOptimization();
    optimizer->optimize(10);

    for (size_t i = 0; i < vertices.size(); ++i) {
      Eigen::Vector3d opt_pose = vertices[i]->estimate();
      segment[i].pose.position.x = opt_pose[0];
      segment[i].pose.position.y = opt_pose[1];
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TebPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
