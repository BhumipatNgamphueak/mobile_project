#include "teb_planner/optimal_planner.hpp"
#include "teb_planner/g2o_types.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <algorithm>
#include <limits>

namespace teb_planner {

using BlockSolverType  = g2o::BlockSolverX;
using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

namespace {

std::unique_ptr<g2o::SparseOptimizer> makeOptimizer() {
    auto linear = std::make_unique<LinearSolverType>();
    auto block  = std::make_unique<BlockSolverType>(std::move(linear));
    auto algo   = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    auto opt = std::make_unique<g2o::SparseOptimizer>();
    opt->setAlgorithm(algo);
    opt->setVerbose(false);
    return opt;
}

}  // namespace

OptimalPlanner::OptimalPlanner(TebConfig* cfg) : cfg_(cfg) {}

bool OptimalPlanner::plan(const PoseSE2& start,
                          const PoseSE2& goal,
                          const std::vector<PoseSE2>& initial_plan,
                          const ObstacleContainer& obstacles,
                          const std::vector<Eigen::Vector2d>& via_points) {
    if (initial_plan.size() < 2) return false;

    TimedElasticBand teb;
    if (hasWarmState() && static_cast<int>(warm_poses_.size()) >= cfg_->min_samples) {
        teb.initFromWarm(warm_poses_, warm_dts_);
    } else {
        teb.initFromPath(initial_plan, cfg_->dt_ref, cfg_->max_vel_x);
    }

    if (teb.sizePoses() < 2) return false;

    // Pin endpoints to the current pose / local goal each cycle.
    teb.setPose(0, start);
    teb.setPose(teb.sizePoses() - 1, goal);
    teb.setPoseFixed(0, true);
    teb.setPoseFixed(teb.sizePoses() - 1, true);

    teb.autoResize(cfg_->dt_ref, cfg_->dt_hysteresis,
                   cfg_->min_samples, cfg_->max_samples);

    via_points_storage_ = via_points;

    auto optimizer = makeOptimizer();
    int  vid       = 0;
    for (auto* v : teb.poses())     { v->setId(vid++); optimizer->addVertex(v); }
    for (auto* v : teb.timeDiffs()) { v->setId(vid++); optimizer->addVertex(v); }

    buildGraph(*optimizer, teb, obstacles);

    if (!optimizer->initializeOptimization()) {
        // Optimizer destruction below frees vertices.
        teb.detach();
        warm_poses_.clear();
        warm_dts_.clear();
        return false;
    }

    const int total_iter =
        std::max(1, cfg_->no_outer_iterations) *
        std::max(1, cfg_->no_inner_iterations);
    optimizer->optimize(total_iter);

    // Snapshot the solution into warm state BEFORE the optimizer is destroyed.
    warm_poses_.clear();
    warm_dts_.clear();
    warm_poses_.reserve(teb.sizePoses());
    warm_dts_.reserve(teb.sizeTimeDiffs());
    for (int i = 0; i < teb.sizePoses();     ++i) warm_poses_.push_back(teb.pose(i));
    for (int i = 0; i < teb.sizeTimeDiffs(); ++i) warm_dts_.push_back(teb.timeDiff(i));

    // The optimizer (going out of scope) will delete the vertices; detach
    // the teb's pointers so its destructor doesn't double-free.
    teb.detach();

    return true;
}

void OptimalPlanner::buildGraph(g2o::SparseOptimizer& opt,
                                TimedElasticBand& teb,
                                const ObstacleContainer& obstacles) {
    addVelocityEdges    (opt, teb);
    addAccelerationEdges(opt, teb);
    addTimeOptimalEdges (opt, teb);
    if (cfg_->weight_shortest_path > 0.0) addShortestPathEdges(opt, teb);
    addObstacleEdges    (opt, teb, obstacles);
    addViaPointEdges    (opt, teb);
    if (cfg_->yaw_locked) addYawHoldEdges(opt, teb);
}

void OptimalPlanner::addVelocityEdges(g2o::SparseOptimizer& opt,
                                      TimedElasticBand& teb) {
    Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
    info(0, 0) = cfg_->weight_max_vel_x;
    info(1, 1) = cfg_->weight_max_vel_y;
    info(2, 2) = cfg_->weight_max_vel_theta;

    for (int i = 0; i < teb.sizeTimeDiffs(); ++i) {
        auto* e = new EdgeVelocityHolonomic();
        e->setVertex(0, teb.poses()[i]);
        e->setVertex(1, teb.poses()[i + 1]);
        e->setVertex(2, teb.timeDiffs()[i]);
        e->setInformation(info);
        e->setTebConfig(cfg_);
        opt.addEdge(e);
    }
}

void OptimalPlanner::addAccelerationEdges(g2o::SparseOptimizer& opt,
                                          TimedElasticBand& teb) {
    if (teb.sizePoses() < 3) return;
    Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
    info(0, 0) = cfg_->weight_acc_lim_x;
    info(1, 1) = cfg_->weight_acc_lim_y;
    info(2, 2) = cfg_->weight_acc_lim_theta;

    for (int i = 1; i < teb.sizePoses() - 1; ++i) {
        auto* e = new EdgeAccelerationHolonomic();
        e->setVertex(0, teb.poses()[i - 1]);
        e->setVertex(1, teb.poses()[i]);
        e->setVertex(2, teb.poses()[i + 1]);
        e->setVertex(3, teb.timeDiffs()[i - 1]);
        e->setVertex(4, teb.timeDiffs()[i]);
        e->setInformation(info);
        e->setTebConfig(cfg_);
        opt.addEdge(e);
    }
}

void OptimalPlanner::addTimeOptimalEdges(g2o::SparseOptimizer& opt,
                                         TimedElasticBand& teb) {
    Eigen::Matrix<double, 1, 1> info;
    info(0, 0) = cfg_->weight_optimaltime;
    for (int i = 0; i < teb.sizeTimeDiffs(); ++i) {
        auto* e = new EdgeTimeOptimal();
        e->setVertex(0, teb.timeDiffs()[i]);
        e->setInformation(info);
        opt.addEdge(e);
    }
}

void OptimalPlanner::addShortestPathEdges(g2o::SparseOptimizer& opt,
                                          TimedElasticBand& teb) {
    Eigen::Matrix<double, 1, 1> info;
    info(0, 0) = cfg_->weight_shortest_path;
    for (int i = 0; i < teb.sizePoses() - 1; ++i) {
        auto* e = new EdgeShortestPath();
        e->setVertex(0, teb.poses()[i]);
        e->setVertex(1, teb.poses()[i + 1]);
        e->setInformation(info);
        opt.addEdge(e);
    }
}

void OptimalPlanner::addObstacleEdges(g2o::SparseOptimizer& opt,
                                      TimedElasticBand& teb,
                                      const ObstacleContainer& obstacles) {
    if (obstacles.empty()) return;

    Eigen::Matrix<double, 1, 1> info_obs; info_obs(0, 0) = cfg_->weight_obstacle;
    Eigen::Matrix<double, 1, 1> info_inf; info_inf(0, 0) = cfg_->weight_inflation;

    const double radius2 = cfg_->obstacle_search_radius * cfg_->obstacle_search_radius;

    for (int i = 1; i < teb.sizePoses() - 1; ++i) {
        const Eigen::Vector2d& p = teb.pose(i).position();
        for (const auto& obs : obstacles) {
            const Eigen::Vector2d d = obs->getCentroid() - p;
            if (d.squaredNorm() > radius2) continue;

            auto* e_o = new EdgeObstacle();
            e_o->setVertex(0, teb.poses()[i]);
            e_o->setInformation(info_obs);
            e_o->setTebConfig(cfg_);
            e_o->setObstacle(obs.get());
            opt.addEdge(e_o);

            if (cfg_->weight_inflation > 0.0) {
                auto* e_i = new EdgeInflation();
                e_i->setVertex(0, teb.poses()[i]);
                e_i->setInformation(info_inf);
                e_i->setTebConfig(cfg_);
                e_i->setObstacle(obs.get());
                opt.addEdge(e_i);
            }
        }
    }
}

void OptimalPlanner::addViaPointEdges(g2o::SparseOptimizer& opt,
                                      TimedElasticBand& teb) {
    if (via_points_storage_.empty() || cfg_->weight_viapoint <= 0.0) return;
    if (teb.sizePoses() < 3) return;

    Eigen::Matrix<double, 1, 1> info;
    info(0, 0) = cfg_->weight_viapoint;

    for (size_t k = 0; k < via_points_storage_.size(); ++k) {
        int    best_i = -1;
        double best_d = std::numeric_limits<double>::infinity();
        for (int i = 1; i < teb.sizePoses() - 1; ++i) {
            const double d = (teb.pose(i).position() - via_points_storage_[k]).squaredNorm();
            if (d < best_d) { best_d = d; best_i = i; }
        }
        if (best_i < 0) continue;

        auto* e = new EdgeViaPoint();
        e->setVertex(0, teb.poses()[best_i]);
        e->setInformation(info);
        e->setViaPoint(&via_points_storage_[k]);
        opt.addEdge(e);
    }
}

void OptimalPlanner::addYawHoldEdges(g2o::SparseOptimizer& opt,
                                     TimedElasticBand& teb) {
    Eigen::Matrix<double, 1, 1> info;
    info(0, 0) = cfg_->weight_kinematics_yaw;

    for (int i = 1; i < teb.sizePoses() - 1; ++i) {  // skip fixed endpoints
        auto* e = new EdgeYawHold();
        e->setVertex(0, teb.poses()[i]);
        e->setInformation(info);
        e->setReferenceYaw(cfg_->yaw_locked_ref);
        opt.addEdge(e);
    }
}

bool OptimalPlanner::getVelocityCommand(double& vx_world, double& vy_world) const {
    if (warm_poses_.size() < 2 || warm_dts_.empty()) return false;
    const double T = std::max(warm_dts_[0], 1e-3);
    vx_world = (warm_poses_[1].x() - warm_poses_[0].x()) / T;
    vy_world = (warm_poses_[1].y() - warm_poses_[0].y()) / T;
    return true;
}

}  // namespace teb_planner
