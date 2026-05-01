#ifndef TEB_PLANNER_OPTIMAL_PLANNER_HPP
#define TEB_PLANNER_OPTIMAL_PLANNER_HPP

#include "teb_planner/teb_config.hpp"
#include "teb_planner/timed_elastic_band.hpp"
#include "teb_planner/obstacles.hpp"
#include "teb_planner/pose_se2.hpp"

#include <g2o/core/sparse_optimizer.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace teb_planner {

// Build the g2o graph from a TimedElasticBand and run Levenberg–Marquardt.
// Mirrors teb_local_planner::TebOptimalPlanner (simplified for holonomic).
//
// Vertex ownership pattern (important):
//   • TimedElasticBand creates vertices fresh each plan() call.
//   • g2o::SparseOptimizer takes ownership when addVertex is called.
//   • Before the optimizer is destroyed at the end of plan(), we copy the
//     final pose / dt values into `warm_poses_` / `warm_dts_` (plain values,
//     not vertex pointers).  Next plan() call rebuilds vertices from these.
class OptimalPlanner {
public:
    explicit OptimalPlanner(TebConfig* cfg);
    ~OptimalPlanner() = default;

    OptimalPlanner(const OptimalPlanner&)            = delete;
    OptimalPlanner& operator=(const OptimalPlanner&) = delete;

    // Plan one cycle.  `start` and `goal` pin band endpoints; `initial_plan`
    // seeds the band on cold start; `obstacles` and `via_points` shape it.
    bool plan(const PoseSE2& start,
              const PoseSE2& goal,
              const std::vector<PoseSE2>& initial_plan,
              const ObstacleContainer& obstacles,
              const std::vector<Eigen::Vector2d>& via_points);

    // Velocity from the first segment of the optimised band, in WORLD frame.
    bool getVelocityCommand(double& vx_world, double& vy_world) const;

    // Read the optimised band (for visualization / publishing).
    const std::vector<PoseSE2>& getOptimizedPoses() const { return warm_poses_; }
    const std::vector<double>&  getTimeDiffs()      const { return warm_dts_; }

    bool hasWarmState() const { return !warm_poses_.empty(); }
    void clearWarmState() { warm_poses_.clear(); warm_dts_.clear(); }

private:
    TebConfig* cfg_;

    // Persistent warm state (values, not g2o vertex pointers).
    std::vector<PoseSE2> warm_poses_;
    std::vector<double>  warm_dts_;

    // Stable storage for via-points so EdgeViaPoint pointers remain valid
    // for the duration of optimization.  Refilled each plan() call.
    std::vector<Eigen::Vector2d> via_points_storage_;

    // Build the graph against an existing optimizer + band.
    void buildGraph(g2o::SparseOptimizer& opt,
                    TimedElasticBand& teb,
                    const ObstacleContainer& obstacles);

    void addVelocityEdges    (g2o::SparseOptimizer&, TimedElasticBand&);
    void addAccelerationEdges(g2o::SparseOptimizer&, TimedElasticBand&);
    void addTimeOptimalEdges (g2o::SparseOptimizer&, TimedElasticBand&);
    void addShortestPathEdges(g2o::SparseOptimizer&, TimedElasticBand&);
    void addObstacleEdges    (g2o::SparseOptimizer&, TimedElasticBand&,
                              const ObstacleContainer&);
    void addViaPointEdges    (g2o::SparseOptimizer&, TimedElasticBand&);
    void addYawHoldEdges     (g2o::SparseOptimizer&, TimedElasticBand&);
};

}  // namespace teb_planner

#endif  // TEB_PLANNER_OPTIMAL_PLANNER_HPP
