#ifndef TEB_PLANNER_TIMED_ELASTIC_BAND_HPP
#define TEB_PLANNER_TIMED_ELASTIC_BAND_HPP

#include "teb_planner/g2o_types.hpp"
#include "teb_planner/pose_se2.hpp"

#include <vector>

namespace teb_planner {

using PoseVertexContainer     = std::vector<VertexPose*>;
using TimeDiffVertexContainer = std::vector<VertexTimeDiff*>;

// Container of pose + time-diff vertex *pointers* used while building a g2o
// graph.  After `OptimalPlanner::plan()` returns, the optimizer owns these
// vertices and the containers below are emptied (pointers become dangling
// otherwise).  Persistent state across cycles lives in OptimalPlanner as
// std::vector<PoseSE2>/double, NOT here.
class TimedElasticBand {
public:
    TimedElasticBand() = default;
    ~TimedElasticBand() { releaseAllVertices(); }

    TimedElasticBand(const TimedElasticBand&)            = delete;
    TimedElasticBand& operator=(const TimedElasticBand&) = delete;

    int sizePoses()     const { return static_cast<int>(poses_.size()); }
    int sizeTimeDiffs() const { return static_cast<int>(time_diffs_.size()); }

    PoseVertexContainer&           poses()           { return poses_; }
    const PoseVertexContainer&     poses()     const { return poses_; }
    TimeDiffVertexContainer&       timeDiffs()       { return time_diffs_; }
    const TimeDiffVertexContainer& timeDiffs() const { return time_diffs_; }

    const PoseSE2& pose(int i)     const { return poses_[i]->estimate(); }
    double         timeDiff(int i) const { return time_diffs_[i]->estimate(); }

    void setPose(int i, const PoseSE2& p)    { poses_[i]->setEstimate(p); }
    void setTimeDiff(int i, double v)        { time_diffs_[i]->setEstimate(v); }
    void setPoseFixed(int i, bool fixed)     { poses_[i]->setFixed(fixed); }
    void setTimeDiffFixed(int i, bool fixed) { time_diffs_[i]->setFixed(fixed); }

    // Delete all owned vertices (use only when they are NOT in an optimizer).
    void releaseAllVertices();

    // Forget pointers WITHOUT deleting (use after optimizer takes ownership).
    void detach() { poses_.clear(); time_diffs_.clear(); }

    // Initialise the band from a sparse polyline reference, dropping samples
    // about `dt_ref * max_vel` apart along the arc length.
    void initFromPath(const std::vector<PoseSE2>& path, double dt_ref, double max_vel);

    // Initialise the band by transplanting a previous solution.
    void initFromWarm(const std::vector<PoseSE2>& warm_poses,
                      const std::vector<double>& warm_dts);

    // Insert / drop poses so Δt stays roughly within [dt_ref - hyst, dt_ref + hyst].
    void autoResize(double dt_ref, double dt_hyst, int min_samples, int max_samples);

private:
    PoseVertexContainer     poses_;
    TimeDiffVertexContainer time_diffs_;
};

}  // namespace teb_planner

#endif  // TEB_PLANNER_TIMED_ELASTIC_BAND_HPP
