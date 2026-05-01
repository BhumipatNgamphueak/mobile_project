#include "teb_planner/timed_elastic_band.hpp"

#include <algorithm>
#include <cmath>

namespace teb_planner {

void TimedElasticBand::releaseAllVertices() {
    for (auto* v : poses_)      delete v;
    for (auto* v : time_diffs_) delete v;
    poses_.clear();
    time_diffs_.clear();
}

void TimedElasticBand::initFromPath(const std::vector<PoseSE2>& path,
                                    double dt_ref,
                                    double max_vel) {
    releaseAllVertices();
    if (path.size() < 2) return;

    const double step_len = std::max(0.05, dt_ref * std::max(max_vel, 0.05));

    // Walk the polyline, dropping samples roughly `step_len` apart.
    std::vector<PoseSE2> samples;
    samples.push_back(path.front());

    double accumulated = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        const Eigen::Vector2d a = path[i - 1].position();
        const Eigen::Vector2d b = path[i].position();
        const double seg = (b - a).norm();
        if (seg < 1e-9) continue;

        double remaining = step_len - accumulated;
        double t = 0.0;
        while (remaining <= seg - t) {
            t += remaining;
            const Eigen::Vector2d p = a + (b - a) * (t / seg);
            samples.emplace_back(p.x(), p.y(), path[i].theta());
            accumulated = 0.0;
            remaining   = step_len;
        }
        accumulated += (seg - t);
    }

    if ((samples.back().position() - path.back().position()).norm() > 1e-3) {
        samples.push_back(path.back());
    }
    if (samples.size() < 2) {
        samples.push_back(path.back());
    }

    for (size_t i = 0; i < samples.size(); ++i) {
        poses_.push_back(new VertexPose(samples[i], false));
    }
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        const double seg = (samples[i + 1].position() - samples[i].position()).norm();
        const double dt  = std::max(seg / std::max(max_vel, 0.05), 0.05);
        time_diffs_.push_back(new VertexTimeDiff(dt, false));
    }
}

void TimedElasticBand::initFromWarm(const std::vector<PoseSE2>& warm_poses,
                                    const std::vector<double>& warm_dts) {
    releaseAllVertices();
    if (warm_poses.size() < 2 || warm_dts.size() != warm_poses.size() - 1) return;

    for (size_t i = 0; i < warm_poses.size(); ++i) {
        poses_.push_back(new VertexPose(warm_poses[i], false));
    }
    for (size_t i = 0; i < warm_dts.size(); ++i) {
        time_diffs_.push_back(new VertexTimeDiff(warm_dts[i], false));
    }
}

void TimedElasticBand::autoResize(double dt_ref, double dt_hyst,
                                  int min_samples, int max_samples) {
    if (sizePoses() < 3) return;

    bool modified = true;
    int  guard    = 100;
    while (modified && guard-- > 0) {
        modified = false;

        // Insert pose where Δt is too large.
        for (int i = 0; i < sizeTimeDiffs(); ++i) {
            if (sizePoses() >= max_samples) break;
            if (time_diffs_[i]->estimate() > dt_ref + dt_hyst) {
                const PoseSE2& a = poses_[i]->estimate();
                const PoseSE2& b = poses_[i + 1]->estimate();
                const PoseSE2  mid = PoseSE2::average(a, b);

                auto* new_pose = new VertexPose(mid, false);
                auto* new_dt   = new VertexTimeDiff(time_diffs_[i]->estimate() / 2.0, false);
                time_diffs_[i]->setEstimate(time_diffs_[i]->estimate() / 2.0);

                poses_.insert(poses_.begin() + i + 1, new_pose);
                time_diffs_.insert(time_diffs_.begin() + i + 1, new_dt);
                modified = true;
                break;
            }
        }
        if (modified) continue;

        // Remove an interior pose where Δt is too small.
        for (int i = 0; i < sizeTimeDiffs() - 1; ++i) {
            if (sizePoses() <= min_samples) break;
            if (time_diffs_[i]->estimate() < dt_ref - dt_hyst) {
                if (i + 1 == 0 || i + 1 == sizePoses() - 1) continue;
                time_diffs_[i + 1]->setEstimate(
                    time_diffs_[i + 1]->estimate() + time_diffs_[i]->estimate());

                delete poses_[i + 1];
                delete time_diffs_[i];
                poses_.erase(poses_.begin() + i + 1);
                time_diffs_.erase(time_diffs_.begin() + i);
                modified = true;
                break;
            }
        }
    }
}

}  // namespace teb_planner
