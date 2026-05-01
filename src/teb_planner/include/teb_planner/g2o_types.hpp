#ifndef TEB_PLANNER_G2O_TYPES_HPP
#define TEB_PLANNER_G2O_TYPES_HPP

#include "teb_planner/pose_se2.hpp"
#include "teb_planner/penalties.hpp"
#include "teb_planner/obstacles.hpp"
#include "teb_planner/teb_config.hpp"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>

#include <Eigen/Core>
#include <cmath>

namespace teb_planner {

// ─── VertexPose: SE(2) pose, 3 DOF ────────────────────────────────────

class VertexPose : public g2o::BaseVertex<3, PoseSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit VertexPose(bool fixed = false) {
        setToOriginImpl();
        setFixed(fixed);
    }
    VertexPose(const PoseSE2& pose, bool fixed = false) {
        setEstimate(pose);
        setFixed(fixed);
    }

    void setToOriginImpl() override { _estimate = PoseSE2(); }
    void oplusImpl(const double* update) override { _estimate.plus(update); }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// ─── VertexTimeDiff: scalar Δt, 1 DOF ─────────────────────────────────

class VertexTimeDiff : public g2o::BaseVertex<1, double> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit VertexTimeDiff(double dt = 0.1, bool fixed = false) {
        setEstimate(dt);
        setFixed(fixed);
    }

    void setToOriginImpl() override { _estimate = 0.1; }
    void oplusImpl(const double* update) override {
        _estimate += update[0];
        if (_estimate < 1e-3) _estimate = 1e-3;
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// ─── EdgeVelocityHolonomic ────────────────────────────────────────────
// Vertices: [pose_i, pose_{i+1}, dt_i].  Errors: (vx, vy, ω) bound penalties.

class EdgeVelocityHolonomic : public g2o::BaseMultiEdge<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeVelocityHolonomic() { resize(3); }

    void setTebConfig(const TebConfig* cfg) { cfg_ = cfg; }

    void computeError() override {
        const auto* p1 = static_cast<const VertexPose*>(_vertices[0]);
        const auto* p2 = static_cast<const VertexPose*>(_vertices[1]);
        const auto* dt = static_cast<const VertexTimeDiff*>(_vertices[2]);

        const Eigen::Vector2d delta = p2->estimate().position() - p1->estimate().position();
        const double avg_theta = 0.5 * (p1->estimate().theta() + p2->estimate().theta());
        const double c = std::cos(avg_theta);
        const double s = std::sin(avg_theta);

        const double T = std::max(dt->estimate(), 1e-6);
        const double vx =  ( delta.x() * c + delta.y() * s) / T;
        const double vy =  (-delta.x() * s + delta.y() * c) / T;

        double dtheta = p2->estimate().theta() - p1->estimate().theta();
        dtheta = std::atan2(std::sin(dtheta), std::cos(dtheta));
        const double omega = dtheta / T;

        const double eps = 0.01;
        _error[0] = penaltyBoundToInterval(vx,    cfg_->max_vel_x,     eps);
        _error[1] = penaltyBoundToInterval(vy,    cfg_->max_vel_y,     eps);
        _error[2] = penaltyBoundToInterval(omega, cfg_->max_vel_theta, eps);
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    const TebConfig* cfg_ = nullptr;
};

// ─── EdgeAccelerationHolonomic ────────────────────────────────────────
// Vertices: [pose_{i-1}, pose_i, pose_{i+1}, dt_{i-1}, dt_i]
// Errors: (ax, ay, α) bound penalties.

class EdgeAccelerationHolonomic : public g2o::BaseMultiEdge<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeAccelerationHolonomic() { resize(5); }

    void setTebConfig(const TebConfig* cfg) { cfg_ = cfg; }

    void computeError() override {
        const auto* p0  = static_cast<const VertexPose*>(_vertices[0]);
        const auto* p1  = static_cast<const VertexPose*>(_vertices[1]);
        const auto* p2  = static_cast<const VertexPose*>(_vertices[2]);
        const auto* dt0 = static_cast<const VertexTimeDiff*>(_vertices[3]);
        const auto* dt1 = static_cast<const VertexTimeDiff*>(_vertices[4]);

        const double T0 = std::max(dt0->estimate(), 1e-6);
        const double T1 = std::max(dt1->estimate(), 1e-6);

        const Eigen::Vector2d d01 = p1->estimate().position() - p0->estimate().position();
        const Eigen::Vector2d d12 = p2->estimate().position() - p1->estimate().position();

        const double ta = 0.5 * (p0->estimate().theta() + p1->estimate().theta());
        const double tb = 0.5 * (p1->estimate().theta() + p2->estimate().theta());
        const double ca = std::cos(ta), sa = std::sin(ta);
        const double cb = std::cos(tb), sb = std::sin(tb);

        const double vxa = ( d01.x() * ca + d01.y() * sa) / T0;
        const double vya = (-d01.x() * sa + d01.y() * ca) / T0;
        const double vxb = ( d12.x() * cb + d12.y() * sb) / T1;
        const double vyb = (-d12.x() * sb + d12.y() * cb) / T1;

        double dth_a = p1->estimate().theta() - p0->estimate().theta();
        double dth_b = p2->estimate().theta() - p1->estimate().theta();
        dth_a = std::atan2(std::sin(dth_a), std::cos(dth_a));
        dth_b = std::atan2(std::sin(dth_b), std::cos(dth_b));
        const double wa = dth_a / T0;
        const double wb = dth_b / T1;

        const double dt_avg = 0.5 * (T0 + T1);
        const double ax    = (vxb - vxa) / dt_avg;
        const double ay    = (vyb - vya) / dt_avg;
        const double alpha = (wb  - wa ) / dt_avg;

        const double eps = 0.01;
        _error[0] = penaltyBoundToInterval(ax,    cfg_->acc_lim_x,     eps);
        _error[1] = penaltyBoundToInterval(ay,    cfg_->acc_lim_y,     eps);
        _error[2] = penaltyBoundToInterval(alpha, cfg_->acc_lim_theta, eps);
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    const TebConfig* cfg_ = nullptr;
};

// ─── EdgeTimeOptimal: minimise Δt ─────────────────────────────────────

class EdgeTimeOptimal : public g2o::BaseUnaryEdge<1, double, VertexTimeDiff> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const auto* dt = static_cast<const VertexTimeDiff*>(_vertices[0]);
        _error[0] = dt->estimate();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// ─── EdgeShortestPath: minimise inter-pose distance ───────────────────

class EdgeShortestPath : public g2o::BaseBinaryEdge<1, double, VertexPose, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const auto* p1 = static_cast<const VertexPose*>(_vertices[0]);
        const auto* p2 = static_cast<const VertexPose*>(_vertices[1]);
        _error[0] = (p2->estimate().position() - p1->estimate().position()).norm();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// ─── EdgeObstacle: hard clearance d ≥ min_obstacle_dist ───────────────

class EdgeObstacle : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setTebConfig(const TebConfig* cfg) { cfg_ = cfg; }
    void setObstacle(const Obstacle* obs) { obs_ = obs; }

    void computeError() override {
        const auto* p = static_cast<const VertexPose*>(_vertices[0]);
        const double d = obs_->getMinimumDistance(p->estimate().position());
        _error[0] = penaltyBoundFromBelow(d, cfg_->min_obstacle_dist, 0.0);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    const TebConfig* cfg_ = nullptr;
    const Obstacle*  obs_ = nullptr;
};

// ─── EdgeInflation: soft cushion d ≥ inflation_dist ───────────────────

class EdgeInflation : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setTebConfig(const TebConfig* cfg) { cfg_ = cfg; }
    void setObstacle(const Obstacle* obs) { obs_ = obs; }

    void computeError() override {
        const auto* p = static_cast<const VertexPose*>(_vertices[0]);
        const double d = obs_->getMinimumDistance(p->estimate().position());
        _error[0] = penaltyBoundFromBelow(d, cfg_->inflation_dist, 0.0);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    const TebConfig* cfg_ = nullptr;
    const Obstacle*  obs_ = nullptr;
};

// ─── EdgeViaPoint: pull pose toward reference path point ──────────────

class EdgeViaPoint : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setViaPoint(const Eigen::Vector2d* via) { via_ = via; }

    void computeError() override {
        const auto* p = static_cast<const VertexPose*>(_vertices[0]);
        _error[0] = (p->estimate().position() - *via_).norm();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    const Eigen::Vector2d* via_ = nullptr;
};

// ─── EdgeYawHold: pull θ toward reference yaw (yaw lock) ──────────────

class EdgeYawHold : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setReferenceYaw(double yaw) { ref_yaw_ = yaw; }

    void computeError() override {
        const auto* p = static_cast<const VertexPose*>(_vertices[0]);
        double d = p->estimate().theta() - ref_yaw_;
        d = std::atan2(std::sin(d), std::cos(d));
        _error[0] = d;
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    double ref_yaw_ = 0.0;
};

}  // namespace teb_planner

#endif  // TEB_PLANNER_G2O_TYPES_HPP
