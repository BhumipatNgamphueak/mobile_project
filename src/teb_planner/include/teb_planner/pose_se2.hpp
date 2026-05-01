#ifndef TEB_PLANNER_POSE_SE2_HPP
#define TEB_PLANNER_POSE_SE2_HPP

#include <Eigen/Core>
#include <cmath>

namespace teb_planner {

// SE(2) pose (x, y, θ). Mirrors teb_local_planner::PoseSE2.
class PoseSE2 {
public:
    PoseSE2() : pos_(0.0, 0.0), theta_(0.0) {}
    PoseSE2(double x, double y, double theta) : pos_(x, y), theta_(theta) {}
    PoseSE2(const Eigen::Vector2d& pos, double theta) : pos_(pos), theta_(theta) {}

    Eigen::Vector2d&       position()       { return pos_; }
    const Eigen::Vector2d& position() const { return pos_; }

    double&       x()           { return pos_[0]; }
    double&       y()           { return pos_[1]; }
    double&       theta()       { return theta_; }
    const double& x()     const { return pos_[0]; }
    const double& y()     const { return pos_[1]; }
    const double& theta() const { return theta_; }

    void plus(const double* update) {
        pos_[0] += update[0];
        pos_[1] += update[1];
        theta_  += update[2];
        theta_   = std::atan2(std::sin(theta_), std::cos(theta_));
    }

    static PoseSE2 average(const PoseSE2& a, const PoseSE2& b) {
        const double s = 0.5 * (std::sin(a.theta_) + std::sin(b.theta_));
        const double c = 0.5 * (std::cos(a.theta_) + std::cos(b.theta_));
        return PoseSE2(0.5 * (a.pos_ + b.pos_), std::atan2(s, c));
    }

private:
    Eigen::Vector2d pos_;
    double theta_;
};

}  // namespace teb_planner

#endif  // TEB_PLANNER_POSE_SE2_HPP
