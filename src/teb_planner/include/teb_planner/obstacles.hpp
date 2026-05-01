#ifndef TEB_PLANNER_OBSTACLES_HPP
#define TEB_PLANNER_OBSTACLES_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace teb_planner {

// Abstract obstacle (mirrors teb_local_planner::Obstacle).
class Obstacle {
public:
    virtual ~Obstacle() = default;
    virtual double          getMinimumDistance(const Eigen::Vector2d& position) const = 0;
    virtual Eigen::Vector2d getClosestPoint(const Eigen::Vector2d& position) const = 0;
    virtual const Eigen::Vector2d& getCentroid() const = 0;
};

// Single-point obstacle (only obstacle type used in this package).
class PointObstacle : public Obstacle {
public:
    PointObstacle() : pos_(0.0, 0.0) {}
    PointObstacle(double x, double y) : pos_(x, y) {}
    explicit PointObstacle(const Eigen::Vector2d& p) : pos_(p) {}

    double getMinimumDistance(const Eigen::Vector2d& position) const override {
        return (position - pos_).norm();
    }
    Eigen::Vector2d getClosestPoint(const Eigen::Vector2d& /*position*/) const override {
        return pos_;
    }
    const Eigen::Vector2d& getCentroid() const override { return pos_; }

private:
    Eigen::Vector2d pos_;
};

using ObstaclePtr       = std::shared_ptr<Obstacle>;
using ObstacleContainer = std::vector<ObstaclePtr>;

}  // namespace teb_planner

#endif  // TEB_PLANNER_OBSTACLES_HPP
