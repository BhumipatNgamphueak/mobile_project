#ifndef TEB_PLANNER_PENALTIES_HPP
#define TEB_PLANNER_PENALTIES_HPP

#include <cmath>

namespace teb_planner {

// Penalty: var should be ≤ a.  Zero inside, grows linearly outside.
inline double penaltyBoundFromAbove(double var, double a, double epsilon) {
    if (var <= a - epsilon) return 0.0;
    return var - (a - epsilon);
}

// Penalty: var should be ≥ a.  Zero inside, grows linearly outside.
inline double penaltyBoundFromBelow(double var, double a, double epsilon) {
    if (var >= a + epsilon) return 0.0;
    return -var + (a + epsilon);
}

// Penalty: a ≤ var ≤ b.  Zero inside the open interval.
inline double penaltyBoundToInterval(double var, double a, double b, double epsilon) {
    if (var < a + epsilon) return -var + (a + epsilon);
    if (var > b - epsilon) return  var - (b - epsilon);
    return 0.0;
}

// Symmetric: |var| ≤ b.
inline double penaltyBoundToInterval(double var, double b, double epsilon) {
    return penaltyBoundToInterval(var, -b, b, epsilon);
}

}  // namespace teb_planner

#endif  // TEB_PLANNER_PENALTIES_HPP
