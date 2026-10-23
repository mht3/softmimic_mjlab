#pragma once
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>

namespace isaaclab {
// Pelvis IMU only: no unobservable world height or external-force measurement.
inline bool fallen_or_invalid(const Eigen::Quaternionf& q, float limit_angle) {
    const float norm = q.norm();
    if (!std::isfinite(norm) || norm < .9f || norm > 1.1f) return true;
    const auto gravity = q.normalized().conjugate() * Eigen::Vector3f(0,0,-1);
    const float tilt = std::acos(std::clamp(-gravity.z(), -1.f, 1.f));
    return !std::isfinite(tilt) || tilt > limit_angle;
}
}
