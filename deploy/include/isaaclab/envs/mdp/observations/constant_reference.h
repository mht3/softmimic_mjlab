#pragma once

#include <yaml-cpp/yaml.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace isaaclab::mdp {

inline std::vector<float> constant_nominal_reference(
    const YAML::Node& params, const std::string& name, std::size_t dimension)
{
    if (!params["reference_mode"] ||
        params["reference_mode"].as<std::string>() != "constant_nominal" ||
        !params["value"] || !params["value"].IsSequence()) {
        throw std::runtime_error(name + ": missing verified constant nominal reference; re-export this policy");
    }
    auto value = params["value"].as<std::vector<float>>();
    if (value.size() != dimension) {
        throw std::runtime_error(name + ": reference dimension mismatch");
    }
    for (float x : value) {
        if (!std::isfinite(x)) throw std::runtime_error(name + ": non-finite reference");
    }
    return value;
}

}  // namespace isaaclab::mdp
