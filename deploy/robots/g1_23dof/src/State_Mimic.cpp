#include "State_Mimic.h"
#include "isaaclab/envs/mdp/observations/nominal_motion.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include "isaaclab/utils/fall_guard.h"
#include <algorithm>
#include <cmath>

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;

// Desired compliance stiffness observed by SoftMimic-style compliant policies.
// Initialized from the FSM config and adjustable online via the remote
// (up/down: translational, left/right: rotational). Ranges follow the
// SoftMimic training distribution.
static std::atomic<float> desired_stiffness{200.0f};           // [N/m], range [40, 1000]
static std::atomic<float> desired_rotational_stiffness{1.0f};  // [Nm/rad], range [0.1, 10]


Eigen::Quaternionf robot_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    auto & motors = robot->lowstate->msg_.motor_state();

    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ());


//    return root_quat;
    return torso_quat;
}

Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ());


//    return root_quat;
    return torso_quat;
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_command)
{
    auto loader = State_Mimic::motion;
    std::vector<float> data;

    auto motion_joint_pos = loader->joint_pos();
    auto motion_joint_vel = loader->joint_vel();

    data.insert(data.end(),
                motion_joint_pos.data(),
                motion_joint_pos.data() + motion_joint_pos.size());
    data.insert(data.end(),
                motion_joint_vel.data(),
                motion_joint_vel.data() + motion_joint_vel.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_Mimic::motion;
    std::vector<float> out;

    auto real_quat_w = robot_quat_w(env);
    auto ref_quat_w  = motion_anchor_quat_w(loader);

    auto rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    auto rot = rot_.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(desired_stiffness_log)
{
    return { std::log(desired_stiffness + 1e-6f) };
}

REGISTER_OBSERVATION(desired_rotational_stiffness_log)
{
    return { std::log(desired_rotational_stiffness + 1e-6f) };
}

}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    const bool constant = deploy_cfg["reference_mode"] &&
        deploy_cfg["reference_mode"].as<std::string>() == "constant_nominal";
    std::filesystem::path motion_file;
    time_range_ = {0.f, 0.f};
    // Standing nominal channels are fully embedded in deploy.yaml. No ignored
    // training NPZ is needed on a fresh clone or on the robot.
    if (!constant) {
        motion_file = cfg["motion_file"].as<std::string>();
        if (!motion_file.is_absolute()) motion_file = param::proj_dir / motion_file;
        motion_ = std::make_shared<MotionLoader_>(motion_file.string());
        motion = motion_;
        time_range_[0] = cfg["time_start"] ? cfg["time_start"].as<float>() : 0.f;
        time_range_[1] = cfg["time_end"] ? cfg["time_end"].as<float>() : motion_->duration;
        if (time_range_[0] < 0 || time_range_[1] <= time_range_[0] || time_range_[1] > motion_->duration)
            throw std::runtime_error("Invalid reference playback range");
    }
    if (cfg["policy_watchdog_ms"]) watchdog_ms_ = cfg["policy_watchdog_ms"].as<int>();
    if (watchdog_ms_ < 40 || watchdog_ms_ > 200) throw std::runtime_error("Policy watchdog must be 40..200 ms");
    std::string end_state = "Velocity";
    if (cfg["end_state"]) {
        end_state = cfg["end_state"].as<std::string>();
    }
    if (cfg["loop"]) {
        loop_ = cfg["loop"].as<bool>();
    }
    if (cfg["desired_stiffness"]) {
        desired_stiffness = cfg["desired_stiffness"].as<float>();
    }
    if (cfg["desired_rotational_stiffness"]) {
        desired_rotational_stiffness = cfg["desired_rotational_stiffness"].as<float>();
    }
    spdlog::info("Desired stiffness: {:.1f} N/m, rotational: {:.2f} Nm/rad",
                 desired_stiffness.load(), desired_rotational_stiffness.load());

    std::function<std::vector<float>(const std::string&)> reference_provider;
    if (deploy_cfg["reference_mode"] && deploy_cfg["reference_mode"].as<std::string>() == "nominal_motion") {
        auto nominal = std::make_shared<isaaclab::NominalMotion>(motion_file.string());
        if (nominal->frames != static_cast<size_t>(motion_->num_frames)) throw std::runtime_error("Nominal/playback frame mismatch");
        reference_provider = [nominal, loader=motion_](const std::string& key) {return nominal->at(key, loader->frame);};
    }
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy_cfg, articulation, reference_provider);
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");


    if (cfg["limit_angle"]) {
        limit_angle_ = cfg["limit_angle"].as<float>();
    }

    if (!std::isfinite(limit_angle_) || limit_angle_ <= 0 || limit_angle_ >= 3.14159265f)
        throw std::runtime_error("Invalid fall cutoff angle");

    this->registered_checks.emplace_back(
        std::make_pair([this]()->bool { return policy_fault.load(); },
                       FSMStringMap.right.at("Passive"))
    );
    if (!loop_ && motion_) {
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{ return motion_finished_.load(); }, // time out
                FSMStringMap.right.at(end_state)
            )
        );
    }

}

void State_Mimic::enter()
{
    policy_fault = false;
    policy_ready_ = false;
    motion_finished_ = false;
    last_policy_ms_ = steady_ms();
    try {
        env->robot->update();
        if (isaaclab::fallen_or_invalid(env->robot->data.root_quat_w, limit_angle_))
            throw std::runtime_error("Robot is tilted beyond the fall cutoff");
    } catch (const std::exception& error) {
        spdlog::error("Invalid sensor state on policy entry: {}", error.what());
        policy_fault = true;
        return;
    }
    // Hold a validated measured pose until the first inference completes.
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = env->robot->data.joint_pos[i];
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].dq() = 0;
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].tau() = 0;
    }

    motion = motion_; // set for specific motion
    env->action_manager->reset();
    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::steady_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        try {
            if (motion_) {
                motion_->reset(env->robot->data, time_range_[0]);
                auto ref_yaw = isaaclab::yawQuaternion(motion_->root_quaternion()).toRotationMatrix();
                auto robot_yaw = isaaclab::yawQuaternion(robot_quat_w(env.get())).toRotationMatrix();
                init_quat = robot_yaw * ref_yaw.transpose();
            }
            env->reset();
        } catch (const std::exception& error) {
            spdlog::error("Policy initialization failed: {}", error.what());
            policy_fault = true;
            return;
        }
        while (policy_thread_running)
        {
            if (motion_) {
                float motion_time = env->episode_length * env->step_dt + time_range_[0];
                if (loop_) motion_time = time_range_[0] + std::fmod(env->episode_length * env->step_dt, time_range_[1] - time_range_[0]);
                motion_->update(motion_time);
                motion_finished_ = !loop_ && motion_time >= time_range_[1];
            }
            try {
                env->step();
                last_policy_ms_ = steady_ms();
                policy_ready_ = true;
            } catch (const std::exception& error) {
                spdlog::error("Compliance policy failed; requesting Passive: {}", error.what());
                policy_fault = true;
                policy_thread_running = false;
                break;
            }

            // Sleep
            // Never replay a burst of stale history after an inference overrun.
            if (clock::now() > sleepTill) sleepTill = clock::now() + dt;
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Mimic::run()
{
    // Check a fresh IMU snapshot in the command loop, before any policy target
    // is published. Do not wait for inference or read its robot-state buffer.
    Eigen::Quaternionf orientation;
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto &q = lowstate->msg_.imu_state().quaternion();
        orientation = Eigen::Quaternionf(q[0],q[1],q[2],q[3]);
    }
    if (isaaclab::fallen_or_invalid(orientation, limit_angle_)) {
        if (!policy_fault.exchange(true)) spdlog::warn("Fall/invalid IMU detected; latching policy OFF and entering Passive");
    }
    if (lowstate->isTimeout() || steady_ms() - last_policy_ms_.load() > watchdog_ms_)
        policy_fault = true;
    if (policy_fault) {
        policy_thread_running = false;
        // Stop driving targets immediately, before the FSM publishes this tick.
        for (auto id : env->robot->data.joint_ids_map) {
            auto &motor = lowcmd->msg_.motor_cmd()[static_cast<int>(id)];
            motor.q() = 0; motor.kp() = 0; motor.kd() = 3; motor.dq() = 0; motor.tau() = 0;
        }
        return;
    }
    // Online stiffness adjustment (clamped to the SoftMimic training ranges).
    const auto & joy = FSMState::lowstate->joystick;
    if (joy.up.on_pressed) {
        desired_stiffness = std::min(desired_stiffness * 1.25f, 1000.0f);
        spdlog::info("desired_stiffness -> {:.1f} N/m", desired_stiffness.load());
    }
    if (joy.down.on_pressed) {
        desired_stiffness = std::max(desired_stiffness / 1.25f, 40.0f);
        spdlog::info("desired_stiffness -> {:.1f} N/m", desired_stiffness.load());
    }
    if (joy.right.on_pressed) {
        desired_rotational_stiffness = std::min(desired_rotational_stiffness * 1.25f, 10.0f);
        spdlog::info("desired_rotational_stiffness -> {:.2f} Nm/rad", desired_rotational_stiffness.load());
    }
    if (joy.left.on_pressed) {
        desired_rotational_stiffness = std::max(desired_rotational_stiffness / 1.25f, 0.1f);
        spdlog::info("desired_rotational_stiffness -> {:.2f} Nm/rad", desired_rotational_stiffness.load());
    }

    if (!policy_ready_) return;
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
