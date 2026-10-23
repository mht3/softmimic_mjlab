"""Exercise the actual native sensor adapter without DDS or hardware."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_native_sensor_adapter(tmp_path):
    source = tmp_path / 'sensors.cpp'
    source.write_text(r'''
#include <array>
#include <vector>
#include "unitree_articulation.h"
#include <limits>
#include <cassert>
#include <mutex>
#include <memory>
struct IMU {
    std::array<float,4> q{};std::array<float,3> w{};
    auto& quaternion(){return q;}auto& gyroscope(){return w;}
};
struct Motor {
    float pos=0,vel=0;
    float& q(){return pos;}float& dq(){return vel;}
};
struct Message {
    IMU imu;std::array<Motor,29> motors;
    auto& imu_state(){return imu;}auto& motor_state(){return motors;}
};
struct Fake {
    Message msg_;
    unitree::common::UnitreeJoystick joystick;
    std::mutex mutex_;
};
int main() {
    auto sample=std::make_shared<Fake>();
    sample->msg_.imu_state().quaternion()={1.01f,0.f,0.f,0.f};
    sample->msg_.imu_state().gyroscope()={.1f,.2f,.3f};
    sample->msg_.motor_state()[15].q()=.25f;
    unitree::BaseArticulation<std::shared_ptr<Fake>> robot(sample);
    robot.data.joint_ids_map={15};robot.data.joint_pos.resize(1);robot.data.joint_vel.resize(1);
    robot.update();assert(robot.data.joint_pos[0]==.25f);
    assert(std::abs(robot.data.projected_gravity_b[2]+1)<1e-6);
    assert(robot.data.root_ang_vel_b[1]==.2f);
    int rejected=0;
    sample->msg_.imu_state().quaternion()={0,0,0,0};
    try{robot.update();}catch(const std::runtime_error&){rejected++;}
    sample->msg_.imu_state().quaternion()={1,0,0,0};
    sample->msg_.motor_state()[15].dq()=std::numeric_limits<float>::infinity();
    try{robot.update();}catch(const std::runtime_error&){rejected++;}
    sample->msg_.motor_state()[15].dq()=0;
    sample->msg_.imu_state().gyroscope()[0]=std::numeric_limits<float>::quiet_NaN();
    try{robot.update();}catch(const std::runtime_error&){rejected++;}
    assert(rejected==3);
}
''')
    exe = tmp_path / 'sensors'
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'deploy/include'),'-I/usr/include/eigen3',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
