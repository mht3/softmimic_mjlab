#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include <net/if.h>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        throw std::runtime_error("Another controller owns rt/lowcmd; refusing to take over");
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    FSMState::lowstate->set_timeout_ms(100);
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Fail before opening DDS when the operator has not selected an adapter.
    const auto network = vm["network"].as<std::string>();
    if (network.empty() || network == "lo" || if_nametoindex(network.c_str()) == 0) {
        spdlog::critical("Specify an existing robot Ethernet adapter with --network=INTERFACE (not loopback).");
        return 2;
    }
    if (!param::config["FSM"]) {
        spdlog::critical("Missing controller config: {}", (param::config_dir / "config.yaml").string());
        return 2;
    }

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Console input starts only after CLI validation and robot-mode checks.
    FSMState::keyboard = std::make_shared<Keyboard>();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "Then [R1 + A] enters compliant standing; [L2 + B] returns to Passive.\n";


    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

