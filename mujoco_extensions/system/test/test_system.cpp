// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/LU>

#include "dexterity/mujoco_extensions/system/system_class.h"

void test_rollout(const std::string& model_filepath, const std::string& policy_filepath) {
  // parameters
  const int physics_substeps = 2;
  const int command_dim = 25;
  const int state_dim = 51;
  int num_commands = 200;
  int num_steps = num_commands * physics_substeps;
  static_cast<void>(num_steps);

  // initialization
  EigenTypes::VectorT initial_state(state_dim);
  // qpos (26 elements): base_pos (3), base_quat (4), leg_joints (12), arm_joints (7)
  initial_state << 0.0, 0.0, 0.51, 1.0, 0.0, 0.0, 0.0,                   // base qpos (7)
      0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5,  // leg joints qpos (12)
      0.0, -0.9, 1.8, 0.0, -0.9, 0.0, 0.0,                               // arm joints qpos (7)
      // qvel (25 elements): base_vel (6), leg_joint_vel (12), arm_joint_vel (7)
      0.0, 0.26, 1.0, 0.0, 0.0, 0.0,                               // base qvel (6)
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // leg joints qvel (12)
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;                           // arm joints qvel (7)
  // Replace this with the actual initial system
  EigenTypes::VectorT default_command(command_dim);
  default_command << 0, 0, 0, 0, -0.9, 1.8, 0, -0.9, 0, -1.54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55;

  EigenTypes::MatrixT command(num_commands, command_dim);
  for (int i = 0; i < num_commands; i++) {
    command.row(i) = default_command;
  }
  SystemClass::System system(model_filepath, policy_filepath);

  // Rollouts of the policy are deterministic
  auto [states0, sensors0] = system.rollout(initial_state, command, physics_substeps);
  auto [states1, sensors1] = system.rollout(initial_state, command, physics_substeps);

  // Dimensions are correct
  assert(states0.rows() == num_steps);
  assert(states0.cols() == system.model->nq + system.model->nv);
  assert(sensors0.rows() == num_steps);
  assert(sensors0.cols() == system.model->nsensordata);

  std::cout << "delta_states     " << (states0 - states1).norm() << std::endl;
  std::cout << "delta_sensors    " << (sensors0 - sensors1).norm() << std::endl;

  // Rollouts are deterministic
  assert((states0 - states1).norm() < 1e-10);
  assert((sensors0 - sensors1).norm() < 1e-10);

  assert(states0 == states1);
  assert(sensors0 == sensors1);

  // Rolling a sequence of commands is equivalent to running them separately
  auto [states2, sensors2] = system.rollout(initial_state, command.middleRows(0, 100), physics_substeps);

  bool reset_last_output = false;
  auto [states3, sensors3] = system.rollout(states2.row(states2.rows() - 1), command.middleRows(100, 100),
                                            physics_substeps, reset_last_output);

  auto [states4, sensors4] = system.rollout(initial_state, command, physics_substeps);

  // Dimensions are correct
  std::cout << "states2     " << states2.rows() << "   " << states2.cols() << std::endl;
  std::cout << "states3     " << states3.rows() << "   " << states3.cols() << std::endl;
  std::cout << "states4     " << states4.rows() << "   " << states4.cols() << std::endl;

  int half_steps = num_commands / 2 * physics_substeps;
  std::cout << "delta_states   1st half  " << (states2 - states4.middleRows(0, half_steps)).norm() << std::endl;
  std::cout << "delta_states   2nd half  " << (states3 - states4.middleRows(half_steps, half_steps)).norm()
            << std::endl;
  std::cout << "delta_sensors  1st half  " << (sensors2 - sensors4.middleRows(0, half_steps)).norm() << std::endl;
  std::cout << "delta_sensors  2nd half  " << (sensors3 - sensors4.middleRows(half_steps, half_steps)).norm()
            << std::endl;

  assert((states2 - states4.middleRows(0, half_steps)).norm() < 1e-8);
  assert((states3 - states4.middleRows(half_steps, half_steps)).norm() < 1e-8);
  assert((sensors2 - sensors4.middleRows(0, half_steps)).norm() < 1e-8);
  assert((sensors3 - sensors4.middleRows(half_steps, half_steps)).norm() < 1e-8);
  std::cout << "finish" << std::endl;
}

void test_system_constructor(const std::string& model_filepath, const std::string& policy_filepath) {
  SystemClass::System system1(model_filepath, policy_filepath);

  char error_buffer[1000] = "Could not load XML file";
  mjModel* model = mj_loadXML(model_filepath.c_str(), nullptr, error_buffer, sizeof(error_buffer));
  std::shared_ptr<OnnxInterface::Session> reference_session =
      std::shared_ptr<OnnxInterface::Session>(OnnxInterface::allocateOrtSession(policy_filepath));
  SystemClass::System system2(policy_filepath, model, reference_session);
}

int main() {
  // Paths
  const std::string model_filepath = "assets/robots/spot_fast/xml/robot.xml";
  const std::string policy_filepath = "dexterity/data/policies/xinghao_policy_v1.onnx";

  test_system_constructor(model_filepath, policy_filepath);
  test_rollout(model_filepath, policy_filepath);
  return 0;
}
