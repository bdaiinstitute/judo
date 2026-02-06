/* Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved. */
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <unsupported/Eigen/CXX11/Tensor>
#include "dexterity/mujoco_extensions/system/system_class.h"
#include "dexterity/mujoco_extensions/system/system_utils.h"

using EigenTypes::MatrixT;
using EigenTypes::VectorT;

void setState(const mjModel* model, mjData* data, const VectorT state) {
  int nq = model->nq;
  int nv = model->nv;
  for (int i = 0; i < nq; ++i) {
    data->qpos[i] = state[i];
  }
  for (int i = 0; i < nv; ++i) {
    data->qvel[i] = state[nq + i];
  }
}

int main() {
  // Paths
  const std::string model_filename = "dexterity/models/xml/scenes/test_scenes/spot_wheel_rim.xml";
  const std::string policy_filename = "dexterity/data/policies/xinghao_policy_v1.onnx";

  std::cout << "Rolling out policy." << std::endl;
  const int command_dim = 25;
  VectorT reference_command =
      VectorT::Zero(command_dim);  // 3 for base, 7 for arm, 12 for legs, 3 for pitch, roll, height of the base
  reference_command[4] = -3.0;
  reference_command[5] = 3.0;
  reference_command[24] = 0.55;
  int physics_substeps = 2;

  int num_commands = 1;
  MatrixT command(num_commands, command_dim);
  command.row(0) = reference_command.transpose();
  command.row(0).segment(0, 3) << 1, 0, 0;  // x, y, theta

  MatrixT posref(num_commands * 2, 3);
  posref.row(0) << 1, 1, 0.;  // x, y, theta
  posref.row(1) << 1, 1, 0.;  // x, y, theta

  SystemClass::System system(model_filename, policy_filename);

  const int nq = system.model->nq;
  const int nv = system.model->nv;
  VectorT initial_state(nq + nv);
  initial_state << 1.0, 1.0, 0.51, 1, 0, 0, 0,                           // base pos
      0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5,  // legs pos
      0, -0.9, 1.8, 0, -0.9, 0, -1.54,                                   // arm pos
      2.0, 0, 0.275, 1, 0, 0, 0,                                         // object pos
      0, 0, 0, 0, 0, 0,                                                  // base vel
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                // legs vel
      0, 0, 0, 0, 0, 0, 0,                                               // arm vel
      0, 0, 0, 0, 0, 0;
  setState(system.model, system.data, initial_state);

  VectorT p_gains(3);
  p_gains << 1., 1., 0.05;

  auto [states, sensors, closed_loop_ctrls] =
      system.rollout_world_frame_feedback(initial_state, command, posref, p_gains, physics_substeps);

  // std::tie(states, sensors) = system.rollout_world_frame(
  //   initial_state, command, physics_substeps);
}
