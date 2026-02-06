// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "dexterity/mujoco_extensions/onnx_interface/onnx_interface.h"

void test_inference() {
  const std::string policy_path = "dexterity/data/policies/xinghao_policy_v1.onnx";
  OnnxInterface::VectorT real_output(12);
  real_output << -14.9806f, -15.5571f, 22.7272f, 3.18844f, -3.29105f, 29.6997f, -5.07792f, -3.55804f, 8.1211f,
      -33.9089f, -7.37436f, 10.0438f;
  OnnxInterface::Policy onnx_policy(policy_path);
  OnnxInterface::VectorT input_placeholder_(onnx_policy.input_size);
  input_placeholder_.setOnes();
  assert((onnx_policy.input_size == 84) && (onnx_policy.output_size == 12));
  assert(onnx_policy.getInputName() == "obs");
  assert(onnx_policy.getOutputName() == "actions");
  assert((real_output - onnx_policy.policyInference(&input_placeholder_)).norm() < 1e-4f);
}

int main() {
  test_inference();
  return 0;
}
