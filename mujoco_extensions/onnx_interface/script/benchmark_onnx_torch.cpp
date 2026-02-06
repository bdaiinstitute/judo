// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#include <bits/stdc++.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include <torch/script.h>
#include <torch/torch.h>
#include <Eigen/Dense>

#include "dexterity/mujoco_extensions/onnx_interface/onnx_interface.h"

const int64_t input_size = 84;
const int64_t output_size = 12;
const float max_val = 20.0;

int64_t time_point_to_long(std::chrono::time_point<std::chrono::system_clock> time_point) {
  auto duration = std::chrono::time_point_cast<std::chrono::nanoseconds>(time_point);
  auto epoch = duration.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
  return value.count();
}

// evaluate neural network policy
OnnxInterface::VectorT torch_inference(torch::jit::Module model, const OnnxInterface::VectorT& input_obs) {
  OnnxInterface::VectorT torch_output_vector(output_size);

  auto input_tensor = torch::zeros({1, input_size}, torch::kFloat32);
  float* data_ptr = input_tensor.data_ptr<float>();
  for (int i = 0; i < input_size; ++i) {
    data_ptr[i] = input_obs[i];
  }

  std::vector<torch::jit::IValue> torch_input = {input_tensor};
  at::Tensor torch_output;

  // Execute the model and turn its output into a tensor.
  torch_output = model.forward(torch_input).toTensor();
  torch_output = torch_output.contiguous();
  float* output_data_ptr = torch_output.data_ptr<float>();
  for (int i = 0; i < output_size; i++) {
    torch_output_vector[i] = static_cast<float>(output_data_ptr[i]);
  }
  return torch_output_vector;
}

int main() {
  int num_iterations = 3001;
  OnnxInterface::VectorT input_obs_vector(input_size);
  OnnxInterface::VectorT last_input_vec(input_size);
  last_input_vec.setZero();
  OnnxInterface::VectorT onnx_output;
  OnnxInterface::VectorT torch_output;
  const std::string onnx_policy_path = "dexterity/data/policies/xinghao_policy_v1.onnx";
  const std::string torch_policy_path = "dexterity/data/policies/xinghao_policy_v1.pt";
  uint seed = 29;
  srand(seed);

  // Loads the torch module
  torch::jit::Module model = torch::jit::load(torch_policy_path);

  // Loads the Onnx inference
  OnnxInterface::Policy onnx_policy(onnx_policy_path, OnnxInterface::allocateOrtSession(onnx_policy_path));

  int64_t onnx_inf_avg = 0;
  int64_t torch_inf_avg = 0;
  bool first = true;

  for (int i = 0; i < num_iterations; i++) {
    // Creates a random vector
    for (int j = 0; j < input_size; j++) {
      input_obs_vector[j] = (static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) - 0.5f) * max_val;
    }

    auto start_torch_time = std::chrono::system_clock::now();
    torch_output = torch_inference(model, input_obs_vector);
    auto end_torch_time = std::chrono::system_clock::now();
    if (!first) {
      torch_inf_avg = torch_inf_avg + time_point_to_long(end_torch_time) - time_point_to_long(start_torch_time);
    }

    auto start_onnx_time = std::chrono::system_clock::now();
    onnx_output = onnx_policy.policyInference(&input_obs_vector);
    auto end_onnx_time = std::chrono::system_clock::now();
    if (!first) {
      onnx_inf_avg = onnx_inf_avg + time_point_to_long(end_onnx_time) - time_point_to_long(start_onnx_time);
    }

    if (first) {
      first = !first;
    }

    assert((onnx_output - torch_output).norm() < 0.005f);
  }

  std::cout << "Average torch inference time: " << torch_inf_avg / (num_iterations - 1) << " ns" << std::endl;
  std::cout << "Average onnx interface inference time: " << onnx_inf_avg / (num_iterations - 1) << " ns" << std::endl;

  return 0;
}
