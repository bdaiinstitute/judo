// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <unsupported/Eigen/CXX11/Tensor>

#include "dexterity/mujoco_extensions/system/eigen_types.h"

void test_tensor_to_matrix_list() {
  EigenTypes::Tensor3d tensor(3, 3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        tensor(i, j, k) = 100 * i + 10 * j + k;
      }
    }
  }
  EigenTypes::MatrixTList matrix_list = EigenTypes::tensor_to_matrix_list(tensor);
  assert(std::abs(matrix_list[0](1, 2) - 12) < 1e-7);
  assert(std::abs(matrix_list[2](2, 2) - 222) < 1e-7);
  // Time the creation of a large matrix
  EigenTypes::Tensor3d tensor2(100, 100, 100);
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 100; ++j) {
      for (int k = 0; k < 100; ++k) {
        tensor2(i, j, k) = i + j + k;
      }
    }
  }
  auto start_fn = std::chrono::high_resolution_clock::now();
  EigenTypes::MatrixTList matrix_list2 = EigenTypes::tensor_to_matrix_list(tensor2);
  auto end_fn = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_fn - start_fn);
  std::cout << "Conversion time for 100 x 100 x 100 tensor to std::vector of matrices: " << duration.count()
            << " microseconds" << std::endl;
}

void test_matrix_to_vector_list() {
  EigenTypes::MatrixT matrix{{0, 1, 2}, {3, 4, 5}};
  EigenTypes::VectorTList vector_list = EigenTypes::matrix_to_vector_list(matrix);
  assert(std::abs(vector_list[0](0) - 0) < 1e-7);
  assert(std::abs(vector_list[1](1) - 4) < 1e-7);
  EigenTypes::MatrixT matrix2(100, 100);
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 100; ++j) {
      matrix2(i, j) = i + j;
    }
  }
  auto start_fn = std::chrono::high_resolution_clock::now();
  EigenTypes::VectorTList vector_list2 = EigenTypes::matrix_to_vector_list(matrix2);
  auto end_fn = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_fn - start_fn);
  std::cout << "Conversion time for 100 x 100 matrix to list of vectors: " << duration.count() << " microseconds"
            << std::endl;
}

int main() {
  test_tensor_to_matrix_list();
  test_matrix_to_vector_list();
  return 0;
}
