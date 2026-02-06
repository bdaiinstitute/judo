// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

#include "system/eigen_types.h"

namespace EigenTypes {
VectorTList matrix_to_vector_list(const MatrixT& matrix) {
  VectorTList vector_list;
  auto num_vecs = matrix.rows();
  vector_list.reserve(num_vecs);
  for (auto mat_row = 0; mat_row < num_vecs; ++mat_row) {
    vector_list.emplace_back(matrix.row(mat_row));
  }
  return vector_list;
}

MatrixTList tensor_to_matrix_list(const Tensor3d& tensor) {
  MatrixTList matrix_list;
  const auto& dims = tensor.dimensions();
  matrix_list.reserve(dims[0]);
  for (auto mat_ind = 0; mat_ind < dims[0]; ++mat_ind) {
    MatrixT matrix_slice = Eigen::Map<const MatrixT>(tensor.data() + mat_ind * dims[1] * dims[2], dims[1], dims[2]);
    matrix_list.emplace_back(matrix_slice);
  }
  return matrix_list;
}
}  // namespace EigenTypes
