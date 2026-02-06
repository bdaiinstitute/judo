// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#pragma once

#include <vector>

#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

namespace EigenTypes {
using VectorT = Eigen::VectorXd;
using MatrixT = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorTList = std::vector<VectorT>;
using MatrixTList = std::vector<MatrixT>;
using Tensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>;

VectorTList matrix_to_vector_list(const MatrixT& matrix);
MatrixTList tensor_to_matrix_list(const Tensor3d& tensor);

}  // namespace EigenTypes
