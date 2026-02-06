/* Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved. */

#include <pybind11/pybind11.h>
#include <functional>

namespace mujoco_extensions::pybind {

namespace py = pybind11;

namespace jacobian_smoothing {
void bindJacobianSmoothing(const std::reference_wrapper<py::module>& root);
}

/**
   Creating a pybind module. The first argument should correspond to the module name in bazel build. The second argument
   is an instance of py::module that should be used for all bindings.
 */
PYBIND11_MODULE(jacobian_smoothing_pybind, python_module) {
  // Create module docstring.
  python_module.doc() = "MuJoCo Extensions: jacobian smoothing";

  jacobian_smoothing::bindJacobianSmoothing(python_module);
}
}  // namespace mujoco_extensions::pybind
