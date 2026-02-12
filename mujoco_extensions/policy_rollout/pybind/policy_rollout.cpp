/* Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved. */
#include <mujoco/mujoco.h>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <unsupported/Eigen/CXX11/Tensor>
#include "system/system_class.h"
#include "system/system_utils.h"
#include "pybind11/eval.h"

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

namespace mujoco_extensions::pybind::policy_rollout {

namespace py = pybind11;

/**
 * Setup openmp environment.
 */
constexpr std::size_t OMP_NUM_THREADS{8};

/**
 * Define types that will be used throughout this file.
 */
using EigenTypes::MatrixT;
using EigenTypes::MatrixTList;
using EigenTypes::Tensor3d;
using EigenTypes::VectorT;
using EigenTypes::VectorTList;

/**
 * Get model and data pointers from plant object of `Plant` python type. We need this function because C++ has no
 * control over this python object, and hence we cannot simply access its storage. This function returns two pointers to
 * objects of `mjModel` and `mjData` types.
 *
 * The design follows this suggestion https://github.com/google-deepmind/mujoco/issues/983#issuecomment-1643732152
 */
// Function to convert a Python list of mjModel objects to a std::vector<mjModel_ *>
std::vector<const mjModel*> getModelVector(const py::list& python_model) {
  std::vector<const mjModel*> model_vector;

  for (const auto& item : python_model) {
    // Retrieve the raw pointer from the Python object
    const auto modelPointer = item.attr("_address").cast<std::uintptr_t>();
    const mjModel* model = reinterpret_cast<const mjModel*>(modelPointer);
    model_vector.push_back(model);
  }

  return model_vector;
}

// Function to convert a Python list of mjData objects to a std::vector<mjData_ *>
std::vector<mjData*> getDataVector(const py::list& python_data) {
  std::vector<mjData*> data_vector;

  for (const auto& item : python_data) {
    // Retrieve the raw pointer from the Python object
    const auto dataPointer = item.attr("_address").cast<std::uintptr_t>();
    mjData* data = reinterpret_cast<mjData*>(dataPointer);
    data_vector.push_back(data);
  }

  return data_vector;
}

/**
 * Bindings for `policy_rollout` submodule.
 */
void bindPolicyRollout(const std::reference_wrapper<py::module>& root) {
  using pybind11::literals::operator""_a;

  Eigen::setNbThreads(OMP_NUM_THREADS);
  // Create `policy_rollout` submodule.
  auto python_module = root.get().def_submodule("policy_rollout");

  // SystemUtils
  python_module.def(
      "get_joint_proportional_gains",
      [](const py::object& python_model) -> VectorT {
        const auto modelPointer = python_model.attr("_address").cast<std::uintptr_t>();
        const mjModel* model = reinterpret_cast<const mjModel*>(modelPointer);
        return SystemUtils::getJointProportionalGains(model);
      },
      "Retrieve the proportional gain used by the PD controller from the mujoco model.");

  python_module.def(
      "set_state",
      [](std::shared_ptr<SystemClass::System> systemObject, const VectorT& state) -> void {
        SystemUtils::setState(systemObject->model, systemObject->data, state);
      },
      "Sets the state of a system object", py::arg("system"), py::arg("state"));

  python_module.def(
      "threaded_physics_rollout",
      [](const py::list& python_model, const py::list& python_data, const VectorTList& state,
         const MatrixTList& control) -> std::tuple<MatrixTList, MatrixTList> {
        // Convert Python lists to C++ vectors of raw pointers
        auto model = getModelVector(python_model);
        std::vector<mjData*> data = getDataVector(python_data);

        // Call the threadedPhysicsRollout function
        return SystemUtils::threadedPhysicsRollout(model, data, state, control);
      },
      "model"_a, "data"_a, "state"_a, "control"_a);

  python_module.def(
      "threaded_physics_rollout",
      [](const py::list& python_model, const py::list& python_data, const MatrixT& state,
         const Tensor3d& control) -> std::tuple<MatrixTList, MatrixTList> {
        auto model = getModelVector(python_model);
        std::vector<mjData*> data = getDataVector(python_data);
        auto state_list = EigenTypes::matrix_to_vector_list(state);
        auto control_list = EigenTypes::tensor_to_matrix_list(control);
        return SystemUtils::threadedPhysicsRollout(model, data, state_list, control_list);
      },
      "model"_a, "data"_a, "state"_a, "control"_a);

  // SystemClass
  py::class_<SystemClass::System, std::shared_ptr<SystemClass::System>>(python_module, "System")
      .def(py::init<const std::string&, const std::string&>(), "model_filepath"_a, "policy_filepath"_a)
      .def_readwrite("observation", &SystemClass::System::observation)
      .def_property(
          "policy_output",
          [](SystemClass::System& system) {
            return system.policy_output;
          },
          [](SystemClass::System& system, const VectorT& value) {
            system.policy_output.resize(value.size());
            std::copy(value.begin(), value.end(), system.policy_output.begin());
          })
      .def("reset", &SystemClass::System::reset)
      .def("load_policy", &SystemClass::System::loadPolicy)
      .def("set_observation", &SystemClass::System::setObservation, "command"_a)
      .def("policy_inference", &SystemClass::System::policyInference)
      .def("get_control", &SystemClass::System::getControl)
      .def("get_state", &SystemClass::System::getState)
      .def("rollout", &SystemClass::System::rollout, "state"_a, "command"_a, "physics_substeps"_a = 2,
           "reset_last_output"_a = true, "cutoff_time"_a = SystemClass::kInfiniteTime);  // Match C++

  python_module.def("threaded_rollout", &SystemClass::threadedRollout,
                    "Threaded policy rollout with shared pointers to System objects.", "systems"_a, "states"_a,
                    "command"_a, "last_policy_output"_a, "num_threads"_a, "physics_substeps"_a,
                    "cutoff_time"_a = SystemClass::kInfiniteTime);

  python_module.def(
      "create_systems_vector",
      [](const py::object& python_model, const std::string& policy_filepath,
         const int num_systems) -> std::vector<std::shared_ptr<SystemClass::System>> {
        const auto modelPointer = python_model.attr("_address").cast<std::uintptr_t>();
        const mjModel* reference_model = reinterpret_cast<const mjModel*>(modelPointer);

        return SystemClass::create_systems_vector(reference_model, policy_filepath, num_systems);
      },
      "model"_a, "policy_filepath"_a, "num_systems"_a);
}

}  // namespace mujoco_extensions::pybind::policy_rollout
