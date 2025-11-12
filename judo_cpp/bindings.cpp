#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>

#include "rollout.h"

namespace py = pybind11;

static std::vector<const mjModel*> getModelVector(const py::list& python_models) {
    std::vector<const mjModel*> model_vector;
    model_vector.reserve(python_models.size());
    for (auto&& item : python_models) {
        auto ptr = item.attr("_address").cast<std::uintptr_t>();
        model_vector.push_back(reinterpret_cast<const mjModel*>(ptr));
    }
    return model_vector;
}

static std::vector<mjData*> getDataVector(const py::list& python_data) {
    std::vector<mjData*> data_vector;
    data_vector.reserve(python_data.size());
    for (auto&& item : python_data) {
        auto ptr = item.attr("_address").cast<std::uintptr_t>();
        data_vector.push_back(reinterpret_cast<mjData*>(ptr));
    }
    return data_vector;
}

PYBIND11_MODULE(_judo_cpp, m) {
    // Function to shutdown persistent thread pool
    m.def("shutdown_thread_pool",
          []() {
              ThreadPoolManager::instance().shutdown();
          },
          R"doc(
Shutdown the persistent thread pool.

Call this function to clean up the persistent thread pool when done with rollouts.
The pool will be automatically recreated on the next call to persistent_cpp_rollout.
)doc");

    // Rollout
    m.def("rollout",
          [](const py::list& models,
             const py::list& data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls)
          {
              // turn Python lists into vectors of mjModel*/mjData*
              auto models_cpp = getModelVector(models);
              auto data_cpp   = getDataVector(data);

              // call into your C++ implementation
              return Rollout(models_cpp, data_cpp, x0, controls);
          },
          py::arg("models"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          R"doc(
Run parallel MuJoCo rollouts.

    Args:
    models:                 length-B list of mujoco._structs.MjModel
    data:                   length-B list of mujoco._structs.MjData
    x0:                     2D array of shape (B, nq+nv), batched initial [qpos;qvel]
    controls:               3D array of shape (B, horizon, nu), batched control inputs

Returns:
    tuple of three np.ndarray:
    states  -> shape (B, horizon+1, nq+nv) - MuJoCo states (includes initial state)
    sensors -> shape (B, horizon, nsensordata) - sensor data
)doc");

    // Sim
    m.def("sim",
          [](py::object model,
             py::object data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls)
          {
              auto model_ptr = reinterpret_cast<const mjModel*>(model.attr("_address").cast<std::uintptr_t>());
              auto data_ptr  = reinterpret_cast<mjData*>(data.attr("_address").cast<std::uintptr_t>());
              Sim(model_ptr, data_ptr, x0, controls);
          },
          py::arg("model"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          R"doc(
Run a single MuJoCo simulation step.

    Args:
    model:      mujoco._structs.MjModel
    data:       mujoco._structs.MjData
    x0:         1D array of shape (nq+nv), initial [qpos;qvel]
    controls:   1D array of shape (nu), control input

Returns:
    None
)doc");
}
