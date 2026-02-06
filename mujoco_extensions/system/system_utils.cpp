// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

#include "system/system_utils.h"

namespace SystemUtils {

mjModel* loadModel(const std::string& model_filepath) {
  char error_buffer[1000] = "Could not load XML file";

  // Attempt to load the model from the XML file
  mjModel* model = mj_loadXML(model_filepath.c_str(), nullptr, error_buffer, sizeof(error_buffer));

  // Check if the model failed to load
  if (!model) {
    throw std::runtime_error("Failed to load model XML file: " + model_filepath + " - " + error_buffer);
  }
  return model;
}

void setState(const mjModel* model, mjData* data, const VectorT& state) {
  int nq = model->nq;
  int nv = model->nv;

  // Check that the state size is correct
  if (state.size() != nq + nv) {
    std::cerr << "Error: State size (" << state.size() << ") does not match "
              << "the expected size (" << nq + nv << ")." << std::endl;
    throw std::runtime_error("State size mismatch");
  }

  for (int i = 0; i < nq; ++i) {
    data->qpos[i] = state[i];
  }
  for (int i = 0; i < nv; ++i) {
    data->qvel[i] = state[nq + i];
  }
}

VectorT getJointProportionalGains(const mjModel* model) {
  const auto actuationNumber = model->nu;
  return Eigen::Map<VectorT, 0, Eigen::InnerStride<mjNGAIN>>(model->actuator_gainprm, actuationNumber);
}

std::tuple<MatrixT, MatrixT> physicsRollout(const mjModel* model, mjData* data, const VectorT& state,
                                            const MatrixT& control) {
  // Dimensions
  const auto horizon = control.rows();
  const int nq = model->nq;
  const int nv = model->nv;
  const int nu = model->nu;
  const int state_dim = nq + nv;
  const int nsensordata = model->nsensordata;

  // Copy initial state into mjData.
  for (int l = 0; l < nq; ++l) {
    data->qpos[l] = state[l];
  }
  for (int l = 0; l < nv; ++l) {
    data->qvel[l] = state[l + nq];
  }
  // mj_forward(model, data);

  // Matrices to store states and sensors
  MatrixT states = MatrixT::Zero(horizon, state_dim);
  MatrixT sensors = MatrixT::Zero(horizon, nsensordata);

  // Rollout physics and policy
  for (int i = 0; i < horizon; ++i) {
    // Copy current controls into mjData.
    for (int l = 0; l < nu; ++l) {
      data->ctrl[l] = control(i, l);
    }
    mj_step(model, data);
    for (int l = 0; l < nq; ++l) {
      states(i, l) = data->qpos[l];
    }
    for (int l = 0; l < nv; ++l) {
      states(i, nq + l) = data->qvel[l];
    }
    for (int l = 0; l < nsensordata; ++l) {
      sensors(i, l) = data->sensordata[l];
    }
  }
  return {states, sensors};
}

std::tuple<MatrixTList, MatrixTList> threadedPhysicsRollout(const std::vector<const mjModel*>& model,
                                                            const std::vector<mjData*>& data, const VectorTList& state,
                                                            const MatrixTList& control) {
  // dimensions
  // model [num_threads]
  // data [num_threads]
  // state [num_threads, state_dim]
  // control [num_threads, horizon, state_dim]

  const auto num_threads = model.size();
  const auto horizon = control[0].rows();
  const int state_dim = model[0]->nq + model[0]->nv;
  const int nu = model[0]->nu;
  const int nsensordata = model[0]->nsensordata;
  const auto data_size = data.size();
  const auto state_size = state.size();
  const auto control_size = control.size();

  // Check sizes
  if (data_size != num_threads) {
    std::cerr << "Error: data size (" << data.size() << ") does not match "
              << "expected size (" << num_threads << ")." << std::endl;
    throw std::runtime_error("Data size mismatch");
  }

  if (state_size != num_threads) {
    std::cerr << "Error: state size (" << state.size() << ") do not match "
              << "expected size (" << num_threads << ")." << std::endl;
    throw std::runtime_error("State size mismatch");
  }

  for (const auto& s : state) {
    if (s.size() != state_dim) {
      std::cerr << "Error: state vector size (" << s.size() << ") does not match "
                << "expected size (" << state_dim << ")." << std::endl;
      throw std::runtime_error("State vector size mismatch");
    }
  }

  if (control_size != num_threads) {
    std::cerr << "Error: control size (" << control.size() << ") does not match "
              << "expected size (" << num_threads << ")." << std::endl;
    throw std::runtime_error("Control size mismatch");
  }

  for (const auto& ctrl : control) {
    if (ctrl.rows() != horizon) {
      std::cerr << "Error: control rows (" << ctrl.rows() << ") do not match "
                << "expected size (" << horizon << ")." << std::endl;
      throw std::runtime_error("Control size mismatch");
    }
    if (ctrl.cols() != nu) {
      std::cerr << "Error: control columns (" << ctrl.cols() << ") do not match "
                << "expected size (" << nu << ")." << std::endl;
      throw std::runtime_error("Control size mismatch");
    }
  }

  // Vectors to store states and sensors for each thread
  MatrixTList states(num_threads, MatrixT::Zero(horizon, state_dim));
  MatrixTList sensors(num_threads, MatrixT::Zero(horizon, nsensordata));

  // Vector to store threads
  std::vector<std::thread> threads;

  // Start threads
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([i, &model, &data, &state, &control, &states, &sensors]() {
      // Run the rollout function in each thread
      std::tie(states[i], sensors[i]) = physicsRollout(model[i], data[i], state[i], control[i]);
    });
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Return the results
  return {states, sensors};
}

}  // namespace SystemUtils
