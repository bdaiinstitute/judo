#include "spot_rollout.h"
#include <pybind11/stl.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <chrono>

namespace py = pybind11;

// ONNX Policy wrapper class
class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::shared_ptr<Ort::Session>& session)
        : session_(session), memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        Ort::Allocator allocator(*session_, memory_info_);
        input_name_  = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        input_shape_  = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_shape_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_size_ = static_cast<int>(input_shape_[1]);
        output_size_ = static_cast<int>(output_shape_[1]);
    }

    std::vector<float> run(const std::vector<float>& observation) {
        if ((int)observation.size() != input_size_) {
            throw std::runtime_error("Observation size does not match ONNX input dimension");
        }
        std::array<int64_t, 2> ishape = { 1, static_cast<int64_t>(observation.size()) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float*>(observation.data()), observation.size(), ishape.data(), 2);
        const char* in_names[1] = { input_name_.c_str() };
        const char* out_names[1] = { output_name_.c_str() };
        auto outputs = session_->Run(run_options_, in_names, &input_tensor, 1, out_names, 1);
        auto& out = outputs[0];
        float* ptr = out.GetTensorMutableData<float>();
        auto info = out.GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        return std::vector<float>(ptr, ptr + n);
    }

    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }

private:
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    Ort::RunOptions run_options_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    int input_size_ = 0;
    int output_size_ = 0;
};

// Utility functions
py::array_t<double> make_array_owned(std::vector<double>& buf, int B, int T, int D) {
    std::vector<ssize_t> shape   = { B, T, D };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(double) * T * D),
        static_cast<ssize_t>(sizeof(double) *     D),
        static_cast<ssize_t>(sizeof(double))
    };
    auto heap_buf = new std::vector<double>(std::move(buf));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(shape, strides, heap_buf->data(), free_when_done);
}

// ONNX session allocator
static std::shared_ptr<Ort::Session> allocate_shared_session(const std::string& onnx_path) {
    static Ort::Env env;
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    return std::make_shared<Ort::Session>(env, onnx_path.c_str(), opts);
}

static std::string get_policy_path() {
    return std::string("judo_cpp/policy/spot_relic_policy.onnx");
}

// Constants and helper functions (from rollout_spot.cpp)
static const double ACTION_SCALE = 0.2;
static const double STANDING_POS_RL[19] = {
    0.12, 0.5, -1.0,  -0.12, 0.5, -1.0,  0.12, 0.5, -1.0,  -0.12, 0.5, -1.0,
    0.0, -0.9, 1.8,  0.0, -0.9, 0.0, -1.54
};

static const int ISAAC_TO_MUJOCO_INDICES_12[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
static const int ISAAC_TO_MUJOCO_INDICES_19[19] = {1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18};
static const int MUJOCO_TO_ISAAC_INDICES_12[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
static const int MUJOCO_TO_ISAAC_INDICES_19[19] = {12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18};

static void isaac_to_mujoco_12(const double* isaac, double* mujoco) {
    for (int i = 0; i < 12; i++) {
        mujoco[i] = isaac[ISAAC_TO_MUJOCO_INDICES_12[i]];
    }
}

static void mujoco_to_isaac_19(const double* mujoco, double* isaac) {
    for (int i = 0; i < 19; i++) {
        isaac[i] = mujoco[MUJOCO_TO_ISAAC_INDICES_19[i]];
    }
}

// =============================================================================
// SpotThreadPool Implementation
// =============================================================================

SpotThreadPool::SpotThreadPool(int num_threads)
    : num_threads_(num_threads), stop_(false), active_workers_(0), total_tasks_(0), completed_tasks_(0) {
    if (num_threads_ > 0) {
        threads_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; ++i) {
            threads_.emplace_back(&SpotThreadPool::worker_thread, this);
        }
    }
}

SpotThreadPool::~SpotThreadPool() {
    if (num_threads_ > 0) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread &worker : threads_) {
            worker.join();
        }
    }
}

void SpotThreadPool::execute_parallel(std::function<void(int)> func, int total_work) {
    if (num_threads_ == 0) {
        // Single-threaded execution
        for (int i = 0; i < total_work; ++i) {
            func(i);
        }
    } else {
        // Multi-threaded execution
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            total_tasks_ = total_work;
            completed_tasks_ = 0;

            for (int i = 0; i < total_work; ++i) {
                tasks_.push([func, i]() { func(i); });
            }
        }
        condition_.notify_all();

        std::unique_lock<std::mutex> lock(queue_mutex_);
        finished_.wait(lock, [this]() { return completed_tasks_ == total_tasks_; });
    }
}

void SpotThreadPool::worker_thread() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty())
                return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            completed_tasks_++;
            if (completed_tasks_ == total_tasks_) {
                finished_.notify_one();
            }
        }
    }
}

// =============================================================================
// SpotRollout Implementation
// =============================================================================

SpotRollout::SpotRollout(int nthread, double cutoff_time) : num_threads_(nthread), cutoff_time_(cutoff_time) {
    initialize_policy();
    if (num_threads_ != 0) {
        thread_pool_ = std::make_unique<SpotThreadPool>(num_threads_);
    }
}

SpotRollout::~SpotRollout() {
    close();
}

void SpotRollout::close() {
    if (!closed_) {
        thread_pool_.reset();
        policy_.reset();
        onnx_session_.reset();
        closed_ = true;
    }
}

void SpotRollout::initialize_policy() {
    onnx_session_ = allocate_shared_session(get_policy_path());
    policy_ = std::make_unique<OnnxPolicy>(onnx_session_);
}

py::tuple SpotRollout::rollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>& data,
    const py::array_t<double>& initial_state,
    const py::array_t<double>& controls
) {
    if (closed_) {
        throw std::runtime_error("Rollout requested after object was closed");
    }

    int B = (int)models.size();
    if (B == 0 || B != (int)data.size()) {
        throw std::runtime_error("models/data must have same non-zero length");
    }

    int horizon = (int)controls.shape(1);
    const mjModel* m0 = models[0];
    int nq = m0->nq;
    int nv = m0->nv;
    int nu = m0->nu;
    int nsens = m0->nsensordata;
    int nstate = nq + nv;

    if (initial_state.ndim() != 2 || initial_state.shape(0) != B || initial_state.shape(1) != nstate) {
        throw std::runtime_error("initial_state must be a 2D array of shape (B, nq+nv)");
    }

    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> sens_buf(B * horizon * nsens);

    auto controls_unchecked = controls.unchecked<3>();
    const double* x0_ptr = initial_state.data();

    std::vector<std::vector<float>> prev_policy(B, std::vector<float>(12, 0.0f));

    {
        py::gil_scoped_release release;

        auto execute_work = [&](int i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            mjData* d = data[i];
            const mjModel* m = models[i];

            d->time = 0.0;
            const double* x0_i = x0_ptr + i * nstate;
            mj_setState(m, d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
            mj_forward(m, d);
            mju_zero(d->qacc_warmstart, m->nv);

            double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
            double* se_ptr = &sens_buf[i * horizon * nsens];

            for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
            for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];

            int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
            compute_indices(m, base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start);

            for (int t = 0; t < horizon; t++) {
                // Check timeout before each step
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                if (elapsed > cutoff_time_) {
                    // Timeout reached, fill remaining states with current state and return
                    for (int remaining_t = t; remaining_t < horizon; remaining_t++) {
                        for (int j = 0; j < nq; j++) st_ptr[(remaining_t + 1) * nstate + j] = d->qpos[j];
                        for (int j = 0; j < nv; j++) st_ptr[(remaining_t + 1) * nstate + nq + j] = d->qvel[j];
                        for (int j = 0; j < nsens; j++) se_ptr[remaining_t * nsens + j] = d->sensordata[j];
                    }
                    return;
                }

                std::vector<float> obs;
                double cmd_buf[25];
                for (int j = 0; j < 25; j++) {
                    cmd_buf[j] = static_cast<double>(controls_unchecked(i, t, j));
                }

                build_observation(m, d, cmd_buf, prev_policy[i],
                                base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start, obs);

                auto policy_out_vec = policy_->run(obs);

                std::vector<double> ctrl;
                compute_control_from_policy(policy_out_vec.data(), cmd_buf, ctrl);

                if ((int)ctrl.size() != nu) {
                    throw std::runtime_error("Computed control size does not match model nu");
                }
                for (int j = 0; j < nu; j++) d->ctrl[j] = ctrl[j];

                mj_step(m, d);
                prev_policy[i] = std::move(policy_out_vec);

                for (int j = 0; j < nq; j++) st_ptr[(t + 1) * nstate + j] = d->qpos[j];
                for (int j = 0; j < nv; j++) st_ptr[(t + 1) * nstate + nq + j] = d->qvel[j];
                for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = d->sensordata[j];
            }
        };

        if (num_threads_ == 0) {
            // Single-threaded execution
            for (int i = 0; i < B; ++i) {
                execute_work(i);
            }
        } else {
            // Multi-threaded execution
            thread_pool_->execute_parallel(execute_work, B);
        }
    }

    auto states_arr = make_array_owned(states_buf, B, horizon + 1, nstate);
    auto sens_arr = make_array_owned(sens_buf, B, horizon, nsens);
    return py::make_tuple(states_arr, sens_arr);
}

SpotRollout* SpotRollout::__enter__() {
    return this;
}

void SpotRollout::__exit__(py::object exc_type, py::object exc_val, py::object exc_tb) {
    close();
}

int SpotRollout::get_num_threads() const {
    return num_threads_;
}

// Helper method implementations
void SpotRollout::compute_indices(const mjModel* m, int& base_qpos_start, int& base_qvel_start,
                                  int& leg_qpos_start, int& leg_qvel_start) {
    int free_joint_idx = -1;
    for (int j = 0; j < m->njnt; j++) {
        if (m->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        free_joint_idx = 0;
    }
    base_qpos_start = m->jnt_qposadr[free_joint_idx];
    base_qvel_start = m->jnt_dofadr[free_joint_idx];
    leg_qpos_start = base_qpos_start + 7;
    leg_qvel_start = base_qvel_start + 6;
}

void SpotRollout::build_observation(const mjModel* m, mjData* d, const double* command_ptr,
                                   const std::vector<float>& prev_policy,
                                   int base_qpos_start, int base_qvel_start,
                                   int leg_qpos_start, int leg_qvel_start,
                                   std::vector<float>& obs_out) {
    obs_out.resize(84);
    int off = 0;

    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = d->qpos[base_qpos_start + 3 + i];
    }

    double invq[4];
    mju_negQuat(invq, quat);

    double blin[3];
    mju_rotVecQuat(blin, d->qvel + base_qvel_start, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(blin[i]);

    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(d->qvel[base_qvel_start + 3 + i]);

    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(gvec_rotated[i]);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(command_ptr[0 + i]);
    for (int i = 0; i < 7; i++) obs_out[off++] = static_cast<float>(command_ptr[3 + i]);
    for (int i = 0; i < 12; i++) obs_out[off++] = static_cast<float>(command_ptr[10 + i]);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(command_ptr[22 + i]);

    double jpos_raw[19];
    double jvel_raw[19];
    for (int i = 0; i < 19; i++) {
        jpos_raw[i] = d->qpos[leg_qpos_start + i] - STANDING_POS_RL[i];
        jvel_raw[i] = d->qvel[leg_qvel_start + i];
    }

    double jpos_isaac[19];
    double jvel_isaac[19];
    mujoco_to_isaac_19(jpos_raw, jpos_isaac);
    mujoco_to_isaac_19(jvel_raw, jvel_isaac);

    for (int i = 0; i < 19; i++) obs_out[off++] = static_cast<float>(jpos_isaac[i]);
    for (int i = 0; i < 19; i++) obs_out[off++] = static_cast<float>(jvel_isaac[i]);
    for (int i = 0; i < 12; i++) obs_out[off++] = (i < (int)prev_policy.size() ? prev_policy[i] : 0.0f);
}

void SpotRollout::compute_control_from_policy(const float* policy_out, const double* command_ptr,
                                             std::vector<double>& ctrl_out) {
    ctrl_out.resize(19);
    double action_isaac[12];
    for (int i = 0; i < 12; i++) action_isaac[i] = static_cast<double>(policy_out[i]);

    double target_leg_isaac[12];
    for (int i = 0; i < 12; i++) target_leg_isaac[i] = action_isaac[i] * ACTION_SCALE;

    double target_leg_mj[12];
    isaac_to_mujoco_12(target_leg_isaac, target_leg_mj);
    for (int i = 0; i < 12; i++) target_leg_mj[i] += STANDING_POS_RL[i];

    for (int i = 0; i < 12; i++) ctrl_out[i] = target_leg_mj[i];
    for (int i = 0; i < 7; i++) ctrl_out[12 + i] = static_cast<double>(command_ptr[3 + i]);

    const double* leg_cmd = command_ptr + 10;
    auto has_nonzero = [](const double* v) { return v[0] != 0.0 || v[1] != 0.0 || v[2] != 0.0; };

    if (has_nonzero(leg_cmd + 0)) {
        for (int i = 0; i < 3; i++) ctrl_out[0 + i] = leg_cmd[0 + i];
    } else if (has_nonzero(leg_cmd + 3)) {
        for (int i = 0; i < 3; i++) ctrl_out[3 + i] = leg_cmd[3 + i];
    } else if (has_nonzero(leg_cmd + 6)) {
        for (int i = 0; i < 3; i++) ctrl_out[6 + i] = leg_cmd[6 + i];
    } else if (has_nonzero(leg_cmd + 9)) {
        for (int i = 0; i < 3; i++) ctrl_out[9 + i] = leg_cmd[9 + i];
    }
}

// =============================================================================
// SimSpot - Single-step simulation with Spot policy
// =============================================================================

py::array_t<float> SimSpot(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls,
    const py::array_t<float>& prev_policy
) {
    static std::shared_ptr<Ort::Session> onnx_session = nullptr;
    static std::unique_ptr<OnnxPolicy> policy = nullptr;

    // Initialize policy on first call
    if (!onnx_session || !policy) {
        onnx_session = allocate_shared_session(get_policy_path());
        policy = std::make_unique<OnnxPolicy>(onnx_session);
    }

    int nq = model->nq;
    int nv = model->nv;
    int nu = model->nu;

    if (x0.size() != nq + nv) {
        throw std::runtime_error("x0 size must equal nq + nv");
    }
    if (controls.size() != 25) {
        throw std::runtime_error("controls size must be 25");
    }
    if (prev_policy.size() != 12) {
        throw std::runtime_error("prev_policy size must be 12");
    }

    // Set initial state
    const double* x0_ptr = x0.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);

    // Convert prev_policy to vector
    std::vector<float> prev_policy_vec(12);
    const float* prev_policy_ptr = prev_policy.data();
    for (int i = 0; i < 12; i++) {
        prev_policy_vec[i] = prev_policy_ptr[i];
    }

    // Get command data
    const double* cmd_ptr = controls.data();
    double cmd_buf[25];
    for (int i = 0; i < 25; i++) {
        cmd_buf[i] = cmd_ptr[i];
    }

    // Compute indices
    int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
    int free_joint_idx = -1;
    for (int j = 0; j < model->njnt; j++) {
        if (model->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        free_joint_idx = 0;
    }
    base_qpos_start = model->jnt_qposadr[free_joint_idx];
    base_qvel_start = model->jnt_dofadr[free_joint_idx];
    leg_qpos_start = base_qpos_start + 7;
    leg_qvel_start = base_qvel_start + 6;

    // Build observation
    std::vector<float> obs(84);
    int off = 0;

    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = data->qpos[base_qpos_start + 3 + i];
    }

    double invq[4];
    mju_negQuat(invq, quat);

    double blin[3];
    mju_rotVecQuat(blin, data->qvel + base_qvel_start, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(blin[i]);

    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(data->qvel[base_qvel_start + 3 + i]);

    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(gvec_rotated[i]);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(cmd_buf[0 + i]);
    for (int i = 0; i < 7; i++) obs[off++] = static_cast<float>(cmd_buf[3 + i]);
    for (int i = 0; i < 12; i++) obs[off++] = static_cast<float>(cmd_buf[10 + i]);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(cmd_buf[22 + i]);

    double jpos_raw[19];
    double jvel_raw[19];
    for (int i = 0; i < 19; i++) {
        jpos_raw[i] = data->qpos[leg_qpos_start + i] - STANDING_POS_RL[i];
        jvel_raw[i] = data->qvel[leg_qvel_start + i];
    }

    double jpos_isaac[19];
    double jvel_isaac[19];
    mujoco_to_isaac_19(jpos_raw, jpos_isaac);
    mujoco_to_isaac_19(jvel_raw, jvel_isaac);

    for (int i = 0; i < 19; i++) obs[off++] = static_cast<float>(jpos_isaac[i]);
    for (int i = 0; i < 19; i++) obs[off++] = static_cast<float>(jvel_isaac[i]);
    for (int i = 0; i < 12; i++) obs[off++] = prev_policy_vec[i];

    // Run policy
    auto policy_out_vec = policy->run(obs);

    // Compute control
    std::vector<double> ctrl(19);
    double action_isaac[12];
    for (int i = 0; i < 12; i++) action_isaac[i] = static_cast<double>(policy_out_vec[i]);

    double target_leg_isaac[12];
    for (int i = 0; i < 12; i++) target_leg_isaac[i] = action_isaac[i] * ACTION_SCALE;

    double target_leg_mj[12];
    isaac_to_mujoco_12(target_leg_isaac, target_leg_mj);
    for (int i = 0; i < 12; i++) target_leg_mj[i] += STANDING_POS_RL[i];

    for (int i = 0; i < 12; i++) ctrl[i] = target_leg_mj[i];
    for (int i = 0; i < 7; i++) ctrl[12 + i] = static_cast<double>(cmd_buf[3 + i]);

    const double* leg_cmd = cmd_buf + 10;
    auto has_nonzero = [](const double* v) { return v[0] != 0.0 || v[1] != 0.0 || v[2] != 0.0; };

    if (has_nonzero(leg_cmd + 0)) {
        for (int i = 0; i < 3; i++) ctrl[0 + i] = leg_cmd[0 + i];
    } else if (has_nonzero(leg_cmd + 3)) {
        for (int i = 0; i < 3; i++) ctrl[3 + i] = leg_cmd[3 + i];
    } else if (has_nonzero(leg_cmd + 6)) {
        for (int i = 0; i < 3; i++) ctrl[6 + i] = leg_cmd[6 + i];
    } else if (has_nonzero(leg_cmd + 9)) {
        for (int i = 0; i < 3; i++) ctrl[9 + i] = leg_cmd[9 + i];
    }

    // Apply control and step
    for (int j = 0; j < nu; j++) data->ctrl[j] = ctrl[j];
    mj_step(model, data);

    // Return new policy output as numpy array
    std::vector<ssize_t> shape = {12};
    std::vector<ssize_t> strides = {sizeof(float)};
    auto heap_buf = new std::vector<float>(std::move(policy_out_vec));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<float>*>(p); });
    return py::array_t<float>(shape, strides, heap_buf->data(), free_when_done);
}
