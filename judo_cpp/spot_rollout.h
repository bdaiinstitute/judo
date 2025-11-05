#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <mujoco/mujoco.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

namespace py = pybind11;

// Forward declaration
class OnnxPolicy;

/**
 * Simple thread pool for parallel execution
 */
class SpotThreadPool {
public:
    explicit SpotThreadPool(int num_threads);
    ~SpotThreadPool();

    // Execute function across all threads
    void execute_parallel(std::function<void(int)> func, int total_work);

    // Get thread count
    int get_num_threads() const { return num_threads_; }

private:
    int num_threads_;
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    std::atomic<bool> stop_;
    std::atomic<int> active_workers_;
    std::atomic<int> total_tasks_;
    std::atomic<int> completed_tasks_;

    void worker_thread();
};

/**
 * Spot Rollout class mimicking mujoco.rollout.Rollout API
 */
class SpotRollout {
public:
    /**
     * Constructor
     * @param nthread Number of threads (0 = single-threaded, >0 = multi-threaded)
     * @param cutoff_time Maximum time allowed per rollout in seconds (default 0.2s)
     */
    explicit SpotRollout(int nthread = 0, double cutoff_time = 0.2);

    /**
     * Destructor - cleans up resources
     */
    ~SpotRollout();

    /**
     * Close the rollout object and cleanup resources
     */
    void close();

    /**
     * Perform rollout with Spot-specific ONNX policy
     * Matches mujoco.rollout.Rollout.rollout() API signature
     */
    py::tuple rollout(
        const std::vector<const mjModel*>& models,
        const std::vector<mjData*>& data,
        const py::array_t<double>& initial_state,
        const py::array_t<double>& controls
    );

    /**
     * Context manager support
     */
    SpotRollout* __enter__();
    void __exit__(py::object exc_type, py::object exc_val, py::object exc_tb);

    /**
     * Get the number of threads
     */
    int get_num_threads() const;

private:
    int num_threads_;
    double cutoff_time_;
    std::unique_ptr<SpotThreadPool> thread_pool_;
    std::shared_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<OnnxPolicy> policy_;
    bool closed_ = false;

    // Helper methods for rollout computation
    void compute_indices(const mjModel* m, int& base_qpos_start, int& base_qvel_start,
                        int& leg_qpos_start, int& leg_qvel_start);

    void build_observation(const mjModel* m, mjData* d, const double* command_ptr,
                          const std::vector<float>& prev_policy,
                          int base_qpos_start, int base_qvel_start,
                          int leg_qpos_start, int leg_qvel_start,
                          std::vector<float>& obs_out);

    void compute_control_from_policy(const float* policy_out, const double* command_ptr,
                                   std::vector<double>& ctrl_out);

    // Initialize ONNX session and policy
    void initialize_policy();
};

// Utility function to create numpy arrays from C++ vectors
py::array_t<double> make_array_owned(std::vector<double>& buf, int B, int T, int D);

// Legacy function for backwards compatibility
py::tuple RolloutSpot(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>& data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls
);

// Single-step simulation function
py::array_t<float> SimSpot(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls,
    const py::array_t<float>& prev_policy
);
