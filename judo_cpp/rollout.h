#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <deque>
#include <unordered_map>

namespace py = pybind11;

// Utility function to create numpy arrays from C++ vectors
py::array_t<double> make_array(std::vector<double>& buf, int B, int T, int D);

// Thread pool for persistent thread management
class PersistentThreadPool {
public:
    PersistentThreadPool(int num_threads);
    ~PersistentThreadPool();

    // Execute a function across the thread pool
    void execute_parallel(std::function<void(int)> func, int total_work);

    int get_num_threads() const { return num_threads_; }

private:
    int num_threads_;
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    bool stop_;
    int active_workers_;
    int total_tasks_;
    int completed_tasks_;

    void worker_thread();
};

// Global persistent thread pool manager
class ThreadPoolManager {
public:
    static ThreadPoolManager& instance();
    PersistentThreadPool* get_pool(int num_threads);
    void shutdown();

private:
    std::mutex pool_mutex_;
    std::unique_ptr<PersistentThreadPool> current_pool_;
    int current_num_threads_ = 0;
};

py::tuple Rollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls
);

void Sim(
    const mjModel* model,
    mjData*        data,
    const pybind11::array_t<double>& x0,
    const pybind11::array_t<double>& controls
);
