#include "rollout.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <iostream>
#include <omp.h>
#include <string>

namespace py = pybind11;

// ========== Thread Pool Implementation ==========

PersistentThreadPool::PersistentThreadPool(int num_threads)
    : num_threads_(num_threads), stop_(false), active_workers_(0), total_tasks_(0), completed_tasks_(0) {
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&PersistentThreadPool::worker_thread, this);
    }
}

PersistentThreadPool::~PersistentThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void PersistentThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
                active_workers_++;
            }
        }

        if (task) {
            task();

            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_workers_--;
            completed_tasks_++;
            if (completed_tasks_ == total_tasks_) {
                finished_.notify_one();
            }
        }
    }
}

void PersistentThreadPool::execute_parallel(std::function<void(int)> func, int total_work) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        total_tasks_ = total_work;
        completed_tasks_ = 0;

        for (int i = 0; i < total_work; ++i) {
            tasks_.emplace([func, i] { func(i); });
        }
    }

    condition_.notify_all();

    // Wait for all tasks to complete
    std::unique_lock<std::mutex> lock(queue_mutex_);
    finished_.wait(lock, [this] { return completed_tasks_ == total_tasks_; });
}

// Thread Pool Manager Implementation
ThreadPoolManager& ThreadPoolManager::instance() {
    static ThreadPoolManager instance;
    return instance;
}

PersistentThreadPool* ThreadPoolManager::get_pool(int num_threads) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (!current_pool_ || current_num_threads_ != num_threads) {
        current_pool_.reset();  // Destroy old pool
        current_pool_ = std::make_unique<PersistentThreadPool>(num_threads);
        current_num_threads_ = num_threads;
    }

    return current_pool_.get();
}

void ThreadPoolManager::shutdown() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    current_pool_.reset();
    current_num_threads_ = 0;
}

// ========== End Thread Pool Implementation ==========

py::array_t<double> make_array(std::vector<double>& buf,
                              int B, int T, int D) {
    std::vector<ssize_t> shape   = { B, T, D };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(double) * T * D),
        static_cast<ssize_t>(sizeof(double) *     D),
        static_cast<ssize_t>(sizeof(double))
    };

    // Move buf onto the heap so we can own it in the capsule:
    auto heap_buf = new std::vector<double>(std::move(buf));
    // Create a capsule that will delete the vector when the array is gone:
    py::capsule free_when_done(heap_buf, [](void *p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });

    // Build the array pointing into heap_buf->data() and owning the capsule:
    return py::array_t<double>(
        shape, strides,
        heap_buf->data(),
        free_when_done
    );
}


py::tuple Rollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls
) {
    int B = (int)models.size();
    if (B == 0 || B != (int)data.size()) {
        throw std::runtime_error("models/data must have same non-zero length");
    }

    int horizon = (int)controls.shape(1);  // controls: 3D of shape (B, horizon, nu)

    // dims from first model
    const mjModel* m0 = models[0];
    int nq     = m0->nq;
    int nv     = m0->nv;
    int nu     = m0->nu;
    int nsens  = m0->nsensordata;
    int nstate = nq + nv;

    // x0: 2D of shape (B, nstate) for batched initial states
    if (x0.ndim() != 2 || x0.shape(0) != B || x0.shape(1) != nstate) {
        throw std::runtime_error("x0 must be a 2D array of shape (B, nq+nv)");
    }
    const double* x0_ptr = x0.data();

    // allocate outputs
    std::vector<double> states_buf(B * (horizon + 1) * nstate);  // +1 for initial state
    std::vector<double> sens_buf(B * horizon * nsens);

    {
        py::gil_scoped_release release;

        auto controls_unchecked = controls.unchecked<3>();

        #pragma omp parallel for
        for (int i = 0; i < B; i++) {
            try {
                mjData* d = data[i];

                // set initial qpos+qvel for this batch
                d->time = 0.0;
                const double* x0_i = x0_ptr + i * nstate;
                mj_setState(models[i], d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
                mj_forward(models[i], d);
                mju_zero(d->qacc_warmstart, m0->nv);

                double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
                double* se_ptr = &sens_buf[i * horizon * nsens];

                // Store initial state
                for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];
                std::vector<double> current_state(nstate);
                for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                for (int t = 0; t < horizon; t++) {
                    for (int j = 0; j < nu; j++) d->ctrl[j] = controls_unchecked(i, t, j);

                    // Step simulation
                    mj_step(models[i], d);

                    // Record new state
                    for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                    for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                    // Store in output buffers
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[(t + 1) * nstate + j] = current_state[j];
                    }
                    for (int j = 0; j < nsens; j++) {
                        se_ptr[t * nsens + j] = d->sensordata[j];
                    }
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in rollout thread " << i << ": " << e.what() << std::endl;
                // Fill with zeros as fallback
                double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
                double* se_ptr = &sens_buf[i * horizon * nsens];

                for (int t = 0; t <= horizon; t++) {
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[t * nstate + j] = 0.0;
                    }
                }
                for (int t = 0; t < horizon; t++) {
                    for (int j = 0; j < nsens; j++) {
                        se_ptr[t * nsens + j] = 0.0;
                    }
                }
            }
        }
    }

    auto states_arr = make_array(states_buf, B, horizon + 1, nstate);
    auto sens_arr = make_array(sens_buf, B, horizon, nsens);

    return py::make_tuple(states_arr, sens_arr);
}

void Sim(
    const mjModel* model,
    mjData*        data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls
) {
    // dims from model
    int nq = model->nq;
    int nv = model->nv;
    int nu = model->nu;
    int nstate = nq + nv;

    // x0: 1D of shape (nstate)
    if (x0.ndim() != 1 || x0.shape(0) != nstate) {
        throw std::runtime_error("x0 must be a 1D array of shape (nq+nv)");
    }
    if (controls.ndim() != 1 || controls.shape(0) != nu) {
        throw std::runtime_error("controls must be a 1D array of shape (nu)");
    }
    const double* x0_ptr = x0.data();
    const double* ctrl_ptr = controls.data();

    // set initial qpos+qvel
    data->time = 0.0;
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);
    mju_zero(data->qacc_warmstart, nv);

    // set controls
    for (int j = 0; j < nu; j++) data->ctrl[j] = ctrl_ptr[j];

    // step simulation
    mj_step(model, data);
}
