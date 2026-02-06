# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# %%
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import time
from copy import deepcopy

import dexterity.mujoco_extensions.policy_rollout as mpr
import mujoco
import numpy as np
from dexterity.spot_interface.spot.constants import STANDING_UNSTOWED_POS

MODEL_PATH = "assets/robots/spot_fast/xml/robot.xml"
NUM_THREADS_LOG2 = 4
HORIZON = 400
NUM_TRIALS = 10

# %%
# Create thread counts and model objects.
print("Allocating mujoco models + data...")
thread_counts = np.power(2, np.arange(NUM_THREADS_LOG2))
base_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
models = [deepcopy(base_model) for i in range(max(thread_counts))]

# Create initial state.
nu = models[0].nu
initial_state = np.array([0, 0, 2, 1, 0, 0, 0] + list(STANDING_UNSTOWED_POS) + [0] * 25)[:, None]
states = [initial_state for i in range(max(thread_counts))]
controls = [np.zeros((HORIZON, nu)) for i in range(max(thread_counts))]

for curr_thread_count in thread_counts:
    # Slice current experiment data from preallocated lists.
    curr_models = models[:curr_thread_count]
    curr_states = states[:curr_thread_count]
    curr_controls = controls[:curr_thread_count]

    # Recreate data to ensure no pollution between runs.
    curr_data = [mujoco.MjData(mm) for mm in curr_models]

    # Roll out multiple times to get lower-variance appx. of timing.
    t0 = time.perf_counter()
    for _ in range(NUM_TRIALS):
        curr_batch_states, curr_batch_sensors = mpr.threaded_physics_rollout(  # type: ignore
            curr_models, curr_data, curr_states, curr_controls
        )
    delta_t = time.perf_counter() - t0
    print("NUM_THREADS", curr_thread_count, "    time", np.around(delta_t, decimals=4))

# %%
