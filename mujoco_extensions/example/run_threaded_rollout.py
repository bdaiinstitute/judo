# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import pickle as pkl
import time
from dataclasses import dataclass, field

import dexterity.mujoco_extensions.policy_rollout as mpr
import mujoco
import numpy as np
import tyro
from dexterity.spot_interface.spot.constants import (
    ARM_UNSTOWED_POS,
    LOCOMOTION_POLICY_PATH,
    STANDING_HEIGHT,
    STANDING_HEIGHT_CMD,
    STANDING_UNSTOWED_POS,
)
from dexterity.visualizers.mujoco_visualizer import MujocoRenderer


@dataclass
class RolloutConfig:
    model_path: str = "assets/robots/spot_fast/xml/robot.xml"
    policy_path: str = LOCOMOTION_POLICY_PATH
    num_systems: int = 32
    physics_substeps: int = 2
    command_steps: int = 400
    default_command: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0] + list(ARM_UNSTOWED_POS) + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD]
    )
    rollout_with_fb: bool = False


def main(config: RolloutConfig) -> None:
    t0 = time.time()
    systems = mpr.create_systems_vector(config.model_path, config.policy_path, num_systems=config.num_systems)
    print("Time to create systems", time.time() - t0)

    # Create a list of initial states, one for each system
    state = np.array([0, 0, STANDING_HEIGHT, 1, 0, 0, 0] + list(STANDING_UNSTOWED_POS) + [0] * (12 + 7 + 6))
    states = np.tile(state, (config.num_systems, 1))

    # Initialize last_policy_output with zeros
    last_policy_output = np.zeros((config.num_systems, 12))

    if config.rollout_with_fb:
        p_gains = np.array([10.0, 10.0, 0.1])
        # load example solution
        test_traj_path = "dexterity/data/test_data/test_traj.pkl"
        sol = pkl.load(open(test_traj_path, "rb"))  # resolution is 0.02s (matching 50Hz for policy)
        pos_ref = sol["q_traj"]
        vel_ref = sol["dq_traj"]
        state[:2] = pos_ref[0, :2]  # set initial position of base to the first position in the trajectory
        commands_full = np.tile(
            config.default_command, (vel_ref.shape[0], 1)
        )  # arm, leg, and torso commands are constant
        commands_full[:, :3] = vel_ref[:, :3]  # set the velocity commands to the reference trajectory
        batched_commands = np.tile(commands_full, (config.num_systems, 1, 1))

        pos_ref = np.tile(pos_ref, (config.num_systems, 1, 1))
        print("batched_commands", batched_commands.shape)
        out_states, sensors, ctrls, last_policy_output = mpr.threaded_rollout_feedback_world_frame(
            systems,
            states,
            batched_commands,
            pos_ref,
            p_gains,
            last_policy_output,
            config.physics_substeps,
            config.num_systems,
        )
        print("ctrls", len(ctrls), ctrls[0].shape)
    else:
        batched_commands = np.array(
            [np.tile(config.default_command, (config.command_steps, 1)) for i in range(config.num_systems)]
        )
        out_states, sensors, last_policy_output = mpr.threaded_rollout(
            systems, states, batched_commands, last_policy_output, config.num_systems, config.physics_substeps
        )
        t1 = time.time()
        cutoff_states, cutoff_sensors, _ = mpr.threaded_rollout(
            systems,
            states,
            batched_commands,
            last_policy_output,
            config.num_systems,
            config.physics_substeps,
            cutoff_time=0.2,
        )
        delta_t1 = time.time() - t1
        print(f"cutoff run time: {delta_t1}")

    # Rendering trajectory
    visual_skip_period = 10
    model = mujoco.MjModel.from_xml_path(config.model_path)
    renderer = MujocoRenderer(model, time_step=visual_skip_period * model.opt.timestep)

    trajectory = out_states[0][0:-1:visual_skip_period, :]
    renderer.show(trajectory)


if __name__ == "__main__":
    main(tyro.cli(RolloutConfig))
