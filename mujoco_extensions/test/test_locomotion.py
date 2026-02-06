# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import unittest

import dexterity.mujoco_extensions.policy_rollout as mpr  # type: ignore
import mujoco
import numpy as np
from benedict import benedict
from dexterity.spot_interface.spot.constants import (
    LOCOMOTION_POLICY_PATH,
    STANDING_HEIGHT,
    STANDING_HEIGHT_CMD,
    STANDING_STOWED_POS,
    STANDING_UNSTOWED_POS,
)
from scipy.spatial.transform import Rotation as R

REFERENCE_PATH = "dexterity/mujoco_extensions/test/policy_inference.yml"
MODEL_PATH = "dexterity/models/xml/scenes/test_scenes/spot_wheel_rim.xml"


START_POS = np.array(
    [0, 0, STANDING_HEIGHT] + [0.5] * 4 + [0.12, 0.72, -1.45] * 4 + [0, -3, 3, 0, 0, 0, 0] + [10, 10, 10, 1, 0, 0, 0]
)
START_VEL = np.arange(6 + 19 + 6)
START_STATE = np.concatenate((START_POS, START_VEL))


class TestPolicyRollout(unittest.TestCase):
    """Testing Policy Rollout"""

    def test_torso_velocity_command(self) -> None:
        print("\n==== Testing robot frame command ====")
        system = mpr.System(MODEL_PATH, LOCOMOTION_POLICY_PATH)  # type: ignore
        reference_commands = np.array([[0.0, 0, 0, 0, -3, 3, 0, 0, 0, 0] + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD]])
        num_commands = 150
        physics_substeps = 2
        nq = 7 + 12 + 7 + 7

        start_pos = np.concatenate(
            (np.array([0, 0, STANDING_HEIGHT, 1, 0, 0, 0]), STANDING_STOWED_POS, np.array([100, 0, 1, 1, 0, 0, 0]))
        )
        start_vel = np.zeros(31)
        start_state = np.concatenate((start_pos, start_vel))

        test_cases = [
            {"torso_vel_cmd": np.array([+1.0, 0, 0]), "velocity_index": 0},
            {"torso_vel_cmd": np.array([-1.0, 0, 0]), "velocity_index": 0},
            {"torso_vel_cmd": np.array([0, +0.8, 0]), "velocity_index": 1},
            {"torso_vel_cmd": np.array([0, -0.8, 0]), "velocity_index": 1},
            {"torso_vel_cmd": np.array([0, 0, +0.6]), "velocity_index": 5},
            {"torso_vel_cmd": np.array([0, 0, -0.6]), "velocity_index": 5},
        ]

        for case in test_cases:
            velocity_index = case["velocity_index"]
            torso_vel_cmd = case["torso_vel_cmd"]

            # Find the index of the first non-zero element
            first_nonzero_idx = np.nonzero(torso_vel_cmd)[0][0]
            target_vel = torso_vel_cmd[first_nonzero_idx]
            sign = np.sign(target_vel)

            # rollout policy
            commands = np.copy(reference_commands)
            commands[:, 0:3] = torso_vel_cmd
            commands = np.tile(commands, (num_commands, 1))
            states, _ = system.rollout(start_state, commands, physics_substeps)
            avg_vel = np.mean(states[:, nq + velocity_index])

            # ensure spot is moving along to right direction
            print(
                f"Test passed for torso_vel_cmd {torso_vel_cmd} with average velocity "
                + f"{np.around(sign * avg_vel, decimals=2)}"
            )
            self.assertGreater(sign * avg_vel, 0.6 * sign * target_vel)

    def test_torso_velocity_command_world_frame(self) -> None:
        print("\n==== Testing world frame command ====")
        system = mpr.System(MODEL_PATH, LOCOMOTION_POLICY_PATH)  # type: ignore
        reference_commands = np.array([[0.0, 0, 0, 0, -3, 3, 0, 0, 0, 0] + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD]])
        physics_substeps = 2

        def init_state(theta: float) -> np.ndarray:
            start_orientation = R.from_euler("xyz", [0, 0, theta]).as_quat()  # No rotation
            start_pos = np.concatenate(
                (
                    np.array([1.5, 1.5, STANDING_HEIGHT]),
                    start_orientation,
                    STANDING_STOWED_POS,
                    np.array([10, 0, 1, 1, 0, 0, 0]),
                )
            )
            start_vel = np.zeros(31)
            start_state = np.concatenate((start_pos, start_vel))
            return start_state

        # comparing the results of the world frame rollout with the robot frame rollout by sending the same
        # velocity command in the world frame and the robot frame, i.e. "dq_torso_wf" and "dq_torso_rf" respectively
        test_cases = [
            {"theta": 0, "dq_torso_wf": np.array([1.0, 0, 0]), "dq_torso_rf": np.array([1.0, 0, 0])},
            {"theta": 0, "dq_torso_wf": np.array([0, 1.0, 0]), "dq_torso_rf": np.array([0, 1.0, 0])},
            {"theta": np.pi / 2, "dq_torso_wf": np.array([1.0, 0, 0]), "dq_torso_rf": np.array([0, 1.0, 0])},
            {"theta": np.pi / 2, "dq_torso_wf": np.array([0, 1.0, 0]), "dq_torso_rf": np.array([-1.0, 0, 0])},
            {"theta": np.pi, "dq_torso_wf": np.array([1.0, 0, 0]), "dq_torso_rf": np.array([-1.0, 0, 0])},
            {"theta": np.pi, "dq_torso_wf": np.array([0, 1.0, 0]), "dq_torso_rf": np.array([0, -1.0, 0])},
            {"theta": 3 * np.pi / 2, "dq_torso_wf": np.array([1.0, 0, 0]), "dq_torso_rf": np.array([0, -1.0, 0])},
            {"theta": 3 * np.pi / 2, "dq_torso_wf": np.array([0, 1.0, 0]), "dq_torso_rf": np.array([1.0, 0, 0])},
        ]
        for case in test_cases:
            # Rollout with world frame command
            commands = np.copy(reference_commands)
            commands[:, 0:3] = case["dq_torso_wf"]
            states_world, _ = system.rollout_world_frame(init_state(case["theta"]), commands, physics_substeps)
            commands[:, 0:3] = case["dq_torso_rf"]
            states_robot, _ = system.rollout(
                init_state(case["theta"]), commands, physics_substeps, reset_last_output=True
            )
            # Compare the states to see if the internal transformation matches the expected outcome
            self.assertTrue(
                np.allclose(states_world, states_robot, atol=1e4),
                msg=f"Discrepancy found in test with world command {case['dq_torso_wf']} and theta {case['theta']}",
            )
            print(f"Test passed for world command {case['dq_torso_wf']} and theta {case['theta']}")

    def test_policy_observation_and_action(self) -> None:
        print("\n==== Testing Policy Observation and Action ====")
        system = mpr.System(MODEL_PATH, LOCOMOTION_POLICY_PATH)  # type: ignore
        reference_values = benedict.from_yaml(REFERENCE_PATH)
        obs_reference = reference_values["rl_locomotion"]["observation"]
        action_reference = reference_values["rl_locomotion"]["action"]
        target_q_reference = reference_values["rl_locomotion"]["target_q"]

        command = np.arange(25)
        action = 0.1 * np.arange(12)

        mpr.set_state(system, START_STATE)  # type: ignore
        system.policy_output = action
        system.set_observation(command)
        obs = system.observation[np.newaxis, :]
        system.policy_inference()
        action = system.policy_output
        target_q = system.get_control()

        observation_slices = {
            "torso lin vel": slice(0, 3),
            "torso ang vel": slice(3, 6),
            "projected gravity": slice(6, 9),
            "base vel command": slice(9, 12),
            "arm command": slice(12, 19),
            "leg_command": slice(19, 31),
            "body_state": slice(31, 34),
            "joint pos": slice(34, 53),
            "joint vel": slice(53, 72),
            "last action": slice(72, 84),
        }

        self.assertEqual(obs.shape, (1, 84))
        obs = obs.flatten()
        for name, idx in observation_slices.items():
            self.assertTrue(
                np.allclose(obs[idx], obs_reference[idx], atol=1e-5),
                msg=f"Observation mismatch for {name}: expected {obs_reference[idx]}, got {obs[idx]}",
            )

        self.assertTrue(
            np.allclose(action, action_reference, atol=1e-3),
            msg=f"Action mismatch : expected {action_reference}, got {action}",
        )

        self.assertTrue(
            np.allclose(target_q, target_q_reference, atol=1e-3),
            msg=f"Target q mismatch : expected {target_q_reference}, got {target_q}",
        )

    def test_split_trajectory_consistency(self) -> None:
        print("\n==== Testing split trajectory consistency with threaded rollout ====")

        num_systems = 2
        num_commands = 200
        physics_substeps = 2

        # Initialize systems, state, and commands
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        systems = mpr.create_systems_vector(model, LOCOMOTION_POLICY_PATH, num_systems=num_systems)  # type: ignore
        state = np.array(
            [0, 0, STANDING_HEIGHT, 1, 0, 0, 0]
            + list(STANDING_UNSTOWED_POS)
            + [3, 0, STANDING_HEIGHT, 1, 0, 0, 0]
            + [0] * (6 + 12 + 7 + 6)
        )
        states = np.tile(state, (num_systems, 1))
        commands = np.tile(np.array([[1.0, 0, 0] + [0] * 22]), (num_commands, 1))
        batched_commands = np.array([commands for _ in range(num_systems)])

        # Initialize last_policy_output with zeros
        last_policy_output = np.zeros((num_systems, 12))

        # Generate full trajectory
        full_states, full_sensors, full_last_policy_output = mpr.threaded_rollout(  # type: ignore
            systems, states, batched_commands, last_policy_output, num_systems, physics_substeps
        )

        full_states = np.vstack(full_states).reshape(num_systems, num_commands * physics_substeps, -1)

        # Generate split trajectory
        split_point = num_commands // 2
        first_half_commands = batched_commands[:, :split_point, :]
        second_half_commands = batched_commands[:, split_point:, :]

        first_half_states, _, first_half_last_policy_output = mpr.threaded_rollout(  # type: ignore
            systems, states, first_half_commands, last_policy_output, num_systems, physics_substeps
        )
        first_half_states = np.vstack(first_half_states).reshape(
            num_systems, (num_commands * physics_substeps) // 2, -1
        )

        second_half_start_states = [states[-1] for states in first_half_states]
        second_half_states, second_half_sensors, second_half_last_policy_output = mpr.threaded_rollout(  # type: ignore
            systems,
            second_half_start_states,
            second_half_commands,
            first_half_last_policy_output,
            num_systems,
            physics_substeps,
        )
        second_half_states = np.vstack(second_half_states).reshape(
            num_systems, (num_commands * physics_substeps) // 2, -1
        )

        # Compare trajectory lengths
        for system_idx in range(num_systems):
            # Compare full STATE trajectories
            full_state_first_half = full_states[system_idx, : split_point * physics_substeps]
            self.assertTrue(
                np.allclose(full_state_first_half, first_half_states[system_idx], atol=1e-6),
                msg=f"States do not match for the first half for system {system_idx}",
            )
            full_state_second_half = full_states[system_idx, split_point * physics_substeps :]
            self.assertTrue(
                np.allclose(full_state_second_half, second_half_states[system_idx], atol=1e-6),
                msg=f"States do not match for the second half for system {system_idx}",
            )

            # Compare SECOND half of SENSOR trajectories
            full_sensor_second_half = full_sensors[system_idx][split_point * physics_substeps :]
            self.assertTrue(
                np.allclose(full_sensor_second_half, second_half_sensors[system_idx], atol=1e-6),
                msg=f"Second half of sensors do not match for system {system_idx}",
            )

            # Compare SECOND half of LAST_POLICY_OUTPUTS
            self.assertTrue(
                np.allclose(full_last_policy_output[system_idx], second_half_last_policy_output[system_idx], atol=1e-6),
                msg=f"Second half of last policy outputs do not match for system {system_idx}",
            )

        print("Test passed: Split trajectory is consistent with full trajectory using threaded rollout")
