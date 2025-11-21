# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.spot.spot_constants import (
    ARM_CMD_INDS,
    ARM_SOFT_LOWER_JOINT_LIMITS,
    ARM_SOFT_UPPER_JOINT_LIMITS,
    ARM_STOWED_POS,
    ARM_UNSTOWED_POS,
    BASE_SOFT_LIMITS,
    BASE_VEL_CMD_INDS,
    FRONT_LEG_CMD_INDS,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    LEG_SOFT_LOWER_JOINT_LIMITS,
    LEG_SOFT_UPPER_JOINT_LIMITS,
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    STANDING_HEIGHT_CMD,
)
from judo.utils.mujoco import RolloutBackend, SimBackendSpot

XML_PATH = str(MODEL_PATH / "xml/spot_components/robot.xml")


@dataclass
class GOAL_POSITIONS:  # noqa: N801
    """Goal positions of Spot."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0]))
    origin_decimal: np.ndarray = field(default_factory=lambda: np.array([0, 0.0, 0.01]))
    blue_cross: np.ndarray = field(default_factory=lambda: np.array([2.77, 0.71, 0.3]))
    black_cross: np.ndarray = field(default_factory=lambda: np.array([1.5, -1.5, 0.275]))


@dataclass
class SpotBaseConfig(TaskConfig):
    """Base config for spot tasks."""

    fall_penalty: float = 2500.0
    spot_fallen_threshold = 0.35  # Torso height where Spot is considered "fallen"
    w_goal: float = 60.0
    w_controls: float = 0.0


ConfigT = TypeVar("ConfigT", bound=SpotBaseConfig)


class SpotBase(Task[ConfigT], Generic[ConfigT]):
    """Flexible base task for Spot locomotion/skills.

    Controls are a compact vector mapped to the 25-dim policy command:
    - Base only:            [base_vel(3)]
    - Base + Arm:           [base_vel(3), arm_cmd(7)]
    - Base + Legs:          [base_vel(3), front_leg_cmd(6), leg_selection(1)]
    - Base + Arm + Legs:    [base_vel(3), arm_cmd(7), front_leg_cmd(6), leg_selection(1)]

    The mapping to the 25-dim policy command is done in task_to_sim_ctrl.
    """

    default_backend = "mujoco_spot"  # Use Spot-specific backend

    def __init__(
        self, model_path: str = XML_PATH, use_arm: bool = True, use_gripper: bool = False, use_legs: bool = False
    ) -> None:
        """Initialize Spot base task.

        Args:
            model_path: Path to the XML model file
            use_arm: Whether to use the arm
            use_gripper: Whether to use the gripper
            use_legs: Whether to use the legs
        """
        super().__init__(model_path)
        self.use_arm = use_arm
        self.use_gripper = use_gripper  # Reserved flag; not used in current mapping
        self.use_legs = use_legs
        self.set_command_values()

        # Use ONNX-based rollout backend
        self.RolloutBackend = RolloutBackend
        self.SimBackend = SimBackendSpot

        self.default_policy_command = np.array(
            [0, 0, 0] + list(ARM_STOWED_POS) + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD]
        )

    @property
    def nu(self) -> int:
        """Number of controls for this task."""
        return len(self.default_command)

    @property
    def ctrlrange(self) -> np.ndarray:
        """Control bounds for the task."""
        BASE_LOWER = -BASE_SOFT_LIMITS
        BASE_UPPER = BASE_SOFT_LIMITS
        GRIPPER_LOWER = GRIPPER_OPEN_POS if self.use_gripper else GRIPPER_CLOSED_POS
        GRIPPER_UPPER = GRIPPER_CLOSED_POS
        ARM_LOWER = np.concatenate((ARM_SOFT_LOWER_JOINT_LIMITS[:-1], [GRIPPER_LOWER]))
        ARM_UPPER = np.concatenate((ARM_SOFT_UPPER_JOINT_LIMITS[:-1], [GRIPPER_UPPER]))
        LEGS_LOWER = LEG_SOFT_LOWER_JOINT_LIMITS[0:6]
        LEGS_UPPER = LEG_SOFT_UPPER_JOINT_LIMITS[0:6]
        SELECTION_LOWER = -np.ones(1)
        SELECTION_UPPER = np.ones(1)

        if not self.use_arm and not self.use_legs:  # Base
            lower_bound = BASE_LOWER
            upper_bound = BASE_UPPER
        elif self.use_arm and not self.use_legs:  # Base and arm
            lower_bound = np.concatenate((BASE_LOWER, ARM_LOWER))
            upper_bound = np.concatenate((BASE_UPPER, ARM_UPPER))
        elif not self.use_arm and self.use_legs:  # Base and legs
            lower_bound = np.concatenate((BASE_LOWER, LEGS_LOWER, SELECTION_LOWER))
            upper_bound = np.concatenate((BASE_UPPER, LEGS_UPPER, SELECTION_UPPER))
        elif self.use_arm and self.use_legs:  # Base, arm, and legs
            lower_bound = np.concatenate((BASE_LOWER, ARM_LOWER, LEGS_LOWER, SELECTION_LOWER))
            upper_bound = np.concatenate((BASE_UPPER, ARM_UPPER, LEGS_UPPER, SELECTION_UPPER))
        else:
            raise ValueError("Invalid combination of use_arm and use_legs")

        return np.stack([lower_bound, upper_bound], axis=-1)

    def set_command_values(self) -> None:
        """Update default_command and command_mask."""
        if not self.use_arm and not self.use_legs:  # Base
            # Base velocity
            self.default_command = np.array([0, 0, 0])
            self.command_mask = np.array(BASE_VEL_CMD_INDS)
        elif self.use_arm and not self.use_legs:  # Base and arm
            # Base velocity, arm joint angles
            self.default_command = np.array([0, 0, 0, *ARM_UNSTOWED_POS])
            self.command_mask = np.array(BASE_VEL_CMD_INDS + ARM_CMD_INDS)
        elif not self.use_arm and self.use_legs:  # Base and legs
            # Base velocity, leg joint angles
            self.default_command = np.array([0, 0, 0, *LEGS_STANDING_POS[0:6], 0])
            self.command_mask = np.array(BASE_VEL_CMD_INDS + FRONT_LEG_CMD_INDS)
        elif self.use_arm and self.use_legs:  # Base, arm, and legs
            # Base velocity, arm joint angles, leg joint angles, leg selection
            self.default_command = np.array([0, 0, 0, *ARM_UNSTOWED_POS, *LEGS_STANDING_POS[0:6], 0])
            self.command_mask = np.array(BASE_VEL_CMD_INDS + ARM_CMD_INDS + FRONT_LEG_CMD_INDS)

    def apply_leg_mask(self, controls: np.ndarray) -> np.ndarray:
        """Activate or deactivate leg commands depending on the leg selection.

        leg selection:
        -1.0 to -0.5: manipulation with left leg
        -0.5 to +0.5: no leg manipulation
        +0.5 to +1.0: manipulation with right leg
        """
        if self.use_legs:
            added_dim = False
            if controls.ndim == 1:  # Expand 1D control vector
                controls = np.expand_dims(controls, axis=0)
                added_dim = True

            selection = controls[..., -1]
            mask_fl = selection < -0.5
            mask_fr = selection > 0.5
            mask_neither = ~(mask_fl | mask_fr)

            controls = controls.copy()
            controls = controls[..., :-1]
            # Last 6 entries are leg commands
            controls[mask_fl, -3:] = 0.0
            controls[mask_fr, -6:-3] = 0.0
            controls[mask_neither, -6:] = 0.0

            if added_dim:  # Squeeze 1D cotrol vector
                controls = controls.squeeze(axis=0)

        return controls

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Map compact controls (..., nu) to 25-dim policy command expected by C++ rollout.

        Layout of 25-dim policy command:
        [0:3]  torso_vel_cmd
        [3:10] arm_cmd
        [10:22] leg_cmd (4 legs x 3) used for override
        [22:25] torso_pos_cmd
        """
        controls = np.asarray(controls)
        added_dim = False
        if controls.ndim == 1:
            controls = controls[None]
            added_dim = True

        _B, T = controls.shape[0], controls.shape[1] if controls.ndim == 3 else 1
        if controls.ndim == 2:
            # assume (..., nu) at sim timestep grid
            controls = controls[:, None, :]
            T = 1

        out = np.zeros((controls.shape[0], controls.shape[1], 25), dtype=controls.dtype)

        base_end = 3
        arm_end = base_end + (7 if self.use_arm else 0)
        legs_end = arm_end + ((6 + 1) if self.use_legs else 0)

        # Base velocity
        out[..., 0:3] = controls[..., 0:base_end]
        out[..., 24] = STANDING_HEIGHT

        # Arm commands
        if self.use_arm:
            out[..., 3:10] = controls[..., base_end:arm_end]

        # Leg override commands (front legs only) + selection
        if self.use_legs:
            leg_block = controls[..., arm_end:legs_end]  # shape (..., 7)
            fl_cmd = leg_block[..., 0:3]
            fr_cmd = leg_block[..., 3:6]
            sel = leg_block[..., 6]
            # selection: < -0.5 -> FL, > 0.5 -> FR, else neither
            mask_fl = sel < -0.5
            mask_fr = sel > 0.5

            # place into policy command leg slots [10:22]
            # groups: FL(10:13), FR(13:16), HL(16:19), HR(19:22)
            out[..., 10:13] = np.where(mask_fl[..., None], fl_cmd, 0.0)
            out[..., 13:16] = np.where(mask_fr[..., None], fr_cmd, 0.0)

        if added_dim:
            out = out.squeeze(axis=0)
        if T == 1 and out.ndim == 3:
            out = out[:, 0, :]
        return out

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Simple standing reward used as a default for base class.

        Penalizes falling and large control magnitudes.
        """
        rewards = np.zeros(states.shape[0])
        return rewards

    @property
    def reset_arm_pos(self) -> np.ndarray:
        """Reset position of the arm."""
        return ARM_UNSTOWED_POS if self.use_arm else ARM_STOWED_POS

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        return np.array(
            [
                *np.random.randn(2),
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
            ]
        )

    def reset(self) -> None:
        """Reset the robot to its initial pose."""
        self.data.qpos = self.reset_pose
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)
