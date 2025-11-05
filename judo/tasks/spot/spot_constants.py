# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from enum import IntEnum

import numpy as np

### Names
LEG_JOINT_NAMES_BOSDYN = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]

LEG_JOINT_NAMES_ISAAC = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

ARM_JOINT_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

JOINT_NAMES_BOSDYN = LEG_JOINT_NAMES_BOSDYN + ARM_JOINT_NAMES

JOINT_NAMES_ISAAC = [
    "arm_sh0",
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "arm_sh1",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "arm_el0",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

LEG_NAMES = [
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
]

FEET_NAMES = [f"{leg_name}_foot" for leg_name in LEG_NAMES]

FL_JOINT_NAMES = ["fl_hx", "fl_hy", "fl_kn"]
FR_JOINT_NAMES = ["fr_hx", "fr_hy", "fr_kn"]
HL_JOINT_NAMES = ["hl_hx", "hl_hy", "hl_kn"]
HR_JOINT_NAMES = ["hr_hx", "hr_hy", "hr_kn"]
LEG_JOINT_NAMES_DICT = {
    "fl": FL_JOINT_NAMES,
    "fr": FR_JOINT_NAMES,
    "hl": HL_JOINT_NAMES,
    "hr": HR_JOINT_NAMES,
}


def get_joint_names_bosdyn(has_arm: bool) -> list:
    """Return the list of Spot joint names in BD API order.

    Args:
        has_arm (bool): If true, include the arm joints.
    """
    if has_arm:
        return JOINT_NAMES_BOSDYN
    return LEG_JOINT_NAMES_BOSDYN


def get_joint_names_isaac(has_arm: bool) -> list:
    """Return the list of Spot joint names in Isaac order.

    Args:
        has_arm (bool): If true, include the arm joints.
    """
    if has_arm:
        return JOINT_NAMES_ISAAC
    return LEG_JOINT_NAMES_ISAAC


### Camera-related names
CAMERA_IMAGE_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
    "hand_color_image",
]
DEPTH_IMAGE_SOURCES = [
    "frontleft_depth",
    "frontright_depth",
    "left_depth",
    "right_depth",
    "back_depth",
    "hand_depth",
]
DEPTH_REGISTERED_IMAGE_SOURCES = [
    "frontleft_depth_in_visual_frame",
    "frontright_depth_in_visual_frame",
    "right_depth_in_visual_frame",
    "left_depth_in_visual_frame",
    "back_depth_in_visual_frame",
    "hand_depth_in_hand_color_frame",
]

IMAGE_TYPES = {"visual", "depth", "depth_registered"}


### Enums and DataClasses
class DOF(IntEnum):
    """Link index and order."""

    # FL_HX = spot_constants_pb2.JOINT_INDEX_FL_HX
    # FL_HY = spot_constants_pb2.JOINT_INDEX_FL_HY
    # FL_KN = spot_constants_pb2.JOINT_INDEX_FL_KN
    # FR_HX = spot_constants_pb2.JOINT_INDEX_FR_HX
    # FR_HY = spot_constants_pb2.JOINT_INDEX_FR_HY
    # FR_KN = spot_constants_pb2.JOINT_INDEX_FR_KN
    # HL_HX = spot_constants_pb2.JOINT_INDEX_HL_HX
    # HL_HY = spot_constants_pb2.JOINT_INDEX_HL_HY
    # HL_KN = spot_constants_pb2.JOINT_INDEX_HL_KN
    # HR_HX = spot_constants_pb2.JOINT_INDEX_HR_HX
    # HR_HY = spot_constants_pb2.JOINT_INDEX_HR_HY
    # HR_KN = spot_constants_pb2.JOINT_INDEX_HR_KN
    # # Arm
    # A0_SH0 = spot_constants_pb2.JOINT_INDEX_A0_SH0
    # A0_SH1 = spot_constants_pb2.JOINT_INDEX_A0_SH1
    # A0_EL0 = spot_constants_pb2.JOINT_INDEX_A0_EL0
    # A0_EL1 = spot_constants_pb2.JOINT_INDEX_A0_EL1
    # A0_WR0 = spot_constants_pb2.JOINT_INDEX_A0_WR0
    # A0_WR1 = spot_constants_pb2.JOINT_INDEX_A0_WR1
    # # Hand
    # A0_F1X = spot_constants_pb2.JOINT_INDEX_A0_F1X

    # DOF count for strictly the legs.
    N_DOF_LEGS = 12
    # DOF count for all DOF on robot (arms and legs).
    N_DOF = 19


class LEGS(IntEnum):
    """Leg links index and order."""

    # FL = spot_constants_pb2.LEG_INDEX_FL
    # FR = spot_constants_pb2.LEG_INDEX_FR
    # HL = spot_constants_pb2.LEG_INDEX_HL
    # HR = spot_constants_pb2.LEG_INDEX_HR

    N_LEGS = 4


class LegDofOrder(IntEnum):
    """Legs DoF order."""

    # HX = spot_constants_pb2.HX
    # HY = spot_constants_pb2.HY
    # KN = spot_constants_pb2.KN

    # The number of leg dof
    N_LEG_DOF = 3


### Gains
# Bosdyn
LEG_K_Q_P_BOSDYN = [624, 936, 286.0] * LEGS.N_LEGS
LEG_K_QD_P_BOSDYN = [5.20, 5.20, 2.04] * LEGS.N_LEGS
ARM_K_Q_P_BOSDYN = [1020, 255, 204, 102, 102, 102, 16.0]
ARM_K_QD_P_BOSDYN = [10.2, 15.3, 10.2, 2.04, 2.04, 2.04, 0.32]

# RL locomotion policy during training
LEG_K_Q_P_RL = [60, 60, 60.0] * LEGS.N_LEGS
LEG_K_QD_P_RL = [1.5, 1.5, 1.5] * LEGS.N_LEGS
ARM_K_Q_P_RL = [120, 120, 120, 100, 100, 100, 16.0]
ARM_K_QD_P_RL = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.32]
K_Q_P_RL = LEG_K_Q_P_RL + ARM_K_Q_P_RL
K_QD_P_RL = LEG_K_QD_P_RL + ARM_K_QD_P_RL

# RL locomotion policy on hardware
LEG_K_Q_P_HW = [75, 75, 75.0] * LEGS.N_LEGS
LEG_K_QD_P_HW = LEG_K_QD_P_RL
ARM_K_Q_P_HW = ARM_K_Q_P_RL
ARM_K_QD_P_HW = ARM_K_QD_P_RL

### Initial Configurations
# Standard joint positions
LEGS_SITTING_POS = np.array(
    [
        0.48,
        1.26,
        -2.7929,
        -0.48,
        1.26,
        -2.7929,
        0.48,
        1.26,
        -2.7929,
        -0.48,
        1.26,
        -2.7929,
    ]
)

LEGS_STANDING_POS = np.array(
    [
        0.12,
        0.72,
        -1.45,
        -0.12,
        0.72,
        -1.45,
        0.12,
        0.72,
        -1.45,
        -0.12,
        0.72,
        -1.45,
    ]
)

LEGS_STANDING_POS_RL = np.array(
    [
        0.12,
        0.5,
        -1.0,
        -0.12,
        0.5,
        -1.0,
        0.12,
        0.5,
        -1.0,
        -0.12,
        0.5,
        -1.0,
    ]
)

GRIPPER_CLOSED_POS = 0
GRIPPER_OPEN_POS = -1.54

ARM_STOWED_POS = np.array(
    [
        0,
        -3.11,
        3.13,
        1.56,
        0,
        -1.56,
        GRIPPER_CLOSED_POS,
    ]
)

ARM_MIDDLE_POS = np.array(
    [
        0,
        -2.8,
        2.8,
        1.56,
        0,
        -1.56,
        GRIPPER_CLOSED_POS,
    ]
)

ARM_UNSTOWED_POS = np.array(
    [
        0,
        -0.9,
        1.8,
        0,
        -0.9,
        0,
        GRIPPER_CLOSED_POS,
    ]
)

SITTING_STOWED_POS = np.concatenate((LEGS_SITTING_POS, ARM_STOWED_POS))

SITTING_UNSTOWED_POS = np.concatenate((LEGS_SITTING_POS, ARM_UNSTOWED_POS))

STANDING_STOWED_POS = np.concatenate((LEGS_STANDING_POS, ARM_STOWED_POS))

STANDING_UNSTOWED_POS = np.concatenate((LEGS_STANDING_POS, ARM_UNSTOWED_POS))

STANDING_POS_RL = np.concatenate((LEGS_STANDING_POS_RL, ARM_UNSTOWED_POS[:-1], np.array([GRIPPER_OPEN_POS])))

SITTING_HEIGHT = 0.1
STANDING_HEIGHT = 0.52
STANDING_HEIGHT_CMD = 0.55

### Limits
# RL locomotion policy limits
LEG_TORQUE_LIMITS_RL = [45.0, 45.0, 60.0] * LEGS.N_LEGS
ARM_TORQUE_LIMITS_RL = [90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32]
TORQUE_LIMITS_RL = LEG_TORQUE_LIMITS_RL + ARM_TORQUE_LIMITS_RL

# position limits
LEG_LOWER_JOINT_LIMITS = np.array([-0.785398, -0.898845, -2.7929] * LEGS.N_LEGS)
LEG_UPPER_JOINT_LIMITS = np.array([0.785398, 2.29511, -0.254801] * LEGS.N_LEGS)
ARM_LOWER_JOINT_LIMITS = np.array([-2.618, -3.1416, 0, -2.7925, -1.8326, -2.8798, -1.57])
ARM_UPPER_JOINT_LIMITS = np.array([3.1416, 0.5236, 3.1416, 2.7925, 1.8326, 2.8798, 0])
LOWER_JOINT_LIMITS = np.concatenate((LEG_LOWER_JOINT_LIMITS, ARM_LOWER_JOINT_LIMITS))
UPPER_JOINT_LIMITS = np.concatenate((LEG_UPPER_JOINT_LIMITS, ARM_UPPER_JOINT_LIMITS))

LEG_SOFT_LOWER_JOINT_LIMITS = np.array([-0.6, -0.8, -2.7] * LEGS.N_LEGS)
LEG_SOFT_UPPER_JOINT_LIMITS = np.array([0.6, 1.65, -0.3] * LEGS.N_LEGS)
ARM_SOFT_LOWER_JOINT_LIMITS = ARM_UNSTOWED_POS - np.array([1.0, 1.0, 0.7, 0.7, 0.7, 0.7, 0])  # closed gripper
ARM_SOFT_UPPER_JOINT_LIMITS = ARM_UNSTOWED_POS + np.array([0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0])

# velocity limits
VELOCITY_TASK_SAFETY_LIMIT: float = 30.0  # Limit when task execution should be stopped
VELOCITY_HW_SAFETY_LIMIT: float = 40.0  # Limit when Spot will shut down


# base limits
BASE_SOFT_LIMITS = 0.7 * np.ones(3)

# Command lengths
# velocity_cmd: 3, arm_joint_cmd: 7, leg_joint_cmd: 12, torso_rph: 3 --> total = 25
RL_LOCOMOTION_COMMAND_LENGTH = 25
RL_LOCOMOTION_ACTION_LENGTH = 12

# TODO generate with scene generation
# Judo task action indicies
BASE_VEL_CMD_INDS = [0, 1, 2]
FRONT_LEG_CMD_INDS = [10, 11, 12, 13, 14, 15]
ARM_CMD_INDS = [3, 4, 5, 6, 7, 8, 9]

# Rollout related
DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME: float = 0.125  # seconds

SECOND_TO_NANOSECOND = 1_000_000_000

### Isaac <-> Mujoco conversion
ISAAC_TO_MUJOCO_INDICES_12 = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
ISAAC_TO_MUJOCO_INDICES_19 = [1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18]
MUJOCO_TO_ISAAC_INDICES_12 = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
MUJOCO_TO_ISAAC_INDICES_19 = [12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18]
