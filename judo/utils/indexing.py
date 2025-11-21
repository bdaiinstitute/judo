# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Tuple

import mujoco
import numpy as np


def get_joint_dof(joint_type: mujoco.mjtJoint) -> Tuple[int, int]:
    """Get the number of position and velocity coordinates for a joint type.

    Args:
        joint_type: MuJoCo joint type (mjJNT_FREE, mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE)

    Returns:
        Tuple containing:
        - Number of position coordinates (qpos)
        - Number of velocity coordinates (qvel)

    Notes:
        Joint types and their DOFs:
        - free:   7 qpos (x,y,z, qw,qx,qy,qz), 6 qvel (vx,vy,vz, wx,wy,wz)
        - ball:   4 qpos (qw,qx,qy,qz),        3 qvel (wx,wy,wz)
        - slide:  1 qpos (translation),         1 qvel (linear velocity)
        - hinge:  1 qpos (rotation),           1 qvel (angular velocity)
    """
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6  # position: 3 translation + 4 quaternion, velocity: 3 linear + 3 angular
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3  # position: 4 quaternion, velocity: 3 angular
    elif joint_type in [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]:
        return 1, 1  # position: 1 coordinate, velocity: 1 rate
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")


def get_control_indices(model: mujoco.MjModel, joint_name: str | list[str]) -> np.ndarray:
    """Get the indices in data.ctrl for the actions.

    Args:
        model: MuJoCo model
        joint_name: Name of the joint or list of joint names to find control indices for

    Returns:
        List of indices in data.ctrl corresponding to the joint's actuators

    Raises:
        ValueError: If joint name is not found in model
    """
    if isinstance(joint_name, list):
        return np.concatenate([get_control_indices(model, name) for name in joint_name])
    elif isinstance(joint_name, str):
        joint_id = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_JOINT), joint_name)
        if joint_id == -1:
            raise ValueError(f"Could not find joint named '{joint_name}' in model")

        # Find actuators that control this joint
        actuator_indices = []
        for i in range(model.nu):
            if model.actuator_trnid[i, 0] == joint_id:
                actuator_indices.append(i)

        return np.array(actuator_indices)
    else:
        raise ValueError(f"Invalid joint name type: provided {type(joint_name)}")


def get_vel_indices(model: mujoco.MjModel, joint_name: str | list[str]) -> np.ndarray:
    """Get velocity indices for any joint type.

    Args:
        model: MuJoCo model
        joint_name: Name of the joint or list of joint names to find velocity indices for

    Returns:
        Tuple containing:
        - List of velocity indices (qvel)
    """
    if isinstance(joint_name, list):
        return np.concatenate([get_vel_indices(model, name) for name in joint_name])
    elif isinstance(joint_name, str):
        joint_id = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_JOINT), joint_name)
        if joint_id == -1:
            raise ValueError(f"Could not find joint named '{joint_name}' in model")

        # Get the starting indices for this joint's DOFs
        qvel_start_idx = model.jnt_dofadr[joint_id]

        # Get the number of position and velocity coordinates for this joint type
        joint_type = model.jnt_type[joint_id]
        _, nqvel = get_joint_dof(joint_type)

        qvel_indices = np.array(range(qvel_start_idx, qvel_start_idx + nqvel))

        return qvel_indices
    else:
        raise ValueError(f"Invalid joint name type: provided {type(joint_name)}")


def get_pos_indices(model: mujoco.MjModel, joint_name: str | list[str]) -> np.ndarray:
    """Get position indices for any joint type.

    Args:
        model: MuJoCo model
        joint_name: Name of the joint or list of joint names to find position indices for

    Returns:
        Tuple containing:
        - List of position indices (qpos)
    """
    if isinstance(joint_name, list):
        return np.concatenate([get_pos_indices(model, name) for name in joint_name])
    elif isinstance(joint_name, str):
        joint_id = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_JOINT), joint_name)
        if joint_id == -1:
            print(model.names)
            raise ValueError(f"Could not find joint named '{joint_name}' in model.")

        # Get the starting indices for this joint's DOFs
        qpos_start_idx = model.jnt_qposadr[joint_id]

        # Get the number of position and velocity coordinates for this joint type
        joint_type = model.jnt_type[joint_id]
        nqpos, _ = get_joint_dof(joint_type)

        qpos_indices = np.array(range(qpos_start_idx, qpos_start_idx + nqpos))

        return qpos_indices
    else:
        raise ValueError(f"Invalid joint name type: provided {type(joint_name)}")


def get_state_indices(model: mujoco.MjModel, joint_name: str | list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Get position and velocity indices for any joint type.

    Args:
        model: MuJoCo model
        joint_name: Name of the joint or list of joint names to find position and velocity indices for

    Returns:
        Tuple containing:
        - List of position indices (qpos)
        - List of velocity indices (qvel)
    """
    qpos_indices = get_pos_indices(model, joint_name)
    qvel_indices = get_vel_indices(model, joint_name)

    return qpos_indices, qvel_indices


def get_sensor_indices(model: mujoco.MjModel, sensor_name: str | list[str]) -> np.ndarray:
    """Get the indices in data.sensordata for a given sensor name.

    Args:
        model: MuJoCo model
        sensor_name: Name of the sensor or list of sensor names to find indices for

    Returns:
        List of indices in data.sensordata corresponding to the sensor

    Raises:
        ValueError: If sensor name is not found in model
    """
    if isinstance(sensor_name, list):
        return np.concatenate([get_sensor_indices(model, name) for name in sensor_name])
    elif isinstance(sensor_name, str):
        sensor_id = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_SENSOR), sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Could not find sensor named '{sensor_name}' in model")

        # Get the starting index and dimension for this sensor's data
        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]

        return np.array(range(adr, adr + dim))
    else:
        raise ValueError(f"Invalid sensor name type: provided {type(sensor_name)}")


def get_control_range(model: mujoco.MjModel, joint_name: str | list[str]) -> np.ndarray:
    """Get the control range of a joint.

    Args:
        model: MuJoCo model
        joint_name: Name of the joint or list of joint names to find control range for

    Returns:
        Control range of the joint
    """
    if isinstance(joint_name, list):
        return np.array([get_control_range(model, name) for name in joint_name])
    elif isinstance(joint_name, str):
        joint_id = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_JOINT), joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in the model.")

        # Find the actuator controlling this joint
        for i in range(model.nu):
            if model.actuator_trnid[i, 0] == joint_id:  # First column is joint ID
                return model.actuator_ctrlrange[i]

        raise ValueError(f"No actuator found for joint '{joint_name}'.")
    else:
        raise ValueError(f"Invalid joint name type: provided {type(joint_name)}")


def get_joint_names(model: mujoco.MjModel) -> list[str]:
    """Returns a list of joint names from the model.

    Args:
        model: MuJoCo model

    Returns:
        List of joint names
    """
    return [mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_JOINT), i) for i in range(model.njnt)]


def get_joint_type(model: mujoco.MjModel, joint_name: str) -> mujoco.mjtJoint:
    """Returns the type of a joint.

    Args:
        model: MuJoCo model
        joint_name: Name of the joint

    Returns:
        Type of the joint
    """
    return model.jnt_type[mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_JOINT), joint_name)]


def get_joint_pos_index_map(model: mujoco.MjModel) -> dict[str, list[int]]:
    """Get all the position indices from the model.

    Args:
        model: MuJoCo model

    Returns:
        Map of position indices
    """
    return {
        joint_name: get_pos_indices(model, joint_name).astype(int).tolist() for joint_name in get_joint_names(model)
    }


def get_joint_vel_index_map(model: mujoco.MjModel) -> dict[str, list[int]]:
    """Get all the velocity indices from the model.

    Args:
        model: MuJoCo model

    Returns:
        Map of velocity indices
    """
    return {
        f"{joint_name}_vel": (model.nq + get_vel_indices(model, joint_name)).astype(int).tolist()
        for joint_name in get_joint_names(model)
    }
