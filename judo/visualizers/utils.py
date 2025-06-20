# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from pathlib import Path
from typing import List

import mujoco
import numpy as np
from mujoco import MjModel, MjsGeom, MjSpec


def get_sensor_name(model: MjModel, sensorid: int) -> str:
    """Return name of the sensor with given ID from MjModel."""
    index = model.name_sensoradr[sensorid]
    end = model.names.find(b"\x00", index)
    name = model.names[index:end].decode("utf-8")
    if len(name) == 0:
        name = f"sensor{sensorid}"
    return name


def get_mesh_data(model: MjModel, meshid: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the vertices and faces of a specified mesh from a MuJoCo model.

    Args:
        model : MjModel The MuJoCo model containing the mesh data.
        meshid : int The index of the mesh to retrieve.

    Result:
        tuple[np.ndarray, np.ndarray]
        Vertices (N, 3) and faces (M, 3) of the mesh.
    """
    vertadr = model.mesh_vertadr[meshid]
    vertnum = model.mesh_vertnum[meshid]
    vertices = model.mesh_vert[vertadr : vertadr + vertnum, :]

    faceadr = model.mesh_faceadr[meshid]
    facenum = model.mesh_facenum[meshid]
    faces = model.mesh_face[faceadr : faceadr + facenum]
    return vertices, faces


def get_mesh_file(spec: MjSpec, geom: MjsGeom) -> Path:
    """Extracts the mesh filepath for a particular geom from an MjSpec."""
    assert geom.type == mujoco.mjtGeom.mjGEOM_MESH, f"Can only get mesh files for meshes, got type {geom.type}"

    meshname = geom.meshname
    mesh = spec.mesh(meshname)

    mesh_path = Path(spec.modelfiledir) / spec.meshdir / mesh.file
    return mesh_path


def get_mesh_scale(spec: MjSpec, geom: MjsGeom) -> np.ndarray:
    """Extracts the relevant scale parameters for a given geom in the MjSpec."""
    assert geom.type == mujoco.mjtGeom.mjGEOM_MESH, (
        f"Can only get mesh scale for mesh-type geoms, got type {geom.type}."
    )

    meshname = geom.meshname
    mesh = spec.mesh(meshname)

    return mesh.scale


def is_trace_sensor(model: MjModel, sensorid: int) -> bool:
    """Check if a sensor is a trace sensor."""
    sensor_name = get_sensor_name(model, sensorid)
    return (
        model.sensor_type[sensorid] == mujoco.mjtSensor.mjSENS_FRAMEPOS
        and model.sensor_datatype[sensorid] == mujoco.mjtDataType.mjDATATYPE_REAL
        and model.sensor_dim[sensorid] == 3
        and "trace" in sensor_name
    )


def count_trace_sensors(model: MjModel) -> int:
    """Count the number of trace sensors of a given mujoco model."""
    num_traces = 0
    for id in range(model.nsensor):
        num_traces += is_trace_sensor(model, id)
    return num_traces


def get_trace_sensors(model: MjModel) -> List[int]:
    """Get the IDs of all trace sensors in a given mujoco model."""
    return [id for id in range(model.nsensor) if is_trace_sensor(model, id)]
