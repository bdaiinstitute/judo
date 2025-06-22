# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from pathlib import Path
from typing import List

import mujoco
import numpy as np
import trimesh
from mujoco import MjModel, MjsGeom, MjsMaterial, MjSpec
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


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

    return mesh.scale  # type: ignore


def apply_mujoco_material(
    mesh: trimesh.Trimesh,
    material: MjsMaterial,
) -> None:
    """Applies a MuJoCo material to a trimesh mesh.

    This sets up PBR parameters and handles RGBA conversion.

    Args:
        mesh: the trimesh.Trimesh to modify
        model: the Mujoco MjModel to read textures (spec.texturedir if available)
        material: an object matching the mjsMaterial struct
        texture_dir: optional override of the directory for texture files
    """
    # prepare PBR material
    pbr = PBRMaterial()

    # get RGBA, convert if needed
    rgba = np.array(material.rgba)
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = rgba_float_to_int(rgba)
    color = tuple(int(x) for x in rgba.tolist())
    pbr.alphaMode = "BLEND" if rgba[3] < 255 else "OPAQUE"

    # set PBR values
    pbr.metallicFactor = float(material.metallic)
    pbr.roughnessFactor = float(material.roughness)
    pbr.emissiveFactor = [material.emission] * 3
    if material.roughness == 0.0 and getattr(material, "shininess", 0) > 0:
        pbr.roughnessFactor = np.sqrt(2.0 / (material.shininess + 2.0)).item()

    dummy = Image.new("RGBA", (1, 1), color)
    pbr.baseColorTexture = dummy
    uv = getattr(mesh.visual, "uv", None)
    mesh.visual = TextureVisuals(material=pbr, uv=uv)

    if getattr(material, "textures", None):
        warnings.warn("Textured meshes are currently unsupported. Loading with RGBA color instead.", stacklevel=2)

    mesh.visual = TextureVisuals(material=pbr, uv=None)


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


def rgba_float_to_int(rgba_float: np.ndarray) -> np.ndarray:
    """Convert RGBA float values in [0, 1] to int values in [0, 255]."""
    return (255 * rgba_float).astype("int")


def rgba_int_to_float(rgba_int: np.ndarray) -> np.ndarray:
    """Convert RGBA int values in [0, 255] to float values in [0, 1]."""
    return rgba_int / 255.0
