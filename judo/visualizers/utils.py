# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from pathlib import Path
from typing import List

import mujoco
import numpy as np
import trimesh
from mujoco import MjModel, MjsGeom, MjsMaterial, MjSpec
from PIL import Image
from trimesh.visual import ColorVisuals, TextureVisuals
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

    return mesh.scale


def apply_mujoco_material(
    mesh: trimesh.Trimesh,
    material: MjsMaterial,
    texture_dir: Path = Path("."),
) -> None:
    """
    Applies a MuJoCo material specification to a trimesh mesh.

    This function translates material properties, including color, PBR values,
    and textures, into a trimesh-compatible visual format.

    Args:
        mesh: The trimesh.Trimesh object to modify.
        material: An object with attributes matching the mjsMaterial struct.
        texture_dir: The directory where texture files are located.
    """

    # Helper to convert float RGBA (0-1) to int RGBA (0-255)
    def rgba_float_to_int(rgba_f: np.ndarray) -> np.ndarray:
        return (np.array(rgba_f) * 255).astype(np.uint8)

    # --- 1. Create and configure the PBR material ---
    pbr_material = PBRMaterial()

    # Get RGBA and ensure it's in the integer format trimesh expects
    rgba = np.array(material.rgba)
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = rgba_float_to_int(rgba)

    # Set base color and alpha mode (preserving logic from your original function)
    pbr_material.baseColorFactor = rgba
    pbr_material.alphaMode = "BLEND" if rgba[-1] < 255 else "OPAQUE"

    # Map PBR properties
    pbr_material.metallicFactor = material.metallic
    pbr_material.roughnessFactor = material.roughness
    pbr_material.emissiveFactor = np.array([material.emission] * 3)

    # Fallback for legacy shininess value if roughness is not set
    if material.roughness == 0.0 and hasattr(material, "shininess") and material.shininess > 0:
        pbr_material.roughnessFactor = np.sqrt(2.0 / (material.shininess + 2.0))

    # --- 2. Handle Textured vs. Simple Color cases ---
    has_texture = hasattr(material, "textures") and material.textures and material.textures[0]
    texture_loaded = False

    if has_texture:
        texture_path = texture_dir / material.textures[0]
        try:
            texture_image = Image.open(texture_path)
            pbr_material.baseColorTexture = texture_image

            # For a texture to be applied, the mesh needs UV coordinates.
            # We create a TextureVisuals object and assign the material.
            mesh.visual = TextureVisuals(material=pbr_material)

            # Handle texture repetition by scaling the UVs
            if hasattr(material, "texrepeat") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
                mesh.visual.uv *= material.texrepeat

            texture_loaded = True
        except FileNotFoundError:
            print(f"Warning: Texture not found at '{texture_path}'. Falling back to simple color.")
        except Exception as e:
            print(f"Warning: Failed to load texture. Error: {e}. Falling back to simple color.")

    # If no texture was specified or if it failed to load, apply a simple color
    if not texture_loaded:
        # For simple colors, ColorVisuals is the correct type.
        # It sets a color for each face or vertex.
        mesh.visual = ColorVisuals()
        mesh.visual.face_colors = rgba


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
