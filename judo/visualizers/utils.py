# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
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

    return mesh.scale  # type: ignore


def has_material(mesh: trimesh.Trimesh) -> bool:
    """Check if mesh needs MuJoCo material applied (no texture or has placeholder)."""
    if isinstance(mesh.visual, ColorVisuals):
        return True

    # Check if mesh has a tiny placeholder texture (< 100px) from failed trimesh load
    if not isinstance(mesh.visual, TextureVisuals):
        return False
    img = getattr(getattr(mesh.visual, "material", None), "image", None)
    return img is not None and img.size[0] < 100


def apply_mujoco_material(
    mesh: trimesh.Trimesh,
    material: MjsMaterial,
    spec: MjSpec | None = None,
    mesh_file: Path | None = None,
) -> None:
    """Applies a MuJoCo material to a trimesh mesh.

    Tries to load texture from: 1) MuJoCo spec, 2) MTL file, 3) solid color fallback.

    Args:
        mesh: the trimesh.Trimesh to modify
        material: an object matching the mjsMaterial struct
        spec: optional MjSpec to look up texture files
        mesh_file: optional mesh file path to find MTL texture
    """
    pbr = PBRMaterial()

    # Get RGBA color
    rgba = np.array(material.rgba)
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = rgba_float_to_int(rgba)
    color = tuple(int(x) for x in rgba.tolist())
    pbr.alphaMode = "BLEND" if rgba[3] < 255 else "OPAQUE"

    # Set PBR values
    pbr.metallicFactor = float(material.metallic)
    pbr.roughnessFactor = float(material.roughness)
    pbr.emissiveFactor = [material.emission] * 3
    if material.roughness == 0.0 and getattr(material, "shininess", 0) > 0:
        pbr.roughnessFactor = np.sqrt(2.0 / (material.shininess + 2.0)).item()

    # Get existing UV coordinates
    uv = getattr(mesh.visual, "uv", None)

    # Try to load texture from MuJoCo spec
    texture_loaded = False
    if spec is not None and getattr(material, "textures", None):
        texture_name = material.textures[1] if len(material.textures) > 1 else None
        if texture_name:
            try:
                texture = spec.texture(texture_name)
                if texture.file:
                    texture_path = Path(spec.modelfiledir) / spec.texturedir / texture.file
                    if texture_path.exists():
                        pbr.baseColorTexture = Image.open(texture_path)
                        texture_loaded = True
            except Exception:
                warnings.warn(f"Failed to load texture '{texture_name}' for material '{material.name}'", stacklevel=2)

    # Try to load texture from MTL file
    if not texture_loaded and mesh_file is not None:
        texture_loaded = _load_texture_from_mtl(pbr, mesh_file)

    # Fall back to solid color
    if not texture_loaded:
        pbr.baseColorTexture = Image.new("RGBA", (1, 1), color)
        uv = None

    mesh.visual = TextureVisuals(material=pbr, uv=uv)


def _load_texture_from_mtl(pbr: PBRMaterial, mesh_file: Path) -> bool:
    """Try to load texture from MTL file. Returns True if successful."""
    # Find MTL file: same name, or base name for split meshes (body_0.obj -> body.mtl)
    mtl_file = mesh_file.with_suffix(".mtl")
    if not mtl_file.exists():
        stem = mesh_file.stem
        if stem[-1].isdigit() and "_" in stem:
            mtl_file = mesh_file.parent / f"{stem.rsplit('_', 1)[0]}.mtl"
    if not mtl_file.exists():
        return False

    try:
        for line in mtl_file.read_text().split("\n"):
            if line.strip().startswith("map_Kd"):
                tex_path = (mtl_file.parent / line.split()[-1]).resolve()
                if tex_path.exists():
                    pbr.baseColorTexture = Image.open(tex_path)
                    return True
                else:
                    warnings.warn(f"Texture not found: {tex_path}", stacklevel=3)
    except Exception as e:
        warnings.warn(f"Failed to load texture from {mtl_file}: {e}", stacklevel=3)
    return False


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
