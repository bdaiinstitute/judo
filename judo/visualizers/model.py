# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from pathlib import Path
from typing import Any, List, Tuple

import mujoco
import numpy as np
import trimesh
from mujoco import MjData, MjsMaterial, MjSpec
from trimesh.creation import box, capsule, cylinder, icosphere
from trimesh.transformations import scale_and_translate
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import PBRMaterial
from viser import (
    ClientHandle,
    LineSegmentsHandle,
    SceneNodeHandle,
    SplineCatmullRomHandle,
    ViserServer,
)

from judo.visualizers.utils import (
    apply_mujoco_material,
    count_trace_sensors,
    get_mesh_file,
    get_mesh_scale,
    rgba_float_to_int,
)

DEFAULT_GRID_SECTION_COLOR = (0.02, 0.14, 0.44)
DEFAULT_GRID_CELL_COLOR = (0.27, 0.55, 1)
DEFAULT_SPHERE_SUBDIVISIONS = 3
DEFAULT_SPLINE_COLOR = (0.8, 0.1, 0.8)
DEFAULT_BEST_SPLINE_COLOR = (0.96, 0.7, 0.0)


class ViserMjModel:
    """Helper for rendering MJCF models in viser.

    Args:
        target: ViserServer or ClientHandle to add MjModel to.
        spec: MjSpec of the model to be visualized.
        show_ground_plane: optional flag to show the default ground plane.
        geom_exclude_substring: optional string to exclude a geom from visualization.
    """

    def __init__(
        self,
        target: ViserServer | ClientHandle,
        spec: MjSpec,
        show_ground_plane: bool = True,
        geom_exclude_substring: str = "",
    ) -> None:
        """Constructor for ViserMjModel."""
        self._target = target
        self._spec = spec
        self._model = spec.compile()

        # Assume first body is root of kinematic tree.
        self._bodies = [self._target.scene.add_frame(self._spec.bodies[0].name, show_axes=False)]
        self._geoms: List = []

        # Show world plane if desired.
        if show_ground_plane:
            self._geoms.append(add_plane(self._target, "ground_plane"))

        # Add coordinate frame for each non-world body in model.
        _geom_placeholder_idx = 0
        _body_placeholder_idx = 0

        for body in self._spec.bodies[1:]:
            # Sharp edge: not using the tree structure of the kinematics ...
            body_name = body.name
            if not body_name:
                body_name = f"body_{_body_placeholder_idx}"
                _body_placeholder_idx += 1
            self._bodies.append(self._target.scene.add_frame(body_name, show_axes=False))

            for geom in body.geoms:
                suffix = geom.name
                if not suffix:  # if geom has no name, use a placeholder.
                    suffix = f"{_geom_placeholder_idx}"
                    _geom_placeholder_idx += 1
                geom_name = f"{body_name}/geom_{suffix}"
                if geom_exclude_substring and geom_exclude_substring in geom_name:
                    continue
                self.add_geom(geom_name, geom)

        # Add traces
        self._num_trace_sensors = count_trace_sensors(self._model)
        self.all_traces_rollout_size = 0
        self.add_traces()

    def add_geom(self, geom_name: str, geom: Any) -> None:
        """Helper function for adding geoms to scene tree."""
        # Store compiled model geom info (handles things like fromto).
        model_geom = self._model.geom(geom.name)
        match geom.type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                # TODO(pculbert): support more color options.
                self._geoms.append(
                    add_plane(
                        self._target,
                        geom_name,
                        pos=geom.pos,
                        quat=geom.quat,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_HFIELD:
                # TODO(pculbert): Implement HField viz as collection of boxes (?).
                raise NotImplementedError("HField is not implemented.")
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                self._geoms.append(
                    add_sphere(
                        self._target,
                        geom_name,
                        radius=model_geom.size[0],
                        pos=geom.pos,
                        quat=geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                self._geoms.append(
                    add_capsule(
                        self._target,
                        geom_name,
                        radius=model_geom.size[0],
                        length=2 * model_geom.size[1],  # MJC has capsule half-lengths.
                        pos=model_geom.pos,
                        quat=model_geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                raise NotImplementedError
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                self._geoms.append(
                    add_cylinder(
                        self._target,
                        geom_name,
                        radius=model_geom.size[0],
                        height=2 * model_geom.size[1],
                        pos=model_geom.pos,
                        quat=model_geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_BOX:
                self._geoms.append(
                    add_box(
                        self._target,
                        geom_name,
                        size=2 * model_geom.size,  # MJC has box half-lengths.
                        pos=model_geom.pos,
                        quat=model_geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_MESH:
                # Get necessary mesh properties.
                mesh_file = get_mesh_file(self._spec, geom)
                mesh_scale = get_mesh_scale(self._spec, geom)

                # Introspect on texture.
                mjs_material = self._spec.material(geom.material)

                # Call the new, robust function to add the mesh.
                handle = add_mesh_from_file(
                    target=self._target,
                    name=geom_name,
                    mesh_file=mesh_file,
                    pos=geom.pos,
                    quat=geom.quat,
                    mesh_scale=mesh_scale,
                    mjs_material=mjs_material,
                )
                self._geoms.append(handle)
            case mujoco.mjtGeom.mjGEOM_SDF:
                raise NotImplementedError("")
            case _:
                raise NotImplementedError(f"Geom type {geom.type} is not supported for visualization.")

    def add_traces(
        self,
        num_traces: int = 0,
        all_traces_rollout_size: int = 0,
        trace_name: str = "trace",
    ) -> None:
        """Add a collection of all traces to the visualizer, done in one go to avoid having too many things.

        We have two sets of traces to care about: the "elite" reward traces and the regular ones. Due to how the line
        segments work, we only need one handle per type.
        """
        # Size is num_traces * size of rollout per trace
        self._traces = []
        self._traces.append(
            add_segments(
                self._target,
                f"best_{trace_name}",
                1e-4 * np.random.rand(4, 2, 3),  # non zero initialization to avoid errors
                rgb=DEFAULT_BEST_SPLINE_COLOR,
            )
        )
        self._traces[0].colors = np.tile(self._traces[0].colors[0, :, :], (all_traces_rollout_size, 1, 1))
        if (rest_trace_size := num_traces - all_traces_rollout_size) > 0:
            self._traces.append(
                add_segments(
                    self._target,
                    f"other_{trace_name}",
                    1e-4 * np.random.rand(4, 2, 3),  # non zero initialization to avoid errors
                    rgb=DEFAULT_SPLINE_COLOR,
                )
            )
            self._traces[1].colors = np.tile(self._traces[1].colors[0, :, :], (rest_trace_size, 1, 1))

    def remove_traces(self) -> None:
        """Remove traces."""
        for trace in self._traces:
            trace.remove()
        self._traces = []

    def set_data(self, data: MjData) -> None:
        """Write updated configuration from mujoco data to viser viewer."""
        # Loop over all bodies, just reading the FK results from data.
        for i in range(1, len(self._bodies)):
            # Use atomic to update both position/orientation synchronously.
            with self._target.atomic():
                # Line up order of bodies in spec with order of bodies in model.
                data_idx = self._spec.bodies[i].id
                self._bodies[i].position = tuple(data.xpos[data_idx])
                self._bodies[i].wxyz = tuple(data.xquat[data_idx])

    def set_traces(self, traces: np.ndarray | None, all_traces_rollout_size: int) -> None:
        """Write updated traces to viser viewer.

        Args:
            traces: trace sensors readings of size (self.num_elite * all_traces_rollout_size, 2, 3).
            all_traces_rollout_size: num_trace_sensors * single_rollout, size of all grouped trace sensor rollouts.
        """
        # Erase all traces if None is received
        if traces is None or self._num_trace_sensors == 0:
            self.remove_traces()
        else:
            num_traces, num_points, trace_dim = traces.shape
            assert trace_dim == 3
            assert num_points == 2, "Number of points in line segment must be 2"

            # check if the number of traces has updated
            if (
                len(self._traces) <= 1
                or self.all_traces_rollout_size != all_traces_rollout_size
                or num_traces != self.num_traces
            ):
                self.remove_traces()
                self.add_traces(num_traces=num_traces, all_traces_rollout_size=all_traces_rollout_size)
                self.all_traces_rollout_size = all_traces_rollout_size
                self.num_traces = num_traces

            # check if there is only the elite trace
            if num_traces == all_traces_rollout_size:
                for trace in self._traces[1:]:
                    trace.remove()
                self._traces = [self._traces[0]]

            # Use atomic to update all traces synchronously
            with self._target.atomic():
                set_segment_points(self._traces[0], traces[:all_traces_rollout_size, :, :])
                if num_traces > all_traces_rollout_size and len(self._traces) > 1:
                    set_segment_points(self._traces[1], traces[all_traces_rollout_size:, :, :])

    def remove(self) -> None:
        """Wrapper function to remove all geometries from Viser."""
        for geom in self._geoms:
            geom.remove()
        self.remove_traces()


def add_plane(
    target: ViserServer | ClientHandle,
    name: str,
    pos: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
) -> SceneNodeHandle:
    """Add a plane geometry to the visualizer with optional position, quaternion, material, and name."""
    return target.scene.add_grid(
        name,
        position=pos,
        wxyz=quat,
        section_color=DEFAULT_GRID_SECTION_COLOR,
        cell_color=DEFAULT_GRID_CELL_COLOR,
    )


def add_sphere(
    target: ViserServer | ClientHandle,
    name: str,
    radius: float,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a sphere geometry to the visualizer with optional position, quaternion, material, and name."""
    sphere_mesh = icosphere(DEFAULT_SPHERE_SUBDIVISIONS, radius)
    set_mesh_color(sphere_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, sphere_mesh, position=pos, wxyz=quat)


def add_cylinder(
    target: ViserServer | ClientHandle,
    name: str,
    radius: float,
    height: float,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a cylinder geometry to the visualizer with optional position, quaternion, material, and name.

    The cylinder is aligned with the z-axis
    """
    cylinder_mesh = cylinder(radius, height)
    set_mesh_color(cylinder_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, cylinder_mesh, position=pos, wxyz=quat)


def add_box(
    target: ViserServer | ClientHandle,
    name: str,
    size: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a box geometry to the visualizer with optional position, quaternion, material, and name."""
    box_mesh = box(size)
    set_mesh_color(box_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, box_mesh, position=pos, wxyz=quat)


def add_capsule(
    target: ViserServer | ClientHandle,
    name: str,
    radius: float,
    length: float,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a capsule geometry to the visualizer with optional position, quaternion, material, and name.

    The capsule is aligned with the z-axis
    """
    # First create a capsule mesh with trimesh since viser doesn't implement it.
    capsule_mesh = capsule(length, radius)
    set_mesh_color(capsule_mesh, rgba)
    return target.scene.add_mesh_trimesh(
        name,
        capsule_mesh,
        wxyz=quat,
        position=pos,
    )


def add_ellipsoid(
    target: ViserServer | ClientHandle,
    name: str,
    scaling: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add an ellipsoid geometry to the visualizer."""
    assert len(scaling) == 3, "Must provide exactly three scalings for ellipsoid."

    # Create sphere mesh.
    ellipsoid_mesh = icosphere(DEFAULT_SPHERE_SUBDIVISIONS, 1.0)

    # Scale this to get an ellipsoid.
    ellipsoid_mesh.apply_transform(scale_and_translate(scaling))

    set_mesh_color(ellipsoid_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, ellipsoid_mesh, position=pos, wxyz=quat)


def add_mesh(
    target: ViserServer | ClientHandle,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a triangular mesh geometry to the visualizer.

    Add a triangular mesh geometry to the visualizer with specified vertices and faces,
    with optional position, quaternion, material, and name.

    Vertices: float (N, 3) and faces: int (M, 3).
    """
    mesh = trimesh.Trimesh(vertices, faces)
    set_mesh_color(mesh, rgba)
    return target.scene.add_mesh_trimesh(name, mesh, position=pos, wxyz=quat)


def add_mesh_from_file(
    target: ViserServer | ClientHandle,
    name: str,
    mesh_file: Path,
    pos: np.ndarray,
    quat: np.ndarray,
    mesh_scale: np.ndarray | None = None,
    mjs_material: MjsMaterial | None = None,
) -> SceneNodeHandle:
    """Add a triangle mesh from file, via trimesh."""
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file {mesh_file} does not exist.")
    mesh = trimesh.load(mesh_file, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), "Loaded geometry is not a mesh type."
    if mesh_scale is not None:
        mesh.apply_scale(mesh_scale)

    # If mesh does not have a good texture, apply MuJoCo one.
    if isinstance(mesh.visual, ColorVisuals) and mjs_material is not None:
        apply_mujoco_material(mesh, mjs_material)

    return target.scene.add_mesh_trimesh(name, mesh, position=pos, wxyz=quat)


def add_spline(
    target: ViserServer | ClientHandle,
    name: str,
    positions: tuple[tuple[float, float, float], ...] | np.ndarray,
    pos: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    rgb: Tuple[float, float, float] = DEFAULT_SPLINE_COLOR,
    line_width: float = 4.0,
    segments: int | None = None,
    visible: bool = True,
) -> SplineCatmullRomHandle:
    """Add a spline to the visualizer with optional position, quaternion."""
    return target.scene.add_spline_catmull_rom(
        name,
        positions,
        position=pos,
        wxyz=quat,
        color=rgb,
        line_width=line_width,
        segments=segments,
        visible=visible,
    )


def add_segments(
    target: ViserServer | ClientHandle,
    name: str,
    points: np.ndarray,
    pos: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    rgb: Tuple[float, float, float] = DEFAULT_SPLINE_COLOR,
    line_width: float = 4.0,
    visible: bool = True,
) -> LineSegmentsHandle:
    """Add line segments to the visualizer with an optional position and orientation.

    TODO(@bhung) Potentially add support for different kinds of segments

    Args:
        target: ViserServer or handle to attach the segments to
        name: name of the segments
        points: size (N x 2 x 3) where index 0 is point, 1 is start vs end, and 2 is 3D coord
        pos: position that the points are defined with respect to. Defaults to origin
        quat: orientation that the points are defined with respect to. Defaults to identity
        rgb: colors of the points. Can be sized (N x 2 x 3) or a broadcastable shape
        line_width: width of the line, in pixels
        visible: whether or not the lines are initially visible
    """
    return target.scene.add_line_segments(
        name,
        points,
        rgb,
        line_width=line_width,
        wxyz=quat,
        position=pos,
        visible=visible,
    )


def set_mesh_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    """Set the color of a trimesh mesh."""
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = rgba_float_to_int(rgba)

    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=rgba,
            main_color=rgba,
            metallicFactor=0.5,
            roughnessFactor=1.0,
            alphaMode="BLEND" if rgba[-1] < 255 else "OPAQUE",
        )
    )


def set_spline_points(
    handle: SplineCatmullRomHandle,
    points: tuple[tuple[float, float, float], ...] | np.ndarray,
) -> None:
    """Set the spline waypoints."""
    points = np.asarray(points)
    assert len(points[0]) == 3 and len(points.shape) == 2
    handle.points = points


def set_segment_points(handle: LineSegmentsHandle, points: np.ndarray) -> None:
    """Set the line waypoints.

    Args:
        handle: handle for the line segments
        points: start and end points of the line segments (N x 2 x 3)
    """
    if isinstance(points, np.ndarray):
        assert len(points.shape) == 3 and points.shape[1] == 2 and points.shape[2] == 3
    handle.points = points
