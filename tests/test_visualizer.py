# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco
import numpy as np
from trimesh.creation import box
from viser import ViserServer

from judo import MODEL_PATH
from judo.visualizers.model import (
    ViserMjModel,
    add_box,
    add_capsule,
    add_cylinder,
    add_ellipsoid,
    add_mesh,
    add_plane,
    add_segments,
    add_sphere,
    add_spline,
    set_mesh_color,
    set_segment_points,
    set_spline_points,
)
from judo.visualizers.utils import rgba_float_to_int, rgba_int_to_float

# Create a global ViserServer instance for use by the tests.
viser_server = ViserServer()
model_path = str(MODEL_PATH / "xml/cylinder_push.xml")
spec = mujoco.MjSpec.from_file(model_path)
model = spec.compile()


def test_model_loading() -> None:
    """Test that the model loads correctly and creates bodies and geometries."""
    viser_model = ViserMjModel(viser_server, spec)
    assert viser_model is not None, "Failed to create ViserMjModel"
    assert len(viser_model._bodies) == model.nbody, "Incorrect number of bodies"
    assert len(viser_model._geoms) > 0, "No geometries created"


def test_set_data() -> None:
    """Test that setting data updates body positions and orientations."""
    viser_model = ViserMjModel(viser_server, spec)
    data = mujoco.MjData(model)
    data.qpos = np.random.randn(model.nq)
    data.qvel = np.random.randn(model.nv)
    mujoco.mj_forward(model, data)
    viser_model.set_data(data)
    for i in range(1, len(viser_model._bodies)):
        assert np.allclose(viser_model._bodies[i].position, data.xpos[i]), f"Position mismatch for body {i}"
        assert np.allclose(viser_model._bodies[i].wxyz, data.xquat[i]), f"Orientation mismatch for body {i}"


def test_ground_plane() -> None:
    """Test that the ground plane can be toggled on and off."""
    viser_model_with_plane = ViserMjModel(viser_server, spec, show_ground_plane=True)
    assert any("ground_plane" in geom.name for geom in viser_model_with_plane._geoms), "Ground plane not found"

    viser_model_no_plane = ViserMjModel(viser_server, spec, show_ground_plane=False)
    assert all("ground_plane" not in geom.name for geom in viser_model_no_plane._geoms), "Unexpected ground plane"


def test_rgba_conversion() -> None:
    """Test conversion between float RGBA and integer RGBA."""
    rgba_float = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    rgba_int = rgba_float_to_int(rgba_float)
    expected_int = (255 * rgba_float).astype(int)
    assert np.array_equal(rgba_int, expected_int), "rgba_float_to_int did not scale correctly"
    rgba_float_converted = rgba_int_to_float(rgba_int)
    # Allow slight differences due to rounding.
    assert np.allclose(rgba_float, rgba_float_converted, atol=1 / 255.0), "rgba_int_to_float did not convert correctly"


def test_set_mesh_color() -> None:
    """Test that set_mesh_color applies the color to a mesh."""
    mesh = box([1, 1, 1])
    rgba = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    set_mesh_color(mesh, rgba)
    expected_int = rgba_float_to_int(rgba)
    # Check a few face colors (all should be set to the same integer color)
    assert mesh.visual is not None, "Mesh visual is None"
    assert np.array_equal(mesh.visual.material.main_color, expected_int), (
        "set_mesh_color did not set the color correctly"
    )


def test_add_plane() -> None:
    """Test that add_plane creates a plane geometry with the correct name."""
    plane = add_plane(
        viser_server,
        "test_plane",
        pos=np.array([1, 2, 3]),
        quat=np.array([1, 0, 0, 0]),
    )
    assert "test_plane" in plane.name, "add_plane did not set correct name"


def test_add_sphere() -> None:
    """Test that add_sphere creates a sphere geometry with the correct name."""
    sphere = add_sphere(
        viser_server,
        "test_sphere",
        radius=1.0,
        pos=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.5, 0.5, 0.5, 1.0]),
    )
    assert "test_sphere" in sphere.name, "add_sphere did not set correct name"


def test_add_cylinder() -> None:
    """Test that add_cylinder creates a cylinder geometry with the correct name."""
    cylinder_geom = add_cylinder(
        viser_server,
        "test_cylinder",
        radius=1.0,
        height=2.0,
        pos=np.ones(3),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.5, 0.5, 0.5, 1.0]),
    )
    assert "test_cylinder" in cylinder_geom.name, "add_cylinder did not set correct name"


def test_add_box() -> None:
    """Test that add_box creates a box geometry with the correct name."""
    box_geom = add_box(
        viser_server,
        "test_box",
        size=np.array([1.0, 1.0, 1.0]),
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.2, 0.3, 0.4, 1.0]),
    )
    assert "test_box" in box_geom.name, "add_box did not set correct name"


def test_add_capsule() -> None:
    """Test that add_capsule creates a capsule geometry with the correct name."""
    capsule_geom = add_capsule(
        viser_server,
        "test_capsule",
        radius=0.5,
        length=2.0,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.3, 0.4, 0.5, 1.0]),
    )
    assert "test_capsule" in capsule_geom.name, "add_capsule did not set correct name"


def test_add_ellipsoid() -> None:
    """Test that add_ellipsoid creates an ellipsoid geometry with the correct name."""
    scaling = np.array([1.0, 2.0, 3.0])
    ellipsoid_geom = add_ellipsoid(
        viser_server,
        "test_ellipsoid",
        scaling=scaling,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.6, 0.7, 0.8, 1.0]),
    )
    assert "test_ellipsoid" in ellipsoid_geom.name, "add_ellipsoid did not set correct name"


def test_add_mesh() -> None:
    """Test that add_mesh creates a mesh geometry with the correct name."""
    mesh = box([1, 1, 1])
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_geom = add_mesh(
        viser_server,
        "test_mesh",
        vertices=vertices,
        faces=faces,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.4, 0.5, 0.6, 1.0]),
    )
    assert "test_mesh" in mesh_geom.name, "add_mesh did not set correct name"


def test_add_spline() -> None:
    """Test that add_spline creates a spline geometry with the correct name."""
    positions = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
    spline_handle = add_spline(
        viser_server,
        "test_spline",
        positions=positions,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgb=(0.1, 0.2, 0.3),
    )
    assert "test_spline" in spline_handle.name, "add_spline did not set correct name"


def test_add_segments() -> None:
    """Test that add_segments creates a segments geometry with the correct name."""
    points = np.random.rand(4, 2, 3)
    segments_handle = add_segments(
        viser_server,
        "test_segments",
        points=points,
        pos=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        rgb=(0.2, 0.3, 0.4),
    )
    assert "test_segments" in segments_handle.name, "add_segments did not set correct name"


class DummySpline:
    """A dummy class to test set_spline_points."""

    def __init__(self) -> None:
        """Initialize the dummy spline with no points."""
        self.points = None


def test_set_spline_points() -> None:
    """Test that set_spline_points sets the points correctly."""
    dummy = DummySpline()
    points_tuple = ((1, 1, 1), (2, 2, 2), (3, 3, 3))
    set_spline_points(dummy, points_tuple)  # type: ignore
    assert np.array_equal(dummy.points, np.array(points_tuple)), "set_spline_points did not set points correctly"  # type: ignore
    # Also test with numpy array input.
    points_np = np.array(points_tuple)
    set_spline_points(dummy, points_np)  # type: ignore
    assert np.array_equal(dummy.points, points_np), "set_spline_points did not convert numpy array correctly"  # type: ignore


class DummySegment:
    """A dummy class to test set_segment_points."""

    def __init__(self) -> None:
        """Initialize the dummy segment with no points."""
        self.points = None


def test_set_segment_points() -> None:
    """Test that set_segment_points sets the points correctly."""
    dummy = DummySegment()
    points = np.random.rand(5, 2, 3)
    set_segment_points(dummy, points)  # type: ignore
    np.testing.assert_allclose(
        dummy.points,  # type: ignore
        points,
        err_msg="set_segment_points did not set points correctly",
    )


def test_remove_traces() -> None:
    """Test that remove_traces clears the traces list."""
    viser_model = ViserMjModel(viser_server, spec)
    # Ensure _traces is defined.
    assert hasattr(viser_model, "_traces"), "Model does not have _traces attribute"
    # Remove the traces.
    viser_model.remove_traces()
    assert viser_model._traces == [], "remove_traces did not clear the traces list"


def test_set_traces() -> None:
    """Test that set_traces correctly sets the traces and handles them."""
    viser_model = ViserMjModel(viser_server, spec)
    # Create a dummy trace array with shape (3, 2, 3)
    traces = np.random.rand(3, 2, 3)
    all_traces_rollout_size = 2
    viser_model.set_traces(traces, all_traces_rollout_size=all_traces_rollout_size)
    # Check that two trace handles have been created.
    assert len(viser_model._traces) == 2, "set_traces did not create 2 trace handles when needed"
    # Check that the points in the 'best' trace handle equal traces[:2]
    np.testing.assert_allclose(
        viser_model._traces[0].points,
        traces[:all_traces_rollout_size],
        err_msg="Best trace points not set correctly",
    )
    # Check that the points in the 'other' trace handle equal traces[2:]
    np.testing.assert_allclose(
        viser_model._traces[1].points,
        traces[all_traces_rollout_size:],
        err_msg="Other trace points not set correctly",
    )
    # Now test with None input to remove traces.
    viser_model.set_traces(None, all_traces_rollout_size=all_traces_rollout_size)
    assert viser_model._traces == [], "set_traces with None did not remove traces"


def test_remove() -> None:
    """Test that remove clears the traces and removes geometries."""
    viser_model = ViserMjModel(viser_server, spec)
    viser_model.remove()
    # Verify that traces are cleared.
    assert viser_model._traces == [], "remove did not clear _traces"
    # Note: geometries are removed via their remove() method; further checking
    # would require inspection of ViserServer's scene state.
