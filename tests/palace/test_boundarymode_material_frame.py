"""A BoundaryMode config writes material tensors in the 2D mesh frame.

The native 2D mesh of an ``x=...`` cross-section has the layout's y on mesh x,
z on mesh y and the propagation direction x on mesh z; for ``y=...`` it is
(x, z, y). Palace reads ``Permittivity`` in the mesh frame, so a layout-axes
``[xx, yy, zz]`` has to be re-ordered or an anisotropic material gets the
wrong component along each axis.
"""

from __future__ import annotations

import json

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import BoundaryModeSim
from gsim.palace.mesh.config_generator import _to_section_frame
from gsim.palace.mesh.generator import generate_mesh
from gsim.palace.models import BoundaryModeConfig
from gsim.palace.models.cross_section import CrossSectionPlaneConfig

EPS = [11.0, 22.0, 33.0]  # layout (xx, yy, zz), all different
ROTATED_AXES = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]


@pytest.mark.parametrize(
    ("axis", "rows"),
    [
        # rows of R: the layout axis behind mesh x, y, z
        ("x", [(0, 1, 0), (0, 0, 1), (1, 0, 0)]),
        ("y", [(1, 0, 0), (0, 0, 1), (0, 1, 0)]),
    ],
)
def test_reordering_is_the_tensor_in_the_mesh_axes(axis, rows):
    rotation = np.array(rows, dtype=float)
    expected = np.diag(rotation @ np.diag(EPS) @ rotation.T)
    got = _to_section_frame({"permittivity": EPS}, axis)["permittivity"]
    assert got == pytest.approx(expected)


def test_scalars_other_planes_and_the_input_are_untouched():
    props = {"permittivity": 3.9, "loss_tangent": [1e-3, 2e-3, 3e-3]}
    assert _to_section_frame(props, "x")["permittivity"] == 3.9
    assert _to_section_frame(props, "x")["loss_tangent"] == [2e-3, 3e-3, 1e-3]
    assert props["loss_tangent"] == [1e-3, 2e-3, 3e-3]
    assert _to_section_frame(props, "z") is props


def test_rotated_material_axes_are_refused_not_guessed():
    props = {"permittivity": EPS, "material_axes": ROTATED_AXES}
    with pytest.raises(ValueError, match="material_axes"):
        _to_section_frame(props, "x")
    identity = {"permittivity": EPS, "material_axes": np.eye(3).tolist()}
    assert _to_section_frame(identity, "x")["permittivity"] == [22.0, 33.0, 11.0]


def _component():
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    comp << gf.c.rectangle((40.0, 10.0), centered=True, layer=(1, 0))
    return comp


def _stack(extra_materials=None, film_material="aniso"):
    stack = LayerStack(pdk_name="aniso", units="um")
    stack.materials = {
        "aniso": {"permittivity": EPS},
        "sio2": {"permittivity": 3.9},
        **(extra_materials or {}),
    }
    stack.layers = {
        "film": Layer(
            name="film",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=1.0,
            thickness=1.0,
            material=film_material,
            layer_type="dielectric",
        )
    }
    stack.dielectrics = [{"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "sio2"}]
    return stack


def _tensors(config):
    return [
        m["Permittivity"]
        for m in config["Domains"]["Materials"]
        if isinstance(m["Permittivity"], list)
    ]


def _config(tmp_path, cut, stack=None):
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(_component())
    sim.set_stack(stack if stack is not None else _stack())
    sim.set_cross_section(cut)
    sim.set_boundary_mode(freq=10e9, num_modes=1)
    sim.set_airbox(margin_x=0.0, margin_y=5.0, z_above=3.0, z_below=0.0)
    sim.mesh(preset="coarse", verbose=False)
    sim.write_config(photonic=True)
    config = json.loads((tmp_path / "config.json").read_text())
    return sim, config


@pytest.mark.parametrize(
    ("cut", "expected", "mesh_first_axis_span"),
    [
        ("x=0", [22.0, 33.0, 11.0], 10.0),  # mesh x spans the layout's y extent
        ("y=0", [11.0, 33.0, 22.0], 40.0),  # mesh x spans the layout's x extent
    ],
)
def test_config_tensor_follows_the_mesh_frame(
    tmp_path, cut, expected, mesh_first_axis_span
):
    sim, config = _config(tmp_path, cut)
    points = np.asarray(meshio.read(sim.mesh_path).points)
    # the frame the tensor has to match: film width along mesh x
    assert points[:, 0].max() - points[:, 0].min() >= mesh_first_axis_span
    assert _tensors(config) == [expected]
    # the stack the caller handed in is not rewritten
    assert sim.stack.materials["aniso"]["permittivity"] == EPS


def test_unused_rotated_material_does_not_block_the_config(tmp_path):
    """Only the materials in the mesh are checked, not every stack entry."""
    unused = {"permittivity": [1.0, 2.0, 3.0], "material_axes": ROTATED_AXES}
    _, config = _config(tmp_path, "x=0", _stack({"unused_rotated": unused}))
    assert _tensors(config) == [[22.0, 33.0, 11.0]]
    assert all("MaterialAxes" not in m for m in config["Domains"]["Materials"])


def test_used_rotated_material_is_still_refused(tmp_path):
    used = {"permittivity": EPS, "material_axes": ROTATED_AXES}
    stack = _stack({"rotated": used}, film_material="rotated")
    with pytest.raises(ValueError, match="material_axes"):
        _config(tmp_path, "x=0", stack)


@pytest.mark.parametrize(
    ("cut", "expected"),
    [("x=0", [22.0, 33.0, 11.0]), ("y=0", [11.0, 33.0, 22.0])],
)
def test_generate_mesh_writes_the_config_in_the_mesh_frame(tmp_path, cut, expected):
    """The public generate_mesh(write_config=True) path gets the cut axis too."""
    result = generate_mesh(
        component=_component(),
        stack=_stack(),
        ports=[],
        output_dir=tmp_path,
        refined_mesh_size=2.0,
        max_mesh_size=10.0,
        margin_x=0.0,
        margin_y=5.0,
        airbox_margin_x=0.0,
        airbox_margin_y=5.0,
        airbox_z_above=3.0,
        airbox_z_below=0.0,
        simulation_type="boundarymode",
        boundary_mode_config=BoundaryModeConfig(freq=10e9, num_modes=1),
        cross_section=CrossSectionPlaneConfig.from_spec(cut),
        write_config=True,
        verbosity=0,
    )
    assert result.config_path is not None
    config = json.loads(result.config_path.read_text())
    assert _tensors(config) == [expected]
