"""Check geometric degeneracy and actionable mesh diagnostics."""

from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gsim.palace.mesh import validation
from gsim.palace.ports.config import PalacePort


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([[0, 0, 0]] * 3, "at least 4"),
        ([[0, 0, 0]] * 4, "p_000"),
        ([[i, 0, 0] for i in range(4)], "collinear"),
    ],
)
def test_degenerate_oriented_bounds(points, message):
    with pytest.raises(ValueError, match=message):
        validation._palace_obb(np.array(points, dtype=float))


def test_spatial_oriented_bounds_preserve_cuboid():
    points = np.array(list(product([0, 2], [0, 3], [0, 4])), dtype=float)
    center, axes, planar = validation._palace_obb(points)
    assert not planar
    np.testing.assert_allclose(center, [1, 1.5, 2])
    assert sorted(2 * np.linalg.norm(axis) for axis in axes) == pytest.approx([2, 3, 4])


@pytest.mark.parametrize(
    ("points", "direction", "message"),
    [
        ([], "invalid", "unknown direction"),
        (None, "+X", "could not read mesh"),
        ([[0, 0, 0]] * 3, "+X", "only 3 vertices"),
        ([[0, 0, 0]] * 4, "+X", "OBB computation failed"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], "+Z", "geometry check FAILED"),
        ([[0, 0, 0], [2, 1, 0], [1, 3, 0], [-1, 2, 0]], "+X", "geometry check FAILED"),
    ],
)
def test_surface_geometry_diagnostics(
    tmp_path, monkeypatch, points, direction, message
):
    reader = (
        Mock(side_effect=OSError("unreadable"))
        if points is None
        else Mock(return_value=np.array(points, dtype=float))
    )
    monkeypatch.setattr(validation, "_get_port_vertices", reader)
    errors = validation._check_surface_geometry(
        mesh_path=tmp_path / "mesh.msh",
        phys_tag=1,
        direction_str=direction,
        context="Port p",
        rel_tol=1e-6,
    )
    assert len(errors) == 1
    assert message in errors[0]


@pytest.mark.parametrize("group", [None, {}, {"type": "cpw", "elements": [{}]}])
def test_missing_port_group_ids_are_reported(tmp_path, group):
    errors = validation.check_lumped_port_geometry(
        mesh_path=tmp_path / "mesh.msh",
        port_surfaces={} if group is None else {"P1": group},
        palace_ports=[PalacePort(name="p", multi_element=bool(group))],
    )
    assert len(errors) == (0 if group is None else 1)
    if errors:
        assert "missing physical group id" in errors[0]


@pytest.mark.parametrize(
    "config_kind", ["empty", "invalid", "writer-failure", "no-output"]
)
@pytest.mark.parametrize("cpw", [False, True])
def test_incomplete_mesh_and_configuration(tmp_path, config_kind, cpw):
    config = tmp_path / "config.json"
    config.write_text("invalid" if config_kind == "invalid" else "{}")
    sim = SimpleNamespace(
        simulation_type="driven",
        _output_dir=None if config_kind == "no-output" else tmp_path,
        _last_mesh_result=SimpleNamespace(
            groups={"port_surfaces": {"P1": {"type": "cpw", "elements": []}}}
            if cpw
            else {},
            mesh_path=tmp_path / "missing.msh",
        ),
        write_config=Mock(side_effect=RuntimeError("writer failed"))
        if config_kind == "writer-failure"
        else Mock(),
    )
    result = validation.validate_mesh(sim)
    assert not result.valid
    assert "No dielectric volumes in mesh." in result.errors
    assert any("No conductor surfaces" in error for error in result.errors)
    assert any(
        "CPW port" if cpw else "No port surfaces" in error for error in result.errors
    )
    if config_kind == "invalid":
        assert any("invalid JSON" in error for error in result.errors)
    if config_kind == "writer-failure":
        assert any("writer failed" in error for error in result.errors)
