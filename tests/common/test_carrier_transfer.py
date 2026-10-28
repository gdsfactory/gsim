"""Carrier maps evaluated on a mesh other than the one they were solved on."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import meshio
import numpy as np
import pytest

from gsim.common.carrier_transfer import transfer_carriers


def _unit_square_mesh(tmp_path, name="square.msh", *, scale=1.0, offset=0.0):
    """Unit square (scaled/shifted) split into 'core' and 'clad' triangles."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    points[:, :2] = points[:, :2] * scale + offset
    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("triangle", np.array([[0, 1, 2]])),
            ("triangle", np.array([[0, 2, 3]])),
        ],
        cell_data={
            "gmsh:physical": [np.array([1]), np.array([2])],
            "gmsh:geometrical": [np.array([1]), np.array([2])],
        },
        field_data={"core": np.array([1, 2]), "clad": np.array([2, 2])},
    )
    path = tmp_path / name
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
    return path


def _linear_carriers(a=3.0, b=-2.0, c=10.0):
    """Carrier map whose concentrations are exactly linear in (x, y).

    Linear interpolation reproduces a linear field exactly, so the
    transferred values have a known analytic answer everywhere inside the
    source domain.
    """
    grid = np.linspace(-0.5, 1.5, 9)
    x, y = (g.ravel() for g in np.meshgrid(grid, grid))
    return SimpleNamespace(
        x_um=x,
        y_um=y,
        electrons_cm3=a * x + b * y + c,
        holes_cm3=c - a * x,
    )


class TestKnownAnalyticAnswer:
    def test_linear_field_is_reproduced_at_nodes(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)
        carriers = _linear_carriers()

        got = transfer_carriers(carriers, mesh_path)

        expected = 3.0 * got.x_um - 2.0 * got.y_um + 10.0
        assert np.allclose(got.electrons_cm3, expected)
        assert np.allclose(got.holes_cm3, 10.0 - 3.0 * got.x_um)
        assert not got.filled.any()

    def test_linear_field_is_reproduced_at_element_centroids(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)

        got = transfer_carriers(_linear_carriers(), mesh_path, at="elements")

        assert got.electrons_cm3.size == 2
        assert np.allclose(got.electrons_cm3, 3.0 * got.x_um - 2.0 * got.y_um + 10.0)


class TestIdenticalMesh:
    def test_transfer_onto_the_source_mesh_is_a_no_op(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)
        mesh = meshio.read(str(mesh_path))
        source = SimpleNamespace(
            x_um=mesh.points[:, 0],
            y_um=mesh.points[:, 1],
            electrons_cm3=np.array([1e18, 2e18, 3e18, 4e18]),
            holes_cm3=np.array([4e17, 3e17, 2e17, 1e17]),
        )

        got = transfer_carriers(source, mesh_path)

        assert np.allclose(got.electrons_cm3, source.electrons_cm3, rtol=1e-12)
        assert np.allclose(got.holes_cm3, source.holes_cm3, rtol=1e-12)


class TestOutsideTheSourceDomain:
    def test_points_outside_take_the_caller_fill(self, tmp_path):
        # Source cloud covers the unit square; the target sits beyond it.
        mesh_path = _unit_square_mesh(tmp_path, "far.msh", scale=1.0, offset=5.0)
        carriers = _linear_carriers()

        got = transfer_carriers(carriers, mesh_path, fill=0.0)

        assert got.filled.all()
        assert np.all(got.electrons_cm3 == 0.0)
        assert np.all(got.holes_cm3 == 0.0)

    def test_fill_accepts_separate_electron_and_hole_values(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path, "far.msh", offset=5.0)

        got = transfer_carriers(_linear_carriers(), mesh_path, fill=(7.0, 9.0))

        assert np.all(got.electrons_cm3 == 7.0)
        assert np.all(got.holes_cm3 == 9.0)

    def test_nearest_fill_extends_the_solved_values(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path, "far.msh", offset=5.0)
        carriers = _linear_carriers()

        got = transfer_carriers(carriers, mesh_path, fill="nearest")

        assert got.filled.all()
        # Nearest source sample is the corner of the source cloud.
        assert np.all(got.electrons_cm3 == pytest.approx(3.0 * 1.5 - 2.0 * 1.5 + 10.0))

    def test_partial_coverage_marks_only_the_outside_points(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path, "half.msh", scale=2.0)
        carriers = _linear_carriers()

        got = transfer_carriers(carriers, mesh_path, fill=0.0)

        assert got.filled.any()
        assert not got.filled.all()


class TestPhysicalGroups:
    def test_regions_are_reported_without_the_caller_touching_tags(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)

        got = transfer_carriers(_linear_carriers(), mesh_path, at="elements")

        assert got.region == ["core", "clad"]

    def test_carriers_are_restricted_to_the_named_regions(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)

        got = transfer_carriers(
            _linear_carriers(), mesh_path, at="elements", regions=["core"]
        )

        assert got.region == ["core", "clad"]
        assert got.electrons_cm3[1] == 0.0
        assert got.electrons_cm3[0] != 0.0
        assert got.filled.tolist() == [False, True]

    def test_a_restricted_point_inside_the_hull_takes_the_nearest_sample(
        self, tmp_path
    ):
        """`fill="nearest"` reaches the region restriction, not only the hull.

        A point the regions exclude sits well inside the solved domain,
        where the interpolant has an answer; the nearest fill has to
        override it, or the restriction does nothing for that point.
        """
        mesh_path = _unit_square_mesh(tmp_path)

        restricted = transfer_carriers(
            _linear_carriers(),
            mesh_path,
            at="elements",
            fill="nearest",
            regions=["core"],
        )
        unrestricted = transfer_carriers(
            _linear_carriers(), mesh_path, at="elements", fill="nearest"
        )

        assert restricted.filled.tolist() == [False, True]
        assert restricted.electrons_cm3[1] != pytest.approx(
            unrestricted.electrons_cm3[1]
        )

    def test_unknown_region_name_is_rejected_with_the_available_ones(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)

        with pytest.raises(ValueError, match="clad"):
            transfer_carriers(_linear_carriers(), mesh_path, regions=["nope"])

    def test_nodes_inherit_the_region_of_an_incident_element(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)

        got = transfer_carriers(_linear_carriers(), mesh_path)

        assert set(got.region) <= {"core", "clad"}
        assert len(got.region) == got.x_um.size


class TestValidation:
    def test_mismatched_sample_lengths_are_rejected(self, tmp_path):
        mesh_path = _unit_square_mesh(tmp_path)
        bad = SimpleNamespace(
            x_um=np.zeros(4),
            y_um=np.zeros(4),
            electrons_cm3=np.zeros(3),
            holes_cm3=np.zeros(4),
        )
        with pytest.raises(ValueError, match="same length"):
            transfer_carriers(bad, mesh_path)

    def test_mesh_without_triangles_is_rejected(self, tmp_path):
        mesh = meshio.Mesh(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cells=[("line", np.array([[0, 1]]))],
        )
        path = tmp_path / "lines.msh"
        meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
        with pytest.raises(ValueError, match="triangle"):
            transfer_carriers(_linear_carriers(), path, at="elements")


def test_runs_without_devsim_or_femwell(monkeypatch, tmp_path):
    for name in ("devsim", "femwell", "skfem"):
        monkeypatch.setitem(sys.modules, name, None)
    mesh_path = _unit_square_mesh(tmp_path)
    got = transfer_carriers(_linear_carriers(), mesh_path)
    assert got.electrons_cm3.size == 4
