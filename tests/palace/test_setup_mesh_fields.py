"""Unit tests for `_setup_mesh_fields` line-collection logic.

These tests bypass gmsh entirely by stubbing `gmsh_utils` helpers, so the
branch-coverage logic (conductor / port / PEC) is verified on any platform,
including macOS where the real gmsh-driven integration tests are skipped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gsim.palace.mesh import generator as mesh_generator
from gsim.palace.mesh.generator import _setup_mesh_fields


@pytest.fixture
def stub_gmsh_utils(monkeypatch):
    """Stub out the gmsh_utils helpers used by `_setup_mesh_fields`.

    Returns a record dict so tests can inspect the calls made.
    """
    record: dict = {
        "setup_mesh_refinement_calls": [],
        "setup_box_refinement_calls": [],
        "finalize_calls": [],
    }

    def fake_get_boundary_lines(tag, kernel):  # noqa: ARG001
        # Deterministic: each surface tag N contributes lines [N*10, N*10+1]
        return [tag * 10, tag * 10 + 1]

    def fake_setup_mesh_refinement(boundary_lines, refined_cellsize, max_cellsize):
        record["setup_mesh_refinement_calls"].append(
            {
                "boundary_lines": list(boundary_lines),
                "refined_cellsize": refined_cellsize,
                "max_cellsize": max_cellsize,
            }
        )
        return 2  # fake field id, matches the real helper

    def fake_setup_box_refinement(*args, **kwargs):
        record["setup_box_refinement_calls"].append({"args": args, "kwargs": kwargs})

    def fake_finalize_mesh_fields(field_ids):
        record["finalize_calls"].append(list(field_ids))

    monkeypatch.setattr(
        mesh_generator.gmsh_utils, "get_boundary_lines", fake_get_boundary_lines
    )
    monkeypatch.setattr(
        mesh_generator.gmsh_utils, "setup_mesh_refinement", fake_setup_mesh_refinement
    )
    monkeypatch.setattr(
        mesh_generator.gmsh_utils, "setup_box_refinement", fake_setup_box_refinement
    )
    monkeypatch.setattr(
        mesh_generator.gmsh_utils, "finalize_mesh_fields", fake_finalize_mesh_fields
    )

    # `_setup_mesh_fields` also calls a few helpers that query the LIVE gmsh
    # model directly (not via gmsh_utils). In a clean environment they return
    # "nothing" (the empty-model calls raise and are swallowed), but leftover
    # global gmsh state from a prior integration test can make them return real
    # curves — observed as flaky extra PEC lines on Windows CI under
    # pytest-randomly. Neutralize them so these unit tests are truly gmsh-free
    # and order-independent.
    monkeypatch.setattr(
        mesh_generator, "_collect_pec_surface_lines", lambda _groups: []
    )
    monkeypatch.setattr(
        mesh_generator, "_get_domain_bbox", lambda: (0.0, 0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(
        mesh_generator, "_line_on_domain_boundary", lambda *_args, **_kwargs: False
    )
    return record


@pytest.fixture
def empty_stack():
    """Minimal stack with no dielectrics so the box-refinement block is a no-op."""
    return SimpleNamespace(dielectrics=[], materials={})


@pytest.fixture
def geometry():
    """Minimal geometry with the required bbox attribute."""
    return SimpleNamespace(bbox=(0.0, 0.0, 100.0, 100.0))


def _groups(
    conductor_tags: list[list[int]] | None = None,
    port_surfaces: list[dict] | None = None,
    pec_tags: list[list[int]] | None = None,
) -> dict:
    """Build a minimal `groups` dict in the shape expected by `_setup_mesh_fields`."""
    conductor_surfaces = {
        f"c{i}": {"tags": tags} for i, tags in enumerate(conductor_tags or [])
    }
    pec_surfaces = {f"p{i}": {"tags": tags} for i, tags in enumerate(pec_tags or [])}
    port_dict = {f"port{i}": surface for i, surface in enumerate(port_surfaces or [])}
    return {
        "conductor_surfaces": conductor_surfaces,
        "pec_surfaces": pec_surfaces,
        "port_surfaces": port_dict,
    }


class TestSetupMeshFields:
    """Tests for line-collection behaviour of `_setup_mesh_fields`."""

    def test_conductor_lines_always_collected(
        self, stub_gmsh_utils, empty_stack, geometry
    ):
        """Conductor-surface edges are always refined."""
        groups = _groups(conductor_tags=[[1, 2]])

        _setup_mesh_fields(
            kernel=None,
            groups=groups,
            geometry=geometry,
            stack=empty_stack,
            refined_cellsize=1.0,
            max_cellsize=10.0,
        )

        calls = stub_gmsh_utils["setup_mesh_refinement_calls"]
        assert len(calls) == 1
        # tag 1 -> [10, 11], tag 2 -> [20, 21]; sorted-deduped
        assert calls[0]["boundary_lines"] == [10, 11, 20, 21]

    def test_pec_always_refined(self, stub_gmsh_utils, empty_stack, geometry):
        """PEC surface lines are always included in the refinement field."""
        groups = _groups(conductor_tags=[[1]], pec_tags=[[5]])

        _setup_mesh_fields(
            kernel=None,
            groups=groups,
            geometry=geometry,
            stack=empty_stack,
            refined_cellsize=1.0,
            max_cellsize=10.0,
        )

        lines = stub_gmsh_utils["setup_mesh_refinement_calls"][0]["boundary_lines"]
        # Conductor tag 1 -> [10, 11]; PEC tag 5 -> [50, 51]
        assert lines == [10, 11, 50, 51]

    def test_ports_simple_shape(self, stub_gmsh_utils, empty_stack, geometry):
        """Ports with plain `tags` shape are refined."""
        groups = _groups(port_surfaces=[{"tags": [3]}])

        _setup_mesh_fields(
            kernel=None,
            groups=groups,
            geometry=geometry,
            stack=empty_stack,
            refined_cellsize=1.0,
            max_cellsize=10.0,
        )

        lines = stub_gmsh_utils["setup_mesh_refinement_calls"][0]["boundary_lines"]
        assert lines == [30, 31]

    def test_ports_cpw_shape(self, stub_gmsh_utils, empty_stack, geometry):
        """Ports with the `type=cpw` shape (nested `elements[].tags`) are refined."""
        groups = _groups(
            port_surfaces=[{"type": "cpw", "elements": [{"tags": [3]}, {"tags": [4]}]}]
        )

        _setup_mesh_fields(
            kernel=None,
            groups=groups,
            geometry=geometry,
            stack=empty_stack,
            refined_cellsize=1.0,
            max_cellsize=10.0,
        )

        lines = stub_gmsh_utils["setup_mesh_refinement_calls"][0]["boundary_lines"]
        # tag 3 -> [30, 31], tag 4 -> [40, 41]
        assert lines == [30, 31, 40, 41]

    def test_ports_only_no_conductors_regression(
        self, stub_gmsh_utils, empty_stack, geometry
    ):
        """Regression guard: with empty conductor groups, ports are still refined.

        This is the exact shape of the earlier regression where the conductor
        branch silently contributed zero lines. If another category's loop gets
        accidentally dropped, the total-lines count regresses and this fails.
        """
        groups = _groups(conductor_tags=[], port_surfaces=[{"tags": [3]}])

        _setup_mesh_fields(
            kernel=None,
            groups=groups,
            geometry=geometry,
            stack=empty_stack,
            refined_cellsize=1.0,
            max_cellsize=10.0,
        )

        calls = stub_gmsh_utils["setup_mesh_refinement_calls"]
        assert len(calls) == 1, "setup_mesh_refinement must still be called"
        assert calls[0]["boundary_lines"] == [30, 31]

    def test_info_log_reports_per_category_counts(
        self,
        stub_gmsh_utils,  # noqa: ARG002 — fixture activates gmsh_utils monkeypatches
        empty_stack,
        geometry,
        caplog,
    ):
        """The INFO log reports per-category counts for regression visibility."""
        groups = _groups(
            conductor_tags=[[1]],
            port_surfaces=[{"tags": [3]}],
            pec_tags=[[5]],
        )

        with caplog.at_level("INFO", logger=mesh_generator.logger.name):
            _setup_mesh_fields(
                kernel=None,
                groups=groups,
                geometry=geometry,
                stack=empty_stack,
                refined_cellsize=1.0,
                max_cellsize=10.0,
            )

        matching = [r for r in caplog.records if "Mesh refinement" in r.message]
        assert len(matching) == 1
        msg = matching[0].getMessage()
        # 1 conductor surface + 1 port + 1 pec, each contributing 2 lines
        assert "conductor=2" in msg
        assert "port=2" in msg
        assert "pec=2" in msg
        assert "6 boundary lines" in msg
