"""Tests for solver-specific mesh semantics."""

from __future__ import annotations

from types import SimpleNamespace

from gsim.fdtd.mesh import _priority_by_mesh_order


def test_lower_pdk_mesh_order_becomes_higher_zap_priority() -> None:
    layers = {
        "core": SimpleNamespace(mesh_order=1),
        "same_order": SimpleNamespace(mesh_order=1),
        "slab": SimpleNamespace(mesh_order=4),
        "cladding": SimpleNamespace(mesh_order=7),
    }

    priorities = _priority_by_mesh_order(layers)

    assert priorities == {
        "core": 3,
        "same_order": 3,
        "slab": 2,
        "cladding": 1,
    }
