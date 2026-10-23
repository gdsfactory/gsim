"""Clipping leaves a real stack's enclosed via slabs as they are (gsim#411).

The CPW via transition of ``test_mesh_regression`` stacks arrays of ``topvia1``
and ``topvia2`` cuts between full metal pads on the IHP stack. Merging joins
them into slabs that lie inside the pads, so ``clip_merged_vias=True`` must
give the same mesh. The snapshot YAML of the regression test is not touched.

Skipped automatically when ``ihp`` is not installed.
"""

from __future__ import annotations

import pytest

from tests.palace.test_mesh_regression import (
    _make_cpw_via_transition_sim,
    _mesh_snapshot,
)

COARSE = {
    "preset": "coarse",
    "margin_x": 0,
    "margin_y": 0,
    "planar_conductors": False,
}


def test_cpw_via_transition_unchanged_by_clipping(tmp_path) -> None:
    pytest.importorskip("ihp")
    plain = _make_cpw_via_transition_sim(tmp_path / "plain")
    plain.mesh(**COARSE)
    clipped = _make_cpw_via_transition_sim(tmp_path / "clipped")
    clipped.mesh(**COARSE, clip_merged_vias=True)

    before = _mesh_snapshot(plain)
    after = _mesh_snapshot(clipped)
    assert after["groups"] == before["groups"]
    assert after["mesh"]["tetrahedra"] == before["mesh"]["tetrahedra"]
    assert after["mesh"]["nodes"] == before["mesh"]["nodes"]
    assert after["mesh"]["invalid_elements"] == 0
