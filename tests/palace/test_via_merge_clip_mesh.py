"""Meshing with ``clip_merged_vias`` on the synthetic stack (gsim#411).

A 2 x 10 array of 0.4 um cuts at 1.5 um pitch on the generic stack of
``_electrode_layouts`` (``m1`` [0, 1], ``v1`` [1, 2], ``m2`` [2, 3]), meshed
with the default merge distance of 2 um, which merges the 20 cuts into one slab.
The number of via volumes in the mesh result tells what clipping did.
"""

from __future__ import annotations

import logging

import pytest

from gsim.palace.mesh import geometry
from tests._electrode_layouts import make_sim

LOGGER = geometry.logger.name
MESH = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0}
SIDE = 0.4
CELLS = [(i * 1.5, j * 1.5) for i in range(10) for j in range(2)]
CUTS = [("v1", x, y, x + SIDE, y + SIDE) for x, y in CELLS]
FAR = ("m1", 30, 0, 34, 4)  # the second electrode
PORTS = [("PLUS", 0.2, 0.2, "m1")]


def _mesh(tmp_path, boxes, clip: bool):
    """Mesh *boxes* with merging on (2 um) and clipping on or off."""
    sim = make_sim(tmp_path / f"es_{clip}", boxes, PORTS)
    sim.add_terminal("A", net="PLUS")
    sim.add_terminal("B", region=(30, 0, 34, 4), layer="m1")
    result = sim.mesh(**MESH, merge_via_distance=2.0, clip_merged_vias=clip)
    mesh_result = sim._last_mesh_result
    return result, mesh_result


def _via_volumes(mesh_result) -> int:
    return len(mesh_result.groups["volumes"]["v1"]["tags"])


def _via_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Via layer" in r.getMessage()
    ]


def test_a_slab_between_full_bars_meshes_the_same_with_clipping(tmp_path) -> None:
    """Enclosed by m1 and m2 bars: clipping changes nothing, down to the mesh."""
    boxes = [
        ("m1", -0.5, -0.5, 15.0, 2.5),
        *CUTS,
        ("m2", -0.5, -0.5, 15.0, 2.5),
        FAR,
    ]
    _, plain = _mesh(tmp_path, boxes, clip=False)
    _, clipped = _mesh(tmp_path, boxes, clip=True)
    assert _via_volumes(plain) == 1
    assert _via_volumes(clipped) == 1
    assert plain.mesh_stats["tetrahedra"] == clipped.mesh_stats["tetrahedra"]
    assert plain.mesh_stats["nodes"] == clipped.mesh_stats["nodes"]


def test_per_cut_pads_under_a_full_bar_give_one_volume_per_cut(
    tmp_path, caplog
) -> None:
    """Landing pads on m1 only where the cuts are: clipping keeps 20 cuts."""
    pads = [("m1", x - 0.1, y - 0.1, x + 0.5, y + 0.5) for x, y in CELLS]
    boxes = [*pads, *CUTS, ("m2", -0.5, -0.5, 15.0, 2.5), FAR]
    _, plain = _mesh(tmp_path, boxes, clip=False)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _, clipped = _mesh(tmp_path, boxes, clip=True)
    assert _via_volumes(plain) == 1
    assert _via_volumes(clipped) == 20
    assert any("Via clipping of 'v1'" in r.getMessage() for r in caplog.records)
    assert clipped.mesh_stats["tetrahedra"] > plain.mesh_stats["tetrahedra"]


def test_no_metal_above_the_cuts_keeps_them_as_drawn(tmp_path, caplog) -> None:
    """No m2 over the vias: the landing area is empty, so the cuts stay as drawn."""
    boxes = [("m1", -0.5, -0.5, 15.0, 2.5), *CUTS, FAR]
    _, plain = _mesh(tmp_path, boxes, clip=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _, clipped = _mesh(tmp_path, boxes, clip=True)
    assert _via_volumes(plain) == 1
    assert _via_volumes(clipped) == 20
    assert _via_warnings(caplog) == []  # clipped back to the drawn area


def test_clipping_off_still_warns_and_points_to_the_option(tmp_path, caplog) -> None:
    boxes = [("m1", -0.5, -0.5, 15.0, 2.5), *CUTS, FAR]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _mesh(tmp_path, boxes, clip=False)
    (text,) = _via_warnings(caplog)
    assert "factor of 8.25" in text
    assert "Set clip_merged_vias=True" in text


@pytest.mark.parametrize("clip", [False, True])
def test_merging_off_ignores_the_flag(tmp_path, clip) -> None:
    boxes = [("m1", -0.5, -0.5, 15.0, 2.5), *CUTS, ("m2", -0.5, -0.5, 15.0, 2.5), FAR]
    sim = make_sim(tmp_path / "es", boxes, PORTS)
    sim.add_terminal("A", net="PLUS")
    sim.add_terminal("B", region=(30, 0, 34, 4), layer="m1")
    sim.mesh(**MESH, merge_via_distance=0, clip_merged_vias=clip)
    assert _via_volumes(sim._last_mesh_result) == 20
