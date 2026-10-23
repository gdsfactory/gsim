"""``clip_merged_vias`` through the net check, the config and the sim classes.

The pure parts (net check, config) need no mesh; the plumbing tests replace
``generate_mesh`` with a function that records its arguments (gsim#411).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gsim.palace import DrivenSim, ElectrostaticSim
from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.mesh.net_groups import check_via_merging
from gsim.palace.mesh.nets import extract_nets
from gsim.palace.models import MeshConfig
from tests._electrode_layouts import component, stack

# ----- check_via_merging -------------------------------------------------------------


def _check(boxes, distance: float, **kwargs) -> None:
    c = component(boxes, ports=[])
    st = stack()
    geometry = extract_geometry(c, st)
    nets = extract_nets(geometry, st, c.ports)
    check_via_merging(geometry, st, nets, distance, **kwargs)


# electrodes 1 um apart on m1; each has a via at its edge, the vias 1 um apart
TWO_NETS = [
    ("m1", 0, 0, 4, 4),
    ("m1", 5, 0, 9, 4),
    ("v1", 3.5, 1, 4, 2),
    ("v1", 5, 1, 5.5, 2),
]
# the same, with a separate m2 bar over each electrode
TWO_NETS_WITH_M2 = [*TWO_NETS, ("m2", 0, 0, 4, 4), ("m2", 5, 0, 9, 4)]


def test_no_metal_above_the_vias_clipping_keeps_them_apart() -> None:
    """TWO_NETS has no m2: the cuts stay as drawn, so nothing is shorted."""
    with pytest.raises(ValueError, match=r"Via layer 'v1'.*different nets"):
        _check(TWO_NETS, 2.0)
    _check(TWO_NETS, 2.0, clip_merged_vias=True)


def test_separate_bars_per_net_clipping_keeps_the_vias_apart() -> None:
    """Without clipping the slab spans both nets; clipped to the m1/m2 overlap
    it falls apart into the two cuts."""
    with pytest.raises(ValueError, match=r"Via layer 'v1'.*different nets") as error:
        _check(TWO_NETS_WITH_M2, 2.0)
    assert "'net1'" in str(error.value)
    assert "'net2'" in str(error.value)
    _check(TWO_NETS_WITH_M2, 2.0, clip_merged_vias=True)


def test_clipping_does_not_hide_a_short_inside_one_bar() -> None:
    """Both vias land on one m1 bar and one m2 bar of a single net: no error, and
    a bar that joins the two nets is still caught."""
    one_net = [
        ("m1", 0, 0, 9, 4),
        ("m2", 0, 0, 9, 4),
        ("v1", 3.5, 1, 4, 2),
        ("v1", 5, 1, 5.5, 2),
    ]
    _check(one_net, 2.0, clip_merged_vias=True)


def test_the_flag_defaults_to_off_for_the_check() -> None:
    with pytest.raises(ValueError, match="different nets"):
        _check(TWO_NETS_WITH_M2, 2.0, clip_merged_vias=False)


# ----- MeshConfig --------------------------------------------------------------------


def test_clip_merged_vias_is_off_by_default() -> None:
    assert MeshConfig().clip_merged_vias is False
    assert MeshConfig.default().clip_merged_vias is False
    assert MeshConfig.coarse().clip_merged_vias is False
    assert MeshConfig.fine().clip_merged_vias is False
    assert MeshConfig.default(clip_merged_vias=True).clip_merged_vias is True


def test_clip_merged_vias_must_be_a_bool() -> None:
    with pytest.raises(ValidationError):
        MeshConfig(clip_merged_vias="maybe")  # type: ignore[arg-type]
    config = MeshConfig()
    with pytest.raises(ValidationError):
        config.clip_merged_vias = "maybe"  # ty: ignore[invalid-assignment]


def test_the_config_documents_both_via_options() -> None:
    doc = MeshConfig.__doc__ or ""
    assert "merge_via_distance:" in doc
    assert "clip_merged_vias:" in doc
    assert "5%" in doc


# ----- the sim classes ---------------------------------------------------------------


def _fake_generate_mesh(captured, tmp_path):
    def _fake(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            mesh_path=tmp_path / "palace.msh",
            config_path=None,
            port_info=[],
            mesh_stats={},
            metadata={},
            groups={},
        )

    return _fake


def test_clip_merged_vias_reaches_generate_mesh(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gsim.palace.mesh.generator.generate_mesh",
        _fake_generate_mesh(captured, tmp_path),
    )
    monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())

    sim = DrivenSim()
    sim.set_output_dir(tmp_path / "sim")
    sim._generate_mesh_internal(
        output_dir=tmp_path / "sim",
        mesh_config=MeshConfig.default(clip_merged_vias=True, merge_via_distance=1.5),
        ports=[],
        driven_config=sim.driven,
        model_name="palace",
        verbose=False,
        write_config=False,
    )
    assert captured["clip_merged_vias"] is True
    assert captured["merge_via_distance"] == 1.5


def test_clip_merged_vias_is_off_in_generate_mesh_by_default(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gsim.palace.mesh.generator.generate_mesh",
        _fake_generate_mesh(captured, tmp_path),
    )
    monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())

    sim = DrivenSim()
    sim.set_output_dir(tmp_path / "sim")
    sim._generate_mesh_internal(
        output_dir=tmp_path / "sim",
        mesh_config=MeshConfig.default(),
        ports=[],
        driven_config=sim.driven,
        model_name="palace",
        verbose=False,
        write_config=False,
    )
    assert captured["clip_merged_vias"] is False


@pytest.mark.parametrize("sim_class", [DrivenSim, ElectrostaticSim])
def test_the_mesh_kwarg_and_the_sim_config_set_the_flag(sim_class) -> None:
    sim = sim_class()
    kwargs = {
        "preset": None,
        "refined_mesh_size": None,
        "max_mesh_size": None,
        "margin": None,
        "airbox_margin": None,
        "fmax": None,
        "planar_conductors": None,
        "show_gui": False,
        "cells_per_feature": 0,
    }
    assert sim._build_mesh_config(**kwargs).clip_merged_vias is False
    sim.mesh_config = MeshConfig.default(clip_merged_vias=True)
    assert sim._build_mesh_config(**kwargs).clip_merged_vias is True
