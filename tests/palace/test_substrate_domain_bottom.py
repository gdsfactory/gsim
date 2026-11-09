"""The substrate thickness in the stack and the bottom of the Palace domain.

A silicon slab in the PDK stack only reaches the mesh when the substrate is
included. With ``include_substrate=False`` (the default) it is left out, and
``set_airbox(z_below=...)`` is then measured from the oxide bottom: the same
domain comes out whatever the slab thickness. These tests pin both behaviours
and the warnings that make the second one visible.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gsim.common.stack import extract_layer_stack
from gsim.palace import DrivenSim
from gsim.palace.mesh.geometry import GeometryData, resolve_mesh_domain_bounds

BOX = 2.0
Z_BELOW = 20.0


def _level(layer, zmin, thickness, material):
    return SimpleNamespace(
        layer=layer,
        zmin=zmin,
        thickness=thickness,
        material=material,
        info={},
        sidewall_angle=0.0,
    )


def _photonic_stack(si_um: float):
    """Generic SOI-like stack: Si slab, BOX, metal and a drawn cladding layer."""
    return cast(
        Any,
        SimpleNamespace(
            layers={
                "substrate": _level((99, 0), -BOX - si_um, si_um, "si"),
                "box": _level((98, 0), -BOX, BOX, "sio2"),
                "metal1": _level((1, 0), 0.0, 1.0, "aluminum"),
                "clad": _level((97, 0), 0.0, 3.0, "sio2"),
            }
        ),
    )


def _rf_stack(si_um: float):
    """Generic RF stack: no shaped dielectrics, a Si slab under one metal."""
    return cast(
        Any,
        SimpleNamespace(
            layers={
                "substrate": _level((99, 0), -si_um, si_um, "si"),
                "metal1": _level((1, 0), 0.0, 1.0, "aluminum"),
            }
        ),
    )


def _domain_bottom(stack, z_below: float = Z_BELOW) -> float:
    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 100.0, 50.0), layer_bboxes={})
    bounds = resolve_mesh_domain_bounds(
        geometry,
        stack,
        margin_x=0.0,
        margin_y=0.0,
        air_margin=0.0,
        airbox_margin_x=None,
        airbox_margin_y=None,
        airbox_z_above=20.0,
        airbox_z_below=z_below,
    )
    return bounds[2]


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# --- what the stack does today, per mode -----------------------------------


@pytest.mark.parametrize("si_um", [150.0, 300.0])
def test_included_substrate_sets_domain_bottom_photonic(si_um):
    stack = extract_layer_stack(_photonic_stack(si_um), include_substrate=True)

    assert _domain_bottom(stack) == pytest.approx(-BOX - si_um - Z_BELOW)


def test_included_substrate_moves_domain_bottom_with_slab_thickness():
    bottoms = [
        _domain_bottom(extract_layer_stack(_photonic_stack(t), include_substrate=True))
        for t in (150.0, 300.0)
    ]

    assert bottoms[0] - bottoms[1] == pytest.approx(150.0)


@pytest.mark.parametrize("build", [_photonic_stack, _rf_stack])
def test_omitted_substrate_does_not_move_domain_bottom(build):
    bottoms = {
        t: _domain_bottom(extract_layer_stack(build(t), include_substrate=False))
        for t in (150.0, 300.0)
    }

    assert bottoms[150.0] == pytest.approx(bottoms[300.0])


def test_rf_substrate_depth_comes_from_substrate_thickness():
    stack = extract_layer_stack(
        _rf_stack(300.0), include_substrate=True, substrate_thickness=300.0
    )

    assert _domain_bottom(stack) == pytest.approx(-300.0 - Z_BELOW)


# --- the dropped substrate is reported -------------------------------------


@pytest.mark.parametrize("build", [_photonic_stack, _rf_stack])
def test_omitted_substrate_layers_are_listed(build):
    stack = extract_layer_stack(build(300.0), include_substrate=False)

    (layer,) = stack.omitted_substrate_layers
    assert layer["name"] == "substrate"
    assert layer["thickness"] == pytest.approx(300.0)
    assert layer["zmin"] == pytest.approx(layer["zmax"] - 300.0)


def test_nothing_listed_when_substrate_is_included():
    stack = extract_layer_stack(_photonic_stack(300.0), include_substrate=True)

    assert stack.omitted_substrate_layers == []


def test_omitted_substrate_layers_do_not_change_serialized_stack():
    stack = extract_layer_stack(_photonic_stack(300.0), include_substrate=False)

    assert "substrate" not in stack.layers
    assert "omitted_substrate_layers" not in stack.to_dict()
    assert "omitted_substrate_layers" not in stack.to_yaml()


def _sim_with(stack, z_below: float | None):
    sim = DrivenSim()
    sim.set_stack(stack)
    if z_below is not None:
        sim.set_airbox(margin_x=0, margin_y=0, z_above=20, z_below=z_below)
    return sim


def test_airbox_z_below_with_omitted_substrate_warns(caplog):
    stack = extract_layer_stack(_photonic_stack(300.0), include_substrate=False)
    sim = _sim_with(stack, z_below=150.0)

    with caplog.at_level(logging.WARNING, logger="gsim"):
        sim._warn_if_substrate_omitted(stack)

    (message,) = _warnings(caplog)
    assert "'substrate' (300 um" in message
    assert "include_substrate=True" in message
    assert "z_below=150" in message


@pytest.mark.parametrize(
    ("include_substrate", "z_below"),
    [(False, None), (False, 0.0), (True, 150.0)],
    ids=["no-airbox", "zero-z-below", "substrate-included"],
)
def test_no_substrate_warning_when_nothing_is_hidden(
    caplog, include_substrate, z_below
):
    stack = extract_layer_stack(
        _photonic_stack(300.0), include_substrate=include_substrate
    )
    sim = _sim_with(stack, z_below=z_below)

    with caplog.at_level(logging.WARNING, logger="gsim"):
        sim._warn_if_substrate_omitted(stack)

    assert _warnings(caplog) == []


def test_stack_without_substrate_layer_never_warns(caplog):
    gf_stack = cast(
        Any,
        SimpleNamespace(layers={"metal1": _level((1, 0), 0.0, 1.0, "aluminum")}),
    )
    stack = extract_layer_stack(gf_stack, include_substrate=False)
    sim = _sim_with(stack, z_below=150.0)

    with caplog.at_level(logging.WARNING, logger="gsim"):
        sim._warn_if_substrate_omitted(stack)

    assert _warnings(caplog) == []


# --- substrate_thickness: unset keeps the 2 um default, and is told apart ---


def test_unset_substrate_thickness_equals_two_um_default():
    unset = extract_layer_stack(_photonic_stack(300.0), include_substrate=False)
    explicit = extract_layer_stack(
        _photonic_stack(300.0), include_substrate=False, substrate_thickness=2.0
    )

    assert unset.dielectrics == explicit.dielectrics
    assert unset.simulation == explicit.simulation
    assert unset.simulation["substrate_thickness"] == 2.0


def test_set_stack_only_forwards_substrate_thickness_when_given():
    sim = DrivenSim()
    sim.set_stack()
    assert "substrate_thickness" not in sim._stack_kwargs

    sim.set_stack(substrate_thickness=180.0, include_substrate=True)
    assert sim._stack_kwargs["substrate_thickness"] == 180.0
    assert sim._stack_kwargs["include_substrate"] is True


def test_rf_substrate_shorter_than_stack_warns_when_thickness_unset(caplog):
    with caplog.at_level(logging.WARNING, logger="gsim"):
        stack = extract_layer_stack(_rf_stack(300.0), include_substrate=True)

    (message,) = _warnings(caplog)
    assert "substrate_thickness=300" in message
    # The default result is unchanged: a 2 um substrate.
    (substrate,) = [d for d in stack.dielectrics if d["name"] == "substrate"]
    assert (substrate["zmin"], substrate["zmax"]) == (-2.0, 0.0)


def test_rf_substrate_thickness_given_is_used_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="gsim"):
        stack = extract_layer_stack(
            _rf_stack(300.0), include_substrate=True, substrate_thickness=180.0
        )

    assert _warnings(caplog) == []
    (substrate,) = [d for d in stack.dielectrics if d["name"] == "substrate"]
    assert substrate["zmin"] == -180.0


def test_photonic_thickness_shallower_than_stack_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="gsim"):
        stack = extract_layer_stack(
            _photonic_stack(300.0), include_substrate=True, substrate_thickness=100.0
        )

    (message,) = _warnings(caplog)
    assert "shallower" in message
    (substrate,) = [d for d in stack.dielectrics if d["name"] == "substrate"]
    assert substrate["zmin"] == pytest.approx(-BOX - 300.0)


def test_photonic_thickness_unset_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="gsim"):
        extract_layer_stack(_photonic_stack(300.0), include_substrate=True)

    assert _warnings(caplog) == []


# --- end to end: the generated mesh, without running Palace ------------------


def _mesh_zrange(tmp_path, si_um: float, *, include_substrate: bool, z_below: float):
    import gdsfactory as gf
    import meshio

    gf.gpdk.PDK.activate()
    c = gf.Component()
    for y0 in (0, 20):
        c.add_polygon([(0, y0), (100, y0), (100, y0 + 10), (0, y0 + 10)], layer=(1, 0))
    for name, x, orient in (("o1", 0, 180), ("o2", 100, 0)):
        c.add_port(
            name, center=(x, 15), width=10, orientation=orient,
            port_type="electrical", layer=(1, 0),
        )  # fmt: skip

    stack = extract_layer_stack(
        _photonic_stack(si_um), include_substrate=include_substrate
    )
    sim = DrivenSim()
    sim.set_output_dir(tmp_path)
    sim.set_geometry(c)
    sim.set_stack(stack)
    sim.set_airbox(margin_x=20, margin_y=20, z_above=20, z_below=z_below)
    for name in ("o1", "o2"):
        sim.add_cpw_port(name, layer="metal1", s_width=10, gap_width=10, length=5)
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=3)
    sim.mesh(preset="coarse", max_mesh_size=400, refined_mesh_size=50)
    z = meshio.read(tmp_path / "palace.msh").points[:, 2]
    return float(z.min()), float(z.max())


def test_mesh_bottom_follows_slab_only_when_substrate_included(tmp_path):
    pytest.importorskip("gmsh")
    kept = {
        t: _mesh_zrange(tmp_path / f"in{t:g}", t, include_substrate=True, z_below=150.0)
        for t in (150.0, 300.0)
    }
    dropped = {
        t: _mesh_zrange(
            tmp_path / f"out{t:g}", t, include_substrate=False, z_below=150.0
        )
        for t in (150.0, 300.0)
    }

    assert kept[150.0][0] - kept[300.0][0] == pytest.approx(150.0)
    assert dropped[150.0] == pytest.approx(dropped[300.0])
