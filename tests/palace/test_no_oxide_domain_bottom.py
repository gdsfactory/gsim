"""``add_oxide_dielectric=False`` and the bottom of the Palace domain.

The synthetic oxide fill is the only region built from ``substrate_thickness``
when the substrate is omitted. Without it the domain bottom comes from the
PDK's remaining dielectric regions or layer extents, so ``substrate_thickness``
changes nothing. These tests pin that behaviour and the warnings that say so.
Generic data only: the generic gdsfactory PDK and a synthetic SOI-like stack.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gsim.common.stack import extract_layer_stack, get_stack
from gsim.palace import DrivenSim
from gsim.palace.mesh.geometry import GeometryData, resolve_mesh_domain_bounds

BOX = 2.0
SI = 300.0
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


def _photonic_stack():
    """Generic SOI-like stack: Si slab, BOX, metal and a drawn cladding layer."""
    return cast(
        Any,
        SimpleNamespace(
            layers={
                "substrate": _level((99, 0), -BOX - SI, SI, "si"),
                "box": _level((98, 0), -BOX, BOX, "sio2"),
                "metal1": _level((1, 0), 0.0, 1.0, "aluminum"),
                "clad": _level((97, 0), 0.0, 3.0, "sio2"),
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


@pytest.fixture
def generic_pdk():
    import gdsfactory as gf

    gf.gpdk.PDK.activate()


# --- the behaviour the docstrings have to describe ---------------------------


@pytest.mark.usefixtures("generic_pdk")
@pytest.mark.parametrize("t", [2.0, 10.0, 50.0])
def test_oxide_fill_sets_bottom_from_substrate_thickness(t):
    stack = get_stack(
        substrate_thickness=t, include_substrate=False, add_oxide_dielectric=True
    )

    assert _domain_bottom(stack) == pytest.approx(-t - Z_BELOW)


@pytest.mark.usefixtures("generic_pdk")
def test_without_oxide_fill_substrate_thickness_does_not_move_bottom():
    bottoms = [
        _domain_bottom(
            get_stack(
                substrate_thickness=t,
                include_substrate=False,
                add_oxide_dielectric=False,
            )
        )
        for t in (2.0, 10.0, 50.0)
    ]

    assert bottoms[0] == pytest.approx(bottoms[1])
    assert bottoms[0] == pytest.approx(bottoms[2])
    # It comes from the PDK layer extents (generic PDK: BOX bottom at -3 um),
    # not from the -2 um default that the oxide fill would have used.
    assert bottoms[0] == pytest.approx(-3.0 - Z_BELOW)


def test_without_oxide_fill_bottom_comes_from_pdk_box():
    stack = extract_layer_stack(
        _photonic_stack(),
        include_substrate=False,
        add_oxide_dielectric=False,
        substrate_thickness=50.0,
    )

    assert stack.dielectrics == []
    assert _domain_bottom(stack) == pytest.approx(-BOX - Z_BELOW)


def test_default_outputs_unchanged_by_thickness_without_oxide():
    unset = extract_layer_stack(
        _photonic_stack(), include_substrate=False, add_oxide_dielectric=False
    )
    given = extract_layer_stack(
        _photonic_stack(),
        include_substrate=False,
        add_oxide_dielectric=False,
        substrate_thickness=2.0,
    )

    assert unset.dielectrics == given.dielectrics
    assert unset.simulation == given.simulation
    assert unset.to_dict() == given.to_dict()


# --- the ineffective substrate_thickness is reported ------------------------


def test_extractor_warns_when_substrate_thickness_has_no_effect(caplog):
    with caplog.at_level(logging.WARNING, logger="gsim"):
        extract_layer_stack(
            _photonic_stack(),
            include_substrate=False,
            add_oxide_dielectric=False,
            substrate_thickness=50.0,
        )

    (message,) = _warnings(caplog)
    assert "substrate_thickness=50 um has no effect" in message
    assert "add_oxide_dielectric=False" in message
    assert "z_below" in message
    # Names where the bottom really is (the BOX bottom), not z=-50.
    assert "lowest at z=-2 um" in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"add_oxide_dielectric": False},  # thickness not given
        {"add_oxide_dielectric": True, "substrate_thickness": 50.0},
        {
            "add_oxide_dielectric": False,
            "include_substrate": True,
            "substrate_thickness": 50.0,
        },
        {
            "add_oxide_dielectric": False,
            "substrate_thickness": 50.0,
            "air_below": 5.0,
        },
    ],
    ids=["unset", "oxide-fill", "substrate-included", "air-below-uses-it"],
)
def test_extractor_does_not_warn_when_substrate_thickness_is_used(caplog, kwargs):
    with caplog.at_level(logging.WARNING, logger="gsim"):
        extract_layer_stack(_photonic_stack(), **kwargs)

    assert not [m for m in _warnings(caplog) if "has no effect" in m]


@pytest.mark.usefixtures("generic_pdk")
def test_set_stack_path_warns_on_resolution(caplog):
    sim = DrivenSim()
    sim.set_stack(substrate_thickness=10.0, add_oxide_dielectric=False)

    with caplog.at_level(logging.WARNING, logger="gsim"):
        sim._resolve_stack()

    (message,) = [m for m in _warnings(caplog) if "has no effect" in m]
    assert "substrate_thickness=10 um" in message
    assert "lowest at z=-3 um" in message


@pytest.mark.usefixtures("generic_pdk")
def test_set_stack_path_silent_with_default_thickness(caplog):
    sim = DrivenSim()
    sim.set_stack(add_oxide_dielectric=False)

    with caplog.at_level(logging.WARNING, logger="gsim"):
        sim._resolve_stack()

    assert not [m for m in _warnings(caplog) if "has no effect" in m]


# --- the substrate warning no longer advises a no-op ------------------------


def _warn_message(caplog, *, add_oxide: bool) -> str:
    stack = extract_layer_stack(
        _photonic_stack(), include_substrate=False, add_oxide_dielectric=add_oxide
    )
    sim = DrivenSim()
    sim.set_stack(stack)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=20, z_below=150.0)
    with caplog.at_level(logging.WARNING, logger="gsim"):
        sim._warn_if_substrate_omitted(stack)
    (message,) = _warnings(caplog)
    return message


def test_substrate_warning_does_not_advise_thickness_without_oxide(caplog):
    message = _warn_message(caplog, add_oxide=False)

    assert "'substrate' (300 um" in message
    assert "z=-2 um (stack bottom)" in message
    assert "include_substrate=True" in message
    assert "PDK dielectric layers" in message
    assert "substrate_thickness has no effect" in message


def test_substrate_warning_with_oxide_fill_keeps_thickness_advice(caplog):
    message = _warn_message(caplog, add_oxide=True)

    assert "substrate_thickness / z_below" in message
    assert "has no effect" not in message


# --- docstrings say what sets the bottom in each case ------------------------


def test_docstrings_qualify_the_no_oxide_case():
    for doc in (
        DrivenSim.set_stack.__doc__,
        DrivenSim.set_airbox.__doc__,
        get_stack.__doc__,
        extract_layer_stack.__doc__,
    ):
        assert doc is not None
        assert "add_oxide_dielectric=False" in doc


# --- end to end: the generated mesh, without running Palace ------------------


def _mesh_zmin(tmp_path, substrate_thickness: float, *, add_oxide: bool) -> float:
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
        _photonic_stack(),
        include_substrate=False,
        add_oxide_dielectric=add_oxide,
        substrate_thickness=substrate_thickness,
    )
    sim = DrivenSim()
    sim.set_output_dir(tmp_path)
    sim.set_geometry(c)
    sim.set_stack(stack)
    sim.set_airbox(margin_x=20, margin_y=20, z_above=20, z_below=Z_BELOW)
    for name in ("o1", "o2"):
        sim.add_cpw_port(name, layer="metal1", s_width=10, gap_width=10, length=5)
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=3)
    sim.mesh(preset="coarse", max_mesh_size=400, refined_mesh_size=50)
    return float(meshio.read(tmp_path / "palace.msh").points[:, 2].min())


def test_mesh_bottom_ignores_substrate_thickness_without_oxide(tmp_path):
    pytest.importorskip("gmsh")
    shallow = _mesh_zmin(tmp_path / "a", 2.0, add_oxide=False)
    deep = _mesh_zmin(tmp_path / "b", 50.0, add_oxide=False)
    with_oxide = _mesh_zmin(tmp_path / "c", 50.0, add_oxide=True)

    assert shallow == pytest.approx(deep)
    assert shallow == pytest.approx(-BOX - Z_BELOW)
    assert with_oxide == pytest.approx(-50.0 - Z_BELOW)
