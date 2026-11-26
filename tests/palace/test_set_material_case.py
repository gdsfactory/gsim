"""``set_material`` matches stack materials ignoring case (gsim#526).

The stack of the generic PDK spells its metal ``Aluminum`` and its oxide
``sio2``. An override spelled ``aluminum`` or ``SiO2`` has to reach the
generated config like the exact spelling does, an unknown name has to be
reported, and of two overrides that differ only in case the last one wins.
Every assertion reads the written ``config.json`` or the log.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import gdsfactory as gf
import pytest

from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

BASE_LOGGER = "gsim.palace.base"

Overrides = list[tuple[str, dict[str, float]]]


@pytest.fixture(autouse=True)
def _gpdk_active():
    """Activate the generic PDK for the test and restore the previous PDK."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.gpdk.PDK.activate()
    try:
        yield
    finally:
        if previous is not None:
            previous.activate()


def _component() -> gf.Component:
    """Metal 1 and metal 2 joined by a via, one electrical port on each metal."""
    layer = gf.gpdk.LAYER
    component = gf.Component()
    component << gf.c.rectangle((60, 10), centered=True, layer=layer.M1)
    component << gf.c.rectangle((8, 8), centered=True, layer=layer.VIA1)
    component << gf.c.rectangle((60, 10), centered=True, layer=layer.M2)
    component.add_port(
        name="o1",
        center=(-30, 0),
        width=10,
        orientation=180,
        port_type="electrical",
        layer=layer.M1,
    )
    component.add_port(
        name="o2",
        center=(30, 0),
        width=10,
        orientation=0,
        port_type="electrical",
        layer=layer.M2,
    )
    return component


def _config(
    tmp_path, overrides: Overrides | None = None, *, prebuilt: bool = False
) -> dict[str, Any]:
    """The ``config.json`` of a coarse driven sim with the given overrides.

    ``set_material`` is called before ``set_stack`` on purpose: the stack does
    not exist yet when the overrides are recorded.
    """
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(_component())
    for name, properties in overrides or []:
        sim.set_material(name, **properties)
    if prebuilt:
        sim.set_stack(get_stack(substrate_thickness=2.0))
    else:
        sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=10, margin_y=10, z_above=20, z_below=10)
    sim.add_port("o1", layer="metal1", length=2.0)
    sim.add_port("o2", layer="metal2", length=2.0)
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=3)
    sim.mesh(preset="coarse", verbose=False)
    return json.loads(sim.write_config().read_text())


def _conductivities(config: dict[str, Any]) -> list[float]:
    return [entry["Conductivity"] for entry in config["Boundaries"]["Conductivity"]]


def _via_conductivities(config: dict[str, Any]) -> list[float]:
    """The conductivity of the conducting volume domains (the via)."""
    return [
        entry["Conductivity"]
        for entry in config["Domains"]["Materials"]
        if "Conductivity" in entry
    ]


def _sigma(value: float) -> dict[str, float]:
    return {"conductivity": value}


# ---------------------------------------------------------------------------
# Spelling of the override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["aluminum", "ALUMINUM"])
def test_conductor_boundaries_ignore_case(tmp_path, spelling):
    """Every conductor boundary carries the override, as for the exact name."""
    config = _config(tmp_path / "case", [(spelling, _sigma(1e7))])
    exact = _config(tmp_path / "exact", [("Aluminum", _sigma(1e7))])

    conductivities = _conductivities(config)
    assert conductivities
    assert set(conductivities) == {1e7}
    assert config["Boundaries"] == exact["Boundaries"]


@pytest.mark.parametrize("spelling", ["aluminum", "ALUMINUM"])
def test_via_domain_ignores_case(tmp_path, spelling):
    """The via volume takes the override too, and the domains agree."""
    config = _config(tmp_path / "case", [(spelling, _sigma(1e7))])
    exact = _config(tmp_path / "exact", [("Aluminum", _sigma(1e7))])

    assert _via_conductivities(config) == [1e7]
    assert config["Domains"]["Materials"] == exact["Domains"]["Materials"]


def test_dielectric_override_ignores_case(tmp_path):
    """``SiO2`` overrides the ``sio2`` domain like ``sio2`` does."""
    config = _config(tmp_path / "case", [("SiO2", {"permittivity": 3.0})])
    exact = _config(tmp_path / "exact", [("sio2", {"permittivity": 3.0})])

    permittivities = [m["Permittivity"] for m in config["Domains"]["Materials"]]
    assert 3.0 in permittivities
    assert config["Domains"]["Materials"] == exact["Domains"]["Materials"]


def test_prebuilt_stack_ignores_case(tmp_path):
    """A stack given to ``set_stack`` is overridden under its own spelling."""
    config = _config(tmp_path, [("aluminum", _sigma(1e7))], prebuilt=True)

    assert set(_conductivities(config)) == {1e7}
    assert _via_conductivities(config) == [1e7]


# ---------------------------------------------------------------------------
# Two spellings of one name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ([("aluminum", _sigma(1e7)), ("ALUMINUM", _sigma(2e7))], 2e7),
        ([("ALUMINUM", _sigma(2e7)), ("aluminum", _sigma(1e7))], 1e7),
    ],
    ids=["upper-last", "lower-last"],
)
def test_last_case_variant_wins(tmp_path, overrides, expected):
    """Two overrides that differ only in case: the later call is the override."""
    config = _config(tmp_path, overrides)

    assert set(_conductivities(config)) == {expected}
    assert _via_conductivities(config) == [expected]


# ---------------------------------------------------------------------------
# The warning for a name that matches nothing
# ---------------------------------------------------------------------------


def test_unknown_name_warns_once_and_changes_nothing(tmp_path, caplog):
    """An unknown name is reported once, with the stack's materials."""
    reference = _config(tmp_path / "reference")

    with caplog.at_level(logging.WARNING, logger=BASE_LOGGER):
        config = _config(tmp_path / "unknown", [("unobtainium", _sigma(1e7))])

    records = [
        r
        for r in caplog.records
        if r.name == BASE_LOGGER and "unobtainium" in r.getMessage()
    ]
    assert len(records) == 1
    assert "Aluminum" in records[0].getMessage()
    assert config["Boundaries"] == reference["Boundaries"]
    assert config["Domains"] == reference["Domains"]


@pytest.mark.parametrize("spelling", ["Aluminum", "aluminum"])
def test_known_name_does_not_warn(tmp_path, caplog, spelling):
    """A name that matches a stack material, in any case, is not reported."""
    with caplog.at_level(logging.WARNING, logger=BASE_LOGGER):
        _config(tmp_path, [(spelling, _sigma(1e7))])

    assert not [r for r in caplog.records if "set_material(" in r.getMessage()]


# ---------------------------------------------------------------------------
# Exact spellings keep their values
# ---------------------------------------------------------------------------


def test_exact_names_are_unchanged(tmp_path):
    """Names spelled as the stack spells them give the values of before."""
    default = _config(tmp_path / "default")
    assert set(_conductivities(default)) == {3.77e7}
    assert _via_conductivities(default) == [3.77e7]
    assert 4.1 in [m["Permittivity"] for m in default["Domains"]["Materials"]]

    config = _config(
        tmp_path / "exact",
        [("Aluminum", _sigma(1e7)), ("sio2", {"permittivity": 3.0})],
    )
    assert set(_conductivities(config)) == {1e7}
    assert _via_conductivities(config) == [1e7]
    assert 3.0 in [m["Permittivity"] for m in config["Domains"]["Materials"]]
