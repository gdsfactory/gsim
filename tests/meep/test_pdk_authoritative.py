"""Meep integration tests for the PDK-authoritative stack."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator

import gdsfactory as gf
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LogicalLayer
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.meep import Simulation


def _make_pdk() -> gf.Pdk:
    return gf.Pdk(
        name="meep_auth_test",
        layers=gf.gpdk.PDK.layers,
        cross_sections=gf.gpdk.PDK.cross_sections,
        layer_stack=GfLayerStack(
            layers={
                "core": LayerLevel(
                    layer=(1, 0),
                    thickness=0.22,
                    zmin=0.0,
                    material="Si",
                    info={"mesh_order": 1},
                    sidewall_angle=10.0,
                ),
                "metal": LayerLevel(
                    layer=(2, 0),
                    thickness=0.2,
                    zmin=0.3,
                    material="Al",
                    info={"mesh_order": 2, "layer_type": "conductor"},
                ),
                "box": LayerLevel(
                    layer=(999, 0),
                    thickness=2.0,
                    zmin=-2.0,
                    material="SiO2",
                    info={"mesh_order": 99},
                ),
            }
        ),
    )


def _make_component() -> gf.Component:
    component = gf.Component()
    component.add_polygon(
        [(0.0, -0.25), (5.0, -0.25), (5.0, 0.25), (0.0, 0.25)],
        layer=(1, 0),
    )
    component.add_port(
        name="o1", center=(0.0, 0.0), orientation=180, width=0.5, layer=(1, 0)
    )
    component.add_port(
        name="o2", center=(5.0, 0.0), orientation=0, width=0.5, layer=(1, 0)
    )
    return component


@pytest.fixture
def _auth_pdk_active() -> Iterator[gf.Pdk]:
    pdk = _make_pdk()
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    pdk.activate()
    try:
        yield pdk
    finally:
        if previous is not None:
            previous.activate()


def _authoritative_sim(pdk: gf.Pdk) -> Simulation:
    sim = Simulation()
    sim.geometry.component = _make_component()
    sim.set_stack(
        pdk_authoritative=True,
        pdk=pdk,
        dielectrics=[{"name": "box", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}],
    )
    sim.source.port = "o1"
    sim.source.wavelength = 1.55
    sim.monitors = ["o1", "o2"]
    sim.solver.resolution = 20
    return sim


def _derived_pdk() -> gf.Pdk:
    return gf.Pdk(
        name="meep_derived_test",
        layers=gf.gpdk.PDK.layers,
        cross_sections=gf.gpdk.PDK.cross_sections,
        layer_stack=GfLayerStack(
            layers={
                "core": LayerLevel(
                    layer=LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(3, 0)),
                    derived_layer=LogicalLayer(layer=(1, 0)),
                    thickness=0.22,
                    zmin=0.0,
                    material="Si",
                    info={"mesh_order": 1},
                ),
                "grating": LayerLevel(
                    layer=LogicalLayer(layer=(1, 0)) & LogicalLayer(layer=(3, 0)),
                    derived_layer=LogicalLayer(layer=(3, 0)),
                    thickness=0.15,
                    zmin=0.0,
                    material="Si",
                    info={"mesh_order": 1},
                ),
                "box": LayerLevel(
                    layer=(999, 0),
                    thickness=2.0,
                    zmin=-2.0,
                    material="SiO2",
                    info={"mesh_order": 99},
                ),
            }
        ),
    )


def _derived_component() -> gf.Component:
    component = gf.Component()
    component.add_polygon(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)], layer=(1, 0)
    )
    component.add_polygon(
        [(4.0, 0.0), (6.0, 0.0), (6.0, 1.0), (4.0, 1.0)], layer=(3, 0)
    )
    component.add_port(
        name="o1", center=(0.0, 0.5), orientation=180, width=0.5, layer=(1, 0)
    )
    return component


def _polygon_area(polygons) -> float:
    total = 0.0
    for polygon in polygons:
        if hasattr(polygon, "area"):
            total += float(polygon.area()) / 1e6
            continue
        points = np.asarray(polygon)
        x = points[:, 0]
        y = points[:, 1]
        total += 0.5 * abs(float((x * np.roll(y, -1) - y * np.roll(x, -1)).sum()))
    return total


@pytest.mark.usefixtures("_auth_pdk_active")
def test_build_config_uses_authoritative_stack_and_materials() -> None:
    pdk = gf.get_active_pdk()
    result = _authoritative_sim(pdk).build_config()

    assert result.stack.simulation["pdk_authoritative"] is True
    assert result.stack.simulation["backend"] == "meep"
    assert result.stack.simulation["pdk_overlay"] is None
    assert result.stack.simulation["stack_hash"]

    entries = {entry.layer_name: entry for entry in result.config.layer_stack}
    assert set(entries) == {"core", "metal", "box"}
    assert entries["core"].mesh_order == 1
    assert entries["metal"].mesh_order == 2
    assert entries["core"].sidewall_angle == 10.0

    # No legacy synthetic oxide, and the explicit PDK dielectric is preserved.
    dielectric_names = {entry["name"] for entry in result.config.dielectrics}
    assert "oxide" not in dielectric_names
    assert dielectric_names == {"box"}

    # Silicon is evaluated from the material database at 1.55 um, not replaced
    # by a placeholder or effective index.
    si_eps = result.config.materials["Si"].epsilon_diag
    assert si_eps is not None
    assert math.sqrt(si_eps[0]) == pytest.approx(3.4757, abs=1e-3)


@pytest.mark.usefixtures("_auth_pdk_active")
def test_sim_config_records_pdk_provenance(tmp_path) -> None:
    pdk = gf.get_active_pdk()
    sim = _authoritative_sim(pdk)
    config_dir = sim.write_config(tmp_path / "sim")
    payload = json.loads((config_dir / "sim_config.json").read_text(encoding="utf8"))

    assert payload["pdk_authoritative"] is True
    assert payload["pdk_name"] == "meep_auth_test"
    assert payload["pdk_stack_hash"]
    assert payload["pdk_provenance"]["layers"]["core"]["mesh_order"] == 1
    assert payload["material_provenance"]["Si"]["epsilon_diag"] is not None
    assert payload["material_provenance"]["SiO2"]["epsilon_diag"] is not None


@pytest.mark.usefixtures("_auth_pdk_active")
def test_authoritative_mode_rejects_legacy_options() -> None:
    sim = Simulation()
    with pytest.raises(ValueError, match="cannot be combined with yaml_path"):
        sim.set_stack(pdk_authoritative=True, yaml_path="stack.yaml")
    with pytest.raises(ValueError, match="ignores legacy stack options"):
        sim.set_stack(pdk_authoritative=True, substrate_thickness=5.0)
    with pytest.raises(ValueError, match="require pdk_authoritative"):
        sim.set_stack(
            dielectrics=[{"name": "box", "material": "SiO2", "zmin": -2, "zmax": 0}]
        )


@pytest.mark.usefixtures("_auth_pdk_active")
def test_legacy_default_path_is_unchanged() -> None:
    sim = Simulation()
    sim.geometry.component = _make_component()
    sim.set_stack()
    stack = sim._resolve_stack_and_materials(wavelength=1.55)[0]

    assert "pdk_authoritative" not in stack.simulation
    assert "oxide" in {dielectric["name"] for dielectric in stack.dielectrics}


def test_meep_materializes_derived_pdk_layers() -> None:
    from gsim.common.pdk import build_layer_stack, resolve_layer_stack
    from gsim.meep.physical_layers import materialize_physical_layers

    pdk = _derived_pdk()
    component = _derived_component()
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    pdk.activate()
    try:
        resolved = resolve_layer_stack(
            component,
            pdk=pdk,
            dielectrics=[
                {"name": "box", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}
            ],
        )
        stack = build_layer_stack(resolved, backend="meep")
        materialized = materialize_physical_layers(component, stack)
    finally:
        if previous is not None:
            previous.activate()

    polygons = materialized.component.get_polygons(by="tuple")
    core_tuple = tuple(materialized.stack.layers["core"].gds_layer)
    grating_tuple = tuple(materialized.stack.layers["grating"].gds_layer)
    assert core_tuple != grating_tuple
    assert _polygon_area(polygons[core_tuple]) == pytest.approx(8.0)
    assert _polygon_area(polygons[grating_tuple]) == pytest.approx(2.0)
