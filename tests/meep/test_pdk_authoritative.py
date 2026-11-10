"""Meep integration tests for the PDK-authoritative stack."""

from __future__ import annotations

import json
import logging
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


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.usefixtures("_auth_pdk_active")
def test_air_spelling_preserves_automatic_mode_domain(air: str) -> None:
    """Canonical gpdk air must be excluded from automatic non-air Z bounds."""
    from gsim.common.stack.extractor import Layer

    sim = _authoritative_sim(gf.get_active_pdk())
    sim.build_config()
    stack = sim.geometry.stack.model_copy(deep=True)
    expected = sim._stack_material_extent(stack)
    stack.layers["air_padding"] = Layer(
        name="air_padding",
        material=air,
        gds_layer=(888, 0),
        zmin=-100,
        zmax=100,
        thickness=200,
        layer_type="dielectric",
    )
    stack.dielectrics.append(
        {"name": "air_box", "material": air, "zmin": -100, "zmax": 100}
    )
    sim.geometry.stack = stack
    sim.mode_solver.port = "o1"
    assert sim._stack_material_extent() == expected
    assert sim._resolve_mode_solver_z_window()[0][0] > -100


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


def _air_only_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "gsim.meep.simulation"
        and record.levelno == logging.WARNING
        and "no bulk dielectric region" in record.getMessage()
    ]


def _simulation(pdk: gf.Pdk, **stack_options) -> Simulation:
    sim = Simulation()
    sim.geometry.component = _make_component()
    sim.set_stack(pdk_authoritative=True, pdk=pdk, **stack_options)
    sim.source.port = "o1"
    sim.source.wavelength = 1.55
    sim.monitors = ["o1", "o2"]
    sim.solver.resolution = 20
    return sim


@pytest.mark.usefixtures("_auth_pdk_active")
def test_air_only_stack_warns_once_and_keeps_config_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sim = _simulation(gf.get_active_pdk())

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        result = sim.build_config()

    assert result.config.dielectrics == []
    messages = _air_only_warnings(caplog)
    assert len(messages) == 1
    assert "FDTD background outside drawn polygons is air" in messages[0]
    assert "levels without polygons" in messages[0]
    assert "dielectrics=[...]" in messages[0]
    assert "explicit air region" in messages[0]


@pytest.mark.usefixtures("_auth_pdk_active")
def test_air_only_warning_is_not_repeated_until_set_stack_rearms_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pdk = gf.get_active_pdk()
    sim = _simulation(pdk)

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        sim.build_config()
        sim.build_config()
        assert len(_air_only_warnings(caplog)) == 1

        sim.set_stack(pdk_authoritative=True, pdk=pdk)
        sim.build_config()
        assert len(_air_only_warnings(caplog)) == 2


@pytest.mark.usefixtures("_auth_pdk_active")
@pytest.mark.parametrize(
    ("region", "expected_material"),
    [
        (
            {"name": "clad", "material": "SiO2", "zmin": -2.0, "zmax": 2.0},
            "SiO2",
        ),
        (
            {"name": "background", "material": "air", "zmin": -2.0, "zmax": 2.0},
            "air",
        ),
    ],
    ids=["sio2", "air"],
)
def test_explicit_region_silences_air_only_warning(
    caplog: pytest.LogCaptureFixture,
    region: dict,
    expected_material: str,
) -> None:
    sim = _simulation(gf.get_active_pdk(), dielectrics=[region])

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        result = sim.build_config()

    assert [entry["material"] for entry in result.config.dielectrics] == [
        expected_material
    ]
    assert _air_only_warnings(caplog) == []
    assert not any("air" in warning for warning in sim.validate_config().warnings)


def test_overlay_dielectrics_silence_air_only_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pdk = _make_pdk()
    pdk.name = "ubcpdk"
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    pdk.activate()
    try:
        sim = _simulation(pdk)
        with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
            result = sim.build_config()
    finally:
        if previous is not None:
            previous.activate()

    assert result.stack.simulation["pdk_overlay"] == "UBC"
    assert [entry["name"] for entry in result.config.dielectrics] == ["box"]
    assert _air_only_warnings(caplog) == []


@pytest.mark.usefixtures("_auth_pdk_active")
def test_legacy_stack_does_not_warn_about_air_only_background(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sim = _authoritative_sim(gf.get_active_pdk())
    sim.set_stack()

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        sim.build_config()

    assert _air_only_warnings(caplog) == []


@pytest.mark.usefixtures("_auth_pdk_active")
def test_prebuilt_authoritative_stack_without_regions_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gsim.common.pdk import build_layer_stack, resolve_layer_stack

    pdk = gf.get_active_pdk()
    prebuilt = build_layer_stack(
        resolve_layer_stack(_make_component(), pdk=pdk), backend="meep"
    )
    assert prebuilt.dielectrics == []

    sim = _authoritative_sim(pdk)
    sim.set_stack(stack=prebuilt)

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        sim.build_config()

    (warning,) = _air_only_warnings(caplog)
    # set_stack(stack=..., dielectrics=...) raises, so do not advise it.
    assert "Pass dielectrics=[...] to set_stack()" not in warning
    assert "stack's dielectrics list" in warning
    assert sim.validate_config().warnings == [warning]


@pytest.mark.usefixtures("_auth_pdk_active")
def test_hand_built_stack_does_not_warn_about_air_only_background(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gsim.common.stack import Layer, LayerStack

    hand_built = LayerStack(
        pdk_name="hand_built",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="Si",
                layer_type="dielectric",
            )
        },
    )
    assert hand_built.dielectrics == []

    sim = _authoritative_sim(gf.get_active_pdk())
    sim.set_stack(stack=hand_built)

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        result = sim.build_config()

    assert result.config.dielectrics == []
    assert _air_only_warnings(caplog) == []
    assert sim.validate_config().warnings == []


@pytest.mark.usefixtures("_auth_pdk_active")
def test_validate_config_reports_air_only_stack_once_resolved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sim = _simulation(gf.get_active_pdk())

    before = sim.validate_config()
    assert before.valid
    assert before.warnings == []

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        sim.build_config()

    assert sim.validate_config().warnings == _air_only_warnings(caplog)


def test_validate_config_without_set_stack_still_reports_no_stack() -> None:
    sim = Simulation()

    assert any("No stack configured" in text for text in sim.validate_config().warnings)


@pytest.mark.usefixtures("_auth_pdk_active")
def test_validate_config_clears_air_only_warning_when_set_stack_adds_a_region() -> None:
    pdk = gf.get_active_pdk()
    sim = _simulation(pdk)
    sim.build_config()
    assert len(sim.validate_config().warnings) == 1

    sim.set_stack(
        pdk_authoritative=True,
        pdk=pdk,
        dielectrics=[{"name": "box", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}],
    )
    assert sim.validate_config().warnings == []
    sim.build_config()
    assert sim.validate_config().warnings == []


@pytest.mark.usefixtures("_auth_pdk_active")
def test_validate_config_keeps_air_only_warning_after_z_crop() -> None:
    sim = _simulation(gf.get_active_pdk())
    sim.domain.z_bounds = (-1.0, 1.0)

    sim.build_config()
    sim.build_config()

    assert len(sim.validate_config().warnings) == 1


@pytest.mark.usefixtures("_auth_pdk_active")
@pytest.mark.parametrize("background", ["air", "SiO2"])
def test_mode_solver_config_does_not_log_the_fdtd_air_only_warning(
    caplog: pytest.LogCaptureFixture,
    tmp_path,
    background: str,
) -> None:
    # The mode solver fills levels without polygons across the cell, so the
    # warning's claim that they add no material would be false here.
    sim = _simulation(gf.get_active_pdk())
    sim.mode_solver.background_material = background
    sim.mode_solver.wavelengths = [1.55]

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        output_dir = sim.write_mode_solver_config(tmp_path / "modes")

    payload = json.loads(
        (output_dir / "mode_solver_config.json").read_text(encoding="utf8")
    )
    assert payload["background_material"] == background
    assert _air_only_warnings(caplog) == []


@pytest.mark.usefixtures("_auth_pdk_active")
@pytest.mark.parametrize("background", ["air", "SiO2"])
def test_build_config_warns_after_a_mode_config_resolved_the_stack(
    caplog: pytest.LogCaptureFixture,
    tmp_path,
    background: str,
) -> None:
    sim = _simulation(gf.get_active_pdk())
    sim.mode_solver.background_material = background
    sim.mode_solver.wavelengths = [1.55]
    sim.write_mode_solver_config(tmp_path / "modes")

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        result = sim.build_config()

    assert result.config.background_material == "air"
    assert result.config.dielectrics == []
    assert len(_air_only_warnings(caplog)) == 1
    assert sim.validate_config().warnings == _air_only_warnings(caplog)


@pytest.mark.usefixtures("_auth_pdk_active")
def test_build_config_warns_when_a_crop_removes_every_region(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sim = _simulation(
        gf.get_active_pdk(),
        dielectrics=[{"name": "top", "material": "SiO2", "zmin": 3.0, "zmax": 4.0}],
    )
    sim.domain.z_bounds = (-1.0, 1.0)

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        result = sim.build_config()

    assert result.config.dielectrics == []
    (warning,) = _air_only_warnings(caplog)
    assert "domain.z_bounds removed all of them" in warning
    assert "Pass dielectrics" not in warning
    assert sim.validate_config().warnings == [warning]


@pytest.mark.usefixtures("_auth_pdk_active")
def test_build_config_logs_air_only_warning_despite_non_air_mode_background(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sim = _simulation(gf.get_active_pdk())
    sim.mode_solver.background_material = "SiO2"

    with caplog.at_level(logging.WARNING, logger="gsim.meep.simulation"):
        result = sim.build_config()

    assert result.config.background_material == "air"
    assert len(_air_only_warnings(caplog)) == 1
