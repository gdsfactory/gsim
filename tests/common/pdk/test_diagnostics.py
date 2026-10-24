"""Tests for the PDK-authoritative stack diagnostics."""

from __future__ import annotations

from collections.abc import Iterator

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel, LogicalLayer
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.pdk import (
    feature_resolution,
    geometry_report,
    material_report,
    resolve_layer_stack,
    stack_report,
)


def _make_pdk() -> gf.Pdk:
    return gf.Pdk(
        name="diagnostics_test",
        layer_stack=GfLayerStack(
            layers={
                "core": LayerLevel(
                    layer=LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(3, 0)),
                    derived_layer=LogicalLayer(layer=(1, 0)),
                    thickness=0.22,
                    zmin=0.0,
                    material="Si",
                    info={"mesh_order": 1},
                    sidewall_angle=10.0,
                ),
                "grating": LayerLevel(
                    layer=LogicalLayer(layer=(1, 0)) & LogicalLayer(layer=(3, 0)),
                    derived_layer=LogicalLayer(layer=(3, 0)),
                    thickness=0.15,
                    zmin=0.0,
                    material="Si",
                    info={"mesh_order": 1},
                ),
                "metal": LayerLevel(
                    layer=(2, 0),
                    thickness=0.2,
                    zmin=1.0,
                    material="Al",
                    info={"mesh_order": 2, "layer_type": "conductor"},
                ),
            }
        ),
    )


def _make_component() -> gf.Component:
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


@pytest.fixture
def _active_pdk() -> Iterator[gf.Pdk]:
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


@pytest.mark.usefixtures("_active_pdk")
def test_stack_report_exposes_authoritative_fields() -> None:
    resolved = resolve_layer_stack(pdk=_make_pdk())
    rows = {row["layer"]: row for row in stack_report(resolved)}

    assert rows["core"]["mesh_order"] == 1
    assert rows["core"]["sidewall_angle_deg"] == 10.0
    assert rows["core"]["material"] == "Si"
    assert rows["metal"]["layer_type"] == "conductor"
    assert rows["core"]["drawn"] is True


@pytest.mark.usefixtures("_active_pdk")
def test_material_report_resolves_optical_and_conductor_properties() -> None:
    resolved = resolve_layer_stack(
        pdk=_make_pdk(),
        dielectrics=[{"name": "box", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}],
    )
    report = material_report(resolved, wavelength_um=1.55)

    assert report["Si"]["found"] is True
    assert report["Si"]["refractive_index"] == pytest.approx(3.4757, abs=1e-3)
    assert report["Si"]["behavior"] == "dielectric"
    assert report["SiO2"]["found"] is True
    assert report["Al"]["behavior"] == "conductive"
    assert report["Al"]["conductivity"] > 1e7


@pytest.mark.usefixtures("_active_pdk")
def test_geometry_report_uses_derived_layer_expression() -> None:
    component = _make_component()
    resolved = resolve_layer_stack(component, pdk=_make_pdk())
    report = geometry_report(component, resolved)

    layers = {row["layer"]: row for row in report["layers"]}
    # core = WG - GRA: 10 um^2 minus the 2 um^2 grating overlap.
    assert layers["core"]["area_um2"] == pytest.approx(8.0)
    assert layers["grating"]["area_um2"] == pytest.approx(2.0)
    assert layers["core"]["polygons"] > 0
    assert report["bbox"] == [0.0, 0.0, 10.0, 1.0]
    assert report["ports"][0]["name"] == "o1"
    assert report["ports"][0]["center"] == [0.0, 0.5]


def test_feature_resolution_flags_under_resolved_features() -> None:
    report = feature_resolution(
        {"gap": 0.1, "width": 0.5, "period": 0.318},
        resolution=20,
        required_cells=4,
    )

    assert report["resolved"] is False
    rows = {row["name"]: row for row in report["features"]}
    assert rows["gap"]["cells"] == pytest.approx(2.0)
    assert rows["gap"]["resolved"] is False
    assert rows["width"]["resolved"] is True
    assert report["suggested_resolution"] == pytest.approx(40.0)


def test_feature_resolution_accepts_adequate_grid() -> None:
    report = feature_resolution({"gap": 0.1}, resolution=50, required_cells=4)

    assert report["resolved"] is True
    assert report["suggested_resolution"] == pytest.approx(40.0)


def test_feature_resolution_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="resolution"):
        feature_resolution({"gap": 0.1}, resolution=0)
    with pytest.raises(ValueError, match="required_cells"):
        feature_resolution({"gap": 0.1}, resolution=20, required_cells=0)
