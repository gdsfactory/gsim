"""Integration checks for the two primary validation PDKs.

UBC and Cornerstone are optional test dependencies.  The tests skip cleanly when
the packages (or a compatible version of them) are unavailable.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

import gdsfactory as gf
import pytest

from gsim.common.pdk import (
    feature_resolution,
    geometry_report,
    material_report,
    resolve_layer_stack,
)

ubcpdk = pytest.importorskip("ubcpdk")


def _load_cspdk() -> Any:
    """Return an importable Cornerstone Si220 PDK, or skip."""
    candidates = ("cspdk.si220", "cspdk.si220.cband")
    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # pragma: no cover - optional dependency
            errors.append(f"{module_name}: {error}")
            continue
        pdk = getattr(module, "PDK", None)
        if pdk is not None:
            return pdk
    pytest.skip("Cornerstone cspdk.si220 PDK unavailable: " + "; ".join(errors))


@pytest.fixture
def _ubc_active() -> Iterator[Any]:
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.clear_cache()
    ubcpdk.PDK.activate()
    try:
        yield ubcpdk
    finally:
        if previous is not None:
            previous.activate()


@pytest.fixture
def _cspdk_active() -> Iterator[Any]:
    pdk = _load_cspdk()
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.clear_cache()
    pdk.activate()
    try:
        yield pdk
    finally:
        if previous is not None:
            previous.activate()


@pytest.mark.usefixtures("_ubc_active")
def test_ubc_stack_is_resolved_from_the_pdk() -> None:
    resolved = resolve_layer_stack()

    assert resolved.pdk_name == "ubcpdk"
    assert resolved.overlay == "UBC"
    assert set(resolved.layers) == {
        "substrate",
        "box",
        "clad",
        "core",
        "core2",
        "heater",
        "metal2",
    }

    assert (resolved.layers["box"].zmin, resolved.layers["box"].zmax) == (-3.0, 0.0)
    assert (
        resolved.layers["clad"].zmin,
        resolved.layers["clad"].zmax,
    ) == pytest.approx((0.0, 1.8))
    assert (resolved.layers["core"].zmin, resolved.layers["core"].zmax) == (
        0.0,
        0.22,
    )
    assert resolved.layers["core"].mesh_order == 1
    assert resolved.layers["core"].sidewall_angle == pytest.approx(10.0)
    assert resolved.layers["substrate"].layer_type == "substrate"
    assert resolved.layers["heater"].layer_type == "conductor"
    assert resolved.layers["metal2"].layer_type == "conductor"

    dielectrics = {d.name: d for d in resolved.dielectrics}
    assert set(dielectrics) == {"substrate", "box", "clad"}
    assert dielectrics["box"].material == "sio2"
    assert (dielectrics["box"].zmin, dielectrics["box"].zmax) == (-3.0, 0.0)


@pytest.mark.usefixtures("_ubc_active")
def test_ubc_materials_and_drawn_geometry_resolve() -> None:
    component = ubcpdk.cells.straight(length=10.0)
    resolved = resolve_layer_stack(component)
    materials = material_report(resolved, wavelength_um=1.55)
    geometry = geometry_report(component, resolved, wavelength_um=1.55)

    assert {"si", "sio2", "TiN", "Aluminum"} <= set(materials)
    assert materials["si"]["refractive_index"] == pytest.approx(3.4757, abs=1e-3)
    assert materials["sio2"]["refractive_index"] == pytest.approx(1.4440, abs=1e-3)
    assert materials["TiN"]["behavior"] == "conductive"
    assert materials["Aluminum"]["behavior"] == "conductive"

    layers = {row["layer"]: row for row in geometry["layers"]}
    assert layers["core"]["drawn"] is True
    assert layers["core"]["area_um2"] > 0
    assert layers["core2"]["drawn"] is False
    assert resolved.drawn_z_range == pytest.approx((0.0, 0.22))


@pytest.mark.usefixtures("_cspdk_active")
def test_cornerstone_stack_uses_derived_layers_and_overlay() -> None:
    resolved = resolve_layer_stack()

    assert resolved.pdk_name.startswith("cspdk.si220")
    assert resolved.overlay == "CSPDK"
    assert set(resolved.layers) == {"core", "grating", "slab", "heater", "metal"}
    assert resolved.layers["core"].zmin == 0.0
    assert resolved.layers["core"].zmax == pytest.approx(0.22)
    assert resolved.layers["core"].mesh_order == 1
    assert resolved.layers["heater"].layer_type == "conductor"
    assert resolved.layers["metal"].layer_type == "conductor"

    dielectrics = {d.name: d for d in resolved.dielectrics}
    assert set(dielectrics) == {"box", "clad"}
    assert (dielectrics["box"].zmin, dielectrics["box"].zmax) == (-2.0, 0.0)
    assert (dielectrics["clad"].zmin, dielectrics["clad"].zmax) == (0.0, 2.0)

    materials = {
        name.lower(): properties
        for name, properties in material_report(resolved, wavelength_um=1.55).items()
    }
    assert materials["si"]["refractive_index"] == pytest.approx(3.4757, abs=1e-3)
    assert materials["sio2"]["refractive_index"] == pytest.approx(1.4440, abs=1e-3)
    assert materials["tin"]["behavior"] == "conductive"
    assert materials["aluminum"]["behavior"] == "conductive"


@pytest.mark.usefixtures("_cspdk_active")
def test_cornerstone_derived_core_geometry_and_port_gap_resolution() -> None:
    module = (
        importlib.import_module("cspdk.si220.cband")
        if "cband" in (gf.get_active_pdk().name)
        else importlib.import_module("cspdk.si220")
    )
    # Use a distinct length so the process-wide gdsfactory cell cache cannot
    # return a same-named cell from another PDK (UBC also defines "straight").
    component = module.cells.straight(length=13.0)
    resolved = resolve_layer_stack(component)

    geometry = geometry_report(component, resolved, wavelength_um=1.55)
    layers = {row["layer"]: row for row in geometry["layers"]}
    assert layers["core"]["drawn"] is True
    assert layers["core"]["area_um2"] > 0
    assert layers["grating"]["drawn"] is False

    # 220 nm Si, 450 nm access waveguides, 100 nm minimum gap: retain the
    # documented four-cell rule so a validation preset cannot silently
    # under-resolve the coupler.
    report = feature_resolution(
        {"core_thickness": 0.22, "waveguide_width": 0.45, "minimum_gap": 0.10},
        resolution=30,
        required_cells=4,
    )
    assert report["resolved"] is False
    assert report["suggested_resolution"] == pytest.approx(40.0)
