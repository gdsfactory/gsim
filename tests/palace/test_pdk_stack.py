"""Palace PDK-authoritative stack tests (gsim#230), IHP first."""

from __future__ import annotations

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.pdk import resolve_layer_stack
from gsim.palace import DrivenSim

ihp = pytest.importorskip("ihp")


@pytest.fixture
def _ihp_active():
    """Activate IHP for the test and restore the previous PDK."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    ihp.PDK.activate()
    try:
        yield
    finally:
        if previous is not None:
            previous.activate()


@pytest.fixture
def _no_overlay_pdk_active():
    """Activate a PDK with no reviewed overlay and no bulk dielectric."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    pdk = gf.Pdk(
        name="no_overlay_test",
        layer_stack=GfLayerStack(
            layers={
                "core": LayerLevel(
                    layer=(1, 0), thickness=0.22, zmin=0.0, material="Si"
                )
            }
        ),
    )
    pdk.activate()
    try:
        yield
    finally:
        if previous is not None:
            previous.activate()


def _component_on_metal1() -> gf.Component:
    metal1_gds = resolve_layer_stack().layers["metal1"].gds_layer
    component = gf.Component()
    component.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=metal1_gds)
    return component


def _straight_topmetal2(length: float = 100.0) -> gf.Component:
    component = gf.Component()
    component.add_polygon(
        [(0, 0), (length, 0), (length, 10), (0, 10)], layer=ihp.LAYER.TopMetal2drawing
    )
    return component


@pytest.mark.usefixtures("_ihp_active")
def test_ihp_overlay_is_applied_by_default() -> None:
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True)
    stack = sim._resolve_stack()

    assert stack.simulation["pdk_overlay"] == "IHP"
    assert {d["name"] for d in stack.dielectrics} == {"ild", "passivation"}
    assert stack.layers["via1"].layer_type == "via"
    assert stack.layers["metal1"].mesh_order == 14


@pytest.mark.usefixtures("_ihp_active")
def test_overlay_alone_produces_mesh_volumes(tmp_path) -> None:
    """Notebooks only need the flag: the overlay supplies the BEOL/passivation."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(_straight_topmetal2())
    sim.set_stack(pdk_authoritative=True)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=100.0, z_below=100.0)
    sim.mesh(preset="coarse", margin_x=0, margin_y=0)

    volumes = sim._last_mesh_result.groups["volumes"]
    assert "sio2" in volumes  # implicit BEOL from the overlay
    assert "sin" in volumes  # passivation from the overlay
    assert "air" in volumes


@pytest.mark.usefixtures("_ihp_active")
def test_explicit_input_overrides_the_overlay() -> None:
    sim = DrivenSim()
    sim.set_stack(
        pdk_authoritative=True,
        dielectrics=[{"name": "ild", "material": "sin", "zmin": -1.0, "zmax": 5.0}],
        role_overrides={"via1": "conductor"},
    )
    stack = sim._resolve_stack()

    by_name = {d["name"]: d for d in stack.dielectrics}
    assert by_name["ild"]["material"] == "sin"
    assert by_name["ild"]["zmin"] == -1.0
    assert stack.layers["via1"].layer_type == "conductor"


@pytest.mark.usefixtures("_no_overlay_pdk_active")
def test_missing_bulk_dielectric_and_airbox_raises() -> None:
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True)
    with pytest.raises(ValueError, match="No dielectric volumes"):
        sim._resolve_stack()


@pytest.mark.usefixtures("_ihp_active")
def test_legacy_kwargs_are_rejected_with_the_flag() -> None:
    sim = DrivenSim()
    with pytest.raises(ValueError, match="ignores legacy stack options"):
        sim.set_stack(pdk_authoritative=True, substrate_thickness=5.0)
    with pytest.raises(ValueError, match="ignores legacy stack options"):
        sim.set_stack(pdk_authoritative=True, include_substrate=True)
    with pytest.raises(ValueError, match="cannot be combined with yaml_path"):
        sim.set_stack(pdk_authoritative=True, yaml_path="stack.yaml")


@pytest.mark.usefixtures("_ihp_active")
def test_bare_overlay_matches_legacy_default_regions() -> None:
    """The IHP overlay reproduces the legacy default dielectric regions."""
    from gsim.common.stack import get_stack

    legacy = sorted(
        (d["material"], round(d["zmin"], 3), round(d["zmax"], 3))
        for d in get_stack().dielectrics
    )
    auth = sorted(
        (d.material, round(d.zmin, 3), round(d.zmax, 3))
        for d in resolve_layer_stack().dielectrics
    )
    assert auth == legacy


@pytest.mark.usefixtures("_ihp_active")
def test_substrate_declaration_matches_legacy_regions() -> None:
    """The explicit substrate override reproduces the legacy 180 um regions."""
    from gsim.common.stack import get_stack

    legacy = sorted(
        (d["material"], round(d["zmin"], 3), round(d["zmax"], 3))
        for d in get_stack(
            include_substrate=True, substrate_thickness=180.0
        ).dielectrics
    )

    sim = DrivenSim()
    sim.set_stack(
        pdk_authoritative=True,
        dielectrics=[
            {"name": "substrate", "material": "silicon", "zmin": -180.0, "zmax": 0.0},
            {"name": "ild", "material": "sio2", "zmin": 0.0, "zmax": 14.03},
        ],
    )
    auth = sorted(
        (d["material"], round(d["zmin"], 3), round(d["zmax"], 3))
        for d in sim._resolve_stack().dielectrics
    )
    assert auth == legacy


@pytest.mark.usefixtures("_ihp_active")
def test_legacy_default_is_unchanged() -> None:
    sim = DrivenSim()
    sim.set_stack()
    stack = sim._resolve_stack()

    # The legacy extractor still inserts its synthetic oxide by default; we are
    # not changing the default behaviour in this increment.
    assert "oxide" in [d["name"] for d in stack.dielectrics]
    assert stack.simulation.get("pdk_authoritative") is None


@pytest.mark.usefixtures("_ihp_active")
def test_drawn_extent_uses_component() -> None:
    sim = DrivenSim()
    sim.set_geometry(_component_on_metal1())
    sim.set_airbox(margin_x=0, margin_y=0, z_above=100.0, z_below=100.0)
    sim.set_stack(pdk_authoritative=True)
    stack = sim._resolve_stack()

    # Only Metal1 (0.84..1.26 um) is drawn; the extent metadata reflects that.
    assert stack.simulation["drawn_z_range"] == pytest.approx([0.84, 1.26])
