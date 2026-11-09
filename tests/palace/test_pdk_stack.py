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
def test_palace_can_disable_pdk_overlay() -> None:
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True, use_pdk_overlay=False)
    sim.set_airbox(margin_x=1, margin_y=1, z_above=1, z_below=1)
    stack = sim._resolve_stack()
    assert stack.simulation.get("pdk_overlay") is None
    assert stack.dielectrics == []

    with pytest.raises(ValueError, match="requires pdk_authoritative"):
        sim.set_stack(use_pdk_overlay=False)


@pytest.mark.usefixtures("_ihp_active")
def test_canonical_material_override_reaches_alias_tokens() -> None:
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True)
    sim.set_material("aluminum", conductivity=2.5e7)
    stack = sim._resolve_stack()
    metal_token = stack.layers["metal1"].material
    assert stack.materials[metal_token]["conductivity"] == 2.5e7


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


@pytest.mark.usefixtures("_no_overlay_pdk_active")
def test_failed_resolution_preserves_authoritative_options() -> None:
    """Following the error's advice must not fall back to the legacy stack."""
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True)
    with pytest.raises(ValueError, match="No dielectric volumes"):
        sim._resolve_stack()

    assert sim._stack_kwargs["pdk_authoritative"] is True
    sim.set_airbox(margin_x=0, margin_y=0, z_above=1.0, z_below=1.0)
    stack = sim._resolve_stack()
    assert stack.simulation["pdk_authoritative"] is True


@pytest.mark.usefixtures("_ihp_active")
def test_explicit_dielectric_material_reaches_palace_materials() -> None:
    """A dielectric-only material token must carry its bulk properties."""
    sim = DrivenSim()
    sim.set_stack(
        pdk_authoritative=True,
        dielectrics=[
            {"name": "substrate", "material": "silicon", "zmin": -180.0, "zmax": 0.0}
        ],
    )
    stack = sim._resolve_stack()

    assert stack.materials["silicon"]["permittivity"] == pytest.approx(11.9)
    assert stack.materials["silicon"]["conductivity"] == pytest.approx(2.0)


def test_synthetic_pdk_layer_names_flow_through_ports(tmp_path) -> None:
    """The tutorial setup never needs IHP-specific layer names."""
    from gsim.palace.stack import build_palace_layer_stack

    synthetic = gf.Pdk(
        name="signal_integrity_synthetic",
        layer_stack=GfLayerStack(
            layers={
                "return_sheet": LayerLevel(
                    layer=(91, 0), zmin=2, thickness=0.5, material="aluminum"
                ),
                "trace_sheet": LayerLevel(
                    layer=(92, 0), zmin=9, thickness=1, material="aluminum"
                ),
            }
        ),
    )
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    try:
        synthetic.activate()
        component = gf.Component()
        component.add_polygon([(0, 0), (200, 0), (200, 10), (0, 10)], layer=(92, 0))
        stack = resolve_layer_stack(
            component,
            pdk=synthetic,
            use_pdk_overlay=False,
            dielectrics=[{"name": "bulk", "material": "sio2", "zmin": 0, "zmax": 11}],
        )

        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "synthetic-setup"))
        sim.set_geometry(component)
        sim.set_stack(build_palace_layer_stack(stack))
        sim.set_airbox(margin_x=10, margin_y=10, z_above=10, z_below=10)
        for name in ("p1", "p2", "p3", "p4"):
            sim.add_port(
                name,
                from_layer="return_sheet",
                to_layer="trace_sheet",
                geometry="interlayer",
                impedance=50,
                excited=True,
            )

        assert sim.stack is not None
        assert sim.stack.layers["trace_sheet"].zmin == 9
        assert all(
            port.from_layer == "return_sheet" and port.to_layer == "trace_sheet"
            for port in sim.ports
        )
        assert all(port.excited for port in sim.ports)
    finally:
        if previous is not None:
            previous.activate()


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
