"""Resolve virtual port markers using small, synthetic PDKs."""

from functools import partial

import gdsfactory as gf
import pytest
from gdsfactory.cross_section import Section
from gdsfactory.technology import LayerLevel, LayerMap, LayerStack, LogicalLayer

from gsim.common.pdk import UnsupportedPortError, resolve_passive_pcell

MARKER = (100, 0)
CORE = (1, 0)
SECOND_CORE = (4, 0)


@pytest.fixture(autouse=True)
def restore_active_pdk():
    """Avoid leaking a synthetic registry into other tests."""
    previous = gf.get_active_pdk()
    yield
    previous.activate()


def marked_cross_section(width=0.5, core_layer=CORE, marker=MARKER, extra=()):
    """Describe a marker and a coincident physical core section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=marker,
        sections=(Section(width=width, layer=core_layer), *extra),
    )


def make_pdk(cross_sections=None):
    """Build a two-band core whose upper band can be etched locally."""
    return gf.Pdk(
        name="cross_section_port_demo",
        cross_sections=(
            {"marked": marked_cross_section}
            if cross_sections is None
            else cross_sections
        ),
        layer_stack=LayerStack(
            layers={
                "lower": LayerLevel(
                    layer=LogicalLayer(layer=CORE),
                    thickness=0.14,
                    zmin=0,
                    material="Si",
                ),
                "upper": LayerLevel(
                    layer=LogicalLayer(layer=CORE) - LogicalLayer(layer=(2, 0)),
                    derived_layer=LogicalLayer(layer=(10, 0)),
                    thickness=0.16,
                    zmin=0.14,
                    material="Si",
                ),
            }
        ),
    )


def component(*, metadata=None, port_layer=MARKER, snapshot=None, port_type="optical"):
    """Return a rectangular guide with two marker-layer ports."""
    result = gf.Component()
    result.add_polygon([(0, -0.25), (3, -0.25), (3, 0.25), (0, 0.25)], layer=CORE)
    for name, center, orientation in (("o1", (0, 0), 180), ("o2", (3, 0), 0)):
        if snapshot is None:
            port = result.add_port(
                name=name,
                center=center,
                orientation=orientation,
                width=0.5,
                layer=port_layer,
                port_type=port_type,
            )
        else:
            port = result.add_port(
                name=name,
                center=center,
                orientation=orientation,
                cross_section=snapshot,
            )
        for key, value in (metadata or {}).items():
            port.info[key] = value
    return result


@pytest.mark.parametrize("description", ["registry", "name", "settings", "snapshot"])
def test_resolves_marker_from_cross_section(description):
    """All supported descriptions identify the complete 300 nm physical core."""
    pdk = make_pdk()
    metadata = None
    snapshot = None
    if description in {"name", "settings"}:
        metadata = {"cross_section": "marked"}
    if description == "settings":
        metadata = {
            "cross_section": "marked",
            "cross_section_settings": {"width": 0.5},
        }
    if description == "snapshot":
        snapshot = marked_cross_section()
    guide = component(metadata=metadata, snapshot=snapshot)
    before_stack = pdk.layer_stack.model_dump()
    before_ports = [port.info.model_dump() for port in guide.ports]

    result = resolve_passive_pcell(guide, pdk=pdk)

    for port in result.ports.values():
        assert port.physical_layer_keys == ("lower", "upper")
        assert port.material == "Si"
        assert port.z_bounds == pytest.approx((0, 0.3))
        assert port.center[2] == pytest.approx(0.15)
        assert port.width == 0.5
    assert pdk.layer_stack.model_dump() == before_stack
    assert [port.info.model_dump() for port in guide.ports] == before_ports


def test_joins_negative_thickness_band():
    """Band orientation does not change the physical cross-section."""
    pdk = make_pdk()
    pdk.layer_stack.layers["upper"].zmin = 0.3
    pdk.layer_stack.layers["upper"].thickness = -0.16
    result = resolve_passive_pcell(component(), pdk=pdk)
    assert result.ports["o1"].z_bounds == pytest.approx((0, 0.3))


def test_uses_only_bands_present_at_each_port():
    """A local etch leaves one thin port and one full-height port."""
    guide = component()
    guide.add_polygon(
        [(-0.1, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.1, 0.4)], layer=(2, 0)
    )
    result = resolve_passive_pcell(guide, pdk=make_pdk())
    assert result.ports["o1"].physical_layer_keys == ("lower",)
    assert result.ports["o1"].z_bounds == pytest.approx((0, 0.14))
    assert result.ports["o2"].z_bounds == pytest.approx((0, 0.3))


def test_clips_inferred_port_bounds_to_domain():
    """Automatic inference honors the same vertical crop as explicit maps."""
    result = resolve_passive_pcell(component(), pdk=make_pdk(), z_bounds=(0.05, 0.2))
    assert result.ports["o1"].z_bounds == pytest.approx((0.05, 0.2))


def test_registry_aliases_are_equivalent_and_factories_are_cached():
    """Equivalent names are accepted without reevaluating them for every port."""
    calls = []

    def factory():
        calls.append("called")
        return marked_cross_section()

    pdk = make_pdk({"first": factory, "alias": factory})
    resolve_passive_pcell(component(), pdk=pdk)
    assert calls == ["called", "called"]


def test_ignores_factories_that_require_unprovided_settings():
    """An unrelated parameterized entry does not disable a usable default."""

    def requires_settings(required):
        return marked_cross_section(width=required)

    pdk = make_pdk({"custom": requires_settings, "marked": marked_cross_section})
    result = resolve_passive_pcell(component(), pdk=pdk)
    assert result.ports["o1"].z_bounds == pytest.approx((0, 0.3))


def test_named_metadata_avoids_unrelated_factories():
    """A port's declared cross-section takes precedence over registry discovery."""

    def unrelated():
        pytest.fail("An explicit cross-section must not inspect unrelated entries")

    pdk = make_pdk({"marked": marked_cross_section, "unrelated": unrelated})
    result = resolve_passive_pcell(
        component(metadata={"cross_section": "marked"}), pdk=pdk
    )
    assert result.ports["o1"].physical_layer_keys == ("lower", "upper")


@pytest.mark.parametrize("explicit", [False, True])
def test_existing_layer_resolution_bypasses_bad_cross_section_metadata(explicit):
    """Explicit overrides and direct physical ports keep their previous behavior."""
    pdk = make_pdk()
    guide = component(
        metadata={"cross_section": "not_registered"},
        port_layer=MARKER if explicit else CORE,
    )
    result = resolve_passive_pcell(
        guide, pdk=pdk, port_layer_map={MARKER: "lower"} if explicit else None
    )
    assert result.ports["o1"].z_bounds == pytest.approx((0, 0.14 if explicit else 0.3))


def test_invalid_explicit_map_is_not_replaced_by_inference():
    """An explicit selection remains authoritative, including its failures."""
    with pytest.raises(UnsupportedPortError, match="has no resolved geometry"):
        resolve_passive_pcell(
            component(), pdk=make_pdk(), port_layer_map={MARKER: "missing"}
        )


@pytest.mark.parametrize("change", ["gap", "overlap", "material"])
def test_rejects_ambiguous_band_groups(change):
    """Never guess between disconnected, overlapping or different-material bands."""
    pdk = make_pdk()
    if change == "gap":
        pdk.layer_stack.layers["upper"].zmin = 0.4
    elif change == "overlap":
        pdk.layer_stack.layers["upper"].zmin = 0.07
    else:
        pdk.layer_stack.layers["upper"].material = "SiO2"
    with pytest.raises(
        UnsupportedPortError, match=r"ambiguous physical bands.*port_layer_map"
    ):
        resolve_passive_pcell(component(), pdk=pdk)


def test_rejects_multiple_different_physical_cross_sections():
    """Different physical cores sharing one marker require an explicit choice."""
    pdk = make_pdk(
        {
            "first": marked_cross_section,
            "second": partial(marked_cross_section, core_layer=SECOND_CORE),
        }
    )
    pdk.layer_stack.layers["other"] = LayerLevel(
        layer=LogicalLayer(layer=SECOND_CORE), thickness=0.25, zmin=0.5, material="Si"
    )
    guide = component()
    guide.add_polygon([(0, -0.25), (3, -0.25), (3, 0.25), (0, 0.25)], layer=SECOND_CORE)
    with pytest.raises(UnsupportedPortError, match="multiple physical cross-sections"):
        resolve_passive_pcell(guide, pdk=pdk)
    guide.ports["o1"].info["cross_section"] = "first"
    guide.ports["o2"].info["cross_section"] = "second"
    result = resolve_passive_pcell(guide, pdk=pdk)
    assert result.ports["o1"].physical_layer_keys == ("lower", "upper")
    assert result.ports["o2"].physical_layer_keys == ("other",)


def test_snapshot_and_factory_settings_select_actual_core():
    """Saved settings override a factory's default core layer."""
    pdk = make_pdk()
    pdk.layer_stack.layers["other"] = LayerLevel(
        layer=LogicalLayer(layer=SECOND_CORE), thickness=0.25, zmin=0.5, material="Si"
    )
    guide = component(
        metadata={
            "cross_section": "marked",
            "cross_section_settings": {"core_layer": SECOND_CORE},
        }
    )
    guide.add_polygon([(0, -0.25), (3, -0.25), (3, 0.25), (0, 0.25)], layer=SECOND_CORE)
    result = resolve_passive_pcell(guide, pdk=pdk)
    assert result.ports["o1"].physical_layer_keys == ("other",)


def test_geometry_disambiguates_registry_entries_at_different_ports():
    """Only the core present at each location contributes to that port."""
    pdk = make_pdk(
        {
            "first": marked_cross_section,
            "second": partial(marked_cross_section, core_layer=SECOND_CORE),
        }
    )
    pdk.layer_stack.layers["other"] = LayerLevel(
        layer=LogicalLayer(layer=SECOND_CORE), thickness=0.25, zmin=0.5, material="Si"
    )
    guide = gf.Component()
    guide.add_polygon([(0, -0.25), (1, -0.25), (1, 0.25), (0, 0.25)], layer=CORE)
    guide.add_polygon([(1, -0.25), (3, -0.25), (3, 0.25), (1, 0.25)], layer=SECOND_CORE)
    for name, center, angle in (("o1", (0, 0), 180), ("o2", (3, 0), 0)):
        guide.add_port(
            name=name, center=center, orientation=angle, width=0.5, layer=MARKER
        )
    result = resolve_passive_pcell(guide, pdk=pdk)
    assert result.ports["o1"].physical_layer_keys == ("lower", "upper")
    assert result.ports["o2"].physical_layer_keys == ("other",)


def test_excludes_substrate_and_noncoincident_sections():
    """Unrelated material at the same XY point is not inferred as the core."""
    extra = (
        Section(width=2, layer=(5, 0)),
        Section(width=0.5, offset=1, layer=(6, 0)),
        Section(width=0.5, hidden=True, layer=(7, 0)),
    )
    pdk = make_pdk({"marked": partial(marked_cross_section, extra=extra)})
    guide = component()
    for number in (3, 5, 6, 7):
        guide.add_polygon([(0, -1), (3, -1), (3, 1), (0, 1)], layer=(number, 0))
        pdk.layer_stack.layers[f"unrelated_{number}"] = LayerLevel(
            layer=LogicalLayer(layer=(number, 0)), thickness=0.2, zmin=-1, material="Si"
        )
    result = resolve_passive_pcell(guide, pdk=pdk)
    assert result.ports["o1"].physical_layer_keys == ("lower", "upper")


@pytest.mark.parametrize(
    "metadata",
    [
        {"cross_section": "unknown"},
        {"cross_section_settings": 3},
        {"cross_section_settings": {"width": 0.5}},
        {"cross_section_settings": {"sections": [{"width": -1}]}},
    ],
)
def test_invalid_metadata_fails_instead_of_guessing(metadata):
    """Bad explicit metadata never falls back to another registered default."""
    with pytest.raises(UnsupportedPortError):
        resolve_passive_pcell(component(metadata=metadata), pdk=make_pdk())


def test_uses_selected_pdk_layers_and_restores_active_pdk():
    """String layer names are interpreted in the requested PDK's namespace."""

    class Layers(LayerMap):
        PIN = MARKER
        DRAW = CORE

    pdk = make_pdk(
        {"marked": partial(marked_cross_section, marker="PIN", core_layer="DRAW")}
    )
    pdk.layers = Layers
    previous = gf.get_active_pdk()
    result = resolve_passive_pcell(component(), pdk=pdk)
    assert result.ports["o1"].z_bounds == pytest.approx((0, 0.3))
    assert gf.get_active_pdk() is previous
    with pytest.raises(UnsupportedPortError):
        resolve_passive_pcell(component(metadata={"cross_section": "unknown"}), pdk=pdk)
    assert gf.get_active_pdk() is previous


def test_virtual_vertical_ports_keep_free_space_semantics():
    """The fallback does not turn vertical optical ports into guided cores."""

    def unrelated():
        pytest.fail("Vertical ports must not trigger cross-section discovery")

    result = resolve_passive_pcell(
        component(port_type="vertical_te"), pdk=make_pdk({"unused": unrelated})
    )
    assert result.ports["o1"].is_vertical
    assert result.ports["o1"].physical_layer_keys == ()


def test_distinct_same_named_pdk_instances_restore_layer_namespace():
    """Same-named PDK copies can assign different concrete layers to one name."""

    class TargetLayers(LayerMap):
        PIN = MARKER
        DRAW = CORE

    class OtherLayers(LayerMap):
        PIN = (101, 0)
        DRAW = SECOND_CORE

    target = make_pdk(
        {"marked": partial(marked_cross_section, marker="PIN", core_layer="DRAW")}
    )
    target.layers = TargetLayers
    previous = gf.Pdk(name=target.name, layers=OtherLayers)
    previous.activate(force=True)
    result = resolve_passive_pcell(component(), pdk=target)
    assert result.ports["o1"].physical_layer_keys == ("lower", "upper")
    assert gf.get_active_pdk() is previous


def test_marker_only_cross_section_does_not_guess_a_material():
    """A virtual section without a drawing section is insufficient evidence."""
    pdk = make_pdk(
        {
            "marker_only": partial(
                gf.cross_section.cross_section, width=0.5, layer=MARKER
            )
        }
    )
    with pytest.raises(UnsupportedPortError, match="does not map"):
        resolve_passive_pcell(component(), pdk=pdk)


def test_named_cross_section_with_different_marker_is_not_replaced():
    """Valid but mismatched metadata does not silently switch to another entry."""
    pdk = make_pdk(
        {
            "marked": marked_cross_section,
            "other_marker": partial(marked_cross_section, marker=(101, 0)),
        }
    )
    with pytest.raises(UnsupportedPortError, match="does not map"):
        resolve_passive_pcell(
            component(metadata={"cross_section": "other_marker"}), pdk=pdk
        )


def test_inferred_core_must_contain_port_location():
    """A cross-section describes ownership rather than creating new material."""
    guide = component()
    guide.ports["o1"].center = (-0.1, 0)
    with pytest.raises(UnsupportedPortError, match="does not map"):
        resolve_passive_pcell(guide, pdk=make_pdk())
