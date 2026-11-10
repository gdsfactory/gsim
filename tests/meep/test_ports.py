"""Tests for gsim.meep.ports."""

from __future__ import annotations

import pytest

from gsim.meep.models.api import PortVerticalOverride
from gsim.meep.models.config import PortData
from gsim.meep.ports import extract_port_info, filter_ports_for_xz


def _port(name: str, x: float, y: float, orientation: float, width: float = 0.5):
    """Build a PortData with minimal fields for filter tests."""
    normal_axis = 0 if orientation in (0, 180) else 1
    direction = "-" if orientation in (0, 90) else "+"
    return PortData(
        name=name,
        center=[x, y, 0.0],
        orientation=orientation,
        width=width,
        normal_axis=normal_axis,
        direction=direction,
    )


def test_port_data_omits_unset_z_span_from_legacy_document():
    dumped = _port("o1", x=0.0, y=0.0, orientation=0).model_dump()

    assert "z_span" not in dumped


class TestFilterPortsForXZ:
    """Tests for filter_ports_for_xz."""

    def test_keeps_port_intersecting_cut(self):
        ports = [_port("o1", x=0.0, y=0.0, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert [p.name for p in kept] == ["o1"]

    def test_drops_port_off_cut(self):
        ports = [_port("o1", x=0.0, y=3.0, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert kept == []

    def test_drops_y_facing_port(self):
        ports = [_port("o1", x=0.0, y=0.0, orientation=90, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert kept == []

    def test_partial_overlap_included(self):
        # Port at y=0.2, width 0.5 -> extends from y=-0.05 to y=0.45;
        # cut at y=0 falls inside.
        ports = [_port("o1", x=0.0, y=0.2, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert [p.name for p in kept] == ["o1"]

    def test_mixed_ports(self):
        ports = [
            _port("wg_in", x=-5.0, y=0.0, orientation=180, width=0.5),
            _port("wg_out", x=5.0, y=0.0, orientation=0, width=0.5),
            _port("y_oriented", x=0.0, y=0.0, orientation=90, width=0.5),
            _port("far_port", x=0.0, y=10.0, orientation=180, width=0.5),
        ]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert {p.name for p in kept} == {"wg_in", "wg_out"}


def _layer(name, gds_layer, zmin, zmax, material):
    from gsim.common.stack import Layer

    return Layer(
        name=name,
        gds_layer=gds_layer,
        zmin=zmin,
        zmax=zmax,
        thickness=zmax - zmin,
        material=material,
        layer_type="dielectric",
    )


def _component(port_layers, drawn_layers=()):
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    component = gf.Component()
    for index, layer in enumerate(drawn_layers):
        component.add_polygon(
            [(0, index), (2, index), (2, index + 0.5), (0, index + 0.5)],
            layer=layer,
        )
    for index, (name, layer) in enumerate(port_layers):
        component.add_port(
            name=name,
            center=(float(index), 0.0),
            width=0.5,
            orientation=180 if index == 0 else 0,
            layer=layer,
        )
    return component


class TestPerPortVerticalGeometry:
    """Resolve vertical mode planes from each port's fabrication layer."""

    def test_nitride_port_ignores_undrawn_silicon(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (174, 0))], drawn_layers=[(174, 0)])
        stack = LayerStack(
            layers={
                "core": _layer("core", (171, 0), 0.0, 0.49, "si"),
                "sin": _layer("sin", (174, 0), 1.34, 1.73, "sin"),
            }
        )

        port = extract_port_info(component, stack, port_margin=0.5)[0]

        assert port.center[2] == pytest.approx(1.535)
        assert port.z_span == pytest.approx(1.39)

    def test_mixed_silicon_and_nitride_ports_are_independent(self):
        from gsim.common.stack import LayerStack

        component = _component(
            [("si", (171, 0)), ("sin", (174, 0))],
            drawn_layers=[(171, 0), (174, 0)],
        )
        stack = LayerStack(
            layers={
                "core": _layer("core", (171, 0), 0.0, 0.49, "si"),
                "sin": _layer("sin", (174, 0), 1.34, 1.73, "sin"),
            }
        )

        ports = {
            port.name: port
            for port in extract_port_info(component, stack, port_margin=0.5)
        }

        assert ports["si"].center[2] == pytest.approx(0.245)
        assert ports["si"].z_span == pytest.approx(1.49)
        assert ports["sin"].center[2] == pytest.approx(1.535)
        assert ports["sin"].z_span == pytest.approx(1.39)

    def test_duplicate_layers_with_identical_extents_are_safe(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (2, 0))], drawn_layers=[(2, 0)])
        stack = LayerStack(
            layers={
                "a": _layer("a", (2, 0), 0.0, 0.22, "si"),
                "b": _layer("b", (2, 0), 0.0, 0.22, "si"),
            }
        )

        port = extract_port_info(component, stack, port_margin=0.5)[0]

        assert port.center[2] == pytest.approx(0.11)
        assert port.z_span == pytest.approx(1.22)

    def test_ambiguous_extents_require_missing_override_fields(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (2, 0))], drawn_layers=[(2, 0)])
        stack = LayerStack(
            layers={
                "lower": _layer("lower", (2, 0), 0.0, 0.2, "si"),
                "upper": _layer("upper", (2, 0), 1.0, 1.4, "si"),
            }
        )

        with pytest.raises(ValueError, match="Could not infer z, z_span for port 'o1'"):
            extract_port_info(component, stack, port_margin=0.5)

        with pytest.raises(ValueError, match="Could not infer z_span for port 'o1'"):
            extract_port_info(
                component,
                stack,
                port_margin=0.5,
                port_overrides={"o1": PortVerticalOverride(z=1.2)},
            )

        port = extract_port_info(
            component,
            stack,
            port_margin=0.5,
            port_overrides={"o1": PortVerticalOverride(z=1.2, z_span=1.4)},
        )[0]
        assert port.center[2] == pytest.approx(1.2)
        assert port.z_span == pytest.approx(1.4)

    def test_override_fields_take_precedence_independently(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (1, 0))], drawn_layers=[(1, 0)])
        stack = LayerStack(layers={"core": _layer("core", (1, 0), 0.0, 0.22, "si")})

        z_only = extract_port_info(
            component,
            stack,
            port_margin=0.5,
            port_overrides={"o1": PortVerticalOverride(z=1.535)},
        )[0]
        span_only = extract_port_info(
            component,
            stack,
            port_margin=0.5,
            port_overrides={"o1": PortVerticalOverride(z_span=1.39)},
        )[0]

        assert z_only.center[2] == pytest.approx(1.535)
        assert z_only.z_span == pytest.approx(1.22)
        assert span_only.center[2] == pytest.approx(0.11)
        assert span_only.z_span == pytest.approx(1.39)

    def test_drawn_port_layer_cropped_from_stack_is_an_error(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (174, 0))], drawn_layers=[(174, 0)])
        stack = LayerStack(layers={"core": _layer("core", (171, 0), 0.0, 0.49, "si")})

        with pytest.raises(ValueError) as excinfo:
            extract_port_info(component, stack, port_margin=0.5, stack_is_cropped=True)

        assert str(excinfo.value) == (
            "Port 'o1' uses drawn layer (174, 0), which has no entry in the active "
            "simulation stack. Stack layers: core (171, 0). Expand domain.z_bounds "
            "only if a crop removed the layer. If the layer is physical material, "
            "add a LayerLevel to the PDK LayerStack or pass a stack that includes "
            "it with set_stack(stack=...). If it is only a marker, move the port "
            "to a stack layer."
        )

    def test_drawn_port_layer_missing_from_stack_says_how_to_fix_it(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (174, 0))], drawn_layers=[(174, 0)])
        stack = LayerStack(layers={"core": _layer("core", (171, 0), 0.0, 0.49, "si")})

        with pytest.raises(ValueError) as excinfo:
            extract_port_info(component, stack, port_margin=0.5)

        assert "z_bounds" not in str(excinfo.value)
        assert str(excinfo.value) == (
            "Port 'o1' uses drawn layer (174, 0), which has no entry in the active "
            "simulation stack. Stack layers: core (171, 0). If the layer is physical "
            "material, add a LayerLevel to the PDK LayerStack or pass a stack that "
            "includes it with set_stack(stack=...). If it is only a marker, move the "
            "port to a stack layer."
        )

    def test_missing_port_layer_error_reports_an_empty_stack(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (174, 0))], drawn_layers=[(174, 0)])

        with pytest.raises(ValueError, match=r"Stack layers: none\. If the layer"):
            extract_port_info(component, LayerStack(layers={}), port_margin=0.5)

    def test_unambiguous_drawn_layer_fallback(self, caplog):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (99, 0))], drawn_layers=[(174, 0)])
        stack = LayerStack(layers={"sin": _layer("sin", (174, 0), 1.34, 1.73, "sin")})

        port = extract_port_info(component, stack, port_margin=0.5)[0]

        assert port.center[2] == pytest.approx(1.535)
        assert port.z_span == pytest.approx(1.39)
        assert "unambiguous drawn-layer Z extent" in caplog.text

    def test_collapsed_z_preserves_zero_center_and_no_span(self):
        from gsim.common.stack import LayerStack

        component = _component([("o1", (1, 0))], drawn_layers=[(1, 0)])
        stack = LayerStack(layers={"core": _layer("core", (1, 0), 0.0, 0.22, "si")})

        port = extract_port_info(component, stack, is_3d=False)[0]

        assert port.center[2] == 0.0
        assert port.z_span is None


@pytest.fixture
def undeclared_layer_pdk():
    """Activate a PDK whose LayerStack declares only ``core`` and ``metal``."""
    import gdsfactory as gf
    from gdsfactory.technology import LayerLevel
    from gdsfactory.technology import LayerStack as GfLayerStack

    pdk = gf.Pdk(
        name="meep_port_layer_test",
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
                ),
                "metal": LayerLevel(
                    layer=(2, 0),
                    thickness=0.2,
                    zmin=0.3,
                    material="Al",
                    info={"mesh_order": 2, "layer_type": "conductor"},
                ),
            }
        ),
    )
    previous = gf.get_active_pdk()
    pdk.activate()
    try:
        yield pdk
    finally:
        previous.activate()


def test_build_config_reports_port_on_undeclared_drawn_layer(undeclared_layer_pdk):
    import gdsfactory as gf

    from gsim.meep import Simulation

    component = gf.Component()
    component.add_polygon([(0, -1), (5, -1), (5, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0, -0.25), (5, -0.25), (5, 0.25), (0, 0.25)], layer=(77, 0))
    component.add_port(
        name="o1", center=(0.0, 0.0), orientation=180, width=0.5, layer=(77, 0)
    )

    sim = Simulation()
    sim.geometry.component = component
    sim.set_stack(
        pdk_authoritative=True,
        pdk=undeclared_layer_pdk,
        dielectrics=[{"name": "box", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}],
    )
    sim.source.port = "o1"
    sim.source.wavelength = 1.55

    with pytest.raises(ValueError) as excinfo:
        sim.build_config()

    assert str(excinfo.value) == (
        "Port 'o1' uses drawn layer (77, 0), which has no entry in the active "
        "simulation stack. Stack layers: core (1, 0), metal (2, 0). If the layer is "
        "physical material, add a LayerLevel to the PDK LayerStack or pass a stack "
        "that includes it with set_stack(stack=...). If it is only a marker, move "
        "the port to a stack layer."
    )


def _simulation_with_port_on_metal(pdk):
    """Build a 3D simulation whose only port is on the ``metal`` level."""
    import gdsfactory as gf

    from gsim.meep import Simulation

    component = gf.Component()
    component.add_polygon([(0, -1), (5, -1), (5, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0, -0.25), (5, -0.25), (5, 0.25), (0, 0.25)], layer=(2, 0))
    component.add_port(
        name="o1", center=(0.0, 0.0), orientation=180, width=0.5, layer=(2, 0)
    )

    sim = Simulation()
    sim.geometry.component = component
    sim.set_stack(
        pdk_authoritative=True,
        pdk=pdk,
        dielectrics=[{"name": "box", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}],
    )
    sim.source.port = "o1"
    sim.source.wavelength = 1.55
    return sim


def test_build_config_explicit_xy_bounds_report_port_layer_removed_by_z_crop(
    undeclared_layer_pdk,
):
    sim = _simulation_with_port_on_metal(undeclared_layer_pdk)
    sim.domain(x_bounds=(-1.0, 6.0), y_bounds=(-2.0, 2.0), z_bounds=(-1.0, 0.25))

    with pytest.raises(ValueError) as excinfo:
        sim.build_config()

    assert str(excinfo.value) == (
        "Port 'o1' uses drawn layer (2, 0), which has no entry in the active "
        "simulation stack. Stack layers: core (1, 0). Expand domain.z_bounds only "
        "if a crop removed the layer. If the layer is physical material, add a "
        "LayerLevel to the PDK LayerStack or pass a stack that includes it with "
        "set_stack(stack=...). If it is only a marker, move the port to a stack "
        "layer."
    )
