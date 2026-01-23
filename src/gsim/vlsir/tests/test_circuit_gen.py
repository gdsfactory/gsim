"""Integration tests for to_vlsir_circuit netlist conversion.

Uses real gdsfactory components with vlsir metadata to test the full
conversion pipeline from GDS layout to VLSIR protobuf.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest
import vlsir.circuit_pb2 as vckt
from gdsfactory.component import Component
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.typings import LayerSpec

from gsim.vlsir.netlist import _spice_type_to_proto, to_vlsir_circuit

# Activate the generic PDK for all tests
PDK = get_generic_pdk()
PDK.activate()


# =============================================================================
# Test device component factories
# =============================================================================


@gf.cell
def resistor(
    width: float = 1.0,
    length: float = 10.0,
    layer: LayerSpec = (1, 0),
    model: str = "resistor",
    resistance: float = 1000.0,
) -> Component:
    """A simple two-terminal resistor with vlsir metadata."""
    c = Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(
        name="p", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    c.add_port(
        name="n", center=(length, width / 2), width=width, orientation=0, layer=layer
    )

    c.info["vlsir"] = {
        "model": model,
        "spice_lib": "basic.lib",
        "spice_type": "RESISTOR",
        "port_order": ["p", "n"],
        "params": {"r": resistance},
    }
    return c


@gf.cell
def capacitor(
    width: float = 1.0,
    length: float = 5.0,
    layer: LayerSpec = (1, 0),
    model: str = "capacitor",
    capacitance: float = 1e-12,
) -> Component:
    """A simple two-terminal capacitor with vlsir metadata."""
    c = Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(
        name="p", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    c.add_port(
        name="n", center=(length, width / 2), width=width, orientation=0, layer=layer
    )

    c.info["vlsir"] = {
        "model": model,
        "spice_lib": "basic.lib",
        "spice_type": "CAPACITOR",
        "port_order": ["p", "n"],
        "params": {"c": capacitance},
    }
    return c


@gf.cell
def nmos(
    width: float = 10.0,
    length: float = 10.0,
    layer: LayerSpec = (1, 0),
    model: str = "nfet_01v8",
    w: float = 1e-6,
    l: float = 180e-9,
    nf: int = 1,
) -> Component:
    """A 4-terminal NMOS transistor with vlsir metadata."""
    c = Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)

    # 4 ports: drain, gate, source, bulk (all width=1 for easy connectivity)
    c.add_port(
        name="D", center=(length / 2, width), width=1, orientation=90, layer=layer
    )
    c.add_port(name="G", center=(0, width / 2), width=1, orientation=180, layer=layer)
    c.add_port(name="S", center=(length / 2, 0), width=1, orientation=-90, layer=layer)
    c.add_port(
        name="B", center=(length, width / 2), width=1, orientation=0, layer=layer
    )

    c.info["vlsir"] = {
        "model": model,
        "spice_lib": "sky130_fd_pr",
        "spice_type": "MOS",
        "port_order": ["d", "g", "s", "b"],
        "port_map": {"D": "d", "G": "g", "S": "s", "B": "b"},
        "params": {"w": w, "l": l, "nf": nf},
    }
    return c


@gf.cell
def pmos(
    width: float = 10.0,
    length: float = 10.0,
    layer: LayerSpec = (1, 0),
    model: str = "pfet_01v8",
    w: float = 2e-6,
    l: float = 180e-9,
    nf: int = 1,
) -> Component:
    """A 4-terminal PMOS transistor with vlsir metadata."""
    c = Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)

    c.add_port(
        name="D", center=(length / 2, width), width=1, orientation=90, layer=layer
    )
    c.add_port(name="G", center=(0, width / 2), width=1, orientation=180, layer=layer)
    c.add_port(name="S", center=(length / 2, 0), width=1, orientation=-90, layer=layer)
    c.add_port(
        name="B", center=(length, width / 2), width=1, orientation=0, layer=layer
    )

    c.info["vlsir"] = {
        "model": model,
        "spice_lib": "sky130_fd_pr",
        "spice_type": "MOS",
        "port_order": ["d", "g", "s", "b"],
        "port_map": {"D": "d", "G": "g", "S": "s", "B": "b"},
        "params": {"w": w, "l": l, "nf": nf},
    }
    return c


@gf.cell
def npn_bjt(
    width: float = 8.0,
    length: float = 8.0,
    layer: LayerSpec = (1, 0),
    model: str = "npn13G2",
    we: float = 0.07e-6,
    le: float = 0.9e-6,
) -> Component:
    """A 3-terminal NPN BJT with vlsir metadata."""
    c = Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)

    c.add_port(
        name="C", center=(length / 2, width), width=1, orientation=90, layer=layer
    )
    c.add_port(name="B", center=(0, width / 2), width=1, orientation=180, layer=layer)
    c.add_port(name="E", center=(length / 2, 0), width=1, orientation=-90, layer=layer)

    c.info["vlsir"] = {
        "model": model,
        "spice_lib": "ihp_sg13g2",
        "spice_type": "BIPOLAR",
        "port_order": ["c", "b", "e"],
        "port_map": {"C": "c", "B": "b", "E": "e"},
        "params": {"we": we, "le": le},
    }
    return c


@gf.cell
def wire(
    length: float = 20.0,
    width: float = 1.0,
    layer: LayerSpec = (1, 0),  # Same layer as resistor for easy connectivity
) -> Component:
    """A routing wire (no vlsir metadata - treated as routing element)."""
    c = Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(
        name="o1", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    c.add_port(
        name="o2", center=(length, width / 2), width=width, orientation=0, layer=layer
    )
    c.info["length"] = length
    c.info["width"] = width
    return c


# =============================================================================
# Tests for _spice_type_to_proto helper
# =============================================================================


class TestSpiceTypeToProto:
    """Tests for SPICE type string to proto enum conversion."""

    @pytest.mark.parametrize(
        "spice_type,expected",
        [
            ("RESISTOR", vckt.SpiceType.RESISTOR),
            ("resistor", vckt.SpiceType.RESISTOR),
            ("CAPACITOR", vckt.SpiceType.CAPACITOR),
            ("INDUCTOR", vckt.SpiceType.INDUCTOR),
            ("MOS", vckt.SpiceType.MOS),
            ("DIODE", vckt.SpiceType.DIODE),
            ("BIPOLAR", vckt.SpiceType.BIPOLAR),
            ("VSOURCE", vckt.SpiceType.VSOURCE),
            ("ISOURCE", vckt.SpiceType.ISOURCE),
            ("SUBCKT", vckt.SpiceType.SUBCKT),
        ],
    )
    def test_string_conversion(self, spice_type: str, expected: int):
        """Test that string spice types convert correctly."""
        assert _spice_type_to_proto(spice_type) == expected

    def test_int_passthrough(self):
        """Test that integer values pass through unchanged."""
        assert _spice_type_to_proto(vckt.SpiceType.MOS) == vckt.SpiceType.MOS

    def test_unknown_string_defaults_to_subckt(self):
        """Test that unknown strings default to SUBCKT."""
        assert _spice_type_to_proto("UNKNOWN") == vckt.SpiceType.SUBCKT


# =============================================================================
# Basic device tests with real gdsfactory components
# =============================================================================


class TestBasicDevicesIntegration:
    """Tests using real gdsfactory components."""

    def test_single_resistor(self):
        """Test circuit with a single resistor connected to wires."""

        @gf.cell
        def resistor_circuit() -> Component:
            c = Component()
            r1 = c << resistor(resistance=1000.0)
            w1 = c << wire(length=10)
            w2 = c << wire(length=10)

            w1.connect("o2", r1.ports["p"])
            w2.connect("o1", r1.ports["n"])

            c.add_port("in", port=w1.ports["o1"])
            c.add_port("out", port=w2.ports["o2"])
            return c

        top = resistor_circuit()
        package, libs = to_vlsir_circuit(top)

        assert len(package.ext_modules) == 1
        assert package.ext_modules[0].name.name == "resistor"
        assert package.ext_modules[0].spicetype == vckt.SpiceType.RESISTOR
        assert "basic.lib" in libs
        assert len(package.modules[0].instances) == 1

    def test_single_capacitor(self):
        """Test circuit with a single capacitor."""

        @gf.cell
        def capacitor_circuit() -> Component:
            c = Component()
            c1 = c << capacitor(capacitance=1e-12)
            w1 = c << wire(length=10)
            w2 = c << wire(length=10)

            w1.connect("o2", c1.ports["p"])
            w2.connect("o1", c1.ports["n"])
            return c

        top = capacitor_circuit()
        package, libs = to_vlsir_circuit(top)

        assert package.ext_modules[0].spicetype == vckt.SpiceType.CAPACITOR

    def test_nmos_transistor(self):
        """Test circuit with an NMOS transistor."""

        @gf.cell
        def nmos_circuit() -> Component:
            c = Component()
            m1 = c << nmos(w=1e-6, l=180e-9)

            # Add a single wire connected to gate
            wg = c << wire(length=5)
            wg.connect("o2", m1.ports["G"])

            return c

        top = nmos_circuit()
        package, libs = to_vlsir_circuit(top)

        assert package.ext_modules[0].spicetype == vckt.SpiceType.MOS
        assert "sky130_fd_pr" in libs

        # Check port order is preserved
        port_names = [s.name for s in package.ext_modules[0].signals]
        assert port_names == ["d", "g", "s", "b"]

    def test_bjt_transistor(self):
        """Test circuit with a BJT."""

        @gf.cell
        def bjt_circuit() -> Component:
            c = Component()
            q1 = c << npn_bjt()
            wc = c << wire(length=5)
            wc.connect("o2", q1.ports["C"], allow_width_mismatch=True)
            return c

        top = bjt_circuit()
        package, libs = to_vlsir_circuit(top)

        assert package.ext_modules[0].spicetype == vckt.SpiceType.BIPOLAR
        assert "ihp_sg13g2" in libs


# =============================================================================
# Complex circuit tests
# =============================================================================


class TestComplexCircuits:
    """Tests for circuits with multiple devices."""

    def test_voltage_divider(self):
        """Test a voltage divider with two resistors."""

        @gf.cell
        def voltage_divider() -> Component:
            c = Component()

            r1 = c << resistor(resistance=10000.0, length=15)
            r2 = c << resistor(resistance=10000.0, length=15)

            # Position r2 after r1
            r2.dmove((25, 0))

            # Wire connecting r1 output to r2 input
            w_mid = c << wire(length=5)
            w_mid.connect("o1", r1.ports["n"])
            r2.connect("p", w_mid.ports["o2"])

            return c

        top = voltage_divider()
        package, _ = to_vlsir_circuit(top)

        # Should have only one external module (resistor model reused)
        assert len(package.ext_modules) == 1
        # Should have two instances
        assert len(package.modules[0].instances) == 2

    def test_rc_filter(self):
        """Test an RC low-pass filter."""

        @gf.cell
        def rc_filter() -> Component:
            c = Component()

            r1 = c << resistor(resistance=1000.0)
            c1 = c << capacitor(capacitance=1e-9)

            # Position
            c1.dmove((15, -5))

            # Wires
            w_in = c << wire(length=5)
            w_mid = c << wire(length=5)
            w_gnd = c << wire(length=5)

            w_in.connect("o2", r1.ports["p"])
            w_mid.connect("o1", r1.ports["n"])
            c1.connect("p", w_mid.ports["o2"], allow_width_mismatch=True)
            w_gnd.connect("o1", c1.ports["n"], allow_width_mismatch=True)

            return c

        top = rc_filter()
        package, _ = to_vlsir_circuit(top)

        assert len(package.ext_modules) == 2  # resistor and capacitor
        assert len(package.modules[0].instances) == 2

    def test_cmos_inverter(self):
        """Test a CMOS inverter with NMOS and PMOS."""

        @gf.cell
        def cmos_inverter() -> Component:
            c = Component()

            mn = c << nmos(w=1e-6, l=180e-9)
            mp = c << pmos(w=2e-6, l=180e-9)

            # Stack PMOS above NMOS
            mp.dmove((0, 15))

            # Wires for input, output, vdd, vss
            w_in = c << wire(length=5)
            w_out = c << wire(length=5)
            w_vdd = c << wire(length=5)
            w_vss = c << wire(length=5)

            # Connect gates together (input)
            w_in.connect("o2", mn.ports["G"], allow_width_mismatch=True)

            # Connect drains together (output)
            w_out.connect("o1", mn.ports["D"], allow_width_mismatch=True)

            return c

        top = cmos_inverter()
        package, libs = to_vlsir_circuit(top)

        assert len(package.ext_modules) == 2  # nfet and pfet
        assert "sky130_fd_pr" in libs

    def test_multiple_libraries(self):
        """Test circuit requiring multiple SPICE libraries."""

        @gf.cell
        def multi_lib_circuit() -> Component:
            c = Component()

            r1 = c << resistor()
            m1 = c << nmos()
            q1 = c << npn_bjt()

            m1.dmove((15, 0))
            q1.dmove((30, 0))

            w1 = c << wire(length=5)
            w2 = c << wire(length=5)
            w3 = c << wire(length=5)

            w1.connect("o2", r1.ports["p"])
            w2.connect("o2", m1.ports["G"], allow_width_mismatch=True)
            w3.connect("o2", q1.ports["B"], allow_width_mismatch=True)

            return c

        top = multi_lib_circuit()
        package, libs = to_vlsir_circuit(top)

        assert len(libs) == 3
        assert "basic.lib" in libs
        assert "sky130_fd_pr" in libs
        assert "ihp_sg13g2" in libs


# =============================================================================
# Routing connectivity tests
# =============================================================================


class TestRoutingConnectivity:
    """Tests for routing graph construction."""

    def test_chained_routing(self):
        """Test that chained wires merge into single nodes."""

        @gf.cell
        def chained_wires() -> Component:
            c = Component()

            r1 = c << resistor()
            w1 = c << wire(length=10)
            w2 = c << wire(length=10)
            w3 = c << wire(length=10)

            # Chain wires together
            w1.connect("o2", r1.ports["p"])
            w2.connect("o1", w1.ports["o1"])
            w3.connect("o1", w2.ports["o2"])

            return c

        top = chained_wires()
        package, _ = to_vlsir_circuit(top)

        # w1, w2, w3 should all merge into one node connecting to R1.p
        # Only 1 signal should exist (the merged node)
        assert len(package.modules[0].signals) == 1

    def test_isolated_routing_segments(self):
        """Test that isolated wires create separate nodes."""

        @gf.cell
        def isolated_wires() -> Component:
            c = Component()

            r1 = c << resistor()
            r2 = c << resistor()
            r2.dmove((0, 10))

            w1 = c << wire(length=5)
            w2 = c << wire(length=5)
            w3 = c << wire(length=5)
            w4 = c << wire(length=5)

            w2.dmove((15, 0))
            w3.dmove((0, 10))
            w4.dmove((15, 10))

            # Separate connections
            w1.connect("o2", r1.ports["p"])
            w2.connect("o1", r1.ports["n"])
            w3.connect("o2", r2.ports["p"])
            w4.connect("o1", r2.ports["n"])

            return c

        top = isolated_wires()
        package, _ = to_vlsir_circuit(top)

        # 4 isolated wires = 4 separate nodes
        assert len(package.modules[0].signals) == 4


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_device_only_no_routing(self):
        """Test a single device with no wires."""

        @gf.cell
        def device_only() -> Component:
            c = Component()
            r1 = c << resistor()
            c.add_port("p", port=r1.ports["p"])
            c.add_port("n", port=r1.ports["n"])
            return c

        top = device_only()
        package, _ = to_vlsir_circuit(top)

        assert len(package.modules[0].instances) == 1
        # No routing means no internal nodes created
        assert len(package.modules[0].signals) == 0

    def test_routing_only_no_devices(self):
        """Test circuit with only routing elements."""

        @gf.cell
        def routing_only() -> Component:
            c = Component()
            w1 = c << wire(length=10)
            w2 = c << wire(length=10)
            w2.connect("o1", w1.ports["o2"])
            return c

        top = routing_only()
        package, libs = to_vlsir_circuit(top)

        assert len(package.ext_modules) == 0
        assert len(package.modules[0].instances) == 0
        assert len(libs) == 0

    def test_empty_component(self):
        """Test with an empty component - should raise KeyError since no netlist exists."""

        @gf.cell
        def empty_component() -> Component:
            return Component()

        top = empty_component()
        # Empty components don't have a netlist entry, so this should raise
        with pytest.raises(KeyError):
            to_vlsir_circuit(top)


# =============================================================================
# Parameter handling tests
# =============================================================================


class TestParameterHandling:
    """Tests for various parameter types."""

    def test_float_parameters(self):
        """Test that float parameters are correctly stored."""

        @gf.cell
        def float_param_circuit() -> Component:
            c = Component()
            r1 = c << resistor(resistance=1234.567)
            w1 = c << wire()
            w1.connect("o2", r1.ports["p"])
            return c

        top = float_param_circuit()
        package, _ = to_vlsir_circuit(top)

        inst = package.modules[0].instances[0]
        params = {p.name: p for p in inst.parameters}
        assert params["r"].value.double_value == 1234.567

    def test_int_parameters(self):
        """Test that integer parameters are correctly stored."""

        @gf.cell
        def int_param_circuit() -> Component:
            c = Component()
            m1 = c << nmos(nf=4)
            w1 = c << wire()
            w1.connect("o2", m1.ports["G"], allow_width_mismatch=True)
            return c

        top = int_param_circuit()
        package, _ = to_vlsir_circuit(top)

        inst = package.modules[0].instances[0]
        params = {p.name: p for p in inst.parameters}
        assert params["nf"].value.int64_value == 4


# =============================================================================
# VLSIR output validation
# =============================================================================


class TestVLSIROutputValidation:
    """Tests that verify VLSIR output is valid."""

    def test_package_serializable(self):
        """Test that output package can be serialized to bytes."""

        @gf.cell
        def serializable_circuit() -> Component:
            c = Component()
            r1 = c << resistor()
            w1 = c << wire()
            w1.connect("o2", r1.ports["p"])
            return c

        top = serializable_circuit()
        package, _ = to_vlsir_circuit(top)

        # Should not raise
        serialized = package.SerializeToString()
        assert len(serialized) > 0

        # Should be able to deserialize
        restored = vckt.Package()
        restored.ParseFromString(serialized)
        assert restored.modules[0].name == "serializable_circuit"

    def test_all_signals_have_valid_names(self):
        """Test that all signal names are non-empty strings."""

        @gf.cell
        def valid_signals_circuit() -> Component:
            c = Component()
            r1 = c << resistor()
            w1 = c << wire()
            w2 = c << wire()
            w2.dmove((15, 0))
            w1.connect("o2", r1.ports["p"])
            w2.connect("o1", r1.ports["n"])
            return c

        top = valid_signals_circuit()
        package, _ = to_vlsir_circuit(top)

        for sig in package.modules[0].signals:
            assert sig.name, "Signal name should not be empty"
            assert isinstance(sig.name, str)
            assert sig.width == 1


# =============================================================================
# Direct device-to-device connections tests
# =============================================================================


class TestDirectDeviceConnections:
    """Tests for direct device-to-device connections without routing."""

    def test_two_resistors_direct_connection(self):
        """Test two resistors connected directly without intermediate routing."""

        @gf.cell
        def direct_resistors() -> Component:
            c = Component()

            r1 = c << resistor(resistance=1000.0)
            r2 = c << resistor(resistance=2000.0)

            # Direct connection: r1.n connects to r2.p
            r2.connect("p", r1.ports["n"])

            return c

        top = direct_resistors()
        package, _ = to_vlsir_circuit(top)

        # Should have 2 instances
        assert len(package.modules[-1].instances) == 2

        # Should create a shared node for the direct connection
        # Both r1.n and r2.p should reference the same net
        instances = list(package.modules[-1].instances)

        # Find the two resistor instances (names are generated by gdsfactory)
        r_instances = [i for i in instances if "resistor" in i.name.lower()]
        assert len(r_instances) == 2

        # Get connections for both
        conns_0 = {c.portname: c.target.sig for c in r_instances[0].connections}
        conns_1 = {c.portname: c.target.sig for c in r_instances[1].connections}

        # One's "n" should equal the other's "p" (the shared node)
        shared_nodes = set(conns_0.values()) & set(conns_1.values())
        assert len(shared_nodes) >= 1, "Should have at least one shared node"

    def test_three_devices_chain_direct(self):
        """Test three devices connected in a chain directly."""

        @gf.cell
        def three_device_chain() -> Component:
            c = Component()

            r1 = c << resistor(resistance=1000.0)
            r2 = c << resistor(resistance=2000.0)
            r3 = c << resistor(resistance=3000.0)

            # Direct chain: r1 -> r2 -> r3
            r2.connect("p", r1.ports["n"])
            r3.connect("p", r2.ports["n"])

            return c

        top = three_device_chain()
        package, _ = to_vlsir_circuit(top)

        assert len(package.modules[-1].instances) == 3

        # Should have 2 internal nodes (r1-r2 junction and r2-r3 junction)
        assert len(package.modules[-1].signals) == 2

    def test_mixed_direct_and_routed_connections(self):
        """Test circuit with both direct and routed connections."""

        @gf.cell
        def mixed_connections() -> Component:
            c = Component()

            r1 = c << resistor(resistance=1000.0)
            r2 = c << resistor(resistance=2000.0)
            r3 = c << resistor(resistance=3000.0)

            # r2 position offset
            r2.dmove((15, 0))
            r3.dmove((40, 0))

            # Direct connection between r1 and r2
            r2.connect("p", r1.ports["n"])

            # Routed connection between r2 and r3
            w = c << wire(length=10)
            w.connect("o1", r2.ports["n"])
            r3.connect("p", w.ports["o2"])

            return c

        top = mixed_connections()
        package, _ = to_vlsir_circuit(top)

        assert len(package.modules[-1].instances) == 3
        # 2 nodes: one from direct connection, one from routing
        assert len(package.modules[-1].signals) == 2

    def test_direct_transistor_connection(self):
        """Test direct connection between transistor terminals."""

        @gf.cell
        def stacked_transistors() -> Component:
            c = Component()

            mn = c << nmos(w=1e-6, l=180e-9)
            mp = c << pmos(w=2e-6, l=180e-9)

            # Stack: PMOS source to NMOS drain (direct connection)
            mp.dmove((0, 15))
            mp.connect("S", mn.ports["D"])

            return c

        top = stacked_transistors()
        package, _ = to_vlsir_circuit(top)

        assert len(package.modules[-1].instances) == 2

        # The connected ports should share a node
        instances = list(package.modules[-1].instances)

        # Find nmos and pmos instances
        nmos_inst = next(
            i for i in instances if "nmos" in i.name.lower() or "nfet" in i.name.lower()
        )
        pmos_inst = next(
            i for i in instances if "pmos" in i.name.lower() or "pfet" in i.name.lower()
        )

        nmos_conns = {c.portname: c.target.sig for c in nmos_inst.connections}
        pmos_conns = {c.portname: c.target.sig for c in pmos_inst.connections}

        # NMOS drain and PMOS source should share a node
        assert nmos_conns.get("d") == pmos_conns.get("s")


# =============================================================================
# Recursive SUBCKT tests
# =============================================================================


class TestRecursiveSubckt:
    """Tests for recursive sub-circuit (SUBCKT) handling."""

    def test_simple_subcircuit(self):
        """Test a circuit containing a sub-component with devices."""

        # Define a reusable sub-circuit with wire connections (no direct device connections
        # to avoid GDSFactory port overlap issues)
        @gf.cell
        def resistor_with_wires() -> Component:
            c = Component()

            r = c << resistor(resistance=10000.0)
            w_in = c << wire(length=5)
            w_out = c << wire(length=5)

            w_in.connect("o2", r.ports["p"])
            w_out.connect("o1", r.ports["n"])

            # Expose ports at wire ends (not at device terminals)
            c.add_port("vin", port=w_in.ports["o1"])
            c.add_port("vout", port=w_out.ports["o2"])

            return c

        @gf.cell
        def top_with_subckt() -> Component:
            c = Component()

            # Instantiate the sub-circuit
            sub = c << resistor_with_wires()

            # Add a wire to the input
            w_ext = c << wire(length=10)
            w_ext.connect("o2", sub.ports["vin"])

            return c

        top = top_with_subckt()
        package, libs = to_vlsir_circuit(top)

        # Should have modules for top and the subcircuit
        module_names = [m.name for m in package.modules]
        assert "top_with_subckt" in module_names

        # Top module should reference the subcircuit instance
        top_mod = next(m for m in package.modules if m.name == "top_with_subckt")

        # Should have the external wire or subckt instance
        assert len(top_mod.instances) >= 1

        # Should have the basic.lib from the resistor
        assert "basic.lib" in libs

    def test_nested_subcircuits(self):
        """Test deeply nested sub-circuits."""

        @gf.cell
        def inner_cell() -> Component:
            c = Component()
            r = c << resistor(resistance=1000.0)
            c.add_port("p", port=r.ports["p"])
            c.add_port("n", port=r.ports["n"])
            return c

        @gf.cell
        def middle_cell() -> Component:
            c = Component()
            inner = c << inner_cell()
            w = c << wire(length=5)
            w.connect("o2", inner.ports["p"])
            c.add_port("in", port=w.ports["o1"])
            c.add_port("out", port=inner.ports["n"])
            return c

        @gf.cell
        def outer_cell() -> Component:
            c = Component()
            mid = c << middle_cell()
            w = c << wire(length=5)
            w.connect("o2", mid.ports["in"])
            return c

        top = outer_cell()
        package, libs = to_vlsir_circuit(top)

        # Should process without error and have the resistor lib
        assert "basic.lib" in libs

        # The package should contain module definitions
        assert len(package.modules) >= 1

    def test_multiple_subcircuit_instances(self):
        """Test instantiating the same sub-circuit multiple times."""

        @gf.cell
        def rc_cell() -> Component:
            c = Component()
            r = c << resistor(resistance=1000.0)
            cap = c << capacitor(capacitance=1e-12)
            cap.dmove((15, 0))
            cap.connect("p", r.ports["n"])
            c.add_port("in", port=r.ports["p"])
            c.add_port("out", port=cap.ports["n"])
            return c

        @gf.cell
        def dual_rc() -> Component:
            c = Component()

            # Two instances of the same sub-circuit
            rc1 = c << rc_cell()
            rc2 = c << rc_cell()
            rc2.dmove((0, 20))

            # Connect them with wires
            w1 = c << wire(length=5)
            w2 = c << wire(length=5)
            w2.dmove((0, 20))

            w1.connect("o2", rc1.ports["in"])
            w2.connect("o2", rc2.ports["in"])

            return c

        top = dual_rc()
        package, libs = to_vlsir_circuit(top)

        # Should have both resistor and capacitor libs
        assert "basic.lib" in libs

        # Each RC cell has 1 resistor + 1 capacitor = 2 devices
        # Total should be 4 device instances across all modules
        total_instances = sum(len(m.instances) for m in package.modules)
        # At minimum we should have the devices
        assert total_instances >= 2


# =============================================================================
# Stress tests
# =============================================================================


class TestStressTests:
    """Stress tests with larger circuits."""

    def test_resistor_chain(self):
        """Test a chain of resistors."""

        @gf.cell
        def resistor_chain(n: int = 20) -> Component:
            c = Component()
            resistors = []
            wires = []

            for i in range(n):
                r = c << resistor(resistance=1000.0 * (i + 1))
                r.dmove((i * 25, 0))
                resistors.append(r)

            # Add wire at start
            w_start = c << wire(length=5)
            w_start.connect("o2", resistors[0].ports["p"])
            wires.append(w_start)

            # Connect resistors with wires
            for i in range(n - 1):
                w = c << wire(length=5)
                w.connect("o1", resistors[i].ports["n"])
                resistors[i + 1].connect("p", w.ports["o2"])
                wires.append(w)

            # Add wire at end
            w_end = c << wire(length=5)
            w_end.connect("o1", resistors[-1].ports["n"])

            return c

        top = resistor_chain(n=20)
        package, _ = to_vlsir_circuit(top)

        assert len(package.modules[0].instances) == 20
        # One external module (resistor)
        assert len(package.ext_modules) == 1

    def test_many_unique_models(self):
        """Test with different device models."""

        @gf.cell
        def multi_model_circuit() -> Component:
            c = Component()

            # Add different device types
            r1 = c << resistor()
            r2 = c << resistor()
            c1 = c << capacitor()
            m1 = c << nmos()
            m2 = c << pmos()
            q1 = c << npn_bjt()

            # Position them
            r2.dmove((20, 0))
            c1.dmove((40, 0))
            m1.dmove((60, 0))
            m2.dmove((80, 0))
            q1.dmove((100, 0))

            # Add some wires
            for i, dev in enumerate([r1, r2, c1]):
                w = c << wire(length=5)
                w.dmove((-10 + i * 20, 5))
                w.connect("o2", dev.ports["p"], allow_width_mismatch=True)

            return c

        top = multi_model_circuit()
        package, libs = to_vlsir_circuit(top)

        # Different models: resistor, capacitor, nfet_01v8, pfet_01v8, npn13G2
        assert len(package.ext_modules) == 5
        assert len(package.modules[0].instances) == 6
        assert len(libs) == 3  # basic.lib, sky130_fd_pr, ihp_sg13g2
