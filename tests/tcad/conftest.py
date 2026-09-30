"""Shared fixtures for the gsim.tcad tests.

The hermetic tests never import the real DEVSIM: a recording fake is
injected into ``sys.modules`` so device/mesh/doping setup can be asserted
without the solver runtime, mirroring how palace-binary tests avoid the
Palace binary.
"""

from __future__ import annotations

import sys
import types

import gdsfactory as gf
import pytest

from tests._helpers import draw_pn_rib


class FakeDevsim(types.ModuleType):
    """Recording stand-in for the ``devsim`` module.

    Records every call as ``(name, kwargs)`` and returns canned values:
    node coordinates come from ``node_coords`` (cm), and the contact charge
    is ``charge_per_volt * bias`` of the contact's bias parameter, so the
    small-signal capacitance of the fake device is exactly
    ``charge_per_volt``.
    """

    def __init__(self):
        super().__init__("devsim")
        self.calls: list[tuple[str, dict]] = []
        self.node_coords = {
            "x": [0.0, 1e-4, 2e-4],
            "y": [0.0, 0.5e-4, 1e-4],
        }
        self.node_values: dict[tuple[str, str], list[float]] = {}
        self.parameters: dict[str, float] = {}
        self.circuit: dict[str, float] = {}
        self.last_ac_frequency = 0.0
        self.charge_per_volt = 2.5e-12

    def _record(self, _call_name, **kwargs):
        self.calls.append((_call_name, kwargs))

    def called(self, call_name):
        return [kwargs for cname, kwargs in self.calls if cname == call_name]

    # -- mesh / device -------------------------------------------------
    def create_gmsh_mesh(self, **kwargs):
        self._record("create_gmsh_mesh", **kwargs)

    def add_gmsh_region(self, **kwargs):
        self._record("add_gmsh_region", **kwargs)

    def add_gmsh_contact(self, **kwargs):
        self._record("add_gmsh_contact", **kwargs)

    def add_gmsh_interface(self, **kwargs):
        self._record("add_gmsh_interface", **kwargs)

    def create_interface_from_nodes(self, **kwargs):
        self._record("create_interface_from_nodes", **kwargs)

    def interface_model(self, **kwargs):
        self._record("interface_model", **kwargs)

    def interface_equation(self, **kwargs):
        self._record("interface_equation", **kwargs)

    def finalize_mesh(self, **kwargs):
        self._record("finalize_mesh", **kwargs)

    def create_device(self, **kwargs):
        self._record("create_device", **kwargs)

    def delete_circuit(self, **kwargs):
        self._record("delete_circuit", **kwargs)
        self.circuit.clear()

    def delete_device(self, **kwargs):
        self._record("delete_device", **kwargs)

    def delete_mesh(self, **kwargs):
        self._record("delete_mesh", **kwargs)

    # -- models / solutions --------------------------------------------
    def node_solution(self, **kwargs):
        self._record("node_solution", **kwargs)

    def set_node_values(self, **kwargs):
        self._record("set_node_values", **kwargs)
        if "values" in kwargs:
            key = (kwargs["region"], kwargs["name"])
            self.node_values[key] = list(kwargs["values"])

    def node_model(self, **kwargs):
        self._record("node_model", **kwargs)

    def edge_average_model(self, **kwargs):
        self._record("edge_average_model", **kwargs)

    def set_parameter(self, **kwargs):
        self._record("set_parameter", **kwargs)
        self.parameters[kwargs["name"]] = kwargs["value"]

    def solve(self, **kwargs):
        self._record("solve", **kwargs)
        # DEVSIM reports every Newton iteration on sys.stdout.
        print(f"Iteration: {kwargs.get('type')}")
        if kwargs.get("type") == "ac":
            self.last_ac_frequency = float(kwargs["frequency"])

    # -- circuit -------------------------------------------------------
    def circuit_element(self, **kwargs):
        self._record("circuit_element", **kwargs)
        self.circuit[kwargs["name"]] = float(kwargs.get("value", 0.0))

    def circuit_alter(self, **kwargs):
        self._record("circuit_alter", **kwargs)
        if kwargs.get("param") in (None, "value"):
            self.circuit[kwargs["name"]] = float(kwargs["value"])

    def get_circuit_node_value(self, **kwargs):
        self._record("get_circuit_node_value", **kwargs)
        if kwargs.get("solution") == "dcop":
            return 0.0
        # Unit-amplitude AC source on a linear capacitor charge_per_volt:
        # Im(I) = 2 pi f C.
        import numpy as np

        return 2.0 * np.pi * self.last_ac_frequency * self.charge_per_volt

    # -- readback ------------------------------------------------------
    def get_node_model_values(self, **kwargs):
        self._record("get_node_model_values", **kwargs)
        name = kwargs["name"]
        if name in self.node_coords:
            return list(self.node_coords[name])
        key = (kwargs["region"], name)
        if key in self.node_values:
            return list(self.node_values[key])
        return [0.0] * len(self.node_coords["x"])

    def get_contact_current(self, **kwargs):
        self._record("get_contact_current", **kwargs)
        return 0.0

    def get_contact_charge(self, **kwargs):
        self._record("get_contact_charge", **kwargs)
        bias = self.parameters.get(f"{kwargs['contact']}_bias", 0.0)
        return self.charge_per_volt * bias


class FakeSimplePhysics(types.ModuleType):
    """Recording stand-in for ``devsim.python_packages.simple_physics``."""

    def __init__(self):
        super().__init__("devsim.python_packages.simple_physics")
        self.calls: list[tuple[str, tuple]] = []

    def called(self, name):
        return [args for cname, args in self.calls if cname == name]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "GetContactBiasName":
            return lambda contact: f"{contact}_bias"
        if name == "CreateContinuousInterfaceModel":

            def _record_model(device, interface, variable):
                self.calls.append((name, (device, interface, variable)))
                return f"continuous{variable}"

            return _record_model

        def _record(*args):
            self.calls.append((name, args))

        return _record


@pytest.fixture
def fake_devsim(monkeypatch):
    """Install the fake devsim modules and return (devsim, simple_physics)."""
    devsim = FakeDevsim()
    packages = types.ModuleType("devsim.python_packages")
    sp = FakeSimplePhysics()
    monkeypatch.setitem(sys.modules, "devsim", devsim)
    monkeypatch.setitem(sys.modules, "devsim.python_packages", packages)
    monkeypatch.setitem(sys.modules, "devsim.python_packages.simple_physics", sp)
    # The devices gsim holds are process-wide: one an earlier test left
    # behind would be released into this test's recorder.
    monkeypatch.setattr("gsim.tcad.sim._LIVE_DEVICES", set())
    return devsim, sp


def build_padded_diode(
    *,
    center_y: float = -20.0,
    half_width: float = 0.2,
    pad_width: float = 0.2,
    zmax: float = 0.22,
):
    """Four-region PN diode: n_pad | n_rib | p_rib | p_pad along y.

    The ohmic contacts sit on the outer pads, well away from the
    metallurgical junction, so the junction electrostatics are not
    distorted by the equilibrium clamp of the contact boundary condition.

    Returns:
        ``(comp, stack, names)`` with ``names`` mapping roles to region
        names.
    """

    from gsim.common.cross_section import build_doped_cross_section
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import make_doped_materials

    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, 0.4), centered=True, layer=(1, 0))
    wg.y = center_y
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    spans = {
        "n_pad": (center_y - half_width - pad_width, center_y - half_width),
        "n_rib": (center_y - half_width, center_y),
        "p_rib": (center_y, center_y + half_width),
        "p_pad": (center_y + half_width, center_y + half_width + pad_width),
    }
    layer_specs = {}
    for i, (name, (y0, y1)) in enumerate(spans.items()):
        gds_layer = (30, i)
        rect = comp << gf.c.rectangle((10.0, y1 - y0), layer=gds_layer)
        rect.y = (y0 + y1) / 2
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=zmax,
            thickness=zmax,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    materials = make_doped_materials(
        [(name, 1.6e3) for name in spans], permittivity=11.9
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )
    return comp, stack, list(spans)


def build_pn_device():
    """Rib with adjacent P/N doped regions on a doped cross-section stack."""
    return draw_pn_rib()
