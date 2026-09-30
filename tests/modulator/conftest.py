"""Shared device fixtures for the gsim.modulator tests.

The device is the lateral PN phase shifter ``demo_phase_shifter`` draws:
a rib with four doped regions (n_pad | n_rib | p_rib | p_pad) along the
in-plane axis and a metal electrode landing on each outer pad. Its drawn
dimensions are re-exported under the names the assertions were written
against, read off the builder rather than restated here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.modes import LineReading
from gsim.modulator.demo import (
    DEFAULT_CENTER_UM,
    DEFAULT_ELECTRODE_THICKNESS_UM,
    DEFAULT_HALF_WIDTH_UM,
    DEFAULT_LENGTH_UM,
    DEFAULT_PAD_WIDTH_UM,
    DEFAULT_RIB_HEIGHT_UM,
    demo_phase_shifter,
)
from gsim.modulator.route import ROUTES, Route
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

CENTER_Y = DEFAULT_CENTER_UM
HALF_WIDTH = DEFAULT_HALF_WIDTH_UM
PAD_WIDTH = DEFAULT_PAD_WIDTH_UM
RIB_HEIGHT = DEFAULT_RIB_HEIGHT_UM
ELECTRODE_THICKNESS = DEFAULT_ELECTRODE_THICKNESS_UM
LENGTH_UM = DEFAULT_LENGTH_UM

#: What either RF Route says when the Mode it selected is the wall Mode:
#: femwell off its two electrode currents, Palace off its gap voltage
#: (ADR 0005).
WALL_MODE_WARNING = "window wall"


@pytest.fixture(autouse=True)
def _generic_pdk_active() -> None:
    """Ensure the generic PDK is active for each test.

    Activation happens before every test rather than only at collection
    time: Palace tests (e.g. the IHP mesh-regression suite) switch the
    active PDK during execution and never restore it, so module-level
    activation alone lets the IHP PDK leak into later modulator tests
    under the random test ordering used by ``pytest-randomly``.
    """
    gf.gpdk.PDK.activate()


def build_demo():
    """Draw the lateral PN phase shifter the tests are written against."""
    return demo_phase_shifter()


@pytest.fixture(scope="module")
def demo():
    """The demo phase shifter, drawn once per module."""
    return build_demo()


@pytest.fixture(scope="module")
def phase_shifter(demo):
    """The lateral PN phase shifter component and its doped stack."""
    return demo.component, demo.stack


@pytest.fixture
def device(demo):
    """The device description matching what :func:`build_demo` drew."""
    return demo.device


@pytest.fixture
def study(phase_shifter, device, tmp_path):
    """A Study over the phase shifter, writing meshes into ``tmp_path``."""
    from gsim.modulator import Study

    component, stack = phase_shifter
    return Study(
        component=component,
        stack=stack,
        device=device,
        plane="x=0",
        output_dir=tmp_path / "study",
    )


#: In-plane extent of the doped slab, pads included (um).
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)


def carriers_at(bias_v: float) -> CarrierMap:
    """A Carrier map across the doped slab, depleting with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 61)
    z = np.linspace(0.0, RIB_HEIGHT, 5)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()
    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(depleted | ~n_side, 1e10, 1e18),
        holes_cm3=np.where(depleted | n_side, 1e10, 1e18),
    )


@pytest.fixture
def biased(study):
    """A Study whose charge Stage already holds a two-point sweep."""
    study.charge.seed(
        BiasSweepResult(
            contact="cathode",
            points=[BiasPoint(bias_v=v, carriers=carriers_at(v)) for v in (0.0, 2.0)],
        )
    )
    return study


#: Frequency the canned small-signal admittances are fitted at (Hz).
JUNCTION_FREQ_HZ = 1e9

#: Canned series-RC junction branch per bias: (bias_v, r_s_ohm_m, c_j_f_per_m).
JUNCTION_BRANCH = [
    (0.0, 1.2e-4, 3.3e-10),
    (1.0, 1.1e-4, 2.8e-10),
    (2.0, 1.0e-4, 2.4e-10),
]


def admittance_point(bias_v: float, r_s_ohm_m: float, c_j_f_per_m: float) -> BiasPoint:
    """A Bias point whose admittance is exactly the given series RC."""
    omega = 2.0 * np.pi * JUNCTION_FREQ_HZ
    y_per_m = 1.0 / (r_s_ohm_m - 1j / (omega * c_j_f_per_m))
    return BiasPoint(
        bias_v=bias_v,
        carriers=carriers_at(bias_v),
        admittance_s_per_cm=y_per_m / 1e2,
        admittance_freq_hz=JUNCTION_FREQ_HZ,
    )


def junction_sweep() -> BiasSweepResult:
    """A canned sweep carrying the series-RC admittances per Bias point."""
    return BiasSweepResult(
        contact="cathode",
        points=[admittance_point(*values) for values in JUNCTION_BRANCH],
    )


class FakeRoute(Route):
    """A scripted Route: solves nothing, answers what the test told it.

    Registered under the ``"palace"`` name by the ``fake_route`` fixture,
    so a Stage configured with ``route="palace"`` reaches it through the
    registry the way it reaches a real route. Every call the Stage
    makes is recorded on the class, because the Stage builds a fresh
    instance per run.
    """

    name: ClassVar = "palace"
    conductor_model: ClassVar = "pec"
    continuous_materials: ClassVar[bool] = False

    #: ``freq_hz -> [n_eff, ...]`` the solve answers with; a single list
    #: answers every frequency alike.
    modes_at: ClassVar[Any] = [3.0 - 0.01j]
    #: The reading of every selected Mode, or a callable of the Mode.
    reading: ClassVar[Any] = None
    #: The boundary-field ratio of every selected Mode.
    boundary: ClassVar[float] = 0.0
    #: The fraction of a Mode outside the Strips.
    outside: ClassVar[float] = 0.0
    #: Every call a Stage made, as ``(method, kwargs)``.
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def reset(cls) -> None:
        """Back to the defaults, with nothing recorded."""
        cls.conductor_model = "pec"
        cls.continuous_materials = False
        cls.modes_at = [3.0 - 0.01j]
        cls.reading = None
        cls.boundary = 0.0
        cls.outside = 0.0
        cls.calls = []

    @classmethod
    def made(cls, method: str) -> list[dict[str, Any]]:
        """The keyword arguments of every recorded call of *method*."""
        return [kwargs for name, kwargs in cls.calls if name == method]

    def _record(self, method: str, **kwargs: Any) -> None:
        type(self).calls.append((method, kwargs))

    def require(self, *, stage_name: str) -> None:
        self._record("require", stage_name=stage_name)

    def check_line_settings(self, **kwargs: Any) -> None:
        self._record("check_line_settings", **kwargs)

    def check_staircase(self, staircase: Any, **kwargs: Any) -> None:
        self._record("check_staircase", staircase=staircase, **kwargs)

    def prepare_line(self, sim: Any, **kwargs: Any) -> None:
        self._record("prepare_line", sim=sim, **kwargs)

    def solve(self, sim: Any, **kwargs: Any) -> list[Any]:
        self._record("solve", sim=sim, **kwargs)
        script = type(self).modes_at
        indices = script(kwargs["freq_hz"]) if callable(script) else script
        return [SimpleNamespace(n_eff=complex(n)) for n in indices]

    def boundary_ratio(self, mode: Any) -> float:
        self._record("boundary_ratio", mode=mode)
        return type(self).boundary

    def strip_fraction_outside(self, mode: Any, span: Any, *, stage_name: str) -> float:
        self._record(
            "strip_fraction_outside", mode=mode, span=span, stage_name=stage_name
        )
        return type(self).outside

    def read_line(self, sim: Any, mode: Any, **kwargs: Any) -> LineReading:
        self._record("read_line", sim=sim, mode=mode, **kwargs)
        reading = type(self).reading
        if callable(reading):
            return reading(mode, kwargs["freq_hz"])
        if reading is not None:
            return reading
        return LineReading(n_eff=mode.n_eff, z0_ohm=50.0 + 0j, wall_mode=False)


@pytest.fixture
def fake_route(monkeypatch):
    """The fake route, registered as the ``"palace"`` Route for one test."""
    FakeRoute.reset()
    monkeypatch.setitem(ROUTES, "palace", FakeRoute)
    return FakeRoute
