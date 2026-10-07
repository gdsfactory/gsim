"""The Palace Route on the real pipeline, and its agreement with femwell.

The cross-Route gate the modulator API owes the originating spec, on both
EM Stages: both first-class Routes solve the identical Staircase on the
identical mesh under the identical outer-wall condition, and must land on
the same effective index — and, on the RF Stage, the same characteristic
impedance — to solver tolerance. It ships here as a runtime-gated test
rather than as a one-off script, so the agreement is re-checked whenever
either Route moves.

What it does not cover: both Routes are handed the *same* Staircase, so
the geometry is identical by construction and the agreement asserted here
says nothing about whether that Staircase is the drawn device. That is
``tests/modulator/test_representation_gate.py``, which compares the
Staircase against the drawn device solved with a continuous ``eps(x, y)``.

Gated on gmsh, the femwell runtime and a Palace binary; deselected by
default, run with ``pytest -m palace_local``. The Carrier maps are
synthetic, so nothing here needs DEVSIM.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.common.modes import NoLineModeError
from gsim.modulator import Device, Study
from gsim.modulator.palace_route import PalaceRoute

from .conftest import (
    CENTER_Y,
    HALF_WIDTH,
    PAD_WIDTH,
    RIB_HEIGHT,
    WALL_MODE_WARNING,
    build_demo,
)

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

pytestmark = pytest.mark.palace_local

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)
# Palace solves order-2 Nedelec elements against femwell's order-1
# Lagrange: the same problem on different element bases, so agreement
# is to the same 1% the shipped Palace/femwell cross-validation pins.
N_EFF_RTOL = 1e-2


def _palace_available() -> bool:
    from gsim.palace.runtime import resolve_palace_binary

    try:
        return resolve_palace_binary() is not None
    except Exception:
        return False


if not _palace_available():  # pragma: no cover - environment dependent
    pytest.skip("Palace binary not available", allow_module_level=True)


def depletion_carriers(bias_v: float) -> CarrierMap:
    """A Carrier map whose depletion region widens with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 121)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(n_side & ~depleted, DOPING_CM3, DEPLETED_CM3),
        holes_cm3=np.where(~n_side & ~depleted, DOPING_CM3, DEPLETED_CM3),
    )


def graded_carriers(bias_v: float = 0.0) -> CarrierMap:
    """A depletion region that widens smoothly with reverse bias.

    A step-like junction is either resolved by a strip edge or not, so the
    staircase error jumps rather than shrinking; a profile that varies
    smoothly across the Junction is the one whose approximation error a
    strip count can actually be said to converge on.
    """
    y = np.linspace(SLAB[0], SLAB[1], 161)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    width = 0.06 * np.sqrt(1.0 + abs(bias_v))
    depletion = 1.0 / (1.0 + np.exp(-(np.abs(yy - CENTER_Y) - width) / 0.04))
    n_side = yy < CENTER_Y
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=DEPLETED_CM3 + DOPING_CM3 * depletion * n_side,
        holes_cm3=DEPLETED_CM3 + DOPING_CM3 * depletion * ~n_side,
    )


def study_at(tmp_path, *, biases=(0.0,), carriers=depletion_carriers) -> Study:
    """A Study whose charge Stage already holds a synthetic Bias sweep."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    study = Study(
        component=component,
        stack=stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
            p_doping_cm3=DOPING_CM3,
            n_doping_cm3=DOPING_CM3,
        ),
        plane="x=0",
        output_dir=tmp_path,
    )
    study.charge.seed(
        BiasSweepResult(
            contact="cathode",
            points=[BiasPoint(bias_v=v, carriers=carriers(v)) for v in biases],
        )
    )
    return study


def optical_n_eff(
    study, *, route: str, n_strips: int, wavelength_um: float = 1.55
) -> complex:
    """Solve the optical Staircase on one Route and return its index.

    No index guess: each Backend's own default tracks the largest
    permittivity of the Cross-section, which is the guided mode on both.
    A fixed guess picks whichever branch happens to sit nearest it, and
    the two Routes then land on different modes and are compared anyway.
    """
    study.optical(
        route=route,
        n_strips=n_strips,
        num_modes=1,
        wavelength_um=wavelength_um,
    )
    return complex(study.optical.run().n_eff[0])


class TestCrossRouteAgreement:
    # 1.55 um is where the default plasma-dispersion coefficients were
    # fitted; 1.31 um is not, and the Staircase used to read its
    # extinction off the fit rather than off the solve, so the Routes
    # could only be trusted to agree at the first of these.
    @pytest.mark.parametrize("wavelength_um", [1.55, 1.31])
    def test_the_optical_routes_agree_on_the_same_staircase(
        self, tmp_path, wavelength_um
    ):
        """Identical Strips, identical mesh: one effective index, two solvers."""
        study = study_at(tmp_path)
        femwell = optical_n_eff(
            study, route="femwell", n_strips=4, wavelength_um=wavelength_um
        )
        palace = optical_n_eff(
            study, route="palace", n_strips=4, wavelength_um=wavelength_um
        )

        # Both must see the guided silicon mode, not the cladding.
        assert femwell.real > 2.0
        assert palace.real > 2.0
        assert abs(palace.real - femwell.real) < N_EFF_RTOL * abs(femwell.real)
        # The staircase carries free-carrier absorption, so both are lossy
        # in the same direction (exp(+i omega t): Im{n_eff} < 0).
        assert palace.imag < 0.0
        assert femwell.imag < 0.0

    def test_the_route_does_not_change_the_result_type(self, tmp_path):
        study = study_at(tmp_path, biases=(0.0, 2.0))
        study.optical(route="palace", n_strips=3, num_modes=1, n_guess=2.5)
        sweep = study.optical.run()

        assert [point.bias_v for point in sweep.points] == [0.0, 2.0]
        assert sweep.reference_bias_v == 0.0
        assert sweep.index_shift[0] == 0.0
        # Depleting the junction removes free carriers, which raises the index.
        assert sweep.index_shift[1] > 0.0
        # Palace reports no mode fields, so containment is not measurable.
        assert np.isnan(sweep.points[0].boundary_field_ratio)

    def test_the_route_says_it_cannot_check_the_window(self, tmp_path):
        """ADR 0002's containment guard cannot run on Palace, and says so."""
        study = study_at(tmp_path)
        study.optical(route="palace", n_strips=3, num_modes=1, n_guess=2.5)
        with pytest.warns(UserWarning, match="cannot check window containment"):
            study.optical.run()


#: The Bias sweep the RF gate's Study holds, and the point it is solved
#: at: the sweep's last. At 0 V the undepleted 1e18 Strips make the
#: loaded line an RC slow wave (``n_eff = 31.9 - 31.9j``) that the loss
#: bound rightly drops, and the slowest candidate left is the Mode
#: between both electrodes together and the metallic Window wall — a
#: real Mode of the shielded line, on which the Routes agree just as
#: well, but not the Traveling-wave electrode's (ticket 23). At 4 V the
#: Staircase's middle Strip is depleted and the line Mode is back inside
#: the bound: ``n_eff = 2.9015 - 0.0014j``, ``Z0`` ~ 41 ohm.
RF_GATE_BIASES = (0.0, 4.0)

#: The guess the gate aims the eigenvalue search at: the loaded line's
#: index, which is also the Stage's own default, rather than the wall
#: Mode's 2.03.
RF_GATE_N_GUESS = 3.0


def rf_line(study, **settings):
    """Solve the RF Stage once and return its line parameters."""
    study.rf(
        frequencies_hz=[10e9],
        n_strips=3,
        num_modes=4,
        n_guess=RF_GATE_N_GUESS,
        **settings,
    )
    return study.rf.run()


def rf_gate_study(output_dir) -> Study:
    """The gate's Study: the depleted Bias point of :data:`RF_GATE_BIASES`."""
    return study_at(output_dir, biases=RF_GATE_BIASES)


class TestRFElectrodeModel:
    """What the Palace Route can express of the electrode metal (ADR 0003)."""

    def test_a_metal_region_leaves_palace_no_line_mode_to_find(self, tmp_path):
        """A region with ``|Im(eps)| ~ 1e7`` returns its own modes.

        This is why the Palace Route does not default to the model the
        femwell Route does. What the search returns depends on where it
        is aimed: at the wall Mode's index every candidate comes back
        losing far more than it advances and the selection refuses them
        all; at the loaded line's index (the gate's guess) it also
        returns the wall Mode carrying the metal's own loss
        (``n_eff = 2.23 - 0.66j``), inside the loss bound, and the
        Route's gap-voltage check names it. Either way, no line Mode.
        """
        study = study_at(tmp_path)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                rf_line(study, route="palace", conductor_model="volume")
            except NoLineModeError:
                return
        assert [w for w in record if WALL_MODE_WARNING in str(w.message)]


# The Marks-Williams integral is exact to 0.4% on the analytic PEC coax,
# but here it is run on two different discretizations of the fields — an
# order-2 Nedelec solve read back off ParaView nodes against an order-2
# Lagrange curl — so the impedance gate is looser than the index one.
Z0_RTOL = 0.05


@pytest.fixture(scope="module")
def rf_gate(tmp_path_factory):
    """One Staircase, one mesh spec, both Routes, one line Mode each.

    Both Routes express the identical Cross-section: perfect-conductor
    electrodes (ADR 0003) inside a metallic Window, femwell applying its
    boundary condition and Palace putting the same wall under
    ``Boundaries.PEC``. The Study is solved at the depleted Bias point
    of :data:`RF_GATE_BIASES`, so that the Mode both Routes select is
    the line Mode between the electrodes and not the shielded line's
    other Mode against the wall (ADR 0005); every warning either Route
    raised is kept, so the gate can check that neither said otherwise.
    One solve of each Route is shared across the gate's assertions,
    because Palace takes tens of seconds per frequency.
    """
    study = rf_gate_study(tmp_path_factory.mktemp("rf_gate"))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        femwell = rf_line(study, route="femwell", conductor_model="pec", order=2)
        palace = rf_line(study, route="palace")
    return femwell, palace, [str(w.message) for w in record]


class TestRFCrossRouteAgreement:
    """The RF Stage's cross-Route numeric gate (ticket 17)."""

    def test_both_routes_select_the_line_mode_not_the_wall_mode(self, rf_gate):
        """What the gate compares is the Traveling-wave electrode's Mode.

        Each Route has its own way of telling the two apart — femwell
        the balance of its two electrode currents, Palace the voltage
        its gap carries against the power — and neither objected.
        """
        _, palace, messages = rf_gate
        assert not [m for m in messages if WALL_MODE_WARNING in m]
        # Palace's diagnostic runs only when its impedance tables were
        # read, and a table reading is real; so a real impedance is what
        # makes the absence of its warning mean something.
        assert palace.z0_ohm[0].imag == 0.0

    def test_both_routes_land_on_the_same_line_mode(self, rf_gate):
        femwell, palace, _ = rf_gate
        n_femwell = float(femwell.n_rf[0])
        n_palace = float(palace.n_rf[0])
        # Both must see the line Mode of the loaded Staircase: not a
        # cladding or box resonance, and not the wall Mode at 2.03.
        assert n_femwell > 2.5
        assert n_palace > 2.5
        assert abs(n_palace - n_femwell) < N_EFF_RTOL * n_femwell

    def test_both_routes_land_on_the_same_impedance(self, rf_gate):
        femwell, palace, _ = rf_gate
        z_femwell = complex(femwell.z0_ohm[0])
        z_palace = complex(palace.z0_ohm[0])
        assert np.isfinite(z_palace.real)
        assert abs(z_palace.real - z_femwell.real) < Z0_RTOL * abs(z_femwell.real)


# Palace's own line integrals against the same integrals run on the fields
# it saved: one solve, two evaluations of ``2P/|I|^2``. The voltage path
# and the loop sample the FE solution exactly where the fallback reads
# ParaView nodes, so the two should sit closer than the two Routes do,
# but the loop hugs the conductor where the field is steepest.
NATIVE_Z0_RTOL = 0.05


@pytest.fixture(scope="module")
def native_solve(tmp_path_factory):
    """The ticket-17 cross-section solved once, both impedance readings kept."""
    from gsim.modulator.palace_route import _palace_hint, solve_palace_modes
    from gsim.palace.line_impedance import field_line_impedance, native_line_impedance
    from gsim.palace.runtime import require_palace_binary

    study = rf_gate_study(tmp_path_factory.mktemp("native_z0"))
    study.rf(
        route="palace",
        frequencies_hz=[10e9],
        n_strips=3,
        num_modes=4,
        n_guess=RF_GATE_N_GUESS,
    )
    stage = study.rf
    staircase = stage.staircase()
    sim = stage.simulation(staircase)
    sim.mesh(**stage.mesh)
    route = PalaceRoute()
    signal, return_ = stage.line_conductors(staircase)
    route.prepare_line(sim, signal=signal, return_=return_, stage_name="rf")
    index = route.impedance_index
    assert index is not None
    solve = solve_palace_modes(
        sim,
        freq_hz=10e9,
        num_modes=4,
        binary=require_palace_binary(hint=_palace_hint("rf")),
        target=RF_GATE_N_GUESS,
        save=4,
    )
    mode, _ratio = stage.select_mode(solve.modes, route, at="f = 10 GHz")
    h_span, v_span = signal.extent
    reading = native_line_impedance(
        solve.results, index=index, mode_id=mode.mode_id, n_eff=mode.n_eff
    )
    native = None if reading is None else reading.z0_ohm
    fields = field_line_impedance(
        sim,
        mode_id=mode.mode_id,
        n_eff=mode.n_eff,
        h_span=h_span,
        v_span=v_span,
        context="The rf stage's palace route",
    )
    return sim.mode_paths, native, fields


class TestNativeImpedance:
    """Ticket 22: Palace's ``mode-Z.csv`` against the field-based fallback."""

    def test_palace_wrote_the_impedance_tables(self, native_solve):
        _, native, _ = native_solve
        assert native is not None
        assert np.isfinite(native.real)
        assert 10.0 < native.real < 1e3

    def test_the_native_and_field_based_impedances_agree(self, native_solve):
        _, native, fields = native_solve
        assert np.isfinite(fields.real)
        assert abs(native.real - fields.real) < NATIVE_Z0_RTOL * abs(fields.real)

    def test_the_stage_reports_the_native_value(self, tmp_path, native_solve):
        """What ``study.rf.run()`` returns on the Palace Route is the table's."""
        _, native, _ = native_solve
        study = rf_gate_study(tmp_path)
        line = rf_line(study, route="palace")
        # Same Cross-section, same settings, a fresh solve: the reading
        # is Palace's, so it is real and lands on the same number.
        assert line.z0_ohm[0].imag == 0.0
        assert line.z0_ohm[0].real == pytest.approx(native.real, rel=1e-3)


class TestStripCountConvergence:
    def test_the_palace_index_shift_converges_in_strip_count(self, tmp_path):
        """Refining the Staircase moves the answer less and less.

        Measured on the index shift across a bias pair, not on the index
        itself. Since the Staircase became the drawn Cross-section with
        its doped silicon binned (ADR 0004), the index is a whole guide's
        worth of material and barely moves with the strip count — the
        residual sits at the level of the mesh noise between one strip
        count and the next. The shift is a sliver's worth, it is what
        ``VpiL`` is computed from, and it is where the binning still
        shows.

        Palace against Palace, so a Staircase converging on the wrong
        answer converges just as neatly. What it converges *to* is
        ``tests/modulator/test_representation_gate.py``'s question.
        """
        study = study_at(tmp_path, biases=(0.0, 4.0), carriers=graded_carriers)

        def shift(n_strips: int) -> float:
            study.optical(route="palace", n_strips=n_strips, num_modes=1)
            return float(study.optical.run().index_shift[-1])

        shifts = {n: shift(n) for n in (1, 2, 4, 8)}
        reference = shifts[8]
        errors = [abs(shifts[n] - reference) for n in (1, 2, 4)]
        assert reference > 0.0
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        # And settling, not merely ordered: four strips land several times
        # closer to the reference than one does.
        assert errors[2] < 0.5 * errors[0]
