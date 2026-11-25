"""The characteristic impedance of a boundary Mode, both ways round.

Sizing the two postprocessing paths from an electrode layout, declaring
them on a solve, reading the impedance back off Palace's own tables, and
falling back to the saved fields when there is no table. None of it
needs a Palace binary: the paths are geometry, and the tables are
whatever ``PalaceTextResults`` was handed.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from gsim.common.modes import Conductor
from gsim.palace import BoundaryModeSim
from gsim.palace.line_impedance import (
    LOOP_STANDOFF_FRACTION,
    declare_impedance_paths,
    line_impedance_paths,
    native_line_impedance,
    palace_line_impedance,
)
from gsim.palace.results import PalaceTextResults

#: A signal electrode, a return electrode and the meshed domain around
#: them, for sizing the impedance paths (um).
SIGNAL = ((-22.6, -20.6), (0.0, 0.5))
GROUND = ((-19.4, -17.4), (0.0, 0.5))
DOMAIN = ((-25.0, -15.0), (-3.0, 2.0))

#: The signal electrode above, as a current integral names it.
SIGNAL_CONDUCTOR = Conductor(name="electrode_low", extent=SIGNAL, model="pec")
#: The same electrode meshed as a region of lossy metal.
LOSSY_SIGNAL_CONDUCTOR = Conductor(name="electrode_low", extent=SIGNAL, model="volume")

#: ``mode -> (Z_PV, Z_VI)`` of the canned ``mode-Z.csv``.
NATIVE_TABLE = {1: (100.0, 80.0), 2: (200.0, 120.0)}


def impedance_paths(**overrides):
    """The paths sized for the electrodes above, with any of them replaced."""
    kwargs = {"signal": SIGNAL, "ground": GROUND, "domain": DOMAIN}
    kwargs.update(overrides)
    return line_impedance_paths(**kwargs)


def impedance_tables(rows=None, *, z_vi: bool = True, index: int = 1):
    """A run's results carrying a ``mode-Z.csv`` under postprocessing *index*."""
    rows = NATIVE_TABLE if rows is None else rows
    table = []
    for m, (z_pv, z_vi_) in rows.items():
        row = {
            "m": str(m),
            f"Z_PV[{index}] (Ohm)": str(z_pv),
            f"L_PV[{index}] (H/m)": "1e-7",
            f"C_PV[{index}] (F/m)": "1e-10",
        }
        if z_vi:
            row[f"Z_VI[{index}] (Ohm)"] = str(z_vi_)
            row[f"L_VI[{index}] (H/m)"] = "1e-7"
            row[f"C_VI[{index}] (F/m)"] = "1e-10"
        table.append(row)
    return PalaceTextResults(
        files={}, csv_tables={"mode-Z.csv": table}, json_data={}, text_data={}
    )


def mode_table(n_modes: int) -> PalaceTextResults:
    """A run's results carrying a ``mode-kn.csv`` and no impedance table."""
    rows = [
        {
            "m": str(m),
            "Re{kn} (1/m)": f"{4.2e7 + m:.6e}",
            "Im{kn} (1/m)": "-1.0e2",
            "Re{n_eff}": f"{2.0 + 0.1 * m:.6e}",
            "Im{n_eff}": "-1.0e-5",
        }
        for m in range(1, n_modes + 1)
    ]
    return PalaceTextResults(
        files={}, csv_tables={"mode-kn.csv": rows}, json_data={}, text_data={}
    )


class FieldlessSim:
    """A run whose saved fields are not there to be read."""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def read_mode_field(self, mode_id):
        raise FileNotFoundError(f"no saved fields for mode {mode_id}")


class TestImpedancePaths:
    """Pure geometry: two electrodes and a domain in, two paths out.

    Palace integrates E along the first and H around the second, so the
    first must run from one electrode face to the other and the second
    must enclose the signal electrode and nothing else.
    """

    def test_the_voltage_path_crosses_the_gap_face_to_face(self):
        paths = impedance_paths()

        (h0, v0), (h1, v1) = paths.voltage
        assert v0 == v1 == pytest.approx(0.25)
        # From the signal's inner face to the return's inner face, on both.
        assert h0 == -20.6
        assert h1 == -19.4

    def test_a_sheets_voltage_path_runs_to_its_edge(self):
        paths = impedance_paths(signal=((-22.6, -20.6), (0.0, 0.0)))

        assert paths.voltage[0] == (-20.6, 0.0)

    def test_a_clearance_fraction_holds_the_ends_off_the_faces(self):
        paths = impedance_paths(clearance_fraction=1e-3)

        (h0, _), (h1, _) = paths.voltage
        # 1e-3 of the smallest clearance, which is the 0.5 um thickness.
        assert h0 == pytest.approx(-20.6 + 0.0005)
        assert h1 == pytest.approx(-19.4 - 0.0005)

    def test_each_end_sits_at_its_own_electrodes_mid_height(self):
        """A return on another metal level is still met on its face."""
        paths = impedance_paths(ground=((-19.4, -17.4), (1.0, 1.5)))

        (_, v0), (_, v1) = paths.voltage
        assert v0 == pytest.approx(0.25)
        assert v1 == pytest.approx(1.25)

    def test_a_return_on_the_low_side_is_crossed_the_other_way(self):
        paths = impedance_paths(signal=GROUND, ground=SIGNAL)

        (h0, _), (h1, _) = paths.voltage
        assert h0 == -19.4
        assert h1 == -20.6

    def test_the_current_loop_stands_off_the_signal_and_clears_everything_else(self):
        paths = impedance_paths()

        h = [p[0] for p in paths.current]
        v = [p[1] for p in paths.current]
        (s0, s1), (t0, t1) = SIGNAL
        # Around the electrode: every corner outside its rectangle...
        assert min(h) < s0
        assert max(h) > s1
        assert min(v) < t0
        assert max(v) > t1
        # ...but well clear of the return and of the domain wall.
        assert max(h) < GROUND[0][0]
        assert min(h) > DOMAIN[0][0]
        assert min(v) > DOMAIN[1][0]
        assert max(v) < DOMAIN[1][1]

    def test_the_loop_stands_off_by_a_fraction_of_the_gap(self):
        """A loop inside the first mesh element misses most of the current."""
        paths = impedance_paths()

        gap = GROUND[0][0] - SIGNAL[0][1]
        standoff = LOOP_STANDOFF_FRACTION * gap
        assert standoff > 0.2 * gap
        (s0, s1), (t0, t1) = SIGNAL
        assert paths.current == (
            (pytest.approx(s0 - standoff), pytest.approx(t0 - standoff)),
            (pytest.approx(s1 + standoff), pytest.approx(t0 - standoff)),
            (pytest.approx(s1 + standoff), pytest.approx(t1 + standoff)),
            (pytest.approx(s0 - standoff), pytest.approx(t1 + standoff)),
        )

    def test_the_loop_standoff_does_not_shrink_with_the_conductor_thickness(self):
        thick = impedance_paths()
        thin = impedance_paths(signal=((-22.6, -20.6), (0.0, 0.01)))

        assert thin.current[1][0] == pytest.approx(thick.current[1][0])
        assert thin.current[0][0] == pytest.approx(thick.current[0][0])

    def test_a_sheet_gets_a_loop_around_its_plane(self):
        paths = impedance_paths(signal=((-22.6, -20.6), (0.0, 0.0)))

        standoff = LOOP_STANDOFF_FRACTION * 1.2
        assert [p[1] for p in paths.current] == pytest.approx(
            [-standoff, -standoff, standoff, standoff]
        )

    def test_the_loop_never_stands_on_the_middle_of_the_gap(self):
        """Mesh vertices tend to sit on an evenly divided gap."""
        assert 0.0 < LOOP_STANDOFF_FRACTION < 0.5
        for divisions in (2, 4, 8, 16):
            assert LOOP_STANDOFF_FRACTION * divisions % 1.0 != 0.0

    def test_a_nearer_conductor_on_the_other_side_limits_the_standoff(self):
        near = ((-23.3, -23.0), (0.0, 0.5))  # 0.4 off the signal's left face
        paths = impedance_paths(others=[near], domain=((-30.0, -15.0), (-3.0, 2.0)))

        left = min(p[0] for p in paths.current)
        assert left > near[0][1]
        assert left == pytest.approx(SIGNAL[0][0] - LOOP_STANDOFF_FRACTION * 0.4)
        assert max(p[0] for p in paths.current) < GROUND[0][0]

    def test_a_diagonal_neighbour_is_cleared_in_both_directions(self):
        above = ((-19.0, -18.0), (0.9, 1.4))
        paths = impedance_paths(others=[above])

        corner = paths.current[2]
        assert not (
            above[0][0] <= corner[0] <= above[0][1]
            and above[1][0] <= corner[1] <= above[1][1]
        )

    def test_an_other_conductor_touching_the_signal_is_refused(self):
        with pytest.raises(ValueError, match="overlaps or touches"):
            impedance_paths(others=[((-20.6, -20.0), (0.0, 0.5))])

    def test_the_loop_fraction_is_adjustable(self):
        paths = impedance_paths(loop_fraction=0.1)

        assert paths.current[1][0] == pytest.approx(SIGNAL[0][1] + 0.12)

    def test_the_declared_solver_input_carries_the_standoff_loop(self):
        """A CPW's generated postprocessing block has the loop off the strip."""
        signal = ((-5.0, 5.0), (0.0, 1.0))
        ground = ((11.0, 511.0), (0.0, 1.0))
        domain = ((-2000.0, 2000.0), (-500.0, 1000.0))
        sim = BoundaryModeSim()
        declare_impedance_paths(
            sim, line_impedance_paths(signal=signal, ground=ground, domain=domain)
        )

        declared = json.loads(json.dumps(sim.mode_postprocessing()))["Impedance"][0]
        corners: list[list[float]] = declared["CurrentPath"]
        voltage: list[list[float]] = declared["VoltagePath"]
        # 0.2531 of the 6 um gap, never a hair of it.
        assert max(c[0] for c in corners) == pytest.approx(5.0 + 0.2531 * 6.0)
        assert max(c[1] for c in corners) == pytest.approx(1.0 + 0.2531 * 6.0)
        assert min(c[1] for c in corners) == pytest.approx(-0.2531 * 6.0)
        assert voltage[0][1] == voltage[1][1] == 0.5

    def test_the_loop_is_closed_by_palace_not_by_repeating_a_point(self):
        """Palace joins the last point back to the first itself."""
        paths = impedance_paths()

        assert len(paths.current) == 4
        assert paths.current[0] != paths.current[-1]

    def test_the_loop_stays_inside_a_tight_domain(self):
        """A wall closer than the gap sets the clearance, not the gap."""
        paths = impedance_paths(domain=((-22.601, -15.0), (-0.001, 2.0)))

        assert min(p[0] for p in paths.current) > -22.601
        assert min(p[1] for p in paths.current) > -0.001

    def test_touching_electrodes_are_refused(self):
        with pytest.raises(ValueError, match="no gap"):
            impedance_paths(ground=((-20.6, -18.6), (0.0, 0.5)))

    def test_electrodes_stacked_over_each_other_are_refused(self):
        with pytest.raises(ValueError, match="no gap"):
            impedance_paths(ground=((-22.0, -20.0), (1.0, 1.5)))

    def test_a_signal_the_domain_clips_is_refused(self):
        """An electrode on or over the wall has no outline to loop around."""
        with pytest.raises(ValueError, match="not inside the meshed domain"):
            impedance_paths(domain=((-22.6, -15.0), (-3.0, 2.0)))


class TestDeclaringThePaths:
    def test_the_paths_become_a_mode_path_of_the_sim(self):
        sim = BoundaryModeSim()
        paths = impedance_paths()

        index = declare_impedance_paths(sim, paths)

        assert index == 1
        (path,) = sim.mode_paths
        assert path.voltage_path == [list(p) for p in paths.voltage]
        assert path.current_path == [list(p) for p in paths.current]
        # Two-dimensional points: cross-section coordinates, not layout.
        assert all(len(p) == 2 for p in path.voltage_path)

    def test_declaring_twice_replaces_rather_than_stacks(self):
        sim = BoundaryModeSim()

        first = declare_impedance_paths(sim, impedance_paths())
        second = declare_impedance_paths(sim, impedance_paths())

        assert len(sim.mode_paths) == 1
        assert first == second == 1

    def test_the_index_is_where_the_sim_put_it(self):
        """A path declared after another is read under its own index."""
        sim = BoundaryModeSim()
        sim.add_impedance_path("probe", voltage=[[-20.0, 0.1], [-20.0, 0.2]])

        assert declare_impedance_paths(sim, impedance_paths()) == 2


class TestNativeImpedance:
    """Reading one Mode's impedance off Palace's own tables."""

    def test_it_is_the_power_current_impedance_of_the_mode_asked_for(self):
        """``Z_VI^2 / Z_PV`` is ``2P/|I|^2``: the definition both Routes use."""
        reading = native_line_impedance(
            impedance_tables(), index=1, mode_id=2, n_eff=2.1 - 1e-5j
        )

        assert reading is not None
        assert reading.z0_ohm == pytest.approx(120.0**2 / 200.0)
        assert reading.n_eff == 2.1 - 1e-5j
        assert reading.wall_mode is False

    def test_it_reads_under_the_index_the_path_was_declared_at(self):
        results = impedance_tables(index=3)
        assert native_line_impedance(results, index=1, mode_id=1, n_eff=2.0) is None
        reading = native_line_impedance(results, index=3, mode_id=1, n_eff=2.0)
        assert reading is not None
        assert reading.z0_ohm == pytest.approx(80.0**2 / 100.0)

    def test_a_gap_carrying_no_voltage_is_reported(self):
        """``|V| |I| << 2P``: the electrodes sit at one potential."""
        reading = native_line_impedance(
            impedance_tables({1: (2.5e-8, 2.1e-3)}),
            index=1,
            mode_id=1,
            n_eff=2.03 - 6e-7j,
        )

        assert reading is not None
        assert reading.wall_mode is True
        assert "between them and the window wall" in reading.diagnostic
        # The impedance itself is still the power-current one.
        assert reading.z0_ohm == pytest.approx(2.1e-3**2 / 2.5e-8)

    def test_a_lossy_line_mode_is_not_reported(self):
        """``Z_PV / Z_PI`` of 0.4 is a lossy line, not a wall mode."""
        reading = native_line_impedance(
            impedance_tables({1: (1.63, 2.62)}), index=1, mode_id=1, n_eff=31.9 - 31.9j
        )
        assert reading is not None
        assert reading.wall_mode is False

    def test_a_loop_enclosing_no_current_is_no_answer(self):
        """``Z_VI = 0`` would make the impedance zero, which is no reading."""
        assert (
            native_line_impedance(
                impedance_tables({1: (100.0, 0.0)}), index=1, mode_id=1, n_eff=2.0
            )
            is None
        )

    def test_no_table_is_no_answer(self):
        assert (
            native_line_impedance(mode_table(2), index=1, mode_id=1, n_eff=2.0) is None
        )

    def test_a_table_without_the_current_is_no_answer(self):
        """``Z_PV`` alone is a different definition, not a fallback."""
        assert (
            native_line_impedance(
                impedance_tables(z_vi=False), index=1, mode_id=1, n_eff=2.0
            )
            is None
        )

    def test_a_mode_the_table_does_not_carry_is_no_answer(self):
        assert (
            native_line_impedance(impedance_tables(), index=1, mode_id=3, n_eff=2.0)
            is None
        )


class TestLossyConductorsReadTheFields:
    """The tables' magnitudes are not a lossy line's impedance.

    Measured on a 10/6 um aluminium CPW at 10 GHz: 85.3 ohm off the
    tables against 74.6 - 8.3j ohm off the fields, so a lossy conductor
    reads the fields and keeps the tables for the wall-Mode diagnostic.
    """

    @staticmethod
    def _fields_say(monkeypatch, z0):
        monkeypatch.setattr(
            "gsim.palace.line_impedance.field_line_impedance",
            lambda *_args, **_kwargs: z0,
        )

    def test_a_lossy_conductor_reads_the_fields_and_keeps_the_diagnostic(
        self, monkeypatch, tmp_path
    ):
        self._fields_say(monkeypatch, 74.6 - 8.3j)

        reading = palace_line_impedance(
            FieldlessSim(tmp_path),
            impedance_tables(),
            index=1,
            mode_id=1,
            n_eff=2.0 - 1e-3j,
            signal=LOSSY_SIGNAL_CONDUCTOR,
            context="The rf stage's palace route",
        )

        assert reading.z0_ohm == 74.6 - 8.3j
        assert reading.wall_mode is False
        assert "saved fields" in reading.diagnostic
        assert "line mode between the electrodes" in reading.diagnostic

    def test_a_lossy_wall_mode_is_still_told_by_the_tables(self, monkeypatch, tmp_path):
        self._fields_say(monkeypatch, 1.0 - 0.1j)

        reading = palace_line_impedance(
            FieldlessSim(tmp_path),
            impedance_tables({1: (2.5e-8, 2.1e-3)}),
            index=1,
            mode_id=1,
            n_eff=2.0,
            signal=LOSSY_SIGNAL_CONDUCTOR,
            context="The rf stage's palace route",
        )

        assert reading.wall_mode is True
        assert reading.z0_ohm == 1.0 - 0.1j

    def test_a_lossy_conductor_without_fields_is_nan_not_the_table(self, tmp_path):
        """The 14.5%-off table value is not handed out as a fallback."""
        with pytest.warns(UserWarning, match="could not read mode 1's saved fields"):
            reading = palace_line_impedance(
                FieldlessSim(tmp_path),
                impedance_tables(),
                index=1,
                mode_id=1,
                n_eff=2.0,
                signal=LOSSY_SIGNAL_CONDUCTOR,
                context="The rf stage's palace route",
            )

        assert np.isnan(reading.z0_ohm.real)
        assert reading.wall_mode is False

    def test_a_perfect_conductor_still_takes_the_table(self, monkeypatch, tmp_path):
        self._fields_say(monkeypatch, 74.6 - 8.3j)

        reading = palace_line_impedance(
            FieldlessSim(tmp_path),
            impedance_tables(),
            index=1,
            mode_id=1,
            n_eff=2.0,
            signal=SIGNAL_CONDUCTOR,
            context="The rf stage's palace route",
        )

        assert reading.z0_ohm == pytest.approx(80.0**2 / 100.0)

    def test_a_sheet_is_perfect_in_the_mesh_so_it_takes_the_table(
        self, monkeypatch, tmp_path
    ):
        self._fields_say(monkeypatch, 74.6 - 8.3j)
        sheet = Conductor(
            name="electrode_low", extent=((-5.0, 5.0), (0.0, 0.0)), model="volume"
        )

        reading = palace_line_impedance(
            FieldlessSim(tmp_path),
            impedance_tables(),
            index=1,
            mode_id=1,
            n_eff=2.0,
            signal=sheet,
            context="The rf stage's palace route",
        )

        assert reading.z0_ohm == pytest.approx(80.0**2 / 100.0)


class TestTablesFirstFieldsAfter:
    """The policy: Palace's own answer, then the saved fields."""

    def test_with_a_declared_index_no_field_file_is_read(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reading = palace_line_impedance(
                FieldlessSim(tmp_path),
                impedance_tables(),
                index=1,
                mode_id=1,
                n_eff=2.0,
                signal=SIGNAL_CONDUCTOR,
                context="The rf stage's palace route",
            )

        assert reading.z0_ohm == pytest.approx(80.0**2 / 100.0)
        assert reading.z0_ohm.imag == 0.0
        assert reading.wall_mode is False

    def test_without_a_declared_path_the_fields_are_read_and_their_absence_reported(
        self, tmp_path
    ):
        """The fallback and its NaN contract stay."""
        with pytest.warns(UserWarning, match="could not read mode 1's saved fields"):
            reading = palace_line_impedance(
                FieldlessSim(tmp_path),
                impedance_tables(),
                index=None,
                mode_id=1,
                n_eff=2.0,
                signal=SIGNAL_CONDUCTOR,
                context="The rf stage's palace route",
            )

        assert np.isnan(reading.z0_ohm.real)
        assert reading.wall_mode is None
        assert reading.n_eff == 2.0

    def test_a_table_that_answers_nothing_falls_through_to_the_fields(self, tmp_path):
        """A declared index whose table carries no current is no answer."""
        with pytest.warns(UserWarning, match="could not read mode 1's saved fields"):
            reading = palace_line_impedance(
                FieldlessSim(tmp_path),
                mode_table(2),
                index=1,
                mode_id=1,
                n_eff=2.0,
                signal=SIGNAL_CONDUCTOR,
                context="The rf stage's palace route",
            )

        assert np.isnan(reading.z0_ohm.real)
        assert reading.wall_mode is None
