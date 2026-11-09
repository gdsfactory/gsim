"""Modal S from driven runs, gauge alignment and the terminal S API.

Nothing here runs Palace. The runs are fake: the files a driven run writes
(``port-S.csv`` with Palace's header, the sidecar ``port_modes.json``, and the
log lines of the wave-port mode solves) are generated from a synthetic modal S,
with the Palace indices laid out as the config generator lays them out (the
active entry of a face on the port's own index, the other modes after all ports)
and with deliberately wrong values in every column that a run measures with the
face terminated for a different mode, so that a wrong choice of column shows.

The reference for the terminal S is the chain-matrix solution of the same
multiconductor line (``chain_terminal_s``), independent of the modal route. The
"Palace gauge" of the synthetic modes is built the way the module assumes
Palace builds it: unit power and the voltage on a pilot terminal real and
positive. The 2D data handed to the code carry a random column order, sign and
phase, and the effective indices a 0.05 % error, as separate 2D solves would.
Generic synthetic values only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import skrf

from gsim.palace import terminal_s as exported_terminal_s
from gsim.palace.ports.modes import PortModeEntry, write_port_modes
from gsim.palace.terminal import modal_to_terminal_s, uniform_line_modal_s
from gsim.palace.terminal_ports import (
    INTERPOLATIONS,
    LumpedTerminal,
    ModalPortMode,
    ModalRun,
    PortModes2D,
    TerminalPortSpec,
    align_gauge,
    build_terminal_s,
    load_modal_run,
    modal_s_from_runs,
    port_modes_2d,
    terminal_s,
)
from gsim.palace.validation import ModeSample
from tests.palace.test_terminal import (
    ELL,
    FREQ,
    OMEGA,
    chain_terminal_s,
    modal_data,
    rlgc,
)
from tests.palace.test_terminal_ports import DOMAIN, G1, G2, S1, S2, modes_from

C_LIGHT = 299_792_458.0
GHZ = FREQ / 1e9
NF = len(FREQ)
TOL_FILE = 1e-12
TOL_S = 1e-9

# -- fake runs ----------------------------------------------------------------------


def layout_entries(ports, active, excite, first_free=None):
    """Sidecar entries as the config generator writes them.

    ``ports`` is ``[(name, number, modes)]``, ``active`` maps a name to its active
    mode, ``excite`` is ``"active"`` or ``"all"``.
    """
    if first_free is None:
        first_free = max(number for _, number, _ in ports) + 1
    rows, extras = [], []
    next_index = first_free
    for name, number, modes in ports:
        rows.append(
            PortModeEntry(
                index=number,
                port=name,
                portnumber=number,
                mode=active[name],
                active=True,
                excited=True,
                attributes=(number + 10,),
            )
        )
        for mode in modes:
            if mode == active[name]:
                continue
            extras.append(
                PortModeEntry(
                    index=next_index,
                    port=name,
                    portnumber=number,
                    mode=mode,
                    active=False,
                    excited=excite == "all",
                    attributes=(number + 10,),
                )
            )
            next_index += 1
    return sorted([*rows, *extras], key=lambda entry: entry.index)


def write_run(folder, entries, s_true, keys, freq_ghz, *, lumped=None, wrong=True):
    """Write ``port-S.csv`` and the sidecar of one fake run.

    ``s_true`` is ``(nf, M, M)`` in the order ``keys`` (``(port, mode)``, lumped
    ports as ``(name, 0)``). A column excited at a non-active entry gets a wrong
    value (``1.1 s + 0.02``) when ``wrong``.
    """
    lumped = lumped or {}
    position = {key: i for i, key in enumerate(keys)}
    index_of = {(e.port, e.mode): e.index for e in entries}
    index_of.update({(name, 0): index for name, index in lumped.items()})
    columns = [(e.index, (e.port, e.mode), e.active) for e in entries if e.excited]
    columns += [(index, (name, 0), True) for name, index in lumped.items()]
    header = ["f (GHz)"]
    data = [np.asarray(freq_ghz, dtype=float)]
    for j, column_key, is_active in sorted(columns):
        for row_key, i in sorted(index_of.items(), key=lambda item: item[1]):
            values = np.array(s_true[:, position[row_key], position[column_key]])
            if wrong and not is_active:
                values = values * 1.1 + 0.02
            with np.errstate(divide="ignore"):
                db = 20.0 * np.log10(np.abs(values))
            header += [f"|S[{i}][{j}]| (dB)", f"arg(S[{i}][{j}]) (deg.)"]
            data += [db, np.degrees(np.angle(values))]
    lines = [", ".join(header)]
    lines += [
        ", ".join(repr(float(column[r])) for column in data)
        for r in range(len(freq_ghz))
    ]
    palace = folder / "output" / "palace"
    palace.mkdir(parents=True)
    (palace / "port-S.csv").write_text("\n".join(lines) + "\n")
    write_port_modes(folder, entries)
    return folder


def log_lines(entries, wave_number_of, freq_ghz):
    """The wave-port lines of a driven log: one block per frequency."""
    lines = []
    for f in freq_ghz:
        lines.append(f"Characterizing wave ports for \u03c9/2\u03c0 = {f:.9g} GHz")
        lines.extend(
            f"Port {e.index}, mode {e.mode}: k\u2099 = "
            f"{wave_number_of(e.port, e.mode, f).real:.12e} "
            f"{wave_number_of(e.port, e.mode, f).imag:+.12e}i"
            for e in entries
        )
    return lines


def random_modal(rng, size, *, zeros=True):
    """A random complex modal S ``(NF, size, size)``, with some exact zeros."""
    s = rng.normal(size=(NF, size, size)) + 1j * rng.normal(size=(NF, size, size))
    s *= 0.3
    if zeros:
        s[:, 0, 0] = 0.0
        s[:, size - 1, 1] = 0.0
    return s


PORTS = [("a", 1, (1, 2, 3)), ("b", 2, (1, 2, 3))]
KEYS = [(name, mode) for name, _, modes in PORTS for mode in modes]
ORDER = tuple(
    ModalPortMode(name, number, mode) for name, number, modes in PORTS for mode in modes
)


def n_eff_of(port, mode):  # noqa: ARG001
    """Distinct quasi-TEM indices: 2.3, 2.0, 1.7."""
    return {1: 2.3, 2: 2.0, 3: 1.7}[mode]


def wave_number_of(port, mode, f_ghz):
    """A lightly lossy wave number of the entry's mode at ``f_ghz``."""
    k0 = 2 * np.pi * f_ghz * 1e9 / C_LIGHT
    return complex(n_eff_of(port, mode) * k0, -1e-3 * n_eff_of(port, mode) * k0)


def cycling_runs(tmp_path, s_true, *, active_of=lambda k, name: k, excite="active"):
    """Three runs, run ``k`` with ``active_of(k, face)`` active on every face."""
    runs = []
    for k in (1, 2, 3):
        entries = layout_entries(
            PORTS, {name: active_of(k, name) for name, _, _ in PORTS}, excite
        )
        folder = write_run(tmp_path / f"run{k}", entries, s_true, KEYS, GHZ)
        lines = log_lines(entries, wave_number_of, GHZ)
        runs.append(load_modal_run(folder, log=lines))
    return runs


# -- Step 5: assembling the modal S ---------------------------------------------------


class TestModalSFromRuns:
    def test_cycling_runs_give_the_matrix_exactly(self, tmp_path):
        s_true = random_modal(np.random.default_rng(1), 6)
        modal = modal_s_from_runs(cycling_runs(tmp_path, s_true))
        assert modal.order == ORDER
        np.testing.assert_allclose(modal.s_modal, s_true, rtol=0, atol=TOL_FILE)
        assert modal.exact == (True,) * 6
        assert modal.source_run == (0, 1, 2, 0, 1, 2)

    def test_exact_zeros_survive_the_file(self, tmp_path):
        s_true = random_modal(np.random.default_rng(2), 6)
        modal = modal_s_from_runs(cycling_runs(tmp_path, s_true))
        assert np.all(modal.s_modal[:, 0, 0] == 0.0)
        assert np.all(modal.s_modal[:, 5, 1] == 0.0)

    def test_the_active_mode_may_differ_from_face_to_face(self, tmp_path):
        """Run 1: a->1, b->2; run 2: a->2, b->3; run 3: a->3, b->1."""
        s_true = random_modal(np.random.default_rng(3), 6)

        def active_of(k, name):
            return k if name == "a" else k % 3 + 1

        modal = modal_s_from_runs(cycling_runs(tmp_path, s_true, active_of=active_of))
        np.testing.assert_allclose(modal.s_modal, s_true, rtol=0, atol=TOL_FILE)
        assert modal.source_run == (0, 1, 2, 2, 0, 1)
        assert all(modal.exact)

    def test_runs_that_excite_every_entry_still_use_the_active_columns(self, tmp_path):
        s_true = random_modal(np.random.default_rng(4), 6)
        runs = cycling_runs(tmp_path, s_true, excite="all")
        modal = modal_s_from_runs(runs)
        np.testing.assert_allclose(modal.s_modal, s_true, rtol=0, atol=TOL_FILE)
        assert all(modal.exact)

    def test_one_run_with_every_entry_excited_marks_the_other_modes_inexact(
        self, tmp_path
    ):
        s_true = random_modal(np.random.default_rng(5), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "single", entries, s_true, KEYS, GHZ)
        modal = modal_s_from_runs([load_modal_run(folder)])
        assert modal.exact == (True, False, False, True, False, False)
        # the active columns are the true ones, the others are the wrong ones
        for column in (0, 3):
            np.testing.assert_allclose(
                modal.s_modal[:, :, column],
                s_true[:, :, column],
                rtol=0,
                atol=TOL_FILE,
            )
        np.testing.assert_allclose(
            modal.s_modal[:, :, 1], s_true[:, :, 1] * 1.1 + 0.02, rtol=0, atol=TOL_FILE
        )

    def test_a_column_nobody_excites_is_refused(self, tmp_path):
        s_true = random_modal(np.random.default_rng(6), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "active")
        folder = write_run(tmp_path / "single", entries, s_true, KEYS, GHZ)
        with pytest.raises(ValueError, match="No run excites mode 2 of port 'a'"):
            modal_s_from_runs([load_modal_run(folder)])

    def test_two_runs_with_the_same_active_mode_are_refused(self, tmp_path):
        s_true = random_modal(np.random.default_rng(7), 6)
        runs = cycling_runs(tmp_path, s_true)
        with pytest.raises(ValueError, match="active mode in 2 runs"):
            modal_s_from_runs([*runs, runs[0]])

    def test_runs_on_different_sweeps_are_refused(self, tmp_path):
        s_true = random_modal(np.random.default_rng(8), 6)
        runs = cycling_runs(tmp_path, s_true)
        shifted = ModalRun(
            freq_ghz=runs[1].freq_ghz * 1.01,
            s=runs[1].s,
            entries=runs[1].entries,
            label="shifted",
        )
        with pytest.raises(ValueError, match="different frequency axis"):
            modal_s_from_runs([runs[0], shifted, runs[2]])

    def test_runs_of_different_simulations_are_refused(self, tmp_path):
        s_true = random_modal(np.random.default_rng(9), 6)
        runs = cycling_runs(tmp_path, s_true)
        smaller = [("a", 1, (1, 2)), ("b", 2, (1, 2))]
        entries = layout_entries(smaller, {"a": 1, "b": 1}, "active")
        keys = [(n, m) for n, _, ms in smaller for m in ms]
        folder = write_run(
            tmp_path / "other",
            entries,
            random_modal(np.random.default_rng(1), 4),
            keys,
            GHZ,
        )
        with pytest.raises(ValueError, match="same simulation"):
            modal_s_from_runs([runs[0], load_modal_run(folder)])

    def test_a_missing_s_entry_names_the_run(self, tmp_path):
        s_true = random_modal(np.random.default_rng(10), 6)
        runs = cycling_runs(tmp_path, s_true)
        broken = ModalRun(
            freq_ghz=runs[1].freq_ghz,
            s={key: v for key, v in runs[1].s.items() if key[0] != 4},
            entries=runs[1].entries,
            label="broken-run",
        )
        with pytest.raises(ValueError, match="broken-run"):
            modal_s_from_runs([runs[0], broken, runs[2]])

    def test_no_run_is_refused(self):
        with pytest.raises(ValueError, match="at least one run"):
            modal_s_from_runs([])

    def test_lumped_ports_are_ordered_by_their_port_number(self, tmp_path):
        """Faces a (1) and b (3) with a lumped port d (2) in between."""
        ports = [("a", 1, (1, 2)), ("b", 3, (1, 2))]
        keys = [("a", 1), ("a", 2), ("d", 0), ("b", 1), ("b", 2)]
        s_true = random_modal(np.random.default_rng(11), 5)
        runs = []
        for k in (1, 2):
            entries = layout_entries(ports, {"a": k, "b": k}, "active", first_free=4)
            runs.append(
                load_modal_run(
                    write_run(
                        tmp_path / f"run{k}",
                        entries,
                        s_true,
                        keys,
                        GHZ,
                        lumped={"d": 2},
                    )
                )
            )
        modal = modal_s_from_runs(runs, lumped=[LumpedTerminal("d", 2, 75.0)])
        assert modal.order == (
            ModalPortMode("a", 1, 1),
            ModalPortMode("a", 1, 2),
            ModalPortMode("d", 2, 0),
            ModalPortMode("b", 3, 1),
            ModalPortMode("b", 3, 2),
        )
        np.testing.assert_allclose(modal.s_modal, s_true, rtol=0, atol=TOL_FILE)
        assert modal.port_columns("d") == [2]

    @pytest.mark.parametrize(
        ("lumped", "match"),
        [
            ([LumpedTerminal("a", 9)], "names must be unique"),
            ([LumpedTerminal("d", 2)], "indices must be unique"),
            ([LumpedTerminal("d", 9), LumpedTerminal("d", 8)], "names must be unique"),
            (
                [LumpedTerminal("d", 9), LumpedTerminal("e", 9)],
                "indices must be unique",
            ),
        ],
    )
    def test_lumped_terminals_may_not_clash_with_the_faces(
        self, tmp_path, lumped, match
    ):
        runs = cycling_runs(tmp_path, random_modal(np.random.default_rng(12), 6))
        with pytest.raises(ValueError, match=match):
            modal_s_from_runs(runs, lumped=lumped)

    def test_a_lumped_port_without_a_column_is_refused(self, tmp_path):
        runs = cycling_runs(tmp_path, random_modal(np.random.default_rng(13), 6))
        with pytest.raises(ValueError, match="No run has an S column"):
            modal_s_from_runs(runs, lumped=[LumpedTerminal("d", 9)])


class TestLumpedTerminal:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "", "index": 1}, "needs a name"),
            ({"name": "d", "index": 0}, "index"),
            ({"name": "d", "index": 1.5}, "index"),
            ({"name": "d", "index": 1, "resistance": 0.0}, "resistance"),
            ({"name": "d", "index": 1, "resistance": float("nan")}, "resistance"),
        ],
    )
    def test_bad_values_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            LumpedTerminal(**kwargs)


class TestReadingARun:
    def test_the_header_decides_the_index_not_the_port_names(self, tmp_path):
        """port_information.json knows ports 1 and 2 only; entries 3..6 stay p3..p6."""
        s_true = random_modal(np.random.default_rng(14), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "r", entries, s_true, KEYS, GHZ)
        run = load_modal_run(folder)
        assert sorted(run.s) == [(i, j) for i in range(1, 7) for j in range(1, 7)]

    def test_the_results_dict_of_sim_run_is_accepted(self, tmp_path):
        s_true = random_modal(np.random.default_rng(15), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "r", entries, s_true, KEYS, GHZ)
        csv = folder / "output" / "palace" / "port-S.csv"
        run = load_modal_run(
            {"port-S.csv": str(csv)}, sidecar=folder / "port_modes.json"
        )
        assert run.freq_ghz.tolist() == pytest.approx(GHZ.tolist())

    def test_a_port_s_file_path_is_accepted(self, tmp_path):
        s_true = random_modal(np.random.default_rng(16), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "r", entries, s_true, KEYS, GHZ)
        run = load_modal_run(folder / "output" / "palace" / "port-S.csv")
        assert len(run.entries) == 6

    def test_a_missing_sidecar_is_explained(self, tmp_path):
        s_true = random_modal(np.random.default_rng(17), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "r", entries, s_true, KEYS, GHZ)
        (folder / "port_modes.json").unlink()
        with pytest.raises(FileNotFoundError, match=r"port_modes\.json"):
            load_modal_run(folder)

    def test_a_folder_without_port_s_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"port-S\.csv"):
            load_modal_run(tmp_path)

    def test_a_magnitude_without_a_phase_is_refused(self, tmp_path):
        folder = tmp_path / "r"
        (folder / "output" / "palace").mkdir(parents=True)
        (folder / "output" / "palace" / "port-S.csv").write_text(
            "f (GHz), |S[1][1]| (dB)\n1.0, -3.0\n"
        )
        write_port_modes(folder, layout_entries([("a", 1, (1,))], {"a": 1}, "active"))
        with pytest.raises(ValueError, match="both a magnitude and a phase"):
            load_modal_run(folder)

    def test_a_malformed_cell_is_refused(self, tmp_path):
        folder = tmp_path / "r"
        (folder / "output" / "palace").mkdir(parents=True)
        (folder / "output" / "palace" / "port-S.csv").write_text(
            "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.)\n1.0, abc, 3.0\n"
        )
        write_port_modes(folder, layout_entries([("a", 1, (1,))], {"a": 1}, "active"))
        with pytest.raises(ValueError, match="Invalid numeric value"):
            load_modal_run(folder)

    def test_the_log_may_be_a_file(self, tmp_path):
        s_true = random_modal(np.random.default_rng(18), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "r", entries, s_true, KEYS, GHZ)
        log = tmp_path / "palace.log"
        log.write_text(
            "\n".join(log_lines(entries, wave_number_of, GHZ)), encoding="utf-8"
        )
        run = load_modal_run(folder, log=log)
        assert len(run.samples) == 6 * NF

    def test_a_run_needs_one_value_per_frequency(self):
        with pytest.raises(ValueError, match="one value per frequency"):
            ModalRun(
                freq_ghz=np.array([1.0, 2.0]),
                s={(1, 1): np.ones(3)},
                entries=(),
            )

    def test_a_run_needs_frequencies(self):
        with pytest.raises(ValueError, match="frequencies"):
            ModalRun(freq_ghz=np.array([]), s={}, entries=())


class TestWaveNumbers:
    def test_every_entry_is_decoded_with_its_own_runs_sidecar(self, tmp_path):
        """The log's port index 4 is a different (face, mode) in every run."""
        s_true = random_modal(np.random.default_rng(19), 6)
        modal = modal_s_from_runs(cycling_runs(tmp_path, s_true))
        assert modal.wave_numbers is not None
        for c, item in enumerate(modal.order):
            expected = [wave_number_of(item.port, item.mode, f) for f in GHZ]
            np.testing.assert_allclose(modal.wave_numbers[:, c], expected, rtol=1e-11)

    def test_a_run_without_a_log_has_none(self, tmp_path):
        s_true = random_modal(np.random.default_rng(20), 6)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "r", entries, s_true, KEYS, GHZ)
        assert modal_s_from_runs([load_modal_run(folder)]).wave_numbers is None

    def test_repeated_solves_are_reduced_to_their_median(self):
        entries = layout_entries([("a", 1, (1,))], {"a": 1}, "active")
        samples = tuple(
            ModeSample(port=1, mode=1, wave_number=complex(k, 0.0), frequency_ghz=10.0)
            for k in (100.0, 101.0, 500.0)
        )
        run = ModalRun(
            freq_ghz=np.array([10.0]),
            s={(1, 1): np.array([0.5 + 0j])},
            entries=entries,
            samples=samples,
        )
        modal = modal_s_from_runs([run])
        assert modal.wave_numbers is not None
        assert modal.wave_numbers[0, 0] == pytest.approx(101.0)

    def test_a_frequency_without_a_solve_is_nan(self):
        entries = layout_entries([("a", 1, (1,))], {"a": 1}, "active")
        run = ModalRun(
            freq_ghz=np.array([10.0, 20.0]),
            s={(1, 1): np.array([0.5 + 0j, 0.4 + 0j])},
            entries=entries,
            samples=(ModeSample(1, 1, 100.0 + 0j, frequency_ghz=10.0),),
        )
        modal = modal_s_from_runs([run])
        assert modal.wave_numbers is not None
        assert modal.wave_numbers[0, 0] == pytest.approx(100.0)
        assert np.isnan(modal.wave_numbers[1, 0])

    def test_a_logged_frequency_is_matched_to_the_nearest_within_the_log_rounding(self):
        entries = layout_entries([("a", 1, (1,))], {"a": 1}, "active")
        run = ModalRun(
            freq_ghz=np.array([10.0001]),
            s={(1, 1): np.array([0.5 + 0j])},
            entries=entries,
            samples=(ModeSample(1, 1, 100.0 + 0j, frequency_ghz=10.0),),
        )
        wave_numbers = modal_s_from_runs([run]).wave_numbers
        assert wave_numbers is not None
        assert wave_numbers[0, 0] == pytest.approx(100.0)

    def test_coinciding_modes_of_a_face_are_flagged(self):
        entries = layout_entries([("a", 1, (1, 2))], {"a": 1}, "all")

        def sample(index, mode, n_eff, f):
            k0 = 2 * np.pi * f * 1e9 / C_LIGHT
            return ModeSample(index, mode, complex(n_eff * k0, 0.0), frequency_ghz=f)

        freq = np.array([10.0, 20.0, 30.0])
        samples = tuple(
            sample(index, mode, n_eff, f)
            for f, n_2 in zip(freq, (2.0, 2.2, 2.0001), strict=True)
            for index, mode, n_eff in ((1, 1, 2.0), (2, 2, n_2))
        )
        eye = np.ones(3, dtype=complex)
        run = ModalRun(
            freq_ghz=freq,
            s={(i, j): 0.1 * eye for i in (1, 2) for j in (1, 2)},
            entries=entries,
            samples=samples,
        )
        flags = modal_s_from_runs([run]).degenerate
        assert len(flags) == 1
        assert flags[0].port == "a"
        assert flags[0].modes == (1, 2)
        assert flags[0].freq_ghz == (10.0, 30.0)

    def test_distinct_modes_raise_no_flag(self, tmp_path):
        modal = modal_s_from_runs(
            cycling_runs(tmp_path, random_modal(np.random.default_rng(21), 6))
        )
        assert modal.degenerate == ()


# -- the synthetic line with a Palace-like gauge -------------------------------------


@dataclass
class World:
    """Unit-power modes of a line, sorted like Palace's (decreasing beta)."""

    rlgc: tuple
    t_v: np.ndarray
    t_i: np.ndarray
    gamma: np.ndarray

    @property
    def count(self):
        return self.t_v.shape[1]

    @property
    def n_eff(self):
        return self.gamma.imag / (OMEGA[:, None] / C_LIGHT)

    @property
    def wave_numbers(self):
        return -1j * self.gamma  # Re = beta, Im = -alpha


def make_world(kind="three", *, lossy=True, homogeneous=False):
    pul = rlgc(kind, lossy=lossy, homogeneous=homogeneous)
    t_v, t_i, gamma = modal_data(*pul)
    order = np.argsort(-gamma.imag, axis=1, kind="stable")
    t_v = np.take_along_axis(t_v, order[:, None, :], axis=2)
    t_i = np.take_along_axis(t_i, order[:, None, :], axis=2)
    gamma = np.take_along_axis(gamma, order, axis=1)
    reaction = np.einsum("fik,fik->fk", t_v.conj(), t_i)
    assert np.all(reaction.real > 0.0)  # the wave carries power into the line
    scale = 1.0 / np.sqrt(np.abs(reaction))  # |2 P| = 1, as port_terminal_matrices
    return World(pul, t_v * scale[:, None, :], t_i * scale[:, None, :], gamma)


def default_pilot_rows(world, f=0):
    """The terminal of largest voltage per mode at frequency index ``f``."""
    return np.argmax(np.abs(world.t_v[f]), axis=0)


def palace_gauge(world, rows):
    """Phase per (frequency, mode) that makes the pilot voltage real and positive."""
    f_idx = np.arange(NF)[:, None]
    k_idx = np.arange(world.count)[None, :]
    pilot_v = world.t_v[f_idx, np.asarray(rows)[None, :], k_idx]
    assert np.all(np.abs(pilot_v) > 1e-3)
    return np.conj(pilot_v) / np.abs(pilot_v)


def palace_modal_s(world, g_a, g_b):
    """The modal S of the line as Palace reports it in the gauges ``g_a``, ``g_b``."""
    s = uniform_line_modal_s(world.gamma, ELL)
    g = np.concatenate([g_a, g_b], axis=1)
    return s * g[:, None, :] / g[:, :, None]


def shuffled_2d(world, _gauge, f_indices, rng, *, eps=5e-4, t_v=None, t_i=None):
    """2D solves with a random column order, phase and a small error in n_eff.

    The gauge argument is only there to read like the calls that build the
    matching Palace-side data: the 2D data deliberately do not use it.
    """
    t_v = world.t_v if t_v is None else t_v
    t_i = world.t_i if t_i is None else t_i
    out = []
    for f in f_indices:
        perm = rng.permutation(world.count)
        phase = np.exp(2j * np.pi * rng.random(world.count))
        out.append(
            PortModes2D(
                freq_ghz=float(GHZ[f]),
                n_eff=world.n_eff[f][perm]
                * (1 + eps * (rng.random(world.count) - 0.5)),
                t_v=t_v[f][:, perm] * phase,
                t_i=t_i[f][:, perm] * phase,
            )
        )
    return out


def three_conductor_specs():
    """Two copies of the three-terminal layout of the part A tests."""
    return [
        TerminalPortSpec(name, terminals=[G1, S1, S2], reference=G2)
        for name in ("a", "b")
    ]


class TestAlignGauge:
    def test_it_recovers_the_palace_gauge_of_shuffled_2d_data(self):
        world = make_world()
        rows = default_pilot_rows(world)
        g = palace_gauge(world, rows)
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(31))
        aligned = align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)
        expected_v = world.t_v * g[:, None, :]
        expected_i = world.t_i * g[:, None, :]
        np.testing.assert_allclose(aligned.t_v, expected_v, rtol=0, atol=1e-12)
        np.testing.assert_allclose(aligned.t_i, expected_i, rtol=0, atol=1e-12)
        assert aligned.pilot == tuple(rows)
        assert aligned.mismatch < 6e-4
        assert aligned.degenerate == ()
        assert len(aligned.pairing) == NF

    def test_the_pairing_follows_the_wave_number_not_the_column_order(self):
        world = make_world()
        solves = shuffled_2d(world, None, [4], np.random.default_rng(32), eps=0.0)
        aligned = align_gauge(
            solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers, extrapolate=True
        )
        perm = aligned.pairing[0]
        np.testing.assert_allclose(
            solves[0].n_eff[list(perm)], world.n_eff[4], rtol=1e-12
        )

    def test_a_named_pilot_terminal_sets_the_phase(self):
        world = make_world()
        rows = np.array([2, 2, 2])
        g = palace_gauge(world, rows)
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(33))
        aligned = align_gauge(
            solves,
            freq_ghz=GHZ,
            wave_numbers=world.wave_numbers,
            terminal_names=("G1", "S1", "S2"),
            pilot=["S2", "S2", "S2"],
        )
        np.testing.assert_allclose(
            aligned.t_v, world.t_v * g[:, None, :], rtol=0, atol=1e-12
        )
        pilot_voltage = aligned.t_v[:, 2, :]
        np.testing.assert_allclose(pilot_voltage.imag, 0.0, atol=1e-12)
        assert np.all(pilot_voltage.real > 0.0)

    def test_the_wrong_pilot_gives_a_different_gauge(self):
        world = make_world()
        g = palace_gauge(world, [2, 2, 2])
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(34))
        wrong = align_gauge(
            solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers, pilot=[0, 0, 0]
        )
        assert np.abs(wrong.t_v - world.t_v * g[:, None, :]).max() > 1e-2

    def test_a_lossless_line_interpolates_exactly_between_two_solves(self):
        world = make_world(lossy=False)
        rows = default_pilot_rows(world)
        g = palace_gauge(world, rows)
        solves = shuffled_2d(world, g, [0, NF - 1], np.random.default_rng(35))
        aligned = align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)
        np.testing.assert_allclose(
            aligned.t_v, world.t_v * g[:, None, :], rtol=0, atol=1e-10
        )

    def test_a_lossy_line_interpolates_best_in_inverse_frequency(self):
        """The loss terms of the modes go as 1/f: that is the default axis.

        Measured on this line (five of nine frequencies, 1-100 GHz): the largest
        error of T_V is 1e-4 in 1/f, 1.7e-2 in log f and 4.8e-2 in f.
        """
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        every = shuffled_2d(world, g, range(NF), np.random.default_rng(36))
        thinned = [every[i] for i in (0, 2, 4, 6, 8)]
        truth = world.t_v * g[:, None, :]
        errors = {
            axis: np.abs(
                align_gauge(
                    thinned,
                    freq_ghz=GHZ,
                    wave_numbers=world.wave_numbers,
                    interpolation=axis,
                ).t_v
                - truth
            ).max()
            for axis in INTERPOLATIONS
        }
        assert errors["inverse"] < 2e-4
        assert 1e-2 < errors["log"] < 3e-2
        assert 3e-2 < errors["linear"] < 7e-2
        default = align_gauge(thinned, freq_ghz=GHZ, wave_numbers=world.wave_numbers)
        assert np.abs(default.t_v - truth).max() == pytest.approx(errors["inverse"])

    def test_an_unknown_interpolation_is_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(36))
        with pytest.raises(ValueError, match="interpolation must be one of"):
            align_gauge(
                solves,
                freq_ghz=GHZ,
                wave_numbers=world.wave_numbers,
                interpolation="cubic",
            )

    def test_the_target_grid_must_be_covered_by_the_2d_solves(self):
        world = make_world(lossy=False)
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, [2, 6], np.random.default_rng(37))
        with pytest.raises(ValueError, match="extrapolate=True"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)
        held = align_gauge(
            solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers, extrapolate=True
        )
        np.testing.assert_allclose(held.t_v, world.t_v * g[:, None, :], atol=1e-10)

    def test_a_2d_mode_that_matches_no_3d_mode_is_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(38), eps=0.0)
        bad = solves[3]
        solves[3] = PortModes2D(bad.freq_ghz, bad.n_eff * 1.02, bad.t_v, bad.t_i)
        with pytest.raises(ValueError, match="cannot be paired"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)
        loose = align_gauge(
            solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers, rtol=0.03
        )
        assert loose.mismatch == pytest.approx(0.02, rel=0.05)

    def test_missing_wave_numbers_are_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(39))
        kn = world.wave_numbers.copy()
        kn[2, 1] = np.nan
        with pytest.raises(ValueError, match="missing at"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=kn)

    def test_wave_numbers_of_the_wrong_shape_are_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(40))
        with pytest.raises(ValueError, match="wave_numbers has shape"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers[:, :2])

    @pytest.mark.parametrize(
        "grid",
        [GHZ[::-1], np.array([1.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.array([])],
    )
    def test_the_target_grid_must_increase(self, grid):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(41))
        with pytest.raises(ValueError, match="strictly increasing"):
            align_gauge(
                solves, freq_ghz=grid, wave_numbers=world.wave_numbers[: len(grid)]
            )

    def test_two_solves_at_one_frequency_are_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, [0, 0, 8], np.random.default_rng(42))
        with pytest.raises(ValueError, match="same frequency"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)

    def test_no_solve_is_refused(self):
        with pytest.raises(ValueError, match="at least one 2D solve"):
            align_gauge([], freq_ghz=GHZ, wave_numbers=np.ones((NF, 3)))

    def test_a_zero_pilot_voltage_is_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(43))
        zeroed = [
            PortModes2D(
                s.freq_ghz,
                s.n_eff,
                np.where(np.arange(3)[:, None] == 1, 0.0, s.t_v),
                s.t_i,
            )
            for s in solves
        ]
        with pytest.raises(ValueError, match="zero voltage on its pilot"):
            align_gauge(
                zeroed, freq_ghz=GHZ, wave_numbers=world.wave_numbers, pilot=[1, 1, 1]
            )

    def test_a_bad_pilot_is_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(44))
        with pytest.raises(ValueError, match="pilot terminal index"):
            align_gauge(
                solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers, pilot=[0, 1, 7]
            )
        with pytest.raises(ValueError, match="is not one of"):
            align_gauge(
                solves,
                freq_ghz=GHZ,
                wave_numbers=world.wave_numbers,
                terminal_names=("G1", "S1", "S2"),
                pilot=["x", "S1", "S2"],
            )

    def test_solves_with_different_mode_counts_are_refused(self):
        world = make_world()
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, range(NF), np.random.default_rng(45))
        small = PortModes2D(solves[1].freq_ghz, np.ones(2), np.eye(2), np.eye(2))
        solves[1] = small
        with pytest.raises(ValueError, match="same number of modes"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)


class TestPortModes2D:
    @pytest.mark.parametrize(
        ("freq", "n_eff", "t_v", "t_i", "match"),
        [
            (0.0, [1.0, 2.0], np.eye(2), np.eye(2), "positive and finite"),
            (1.0, [1.0, 2.0], np.eye(3), np.eye(3), "Expected n_eff"),
            (1.0, [1.0, 2.0], np.eye(2), np.ones((2, 3)), "Expected n_eff"),
            (1.0, [1.0, np.nan], np.eye(2), np.eye(2), "finite"),
            (1.0, [1.0, 2.0], np.full((2, 2), np.inf), np.eye(2), "finite"),
        ],
    )
    def test_bad_input_is_refused(self, freq, n_eff, t_v, t_i, match):
        with pytest.raises(ValueError, match=match):
            PortModes2D(freq, np.asarray(n_eff), t_v, t_i)

    def test_a_complex_n_eff_keeps_its_real_part(self):
        solve = PortModes2D(
            1.0, np.array([2.0 - 0.1j, 1.5 - 0.2j]), np.eye(2), np.eye(2)
        )
        assert solve.n_eff.tolist() == [2.0, 1.5]


class TestDegenerateModes:
    """A homogeneous line: both modes share n_eff, so their basis is not fixed."""

    @staticmethod
    def basis_world():
        world = make_world("sym", lossy=False, homogeneous=True)
        # Any basis of the degenerate pair is a valid mode basis, and for this
        # lossless line T_V and T_I do not depend on frequency; LAPACK's basis
        # for a (numerically) scalar matrix does, by rounding noise. Freeze it.
        world.t_v = np.broadcast_to(world.t_v[0], world.t_v.shape).copy()
        world.t_i = np.broadcast_to(world.t_i[0], world.t_i.shape).copy()
        rng = np.random.default_rng(51)
        g_a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        g_b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        return world, g_a, g_b

    @staticmethod
    def palace_s(world, g_a, g_b):
        g_all = np.zeros((4, 4), dtype=complex)
        g_all[:2, :2], g_all[2:, 2:] = g_a, g_b
        s = uniform_line_modal_s(world.gamma, ELL)
        return np.linalg.inv(g_all) @ s @ g_all

    def test_without_the_3d_voltages_it_is_refused(self):
        world, _, _ = self.basis_world()
        solves = shuffled_2d(world, None, range(NF), np.random.default_rng(52), eps=0.0)
        with pytest.raises(ValueError, match="reference_voltages"):
            align_gauge(solves, freq_ghz=GHZ, wave_numbers=world.wave_numbers)

    def test_allowing_it_runs_but_may_be_wrong(self):
        world, g_a, _ = self.basis_world()
        solves = shuffled_2d(world, None, [0], np.random.default_rng(53), eps=0.0)
        aligned = align_gauge(
            solves,
            freq_ghz=GHZ,
            wave_numbers=world.wave_numbers,
            allow_degenerate=True,
            extrapolate=True,
        )
        assert aligned.degenerate == ((0, 1),)
        # It cannot reproduce Palace's arbitrary basis: that is why it is refused.
        assert np.abs(aligned.t_v - world.t_v @ g_a).max() > 1e-2

    def test_the_fit_to_the_3d_voltages_recovers_the_palace_basis(self):
        world, g_a, _ = self.basis_world()
        solves = shuffled_2d(
            world, None, [0, NF - 1], np.random.default_rng(54), eps=0.0
        )
        reference = [world.t_v[0] @ g_a, world.t_v[NF - 1] @ g_a]
        aligned = align_gauge(
            solves,
            freq_ghz=GHZ,
            wave_numbers=world.wave_numbers,
            reference_voltages=reference,
        )
        np.testing.assert_allclose(aligned.t_v, world.t_v @ g_a, rtol=0, atol=1e-10)
        np.testing.assert_allclose(aligned.t_i, world.t_i @ g_a, rtol=0, atol=1e-10)
        assert aligned.gauge_residual is not None
        assert aligned.gauge_residual.max() < 1e-12
        assert aligned.degenerate == ((0, 1),)
        assert aligned.pilot == ()

    def test_the_terminal_s_is_right_with_different_bases_on_the_two_faces(self):
        world, g_a, g_b = self.basis_world()
        s_pal = self.palace_s(world, g_a, g_b)
        reference = chain_terminal_s(*world.rlgc)
        ends = []
        for g in (g_a, g_b):
            solves = shuffled_2d(
                world, None, range(NF), np.random.default_rng(55), eps=0.0
            )
            ends.append(
                align_gauge(
                    solves,
                    freq_ghz=GHZ,
                    wave_numbers=world.wave_numbers,
                    reference_voltages=[world.t_v[f] @ g for f in range(NF)],
                )
            )
        s_t = modal_to_terminal_s(
            s_pal, [ends[0].t_v, ends[1].t_v], [ends[0].t_i, ends[1].t_i]
        )
        assert np.abs(s_t - reference).max() < TOL_S
        # the basis-blind route (identity basis on both ends) is wrong here
        blind = modal_to_terminal_s(s_pal, [world.t_v] * 2, [world.t_i] * 2)
        assert np.abs(blind - reference).max() > 1e-2

    def test_a_disagreeing_3d_mode_shows_in_the_residual(self):
        world = make_world("three")
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, [0, NF - 1], np.random.default_rng(56), eps=0.0)
        reference = [world.t_v[f] * g[f][None, :] for f in (0, NF - 1)]
        column = reference[0][:, 1].copy()
        stray = np.array([1.0, -2.0, 0.5]) + 0j
        stray -= column * np.vdot(column, stray) / np.vdot(column, column)
        stray /= np.linalg.norm(stray)  # orthogonal to the mode: it cannot be a gauge
        reference[0][:, 1] = column + 0.05 * np.linalg.norm(column) * stray
        aligned = align_gauge(
            solves,
            freq_ghz=GHZ,
            wave_numbers=world.wave_numbers,
            reference_voltages=reference,
        )
        assert aligned.gauge_residual is not None
        assert aligned.gauge_residual[0, 1] == pytest.approx(
            0.05 / np.sqrt(1 + 0.05**2), rel=1e-9
        )
        assert aligned.gauge_residual[0, 0] < 1e-12
        assert aligned.gauge_residual[1].max() < 1e-12
        assert aligned.gauge_scale is not None
        np.testing.assert_allclose(aligned.gauge_scale[1], 1.0, atol=1e-12)

    def test_reference_voltages_must_match_the_solves(self):
        world = make_world("three")
        g = palace_gauge(world, default_pilot_rows(world))
        solves = shuffled_2d(world, g, [0, NF - 1], np.random.default_rng(57), eps=0.0)
        with pytest.raises(ValueError, match="entries for 2 2D solves"):
            align_gauge(
                solves,
                freq_ghz=GHZ,
                wave_numbers=world.wave_numbers,
                reference_voltages=[np.eye(3)],
            )
        with pytest.raises(ValueError, match="expected"):
            align_gauge(
                solves,
                freq_ghz=GHZ,
                wave_numbers=world.wave_numbers,
                reference_voltages=[np.eye(2), np.eye(2)],
            )
        with pytest.raises(ValueError, match="zero terminal voltage"):
            align_gauge(
                solves,
                freq_ghz=GHZ,
                wave_numbers=world.wave_numbers,
                reference_voltages=[np.zeros((3, 3)), np.eye(3)],
            )


# -- Step 6: the terminal S from fake runs -------------------------------------------


def world_wave_numbers(world):
    """``wave_number_of`` for a World: the line's logged wave number per mode."""

    def lookup(port, mode, f_ghz):  # noqa: ARG001
        index = int(np.argmin(np.abs(GHZ - f_ghz)))
        return complex(world.wave_numbers[index, mode - 1])

    return lookup


def two_face_world(tmp_path, *, pilot_b=(2, 2, 2), wrong_pilot_b=None):
    """Palace-like runs of the line, 2D data, and the chain-matrix reference."""
    world = make_world()
    g_a = palace_gauge(world, default_pilot_rows(world))
    g_b = palace_gauge(world, pilot_b)
    s_pal = palace_modal_s(world, g_a, g_b)
    runs = []
    for k in (1, 2, 3):
        entries = layout_entries(PORTS, {"a": k, "b": k}, "active")
        folder = write_run(tmp_path / f"run{k}", entries, s_pal, KEYS, GHZ)
        lines = log_lines(entries, world_wave_numbers(world), GHZ)
        runs.append(load_modal_run(folder, log=lines))
    rng = np.random.default_rng(61)
    modes_2d = {
        "a": shuffled_2d(world, g_a, range(NF), rng),
        "b": shuffled_2d(world, g_b, range(NF), rng),
    }
    pilot = {"b": list(wrong_pilot_b if wrong_pilot_b is not None else pilot_b)}
    return world, runs, modes_2d, pilot, chain_terminal_s(*world.rlgc)


class TestTerminalS:
    def test_cycling_runs_reproduce_the_chain_matrix(self, tmp_path):
        _, runs, modes_2d, pilot, reference = two_face_world(tmp_path)
        result = build_terminal_s(runs, three_conductor_specs(), modes_2d, pilot=pilot)
        assert result.s.shape == reference.shape == (NF, 6, 6)
        assert np.abs(result.s - reference).max() < TOL_S
        assert result.port_names == tuple(
            f"{port}:{terminal}" for port in "ab" for terminal in ("G1", "S1", "S2")
        )
        assert np.all(result.z_ref == 50.0)
        assert set(result.aligned) == {"a", "b"}
        assert result.modal.exact == (True,) * 6

    def test_the_wrong_pilot_terminal_is_a_wrong_answer(self, tmp_path):
        """The check has teeth: a different pilot rule on face b breaks the S."""
        _, runs, modes_2d, _, reference = two_face_world(tmp_path)
        result = build_terminal_s(
            runs, three_conductor_specs(), modes_2d, pilot={"b": [0, 0, 0]}
        )
        assert np.abs(result.s - reference).max() > 1e-2

    def test_without_alignment_the_phase_of_the_2d_modes_matters(self, tmp_path):
        _, runs, modes_2d, _, reference = two_face_world(tmp_path)
        modal = modal_s_from_runs(runs)
        raw = [np.array([m.t_v for m in modes_2d[port]]) for port in ("a", "b")]
        raw_i = [np.array([m.t_i for m in modes_2d[port]]) for port in ("a", "b")]
        s_t = modal_to_terminal_s(modal.s_modal, raw, raw_i)
        assert np.abs(s_t - reference).max() > 1e-2

    def test_the_public_function_returns_a_named_network(self, tmp_path):
        _, runs, modes_2d, pilot, reference = two_face_world(tmp_path)
        net = terminal_s(
            runs, three_conductor_specs(), modes_2d, pilot=pilot, name="gssg"
        )
        assert isinstance(net, skrf.Network)
        assert net.name == "gssg"
        assert list(net.port_names or [])[:3] == ["a:G1", "a:S1", "a:S2"]
        np.testing.assert_allclose(net.f, FREQ, rtol=1e-12)
        assert np.abs(net.s - reference).max() < TOL_S
        np.testing.assert_allclose(net.z0.real, 50.0)

    def test_it_is_exported_from_gsim_palace(self):
        assert exported_terminal_s is terminal_s

    def test_reciprocity_and_passivity_of_the_result(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        s = build_terminal_s(runs, three_conductor_specs(), modes_2d, pilot=pilot).s
        assert np.abs(s - s.transpose(0, 2, 1)).max() < TOL_S
        assert np.linalg.svd(s, compute_uv=False).max() <= 1 + 1e-9

    def test_a_reference_impedance_per_port_and_terminal(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        z_ref = {"a": 50.0, "b": [50.0, 35.0, 35.0]}
        result = build_terminal_s(
            runs, three_conductor_specs(), modes_2d, pilot=pilot, z_ref=z_ref
        )
        assert result.z_ref.tolist() == [50, 50, 50, 50, 35, 35]
        net = result.network()
        np.testing.assert_allclose(net.z0[0].real, [50, 50, 50, 50, 35, 35])
        # compare with converting by hand at the same impedances
        modal = result.modal
        tvs = [result.aligned[p].t_v for p in "ab"]
        tis = [result.aligned[p].t_i for p in "ab"]
        again = modal_to_terminal_s(modal.s_modal, tvs, tis, z_ref=result.z_ref)
        np.testing.assert_allclose(result.s, again, rtol=0, atol=1e-12)

    def test_a_missing_z_ref_entry_is_refused(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        with pytest.raises(ValueError, match="no entry for port 'b'"):
            build_terminal_s(
                runs, three_conductor_specs(), modes_2d, pilot=pilot, z_ref={"a": 50.0}
            )

    @pytest.mark.parametrize("bad", [-5.0, 0.0, float("nan"), 50.0 + 5.0j])
    def test_bad_reference_impedances_are_refused(self, tmp_path, bad):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        with pytest.raises(ValueError, match=r"positive|Complex"):
            build_terminal_s(
                runs, three_conductor_specs(), modes_2d, pilot=pilot, z_ref=bad
            )

    def test_the_specs_must_cover_the_ports_of_the_runs(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        with pytest.raises(ValueError, match="terminal specs are for ports"):
            build_terminal_s(runs, three_conductor_specs()[:1], modes_2d, pilot=pilot)

    def test_a_port_without_2d_solves_is_refused(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        with pytest.raises(ValueError, match="No 2D solves were given"):
            build_terminal_s(
                runs, three_conductor_specs(), {"a": modes_2d["a"]}, pilot=pilot
            )

    def test_terminals_and_modes_must_match_in_number(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        two = [
            TerminalPortSpec(name, terminals=[S1, S2], reference=G2)
            for name in ("a", "b")
        ]
        with pytest.raises(ValueError, match="3 modes in the runs but 2 terminals"):
            build_terminal_s(runs, two, modes_2d, pilot=pilot)

    def test_runs_without_a_log_are_refused(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        bare = [
            ModalRun(freq_ghz=r.freq_ghz, s=r.s, entries=r.entries, label=r.label)
            for r in runs
        ]
        with pytest.raises(ValueError, match="no wave numbers"):
            build_terminal_s(bare, three_conductor_specs(), modes_2d, pilot=pilot)

    def test_the_folders_themselves_can_be_the_runs(self, tmp_path):
        """A path in ``runs`` is read without a log, so it needs the wave numbers."""
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        folders = [tmp_path / f"run{k}" for k in (1, 2, 3)]
        with pytest.raises(ValueError, match="no wave numbers"):
            build_terminal_s(folders, three_conductor_specs(), modes_2d, pilot=pilot)
        # and the pre-loaded runs work
        assert build_terminal_s(runs, three_conductor_specs(), modes_2d, pilot=pilot)

    def test_a_single_run_is_exact_in_its_active_columns_only(self, tmp_path):
        """One run, mode 1 active, all excited: the other columns are wrong ones."""
        world = make_world()
        g_a = palace_gauge(world, default_pilot_rows(world))
        g_b = palace_gauge(world, (2, 2, 2))
        s_pal = palace_modal_s(world, g_a, g_b)
        entries = layout_entries(PORTS, {"a": 1, "b": 1}, "all")
        folder = write_run(tmp_path / "single", entries, s_pal, KEYS, GHZ)
        lines = log_lines(entries, world_wave_numbers(world), GHZ)
        run = load_modal_run(folder, log=lines)
        rng = np.random.default_rng(62)
        modes_2d = {
            "a": shuffled_2d(world, g_a, range(NF), rng),
            "b": shuffled_2d(world, g_b, range(NF), rng),
        }
        result = build_terminal_s(
            [run], three_conductor_specs(), modes_2d, pilot={"b": [2, 2, 2]}
        )
        assert result.modal.exact == (True, False, False, True, False, False)
        reference = chain_terminal_s(*world.rlgc)
        assert np.abs(result.s - reference).max() > 1e-2  # the wrong columns show

    def test_the_cycling_runs_with_a_lumped_port_pass_it_through(self, tmp_path):
        """A face and a lumped port with no coupling: the lumped block is unchanged."""
        ports = [("a", 1, (1, 2))]
        keys = [("a", 1), ("a", 2), ("d", 0)]
        rng = np.random.default_rng(63)
        s_modal = random_modal(rng, 3, zeros=False) * 0.2
        runs = []
        for k in (1, 2):
            entries = layout_entries(ports, {"a": k}, "active", first_free=3)
            folder = write_run(
                tmp_path / f"r{k}", entries, s_modal, keys, GHZ, lumped={"d": 2}
            )
            runs.append(folder)
        spec = TerminalPortSpec("a", terminals=[S1, S2], reference=G2)
        pol = np.linalg.eig(np.array([[2.0, 0.3], [0.3, 1.0]]))[1]
        tv = pol.astype(complex)
        ti = (np.array([[1.0, 0.2], [0.2, 0.7]]) @ tv).astype(complex)
        power = np.einsum("ik,ik->k", tv.conj(), ti).real
        tv, ti = tv / np.sqrt(power), ti / np.sqrt(power)
        lines = []
        for entries_k in (1, 2):
            entries = layout_entries(ports, {"a": entries_k}, "active", first_free=3)
            lines.append(log_lines(entries, wave_number_of, GHZ))
        # n_eff of wave_number_of is 2.3, 2.0 for modes 1, 2: match the 2D solves
        solves = [
            PortModes2D(float(f), np.array([2.3, 2.0]), tv, ti)
            for f in (GHZ[0], GHZ[-1])
        ]
        loaded = [load_modal_run(r, log=lines[i]) for i, r in enumerate(runs)]
        result = build_terminal_s(
            loaded,
            [spec],
            {"a": solves},
            lumped=[LumpedTerminal("d", 2, 75.0)],
            pilot={"a": [0, 0]},
        )
        assert result.port_names == ("a:S1", "a:S2", "d")
        assert result.z_ref.tolist() == [50.0, 50.0, 75.0]
        aligned = result.aligned["a"]
        manual = modal_to_terminal_s(
            result.modal.s_modal,
            [aligned.t_v, np.full((NF, 1, 1), np.sqrt(75.0))],
            [aligned.t_i, np.full((NF, 1, 1), 1 / np.sqrt(75.0))],
            z_ref=[50.0, 50.0, 75.0],
        )
        np.testing.assert_allclose(result.s, manual, rtol=0, atol=1e-12)

    def test_a_network_of_lumped_ports_only_returns_the_modal_s(self):
        """A = 1, B = 0 for a lumped port: nothing changes but the reference."""
        rng = np.random.default_rng(64)
        s = random_modal(rng, 3, zeros=False) * 0.3
        run = ModalRun(
            freq_ghz=GHZ,
            s={(i + 1, j + 1): s[:, i, j] for i in range(3) for j in range(3)},
            entries=(),
        )
        lumped = [
            LumpedTerminal("d1", 1, 50.0),
            LumpedTerminal("d2", 2, 75.0),
            LumpedTerminal("d3", 3, 30.0),
        ]
        result = build_terminal_s([run], [], {}, lumped=lumped)
        np.testing.assert_allclose(result.s, s, rtol=0, atol=1e-13)
        assert result.port_names == ("d1", "d2", "d3")
        assert result.z_ref.tolist() == [50.0, 75.0, 30.0]


class TestFromFields:
    """2D side from saved fields: fields -> port_modes_2d -> align -> terminal S."""

    @pytest.mark.parametrize("current", ["contour", "reaction"])
    def test_the_fields_route_reproduces_the_chain_matrix(self, tmp_path, current):
        world = make_world()
        rows_a = default_pilot_rows(world)
        rows_b = (2, 2, 2)
        g_a = palace_gauge(world, rows_a)
        g_b = palace_gauge(world, rows_b)
        s_pal = palace_modal_s(world, g_a, g_b)
        runs = []
        for k in (1, 2, 3):
            entries = layout_entries(PORTS, {"a": k, "b": k}, "active")
            folder = write_run(tmp_path / f"run{k}", entries, s_pal, KEYS, GHZ)
            lines = log_lines(entries, world_wave_numbers(world), GHZ)
            runs.append(load_modal_run(folder, log=lines))
        spec = three_conductor_specs()[0]
        rng = np.random.default_rng(65)
        modes_2d = {}
        for port in ("a", "b"):
            solves = []
            for f in range(NF):
                perm = rng.permutation(3)
                fields = modes_from(world.t_v[f][:, perm], world.t_i[f][:, perm])
                solves.append(
                    port_modes_2d(
                        fields,
                        spec,
                        freq_ghz=float(GHZ[f]),
                        n_eff=world.n_eff[f][perm],
                        domain=DOMAIN,
                        current=current,
                    )
                )
            modes_2d[port] = solves
        result = build_terminal_s(
            runs,
            three_conductor_specs(),
            modes_2d,
            pilot={"b": list(rows_b)},
        )
        reference = chain_terminal_s(*world.rlgc)
        assert np.abs(result.s - reference).max() < TOL_S


class TestTouchstone:
    def test_a_written_file_reads_back_with_names_and_impedances(self, tmp_path):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path / "runs")
        result = build_terminal_s(runs, three_conductor_specs(), modes_2d, pilot=pilot)
        path = result.write_touchstone(tmp_path / "gssg", name="gssg")
        assert path.name == "gssg.s6p"
        back = skrf.Network(str(path))
        assert back.number_of_ports == 6
        np.testing.assert_allclose(back.f, FREQ, rtol=1e-12)
        assert np.abs(back.s - result.s).max() < 1e-12
        assert list(back.port_names) == list(result.port_names)

    def test_mixed_reference_impedances_survive_the_file(self, tmp_path):
        s = np.zeros((NF, 3, 3), dtype=complex)
        s[:, 0, 1] = s[:, 1, 0] = 0.5
        run = ModalRun(
            freq_ghz=GHZ,
            s={(i + 1, j + 1): s[:, i, j] for i in range(3) for j in range(3)},
            entries=(),
        )
        result = build_terminal_s(
            [run],
            [],
            {},
            lumped=[
                LumpedTerminal("p1", 1, 50.0),
                LumpedTerminal("p2", 2, 75.0),
                LumpedTerminal("p3", 3, 50.0),
            ],
        )
        path = result.write_touchstone(tmp_path / "mixed.s3p")
        back = skrf.Network(str(path))
        np.testing.assert_allclose(back.z0[0].real, [50.0, 75.0, 50.0])
        assert np.abs(back.s - result.s).max() < 1e-12
