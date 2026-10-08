"""Load and map Palace S-parameter results to port names.

Standalone utility — works with a Palace output directory, or the results
dict returned by ``sim.run()``.

Usage::

    from gsim.palace.results import load_sparams

    sp = load_sparams(results)
    sp.plot()  # quick overview
    sp["o1", "o2"].db  # dB array for S(o1, o2)
    sp.s11  # shorthand for 2-port
    sp.freq  # frequency in GHz
"""

from __future__ import annotations

import builtins
import csv
import json
import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, overload

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ModeMetrics(TypedDict):
    """Parsed per-mode metrics from mode-kn.csv."""

    k_n: complex
    n_eff: complex
    eta_eff: complex


_MODE_Z_COL_RE = re.compile(r"^(Z_PV|Z_VI|L_PV|C_PV|L_VI|C_VI)\[(\d+)\]")
_MODE_V_RE_COL_RE = re.compile(r"^Re\{V\[(\d+)\]\}")
_MODE_V_IM_COL_RE = re.compile(r"^Im\{V\[(\d+)\]\}")


class PalaceTextResults:
    """Parsed Palace text output files with pretty-print helpers.

    This is used for simulations that do not emit ``port-S.csv`` (for example,
    BoundaryMode). CSV files are parsed into row dictionaries while JSON files
    are decoded to Python objects.
    """

    ETA0_OHM = 376.730313668

    def __init__(
        self,
        *,
        files: dict[str, Path],
        csv_tables: dict[str, list[dict[str, str]]],
        json_data: dict[str, object],
        text_data: dict[str, list[str]],
    ) -> None:
        """Create parsed text results."""
        self.files = files
        self.csv_tables = csv_tables
        self.json_data = json_data
        self.text_data = text_data
        self.modes = self._parse_modes()
        # BoundaryMode postprocessing results (mode-Z.csv / mode-V.csv):
        # {index: {mode: metrics}}.
        self.mode_impedance = self._parse_mode_impedance()
        self.mode_voltages = self._parse_mode_voltages()

    @staticmethod
    def _to_float(value: object) -> float:
        """Return parsed float from CSV value or NaN when unavailable."""
        if value is None:
            return float("nan")
        try:
            text = str(value).strip()
            if not text:
                return float("nan")
            return float(text)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _format_complex(value: complex, *, sci: bool = True) -> str:
        """Format complex number for compact readable output."""
        if sci:
            return f"{value.real:+.6e}{value.imag:+.6e}j"
        return f"{value.real:.2f}{value.imag:+.2f}j"

    def _parse_modes(self) -> dict[int, ModeMetrics]:
        """Parse mode metrics from mode-kn.csv and compute eta_eff.

        Returns:
            Mapping ``{mode_id: {"k_n": complex, "n_eff": complex,
            "eta_eff": complex}}``.
        """
        rows = self.csv_tables.get("mode-kn.csv", [])
        modes: dict[int, ModeMetrics] = {}

        for idx, raw_row in enumerate(rows, start=1):
            row = {
                str(k).strip(): str(v).strip()
                for k, v in raw_row.items()
                if k is not None
            }

            mode_id_raw = self._to_float(row.get("m"))
            mode_id = int(mode_id_raw) if np.isfinite(mode_id_raw) else idx

            kn_re = self._to_float(row.get("Re{kn} (1/m)"))
            if not np.isfinite(kn_re):
                kn_re = self._to_float(row.get("Re{kn}"))
            if not np.isfinite(kn_re):
                kn_re = self._to_float(row.get("k_n"))
            kn_im = self._to_float(row.get("Im{kn} (1/m)"))
            if not np.isfinite(kn_im):
                kn_im = self._to_float(row.get("Im{kn}"))
            if not np.isfinite(kn_im):
                kn_im = 0.0
            if not np.isfinite(kn_re):
                kn_re = float("nan")
            k_n = complex(kn_re, kn_im)

            n_eff_re = self._to_float(row.get("Re{n_eff}"))
            n_eff_im = self._to_float(row.get("Im{n_eff}"))
            if not np.isfinite(n_eff_im):
                n_eff_im = 0.0
            if not np.isfinite(n_eff_re):
                n_eff_re = float("nan")
            n_eff = complex(n_eff_re, n_eff_im)

            if np.isfinite(n_eff.real) and np.isfinite(n_eff.imag) and abs(n_eff) > 0.0:
                eta_eff = self.ETA0_OHM / n_eff
            else:
                eta_eff = complex(float("nan"), float("nan"))

            modes[mode_id] = {
                "k_n": k_n,
                "n_eff": n_eff,
                "eta_eff": eta_eff,
            }

        return modes

    def _parse_mode_impedance(self) -> dict[int, dict[int, dict[str, float]]]:
        """Parse ``mode-Z.csv`` into ``{index: {mode: metrics}}``.

        Palace writes one row per mode with columns such as
        ``Z_PV[1] (Ohm)``, ``Z_VI[1] (Ohm)``, ``L_PV[1] (H/m)`` and
        ``C_PV[1] (F/m)`` for each postprocessing impedance index.
        """
        rows = self.csv_tables.get("mode-Z.csv", [])
        result: dict[int, dict[int, dict[str, float]]] = {}

        for idx, raw_row in enumerate(rows, start=1):
            row = {
                str(k).strip(): str(v).strip()
                for k, v in raw_row.items()
                if k is not None
            }
            mode_id_raw = self._to_float(row.get("m"))
            mode_id = int(mode_id_raw) if np.isfinite(mode_id_raw) else idx

            for column, raw_value in row.items():
                match = _MODE_Z_COL_RE.match(column)
                if match is None:
                    continue
                quantity, index = match.group(1), int(match.group(2))
                value = self._to_float(raw_value)
                if not np.isfinite(value):
                    continue
                result.setdefault(index, {}).setdefault(mode_id, {})[quantity] = value

        return result

    def _parse_mode_voltages(self) -> dict[int, dict[int, complex]]:
        """Parse ``mode-V.csv`` into ``{index: {mode: complex voltage}}``."""
        rows = self.csv_tables.get("mode-V.csv", [])
        re_parts: dict[int, dict[int, float]] = {}
        im_parts: dict[int, dict[int, float]] = {}

        for idx, raw_row in enumerate(rows, start=1):
            row = {
                str(k).strip(): str(v).strip()
                for k, v in raw_row.items()
                if k is not None
            }
            mode_id_raw = self._to_float(row.get("m"))
            mode_id = int(mode_id_raw) if np.isfinite(mode_id_raw) else idx

            for column, raw_value in row.items():
                value = self._to_float(raw_value)
                re_match = _MODE_V_RE_COL_RE.match(column)
                im_match = _MODE_V_IM_COL_RE.match(column)
                if re_match is not None and np.isfinite(value):
                    index = int(re_match.group(1))
                    re_parts.setdefault(index, {})[mode_id] = value
                elif im_match is not None and np.isfinite(value):
                    index = int(im_match.group(1))
                    im_parts.setdefault(index, {})[mode_id] = value

        result: dict[int, dict[int, complex]] = {}
        for index, modes in re_parts.items():
            for mode_id, re_val in modes.items():
                im_val = im_parts.get(index, {}).get(mode_id, 0.0)
                result.setdefault(index, {})[mode_id] = complex(re_val, im_val)
        return result

    def characteristic_impedance(
        self, *, index: int = 1, mode: int = 1, quantity: str = "Z_PV"
    ) -> float | None:
        """Return a characteristic-impedance quantity from ``mode-Z.csv``.

        Args:
            index: Postprocessing impedance index (``Z_PV[i]``).
            mode: Mode number (the ``m`` column).
            quantity: One of ``"Z_PV"``, ``"Z_VI"``, ``"L_PV"``, ``"C_PV"``,
                ``"L_VI"``, ``"C_VI"``.

        Returns:
            The value, or ``None`` when unavailable.
        """
        return self.mode_impedance.get(index, {}).get(mode, {}).get(quantity)

    def mode_voltage(self, *, index: int = 1, mode: int = 1) -> complex | None:
        """Return the complex mode voltage from ``mode-V.csv``."""
        return self.mode_voltages.get(index, {}).get(mode)

    @overload
    def __getitem__(self, key: Literal["modes"]) -> dict[int, ModeMetrics]: ...

    @overload
    def __getitem__(self, key: int) -> ModeMetrics: ...

    @overload
    def __getitem__(self, key: str) -> ModeMetrics: ...

    def __getitem__(
        self,
        key: str | int,
    ) -> dict[int, ModeMetrics] | ModeMetrics:
        """Dictionary-like access to parsed mode metrics.

        Supported keys:
        - ``"modes"`` -> full ``{mode_id: values}`` mapping
        - integer mode id (for example ``1``)
        - ``"mode_1"`` style string key
        """
        if key == "modes":
            return self.modes
        if isinstance(key, int):
            return self.modes[key]
        if isinstance(key, str) and key.startswith("mode_"):
            mode_id = int(key.split("_", 1)[1])
            return self.modes[mode_id]
        raise KeyError(key)

    @property
    def error_indicators(self) -> dict[str, float] | None:
        """Norm, minimum, maximum and mean of Palace's error indicator.

        Read from ``error-indicators.csv``; None when Palace did not write it.
        """
        rows = self.csv_tables.get("error-indicators.csv")
        if not rows:
            return None
        row = {str(k).strip(): v for k, v in rows[-1].items()}
        columns = {"norm": "Norm", "min": "Minimum", "max": "Maximum", "mean": "Mean"}
        return {key: self._to_float(row.get(column)) for key, column in columns.items()}

    def keys(self) -> list[str]:
        """Return available result file names."""
        return sorted(self.files.keys())

    def _format_csv_table(
        self, name: str, rows: list[dict[str, str]], max_rows: int
    ) -> str:
        """Format one CSV table for terminal-friendly display."""
        if not rows:
            return f"{name}: <empty csv>"

        columns = list(rows[0].keys())
        widths = {col: len(col) for col in columns}
        preview = rows[:max_rows]
        for row in preview:
            for col in columns:
                widths[col] = max(widths[col], len(str(row.get(col, ""))))

        header = " | ".join(col.ljust(widths[col]) for col in columns)
        divider = "-+-".join("-" * widths[col] for col in columns)
        lines = [f"{name} ({len(rows)} rows)", header, divider]
        lines.extend(
            " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
            for row in preview
        )
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows)")
        return "\n".join(lines)

    def _pretty_text(self, *, max_rows: int = 8, max_lines: int = 12) -> str:
        """Build a compact mode summary matching BoundaryMode workflow needs."""
        del max_rows
        del max_lines
        if not self.modes:
            return "No mode data found in mode-kn.csv"

        lines: list[str] = []
        for mode_id in sorted(self.modes):
            mode = self.modes[mode_id]
            lines.append(
                f"mode {mode_id}: "
                f"k_n = {self._format_complex(mode['k_n'])}, "
                f"n_eff = {self._format_complex(mode['n_eff'])}, "
                f"eta_eff ~= {self._format_complex(mode['eta_eff'], sci=False)}"
            )

        for index in sorted(self.mode_impedance):
            for mode_id in sorted(self.mode_impedance[index]):
                metrics = self.mode_impedance[index][mode_id]
                parts = [f"{name} = {value:.6g}" for name, value in metrics.items()]
                lines.append(f"  Z[{index}] mode {mode_id}: " + ", ".join(parts))

        for index in sorted(self.mode_voltages):
            for mode_id in sorted(self.mode_voltages[index]):
                voltage = self.mode_voltages[index][mode_id]
                lines.append(
                    f"  V[{index}] mode {mode_id}: "
                    f"{self._format_complex(voltage, sci=False)} V"
                )

        return "\n".join(lines)

    def print(self, *, max_rows: int = 8, max_lines: int = 12) -> None:
        """Pretty-print text output summaries."""
        builtins.print(  # noqa: T201
            self._pretty_text(max_rows=max_rows, max_lines=max_lines)
        )

    def __repr__(self) -> str:
        """Return concise object representation."""
        return (
            "PalaceTextResults("  # pragma: no cover - trivial repr
            f"files={len(self.files)}, "
            f"csv={len(self.csv_tables)}, "
            f"json={len(self.json_data)}, "
            f"text={len(self.text_data)})"
        )

    def __str__(self) -> str:
        """Return pretty summary for notebook/terminal display."""
        return self._pretty_text()


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


class SParam:
    """A single S-parameter entry (complex-valued vs frequency)."""

    def __init__(self, db: NDArray, deg: NDArray) -> None:
        """Create from dB magnitude and degree phase arrays."""
        self._db = np.asarray(db, dtype=float)
        self._deg = np.asarray(deg, dtype=float)

    @property
    def db(self) -> NDArray:
        """Magnitude in dB."""
        return self._db

    @property
    def deg(self) -> NDArray:
        """Phase in degrees."""
        return self._deg

    @property
    def mag(self) -> NDArray:
        """Linear magnitude."""
        return 10 ** (self._db / 20)

    @property
    def complex(self) -> NDArray:
        """Complex S-parameter values."""
        return self.mag * np.exp(1j * np.deg2rad(self._deg))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SParam(n={len(self._db)})"


class SParams:
    """Palace S-parameter results with named port access.

    Access individual S-parameters by port name pair::

        sp["o1", "o2"]  # -> SParam object
        sp["o1", "o2"].db  # -> dB array
        sp["o1", "o2"].deg  # -> phase array

    For 2-port convenience, RF shorthand works::

        sp.s11  # sp[ports[0], ports[0]]
        sp.s21  # sp[ports[1], ports[0]]
        sp.s12  # sp[ports[0], ports[1]]
        sp.s22  # sp[ports[1], ports[1]]
    """

    def __init__(
        self,
        freq: NDArray,
        data: dict[tuple[str, str], SParam],
        port_names: list[str],
        files: dict[str, Path] | None = None,
        z0: float | None = None,
    ) -> None:
        """Create from frequency array, S-parameter data, and port names."""
        self._freq = freq
        self._data = data
        self._port_names = port_names
        self.files = files or {}
        self._z0 = float(z0) if z0 is not None else None

    @property
    def z0(self) -> float:
        """Reference impedance of the S-parameter normalization [Ohm]."""
        return self._z0 if self._z0 is not None else 50.0

    @property
    def freq(self) -> NDArray:
        """Frequency in GHz."""
        return self._freq

    @property
    def port_names(self) -> list[str]:
        """Ordered list of port names."""
        return list(self._port_names)

    def __getitem__(self, key: tuple[str, str]) -> SParam:
        """Get S-parameter by port name pair: sp["o1", "o2"]."""
        if not isinstance(key, tuple) or len(key) != 2:
            msg = f'Use sp["to", "from"] indexing, got {key!r}'
            raise KeyError(msg)
        if key not in self._data:
            available = [f'("{k[0]}", "{k[1]}")' for k in sorted(self._data)]
            msg = (
                f'S-parameter ("{key[0]}", "{key[1]}") not found. '
                f"Available: {', '.join(available)}"
            )
            raise KeyError(msg)
        return self._data[key]

    def __getattr__(self, name: str) -> SParam:
        """RF shorthand: sp.s11, sp.s21, etc."""
        m = re.fullmatch(r"s(\d)(\d)", name)
        if m and len(self._port_names) >= 2:
            i, j = int(m.group(1)), int(m.group(2))
            if 1 <= i <= len(self._port_names) and 1 <= j <= len(self._port_names):
                to_port = self._port_names[i - 1]
                from_port = self._port_names[j - 1]
                key = (to_port, from_port)
                if key in self._data:
                    return self._data[key]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def keys(self) -> list[tuple[str, str]]:
        """All available (to_port, from_port) pairs."""
        return list(self._data.keys())

    def to_dataframe(self):
        """Export to a flat pandas DataFrame."""
        import pandas as pd

        cols: dict[str, NDArray] = {"freq_ghz": self._freq}
        for (to_p, from_p), sp in self._data.items():
            cols[f"S_{to_p}_{from_p}_db"] = sp.db
            cols[f"S_{to_p}_{from_p}_deg"] = sp.deg
        return pd.DataFrame(cols)

    def to_skrf(self, z0: float | None = None):
        """Convert to a scikit-rf Network object.

        Args:
            z0: Reference impedance in ohms. Defaults to the stored
                reference impedance (``z0``, falling back to 50 Ohm).

        Returns:
            skrf.Network with frequency in Hz and S-parameters in (f, i, j)
            order; port names and reference impedance are preserved.
        """
        import numpy as np
        import skrf as rf

        z0 = float(z0) if z0 is not None else self.z0
        n = len(self._port_names)
        f_hz = self._freq * 1e9
        S = np.zeros((len(f_hz), n, n), dtype=complex)
        for i, pi in enumerate(self._port_names):
            for j, pj in enumerate(self._port_names):
                if (pi, pj) in self._data:
                    S[:, i, j] = self._data[(pi, pj)].complex
                elif (pj, pi) in self._data:
                    # assume reciprocity: S_ij = S_ji
                    S[:, i, j] = self._data[(pj, pi)].complex
        network = rf.Network(f=f_hz, s=S, z0=z0, f_unit="Hz")
        network.port_names = list(self._port_names)
        return network

    def write_touchstone(
        self,
        path: str | Path,
        *,
        z0: float | None = None,
    ) -> Path:
        """Export the S-parameters to a Touchstone file (``.sNp``).

        Frequency is written in Hz, port order and the reference
        impedance are preserved (port names are embedded as ``! Port[i]``
        comments), and the real/imaginary parts are written with 16-digit
        precision so the complex S round-trip error stays far below the
        1e-9 acceptance level.

        Args:
            path: Destination path (``.sNp`` suffix added when missing).
            z0: Reference impedance [Ohm]; defaults to the stored ``z0``
                (falling back to 50 Ohm).

        Returns:
            The resolved file path.
        """
        path = Path(path)
        if not re.fullmatch(r"\.s\d+p", path.suffix.lower()):
            # APPEND (do not with_suffix: "run_1.5GHz" would eat ".5GHz").
            path = Path(f"{path}.s{len(self._port_names)}p")
        network = self.to_skrf(z0=z0)
        fmt = "{:.16e}"
        network.write_touchstone(
            path,
            form="ri",
            format_spec_A=fmt,
            format_spec_B=fmt,
            format_spec_freq=fmt,
            write_z0=False,
        )
        logger.info("Touchstone written to %s", path)
        return path

    @classmethod
    def from_touchstone(cls, path: str | Path) -> SParams:
        """Load S-parameters from a Touchstone file written by this class.

        Port names are restored from the ``! Port[i]`` comments when
        present (otherwise ``p1..pN``), and the reference impedance from
        the file's ``R`` value.
        """
        import skrf as rf

        network = rf.Network(str(path))
        port_names = [
            str(name) if name else f"p{i + 1}"
            for i, name in enumerate(network.port_names or [])
        ]
        if len(port_names) != network.number_of_ports:
            port_names = [f"p{i + 1}" for i in range(network.number_of_ports)]

        freq_ghz = np.asarray(network.f, dtype=float) / 1e9
        data: dict[tuple[str, str], SParam] = {}
        n = network.number_of_ports
        for i in range(n):
            for j in range(n):
                s_ij = network.s[:, i, j]
                data[(port_names[i], port_names[j])] = SParam(
                    db=20 * np.log10(np.clip(np.abs(s_ij), 1e-300, None)),
                    deg=np.degrees(np.unwrap(np.angle(s_ij))),
                )
        z0 = float(np.real(network.z0[0, 0]))
        return cls(
            freq=freq_ghz,
            data=data,
            port_names=port_names,
            z0=z0,
        )

    def _filtered_entries(self, full: bool) -> list[tuple[str, SParam]]:
        """Return ``[(label, SParam), ...]`` filtered by excitation port."""
        from_ports = list(dict.fromkeys(fp for _, fp in self._data))
        first_from = from_ports[0] if from_ports else None
        entries: list[tuple[str, SParam]] = []
        for (to_p, from_p), sp in self._data.items():
            if not full and len(from_ports) > 1 and from_p != first_from:
                continue
            entries.append((f"S({to_p},{from_p})", sp))
        return entries

    def plot(
        self,
        *,
        full: bool = False,
        figsize: tuple[float, float] = (8, 6),
    ) -> None:
        """Plot magnitude and phase with matplotlib (static).

        By default only the first excitation column is plotted
        (e.g. S11, S21 but not S12, S22). Pass ``full=True`` to
        include all entries.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        for label, sp in self._filtered_entries(full):
            ax1.plot(self._freq, sp.db, label=label)
            ax2.plot(self._freq, sp.deg, label=label)

        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(f"S-Parameters (Z0 = {self.z0:g} Ohm)")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.show()

    def plot_plotly(self, *, full: bool = False):
        """Plot S-parameters with Plotly (interactive).

        Returns a ``plotly.graph_objects.Figure`` that renders interactively
        in notebooks and can be saved as standalone HTML via
        ``fig.write_html("sparams.html")``.
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Magnitude (dB)", "Phase (deg)"),
        )
        for label, sp in self._filtered_entries(full):
            fig.add_scatter(
                x=self._freq,
                y=sp.db,
                mode="lines",
                name=label,
                legendgroup=label,
                row=1,
                col=1,
            )
            fig.add_scatter(
                x=self._freq,
                y=sp.deg,
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=False,
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
        fig.update_yaxes(title_text="dB", row=1, col=1)
        fig.update_yaxes(title_text="deg", row=2, col=1)
        fig.update_layout(title="S-Parameters", height=600)
        return fig

    def _port_index_map(self) -> dict[str, int]:
        """Return ``{port_name: 1-based index}`` mapping."""
        return {name: i + 1 for i, name in enumerate(self._port_names)}

    def _sij_label(self, to_port: str, from_port: str) -> str:
        """Return ``Sij`` label for a port pair."""
        idx = self._port_index_map()
        return f"S{idx[to_port]}{idx[from_port]}"

    def plot_interactive(self, phase: bool = False):
        """Plot S-parameters with interactive legend toggling.

        Uses ``Sij`` notation (e.g. S11, S21) and prints the port
        mapping so you know which index corresponds to which port.
        By default shows the first excitation column (S11, S21, S31, ...)
        and hides symmetric/redundant entries. All traces are togglable
        via the legend.

        Args:
            phase: If True, plot phase (deg). Default is magnitude (dB).

        Returns:
            plotly Figure
        """
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        # Print port mapping
        idx = self._port_index_map()
        mapping = ", ".join(f"Port {i}: {name}" for name, i in idx.items())
        print(f"Port mapping: {mapping}")  # noqa: T201

        # Build entries with Sij labels
        entries: list[tuple[str, str, SParam]] = []
        for (to_p, from_p), sp in self._data.items():
            entries.append((self._sij_label(to_p, from_p), from_p, sp))

        # Show first excitation column by default (Si1), hide the rest
        first_from = self._port_names[0] if self._port_names else None
        first_col = [(l, sp) for l, fp, sp in entries if fp == first_from]
        rest = [(l, sp) for l, fp, sp in entries if fp != first_from]
        ordered = first_col + rest
        visible_set = {l for l, _ in first_col}

        fig = go.Figure()

        for label, sp in ordered:
            y = sp.deg if phase else sp.db
            vis = True if label in visible_set else "legendonly"
            fig.add_scatter(
                x=self._freq,
                y=y,
                mode="lines",
                name=label,
                visible=vis,
            )

        ylabel = "Phase (deg)" if phase else "|S| (dB)"
        fig.update_layout(
            title=f"Z0 = {self.z0:g} Ohm",
            xaxis_title="Frequency (GHz)",
            yaxis_title=ylabel,
            width=650,
            height=350,
            margin=dict(t=40, b=40, l=60, r=140),
            modebar=dict(orientation="v"),
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                groupclick="toggleitem",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                itemsizing="constant",
                bordercolor="#888",
                borderwidth=1,
                bgcolor="rgba(245,245,245,0.9)",
                entrywidthmode="pixels",
                entrywidth=70,
            ),
        )
        return fig

    def save_npz(self, filepath: str | Path) -> Path:
        """Save S-parameters to a ``.npz`` file.

        The file can be reloaded with :meth:`SParams.from_file`.

        Args:
            filepath: Destination path (``.npz`` suffix added if missing).

        Returns:
            The resolved file path.
        """
        filepath = Path(filepath).with_suffix(".npz")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, NDArray] = {"freq": self._freq}
        arrays["port_names"] = np.array(self._port_names)
        arrays["z0"] = np.array([self.z0])
        for (to_p, from_p), sp in self._data.items():
            arrays[f"S_{to_p}_{from_p}_db"] = sp.db
            arrays[f"S_{to_p}_{from_p}_deg"] = sp.deg

        np.savez_compressed(filepath, **arrays)  # ty: ignore[invalid-argument-type]
        logger.info("S-parameters saved to %s", filepath)
        return filepath

    @classmethod
    def from_file(cls, filepath: str | Path) -> SParams:
        """Load S-parameters from a ``.npz`` file written by :meth:`save_npz`.

        Args:
            filepath: Path to the ``.npz`` file.

        Returns:
            Reconstructed :class:`SParams` object.
        """
        filepath = Path(filepath).with_suffix(".npz")
        npz = np.load(filepath, allow_pickle=False)

        freq = npz["freq"]
        port_names = list(npz["port_names"])
        z0 = float(npz["z0"][0]) if "z0" in npz else None

        data: dict[tuple[str, str], SParam] = {}
        for to_p in port_names:
            for from_p in port_names:
                db_key = f"S_{to_p}_{from_p}_db"
                deg_key = f"S_{to_p}_{from_p}_deg"
                if db_key in npz and deg_key in npz:
                    data[(to_p, from_p)] = SParam(db=npz[db_key], deg=npz[deg_key])

        logger.info("S-parameters loaded from %s", filepath)
        return cls(freq=freq, data=data, port_names=port_names, z0=z0)

    def __repr__(self) -> str:
        """Return string representation."""
        n_freq = len(self._freq)
        n_ports = len(self._port_names)
        ports_str = ", ".join(self._port_names)
        return (
            f"SParams({n_ports} ports [{ports_str}], "
            f"{n_freq} freq points, "
            f"{len(self._data)} S-parameters)"
        )


def load_sparams(
    source: str | Path | dict,
    *,
    port_info_path: str | Path | None = None,
) -> SParams:
    """Load Palace S-parameter results.

    Args:
        source: One of:

            - **results dict** returned by ``sim.run()``
            - **directory path** — the sim dir or ``output/palace/``
        port_info_path: Explicit path to ``port_information.json``.

    Returns:
        :class:`SParams` object with named port access.

    Raises:
        FileNotFoundError: If ``port-S.csv`` cannot be found.
        ValueError: If a numeric CSV cell is malformed. Negative infinity is
            accepted for dB magnitudes and represents exact zero transmission.
    """
    import pandas as pd

    csv_path, base_dir = _resolve_source(source)
    if csv_path is None:  # pragma: no cover — _resolve_source raises first
        msg = "port-S.csv not found"
        raise FileNotFoundError(msg)

    # Check results dict for port_information.json (injected by DrivenSim)
    if port_info_path is None and isinstance(source, dict):
        pi_val = source.get("port_information.json")
        if pi_val is not None:
            port_info_path = Path(pi_val)

    port_map = _load_port_map(base_dir, csv_path, port_info_path)

    # Preserve cells verbatim until numeric validation, including padded -inf.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    # Extract frequency
    freq_col = next((c for c in df.columns if c.startswith("f")), None)
    freq = (
        _numeric_csv_column(df[freq_col], csv_path, freq_col)
        if freq_col
        else np.arange(len(df), dtype=float)
    )

    # Parse S-parameter columns into SParam objects
    # Group by (i, j) pair — each pair has a dB and deg column
    raw: dict[tuple[int, int], dict[str, NDArray]] = {}
    for col in df.columns:
        parsed = _parse_sparam_col(col)
        if parsed is None:
            continue
        i, j, kind = parsed
        raw.setdefault((i, j), {})[kind] = _numeric_csv_column(
            df[col], csv_path, col, allow_negative_infinity=kind == "db"
        )

    # Build port name list (ordered by index)
    all_indices = set()
    for i, j in raw:
        all_indices.add(i)
        all_indices.add(j)
    port_names = [port_map.get(idx, f"p{idx}") for idx in sorted(all_indices)]

    # Build SParam objects keyed by (port_name, port_name)
    data: dict[tuple[str, str], SParam] = {}
    for (i, j), parts in sorted(raw.items()):
        to_name = port_map.get(i, f"p{i}")
        from_name = port_map.get(j, f"p{j}")
        db = parts.get("db", np.zeros(len(freq)))
        deg = parts.get("deg", np.zeros(len(freq)))
        data[(to_name, from_name)] = SParam(db=db, deg=deg)

    files = dict(source) if isinstance(source, dict) else None

    z0 = _reference_impedance(base_dir, port_info_path)
    return SParams(freq=freq, data=data, port_names=port_names, files=files, z0=z0)


def load_text_results(source: str | Path | dict) -> PalaceTextResults:
    """Load non-S-parameter Palace outputs from text files.

    This parser targets result layouts such as BoundaryMode where Palace writes
    summary CSV/TXT/JSON files under ``output/palace`` but no ``port-S.csv``.

    Args:
        source: Results dict from ``run_local()`` / ``run()``, simulation path,
            or ``output/palace`` path.

    Returns:
        Parsed :class:`PalaceTextResults`.

    Raises:
        FileNotFoundError: If no parseable text output file is found.
    """
    # Reuse existing source normalization; for dict inputs preserve exact map.
    if isinstance(source, dict):
        files = {str(k): Path(v) for k, v in source.items()}
    else:
        _csv_path, base_dir = _resolve_source(source, require_csv=False)
        roots = [
            base_dir,
            base_dir / "output" / "palace",
            base_dir / "output",
        ]
        files = {}
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for p in root.iterdir():
                if p.is_file() and not p.name.startswith("."):
                    files.setdefault(p.name, p)

    parseable_suffixes = {".csv", ".txt", ".json", ".log"}
    parseable = {
        name: path
        for name, path in files.items()
        if path.suffix.lower() in parseable_suffixes
    }
    if not parseable:
        raise FileNotFoundError("No parseable Palace text output files found")

    csv_tables: dict[str, list[dict[str, str]]] = {}
    json_data: dict[str, object] = {}
    text_data: dict[str, list[str]] = {}

    for name, path in sorted(parseable.items()):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                rows = [{k or "": v or "" for k, v in row.items()} for row in reader]
            csv_tables[name] = rows
            continue

        if suffix == ".json":
            with path.open() as f:
                json_data[name] = json.load(f)
            continue

        with path.open(errors="replace") as f:
            text_data[name] = [line.rstrip("\n") for line in f]

    return PalaceTextResults(
        files=files,
        csv_tables=csv_tables,
        json_data=json_data,
        text_data=text_data,
    )


def load_refinement_history(source: str | Path) -> list[PalaceTextResults]:
    """Load the results of every adaptive mesh refinement pass, oldest first.

    With ``SaveAdaptIterations``, Palace keeps the output of pass X in an
    ``iterationX`` subdirectory and writes the last pass at the top level of
    its output directory. A run without refinement gives a single pass.

    Args:
        source: Simulation path, or Palace output directory.

    Returns:
        One :class:`PalaceTextResults` per pass, the final one last.
    """
    base = Path(source)
    passes: list[Path] = []
    for root in (base, base / "output" / "palace", base / "output"):
        passes = sorted(
            (
                d
                for d in root.glob("iteration*")
                if d.is_dir() and d.name.removeprefix("iteration").isdigit()
            ),
            key=lambda d: int(d.name.removeprefix("iteration")),
        )
        if passes:
            break
    return [load_text_results(d) for d in passes] + [load_text_results(base)]


def refinement_convergence(
    history: list[PalaceTextResults],
    metric: Callable[[PalaceTextResults], float],
) -> list[dict[str, float | int | None]]:
    """Tabulate Palace's error estimate and a chosen metric, pass by pass.

    Palace stops refining on its own error estimate, which does not mean that
    a given quantity (a frequency, an S-parameter, a capacitance) has
    converged. This lists both for each pass of :func:`load_refinement_history`.

    The relative change of the metric between two passes is not its error:
    if the changes shrink by a factor r per pass, the error left after the
    last pass is roughly the last change times r / (1 - r).

    Args:
        history: Passes from :func:`load_refinement_history`.
        metric: Function returning the quantity of interest for one pass.

    Returns:
        One row per pass with ``pass``, ``error_norm``, ``value`` and
        ``relative_change``: None for the first pass, NaN after a pass where
        the metric was zero.
    """
    table: list[dict[str, float | int | None]] = []
    previous = None
    for number, results in enumerate(history, start=1):
        value = metric(results)
        indicators = results.error_indicators or {}
        change = None
        if previous is not None:
            change = (value - previous) / abs(previous) if previous else float("nan")
        table.append(
            {
                "pass": number,
                "error_norm": indicators.get("norm"),
                "value": value,
                "relative_change": change,
            }
        )
        previous = value
    return table


def get_port_map(source: str | Path | dict) -> dict[int, str]:
    """Return the ``{port_number: port_name}`` mapping.

    Accepts a directory path or a results dict from ``sim.run()``.
    """
    csv_path, base_dir = _resolve_source(source, require_csv=False)
    return _load_port_map(base_dir, csv_path)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _numeric_csv_column(
    cells,
    csv_path: Path,
    column: str,
    *,
    allow_negative_infinity: bool = False,
) -> NDArray:
    """Parse numeric cells, reporting the source of invalid solver output."""
    values = []
    for row, cell in enumerate(cells, start=1):
        try:
            value = float(cell)
        except (TypeError, ValueError) as exc:
            msg = (
                f"Invalid numeric value {cell!r} in {csv_path}, "
                f"data row {row}, column {column!r}"
            )
            raise ValueError(msg) from exc
        if not np.isfinite(value) and not (
            allow_negative_infinity and np.isneginf(value)
        ):
            msg = (
                f"Invalid numeric value {cell!r} in {csv_path}, "
                f"data row {row}, column {column!r}"
            )
            raise ValueError(msg)
        values.append(value)
    return np.asarray(values, dtype=float)


def _parse_sparam_col(col: str) -> tuple[int, int, str] | None:
    """Parse a Palace S-parameter column header.

    Returns (i, j, "db"|"deg") or None.
    """
    # |S[i][j]| (dB)
    m = re.match(r"\|S\[(\d+)\]\[(\d+)\]\|\s*\((\w+\.?)\)", col)
    if m:
        return int(m.group(1)), int(m.group(2)), "db"

    # arg(S[i][j]) (deg.)
    m = re.match(r"arg\(S\[(\d+)\]\[(\d+)\]\)\s*\((\w+\.?)\)", col)
    if m:
        return int(m.group(1)), int(m.group(2)), "deg"

    return None


def _resolve_source(
    source: str | Path | dict,
    *,
    require_csv: bool = True,
) -> tuple[Path | None, Path]:
    """Turn *source* into ``(csv_path, base_dir)``."""
    if isinstance(source, dict):
        csv_val = source.get("port-S.csv")
        if csv_val is not None:
            csv_path = Path(csv_val)
            return csv_path, csv_path.parent
        for val in source.values():
            p = Path(val)
            if p.exists():
                return None, p.parent
        if require_csv:
            msg = "Results dict has no 'port-S.csv' entry"
            raise FileNotFoundError(msg)
        return None, Path()

    output_dir = Path(source)
    csv_path = _find_file(output_dir, "port-S.csv")
    if csv_path is None and require_csv:
        msg = f"port-S.csv not found in {output_dir} or its subdirectories"
        raise FileNotFoundError(msg)
    return csv_path, output_dir


def _load_port_map(
    output_dir: Path,
    csv_path: Path | None,
    port_info_path: str | Path | None = None,
) -> dict[int, str]:
    """Build ``{port_number: port_name}`` from ``port_information.json``."""
    if port_info_path is not None:
        info_path = Path(port_info_path)
    else:
        info_path = _find_port_info(output_dir, csv_path)

    if info_path is None or not info_path.exists():
        logger.warning(
            "port_information.json not found — using numeric port names (p1, p2, …)"
        )
        return {}

    with open(info_path) as f:
        data = json.load(f)

    port_map: dict[int, str] = {}
    for entry in data.get("ports", []):
        num = entry.get("portnumber")
        name = entry.get("name")
        if num is not None and name is not None:
            port_map[num] = name
        elif num is not None:
            port_map[num] = f"p{num}"

    if port_map and all(
        v.startswith("p") and v[1:].isdigit() for v in port_map.values()
    ):
        logger.info(
            "port_information.json has no 'name' fields — "
            "columns will use numeric names (p1, p2, …). "
            "Re-mesh to get named columns."
        )

    return port_map


def _reference_impedance(
    base_dir: Path,
    port_info_path: str | Path | None,
) -> float | None:
    """Read the unique reference impedance from ``port_information.json``.

    Returns ``None`` (caller falls back to 50 Ohm) when the file is
    missing or the ports declare differing reference impedances.
    """
    if port_info_path is None:
        info_path = _find_port_info(base_dir, None)
    else:
        info_path = Path(port_info_path)
        if not info_path.exists():
            info_path = _find_port_info(base_dir, None)
    if info_path is None or not info_path.exists():
        return None

    import json

    with open(info_path) as f:
        data = json.load(f)

    values = {
        float(entry["Z0"])
        for entry in data.get("ports", [])
        if entry.get("Z0") is not None
    }
    if len(values) == 1:
        return values.pop()
    if values:
        logger.warning(
            "Mixed port reference impedances %s; plots fall back to 50 Ohm labels",
            sorted(values),
        )
    return None


def _find_port_info(output_dir: Path, csv_path: Path | None) -> Path | None:
    """Search common locations for ``port_information.json``."""
    name = "port_information.json"
    candidates = [
        output_dir / name,
        output_dir / "output" / name,
    ]
    if csv_path is not None:
        candidates.insert(0, csv_path.parent / name)
        # Cloud layout: results/output/port-S.csv + results/input/port_info
        candidates.append(csv_path.parent.parent / "input" / name)
        candidates.append(csv_path.parent.parent / name)
        candidates.append(csv_path.parent.parent.parent / name)

    for p in candidates:
        if p.exists():
            return p
    return None


def _find_file(base: Path, name: str) -> Path | None:
    """Find *name* in *base* or common Palace subdirectories."""
    candidates = [
        base / name,
        base / "output" / "palace" / name,
        base / "palace" / name,
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = list(base.rglob(name))
    return matches[0] if matches else None


# -----------------------------------------------------------------------
# Field loading
# -----------------------------------------------------------------------


def load_fields(
    source: str | Path | dict,
    *,
    excitation: int = 1,
    cycle: int | None = None,
    mode: int | None = None,
    boundary: bool = False,
):
    """Load the ParaView volume or boundary dataset for a Palace simulation.

    Requires ``save_step >= 1`` for driven simulations or ``save >= 1`` for
    Eigenmode or BoundaryMode simulations so that field data was written to disk.

    Args:
        source: Results dict from ``sim.run_local()`` / ``sim.run()``,
            or a path to the simulation directory.
        excitation: Excitation index (1-based) to load.
        cycle: Exact ParaView cycle number, including metadata-only cycles.
            Defaults to the last cycle containing solution fields.
        mode: One-based Eigenmode or BoundaryMode mode number. Selects the
            corresponding cycle and requires solution fields. Mutually exclusive
            with ``cycle``.
        boundary: If ``True``, load boundary surface fields
            (for example, ``eigenmode_boundary/``) instead of volume fields.
            Boundary data includes ``J_s_real``,
            ``Q_s_real``, etc.

    Returns:
        ``pyvista.DataSet`` with point data such as
        ``E_real``, ``B_real``, ``S`` (volume) or
        ``J_s_real``, ``Q_s_real`` (boundary).

    Raises:
        FileNotFoundError: If paraview output is missing.
        ValueError: If selectors conflict or no solution fields are available.

    Example::

        vol = load_fields(results)
        bnd = load_fields(results, boundary=True)
        plot_cross_section(vol, normal="x", origin=0)
    """
    import pyvista as pv

    if mode is not None:
        if cycle is not None:
            raise ValueError("Specify mode or cycle, not both")
        if mode < 1:
            raise ValueError("mode must be a one-based positive integer")

    _, base_dir = _resolve_source(source, require_csv=False)
    pvtu_path = _find_paraview_dir(
        base_dir, excitation, cycle, boundary=boundary, mode=mode
    )
    return pv.read(str(pvtu_path))


def _find_paraview_dir(
    base_dir: Path,
    excitation: int,
    cycle: int | None,
    *,
    boundary: bool = False,
    mode: int | None = None,
) -> Path:
    """Locate the ``.pvtu`` file for the requested excitation and cycle.

    Palace uses two output layouts depending on the port type:
    - Multi-excitation (wave ports):  ``paraview/driven/excitation_N/CycleNNNNNN/``
    - Single-excitation (lumped ports): ``paraview/driven/CycleNNNNNN/`` (no subdir)
    Both are searched, with the explicit ``excitation_N`` folder taking priority.
    """
    solvers = (
        ["eigenmode", "boundarymode"]
        if mode is not None
        else ["driven", "boundarymode", "eigenmode"]
    )
    subdirs = [f"{solver}_boundary" if boundary else solver for solver in solvers]
    if mode is not None:
        cycle = mode
    search_roots = [
        base_dir,
        base_dir / "output" / "palace",
        base_dir / "output",
    ]
    exc_dir: Path | None = None
    for root in search_roots:
        for subdir in subdirs:
            # Layout 1: explicit excitation subfolder (wave ports / multi-excitation)
            candidate = root / "paraview" / subdir / f"excitation_{excitation}"
            if candidate.is_dir():
                exc_dir = candidate
                break

            # Layout 2: flat — Cycle dirs sit directly under the solver folder
            flat = root / "paraview" / subdir
            if flat.is_dir() and any(flat.glob("Cycle*")):
                exc_dir = flat
                break

        if exc_dir is not None:
            break

    if exc_dir is None:
        msg = (
            f"ParaView output not found for excitation {excitation}. "
            "Ensure the simulation was run with save_step >= 1 or save >= 1."
        )
        raise FileNotFoundError(msg)

    if cycle is not None:
        pvtu_dir = exc_dir / f"Cycle{cycle:06d}"
        candidates = sorted(pvtu_dir.rglob("*.pvtu"))
        if not candidates:
            msg = f"No .pvtu files found in {pvtu_dir}"
            raise FileNotFoundError(msg)
        selected = candidates[-1]
        if mode is not None and not _has_solution_fields(selected):
            raise ValueError(f"Mode {mode} has no solution fields in {selected}")
        return selected

    # Auto-select last available cycle that contains actual field data.
    # Palace writes a final cycle with mesh diagnostics. Indicator/Rank may
    # be point data or cell data, depending on the MFEM version.
    # Skip these cycles and pick the latest cycle with solution fields.
    candidates = sorted(exc_dir.rglob("*.pvtu"), reverse=True)
    if not candidates:
        msg = (
            f"No .pvtu files found under {exc_dir}. "
            "Ensure the simulation was run with save_step >= 1 or save >= 1."
        )
        raise FileNotFoundError(msg)

    for pvtu in candidates:
        if _has_solution_fields(pvtu):
            return pvtu

    raise ValueError(f"No solution fields found in ParaView cycles under {exc_dir}")


def _has_solution_fields(pvtu: Path) -> bool:
    """Distinguish physical fields from Palace's partition/error metadata."""
    import pyvista as pv

    dataset = pv.read(str(pvtu))
    field_names = set(dataset.point_data) | set(dataset.cell_data)
    return bool(field_names - {"Indicator", "Rank", "attribute"})
