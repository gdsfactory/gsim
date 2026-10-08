"""Palace AC circuit synthesis results and the reusable EM-to-circuit workflow.

This module covers everything between a Palace adaptive driven simulation and
a circuit model:

1. **Circuit synthesis parsing** — Palace >= 0.17 can synthesize a lumped
   L/R/C circuit from the reduced-order model of an adaptive driven
   simulation (``AdaptiveCircuitSynthesis: true`` in the ``Solver/Driven``
   section). The synthesized matrices are written next to the S-parameters
   as ``rom-*.csv`` files:

   - ``rom-Linv-re.csv`` / ``rom-Linv-im.csv`` — inverse inductance L^-1 [1/H]
   - ``rom-Rinv-re.csv`` / ``rom-Rinv-im.csv`` — inverse resistance R^-1 [S]
   - ``rom-C-re.csv`` / ``rom-C-im.csv`` — capacitance C [F]
   - ``rom-orthogonalization-matrix-R.csv`` — Gram-Schmidt R factor
   - ``rom-port-reference.csv`` — per-port reference Y_ref / Z_ref
   - ``rom-portload-<label>-{Linv,Rinv,C}-{re,im}.csv`` — per-port load blocks
   - ``rom-coupled-S.csv`` — S-parameters reconstructed from the circuit
   - ``rom-eigenvalues.csv`` — synthesized-circuit eigenfrequency estimates

   Each matrix file is square with node labels as the header row. Nodes are
   ordered: lumped ports (``port_<idx>_re``), wave ports
   (``waveport_<idx>_re/im``), synthesized interior nodes (``sample_*``), and
   - for frequency-dependent boundary conditions - auxiliary states
   (``<prefix>_p<k>d<j>``).

2. **Fitting** — lumped RLC equivalent-circuit models fitted to simulated
   impedance data (the exported circuit, EM S-parameters, or any complex
   Z(f) source):

   - ``model="rlc1p"``: the one-pole model (series R-L branch in parallel
     with C), parameterized by the resonance frequency ``f0``, quality
     factor ``Q`` and low-frequency resistance ``R``::

         z(w~, Q) = (1 + i w~ Q) / (1 - w~^2 + i w~ / Q)
         Z(f)     = R * z(f / f0, Q)

     solved with JAX/Adam in log space (optional dependencies) or a scipy
     least-squares fallback.
   - ``model="vector_fit"``: scikit-rf's rational multi-pole model
     (:class:`VectorFit`) with stable-pole and passivity checks
     (``passivity_test`` / ``passivity_enforce``) and SPICE export.

3. **Parameter conversions** — batched S <-> Z <-> Y utilities with
   explicit reference impedance (``s_to_z``, ``z_to_s``, ``s_to_y``,
   ``y_to_s``, ``z_to_y``, ``y_to_z``, ``is_complete``).

Usage::

    from gsim.palace import load_circuit_synthesis, fit_rlc

    circuit = load_circuit_synthesis(results)  # sim.run() output or dir
    Y = circuit.Y(results.freq * 1e9)  # (nf, N, N) admittance
    Y_dev = circuit.port_admittance(f, subtract_port_loads=True)  # bare device
    fit = circuit.fit_rlc(f)  # one-pole (R, L, C, f0, Q)
    fit_vf = circuit.fit_rlc(f, model="vector_fit")  # broadband + SPICE
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum Palace version with lumped-port circuit synthesis (rom-*.csv output).
MIN_PALACE_VERSION = "0.17.0"

# Resistances below this are treated as zero when seeding the RLC optimizer.
_MIN_R = 1e-9

Solver = Literal["auto", "jax", "scipy"]

_SKRF_HINT = (
    "scikit-rf is required for model='vector_fit': install it with "
    "`pip install scikit-rf` (or `uv add scikit-rf`)."
)

_PORT_NODE_RE = re.compile(r"^port_(\d+)_re$")
_WAVEPORT_NODE_RE = re.compile(r"^waveport_(\d+)_(re|im)$")
_AUX_NODE_RE = re.compile(
    r"^(?:waveport_\d+|farfield|surfsigma_\d+|rationalz_\d+)_p\d+d\d+$"
)


def _read_matrix_csv(path: Path) -> tuple[list[str], NDArray]:
    """Read a Palace rom matrix CSV: header row of node labels, then values."""
    import pandas as pd

    df = pd.read_csv(path)
    labels = [str(c).strip() for c in df.columns]
    return labels, df.to_numpy(dtype=float)


def _node_kind(label: str) -> Literal["port", "waveport", "sample", "aux"]:
    """Classify a synthesized circuit node label."""
    if _PORT_NODE_RE.match(label):
        return "port"
    if _WAVEPORT_NODE_RE.match(label):
        return "waveport"
    if _AUX_NODE_RE.match(label):
        return "aux"
    return "sample"


class CircuitSynthesis:
    """Synthesized L/R/C circuit from a Palace adaptive driven simulation.

    Attributes:
        nodes: Ordered node labels (ports first, then interior/aux nodes).
        L_inv: Inverse inductance matrix [1/H] (complex, N x N).
        R_inv: Inverse resistance matrix [S] (complex, N x N; zeros if lossless).
        C: Capacitance matrix [F] (complex, N x N).
        port_labels: Labels of the port terminal nodes.
        port_indices: Row/column indices of the port nodes in the matrices.
        port_loads: Per-port load matrices, ``{label: {"L_inv": ..., "R_inv": ...,
            "C": ...}}`` — the termination each port adds to the synthesized
            circuit. Subtract these to recover the bare device.
        orth_R: Gram-Schmidt R factor of the circuit modes (or ``None``).
        port_reference: ``rom-port-reference.csv`` rows (or ``None``).
        coupled_s: ``rom-coupled-S.csv`` rows (or ``None``).
        eigenvalues: ``rom-eigenvalues.csv`` rows (or ``None``).
        files: Mapping of parsed file name -> path.
    """

    def __init__(
        self,
        *,
        nodes: list[str],
        L_inv: NDArray,
        R_inv: NDArray,
        C: NDArray,
        port_loads: dict[str, dict[str, NDArray]] | None = None,
        orth_R: NDArray | None = None,
        port_reference: list[dict[str, str]] | None = None,
        coupled_s: list[dict[str, str]] | None = None,
        eigenvalues: list[dict[str, str]] | None = None,
        files: dict[str, Path] | None = None,
        port_names: dict[str, str] | None = None,
    ) -> None:
        """Create from parsed matrices (prefer :func:`load_circuit_synthesis`)."""
        n = len(nodes)
        for name, mat in (("L_inv", L_inv), ("R_inv", R_inv), ("C", C)):
            if mat.shape != (n, n):
                msg = f"{name} has shape {mat.shape}, expected ({n}, {n})"
                raise ValueError(msg)
        self.nodes = nodes
        self.L_inv = L_inv
        self.R_inv = R_inv
        self.C = C
        self.port_loads = port_loads or {}
        self.orth_R = orth_R
        self.port_reference = port_reference
        self.coupled_s = coupled_s
        self.eigenvalues = eigenvalues
        self.files = files or {}
        self.port_names: dict[str, str] = port_names or {}

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    @property
    def port_labels(self) -> list[str]:
        """Labels of the lumped/wave port terminal nodes."""
        return [n for n in self.nodes if _node_kind(n) in ("port", "waveport")]

    @property
    def port_indices(self) -> list[int]:
        """Row/column indices of the port terminal nodes."""
        return [
            i for i, n in enumerate(self.nodes) if _node_kind(n) in ("port", "waveport")
        ]

    @property
    def internal_indices(self) -> list[int]:
        """Row/column indices of interior (sample + aux) nodes."""
        return [
            i for i, n in enumerate(self.nodes) if _node_kind(n) in ("sample", "aux")
        ]

    def index(self, label: str) -> int:
        """Return the matrix row/column index of *label*."""
        try:
            return self.nodes.index(label)
        except ValueError:
            msg = f"Node {label!r} not in synthesized circuit nodes: {self.nodes}"
            raise KeyError(msg) from None

    # ------------------------------------------------------------------
    # Circuit evaluation
    # ------------------------------------------------------------------

    def _pencil(self, f: NDArray, *, subtract_port_loads: bool = False) -> NDArray:
        """Assemble Y(omega) = L^-1/(iomega) + R^-1 + iomega*C (frequencies in Hz).

        When *subtract_port_loads* is set, the per-port termination blocks
        (``rom-portload-*``) are removed, leaving the bare device pencil —
        the convention Palace documents for cascading and re-termination.
        """
        f = np.atleast_1d(np.asarray(f, dtype=float))
        omega = 2.0 * np.pi * f[:, None, None]
        Y = self.L_inv[None, :, :] / (1j * omega) + self.R_inv[None, :, :]
        Y = Y + 1j * omega * self.C[None, :, :]
        if subtract_port_loads and self.port_loads:
            zero = np.zeros_like(self.L_inv)
            for loads in self.port_loads.values():
                Y = Y - (
                    loads.get("L_inv", zero)[None, :, :] / (1j * omega)
                    + loads.get("R_inv", zero)[None, :, :]
                    + 1j * omega * loads.get("C", zero)[None, :, :]
                )
        return Y

    def Y(self, f: NDArray, *, subtract_port_loads: bool = False) -> NDArray:  # noqa: N802
        """Full synthesized admittance pencil, shape ``(nf, N, N)`` [S].

        Named ``Y`` to match the admittance convention
        ``Y(omega) = L^-1/(iomega) + R^-1 + iomega·C``.
        """
        return self._pencil(f, subtract_port_loads=subtract_port_loads)

    def port_admittance(
        self,
        f: NDArray,
        *,
        subtract_port_loads: bool = False,
    ) -> NDArray:
        """External port admittance seen at the synthesized circuit ports.

        Eliminates interior (sample + auxiliary) nodes via a Schur
        complement, leaving the ``(nf, Np, Np)`` admittance at the port
        terminals [S].

        Args:
            f: Frequencies in Hz.
            subtract_port_loads: Also subtract the per-port termination
                blocks (``rom-portload-*``) so the result is the bare device
                admittance (e.g. without the 50 Ohm port resistor).
        """
        Y = self._pencil(f, subtract_port_loads=subtract_port_loads)
        p_idx = self.port_indices
        i_idx = self.internal_indices
        Yp = Y[:, p_idx, :][:, :, p_idx]
        if i_idx:
            Yii = Y[:, i_idx, :][:, :, i_idx]
            Ypi = Y[:, p_idx, :][:, :, i_idx]
            Yip = Y[:, i_idx, :][:, :, p_idx]
            Yp = Yp - Ypi @ np.linalg.solve(Yii, Yip)
        return Yp

    def port_impedance(
        self, f: NDArray, *, subtract_port_loads: bool = True
    ) -> NDArray:
        """Port impedance matrix ``(nf, Np, Np)`` [Ohm] of the bare device."""
        Y = self.port_admittance(f, subtract_port_loads=subtract_port_loads)
        return np.linalg.inv(Y)

    def s_parameters(
        self,
        f: NDArray,
        z0: float | NDArray | None = None,
        *,
        subtract_port_loads: bool = True,
    ) -> NDArray:
        """S-parameters of the synthesized circuit at its port terminals.

        The bare-device port impedance (port loads subtracted by default) is
        referenced to *z0* via the power-wave relation
        ``S = (Z - Z0)(Z + Z0)^-1``.

        Args:
            f: Frequencies in Hz.
            z0: Reference impedance(s) [Ohm]. Defaults to the per-port
                termination read from the ``rom-portload`` resistance blocks
                (e.g. 50 Ohm), falling back to 50 Ohm when unavailable.
            subtract_port_loads: Subtract port loads before computing S
                (default, gives the device S-parameters).

        Returns:
            Complex S-parameter array of shape ``(nf, Np, Np)``.
        """
        Y = self.port_admittance(f, subtract_port_loads=subtract_port_loads)
        n_p = Y.shape[-1]
        if z0 is None:
            z0 = self.port_reference_impedances(default=50.0)
        z0 = np.asarray(z0, dtype=float) * np.ones(n_p)
        # Power-wave S from the admittance, valid for real diagonal Z0 and
        # equal to (Z - Z0)(Z + Z0)^-1 without inverting (possibly singular) Y.
        return y_to_s(Y, z0=z0)

    def port_reference_impedances(self, default: float = 50.0) -> NDArray:
        """Per-port real reference impedance from the ``rom-portload`` data.

        Falls back to *default* for ports without a resistive load.
        """
        z0 = np.full(len(self.port_labels), float(default))
        for k, label in enumerate(self.port_labels):
            load = self.port_loads.get(label, {})
            r_inv = load.get("R_inv")
            if isinstance(r_inv, np.ndarray):
                value = r_inv.diagonal()[self.index(label)].real
                if value > 0:
                    z0[k] = 1.0 / value
        return z0

    def fit_rlc(
        self,
        f: NDArray,
        *,
        model: Literal["rlc1p", "vector_fit"] = "rlc1p",
        subtract_port_loads: bool = True,
        z0: float | NDArray | None = None,
        steps: int = 1000,
        learning_rate: float = 0.05,
        solver: Literal["auto", "jax", "scipy"] = "auto",
        n_poles_real: int | None = 3,
        n_poles_cmplx: int = 3,
        enforce_passivity: bool = False,
        target_error: float = 1e-2,
    ) -> RLCFit | VectorFit:
        """Fit a circuit model to the exported circuit's terminal response.

        The exported circuit is an optional *model source*: evaluating the
        bare-device port admittance (interior nodes eliminated, port loads
        subtracted) and fitting yields compact parameters without touching
        the EM solver output. Two models are available:

        - ``model="rlc1p"``: one-pole (R, L, C, f0, Q) fit of the
          differential impedance (default). For a one-port circuit the
          driving-point impedance Z11 is used directly.
        - ``model="vector_fit"``: multi-pole rational model (scikit-rf
          VectorFitting) of the port S-parameters, with passivity test /
          enforcement.

        Args:
            f: Frequencies in Hz (e.g. the simulation sweep grid).
            model: ``"rlc1p"`` or ``"vector_fit"``.
            subtract_port_loads: De-embed the port terminations first
                (default) so the fit describes the bare device.
            z0: Reference impedance for the vector fit; defaults to the
                per-port values from the ``rom-portload`` data.
            steps: Optimizer iterations (one-pole, JAX solver only).
            learning_rate: Adam learning rate (one-pole, JAX solver only).
            solver: One-pole solver: ``"auto"`` (JAX if installed, else
                scipy), ``"jax"`` or ``"scipy"``.
            n_poles_real: Vector-fit real poles; ``None`` runs skrf's
                ``auto_fit`` loop with ``target_error``.
            n_poles_cmplx: Vector-fit complex pole pairs.
            enforce_passivity: Run ``passivity_enforce`` after the vector
                fit if the model fails the passivity test.
            target_error: Target RMS error for the ``auto_fit`` loop.

        Returns:
            :class:`~gsim.palace.fitting.RLCFit` (one-pole) or
            :class:`~gsim.palace.fitting.VectorFit` (vector fit).
        """
        if model == "rlc1p":
            z = differential_impedance(
                self.port_impedance(f, subtract_port_loads=subtract_port_loads)
            )
            return fit_rlc(
                f,
                z,
                solver=solver,
                steps=steps,
                learning_rate=learning_rate,
            )
        if model == "vector_fit":
            ref_z0 = z0 if z0 is not None else self.port_reference_impedances()
            s = self.s_parameters(
                f,
                z0=ref_z0,
                subtract_port_loads=subtract_port_loads,
            )
            return fit_rlc(
                f,
                s=s,
                model="vector_fit",
                z0=ref_z0,
                n_poles_real=n_poles_real,
                n_poles_cmplx=n_poles_cmplx,
                enforce_passivity=enforce_passivity,
                target_error=target_error,
            )
        raise ValueError(
            f"unknown fit model {model!r}; expected 'rlc1p' or 'vector_fit'"
        )

    # ------------------------------------------------------------------
    # Convenience readouts
    # ------------------------------------------------------------------

    @property
    def eigenfrequencies(self) -> NDArray:
        """Real parts of the estimated eigenfrequencies [Hz] of the circuit."""
        if not self.eigenvalues:
            return np.empty(0)
        try:
            import pandas as pd

            df = pd.DataFrame(self.eigenvalues)
            col = next((c for c in df.columns if c.strip().startswith("Re{f}")), None)
            if col is None:
                return np.empty(0)
            return df[col].to_numpy(dtype=float) * 1e9
        except Exception:  # pragma: no cover - defensive
            return np.empty(0)

    def __repr__(self) -> str:
        """Return concise object representation."""
        n_int = len(self.internal_indices)
        return (
            f"CircuitSynthesis(nodes={len(self.nodes)} "
            f"[{len(self.port_labels)} ports, {n_int} interior], "
            f"files={len(self.files)})"
        )


def load_circuit_synthesis(
    source: str | Path | dict,
    *,
    port_map: dict[int, str] | None = None,
) -> CircuitSynthesis:
    """Load Palace circuit-synthesis (``rom-*.csv``) results.

    Args:
        source: Results dict from ``sim.run()`` / ``run_local()``, an
            :class:`~gsim.palace.results.SParams` object, or a directory
            containing the Palace output files.
        port_map: Optional ``{palace_port_index: name}`` mapping used to
            attach port names to the parsed port nodes.

    Returns:
        Parsed :class:`CircuitSynthesis`.

    Raises:
        FileNotFoundError: If no ``rom-*.csv`` files are found.
    """
    files = _resolve_rom_files(source)
    if not files:
        msg = (
            "No Palace circuit-synthesis files (rom-*.csv) found. Run the "
            "simulation with adaptive sweep and circuit_synthesis=True."
        )
        raise FileNotFoundError(msg)

    first = files.get("rom-Linv-re.csv") or files.get("rom-C-re.csv")
    if first is None:  # pragma: no cover - rom-Linv is always written
        msg = "rom-Linv-re.csv missing from circuit-synthesis output"
        raise FileNotFoundError(msg)
    nodes, _ = _read_matrix_csv(first)
    n = len(nodes)

    def _load_complex(prefix: str) -> NDArray:
        mat = np.zeros((n, n), dtype=complex)
        re_path = files.get(f"{prefix}-re.csv")
        im_path = files.get(f"{prefix}-im.csv")
        if re_path is not None:
            labels, values = _read_matrix_csv(re_path)
            if [str(c).strip() for c in labels] != nodes:
                logger.warning("%s node labels differ from rom-Linv-re", prefix)
            mat += values
        if im_path is not None:
            mat += 1j * _read_matrix_csv(im_path)[1]
        return mat

    L_inv = _load_complex("rom-Linv")
    R_inv = _load_complex("rom-Rinv")
    C = _load_complex("rom-C")

    # Per-port load blocks: rom-portload-<label>-{Linv,Rinv,C}-{re,im}.csv
    port_loads: dict[str, dict[str, NDArray]] = {}
    for name in files:
        m = re.match(r"^rom-portload-(.+)-(Linv|Rinv|C)-(re|im)\.csv$", name)
        if m is None:
            continue
        label, part, sign = m.group(1), m.group(2), m.group(3)
        _, values = _read_matrix_csv(files[name])
        key = {"Linv": "L_inv", "Rinv": "R_inv", "C": "C"}[part]
        slot = port_loads.setdefault(label, {})
        base = slot.get(key)
        if base is None:
            base = np.zeros((n, n), dtype=complex)
            slot[key] = base
        if sign == "re":
            slot[key] = base + values
        else:
            slot[key] = base + 1j * values

    orth_R = None
    if "rom-orthogonalization-matrix-R.csv" in files:
        orth_R = _read_matrix_csv(files["rom-orthogonalization-matrix-R.csv"])[1]

    port_reference = _read_table(files.get("rom-port-reference.csv"))
    coupled_s = _read_table(files.get("rom-coupled-S.csv"))
    eigenvalues = _read_table(files.get("rom-eigenvalues.csv"))

    port_names: dict[str, str] = {}
    if port_map:
        for label in nodes:
            m = _PORT_NODE_RE.match(label)
            if m is not None and int(m.group(1)) in port_map:
                port_names[label] = port_map[int(m.group(1))]

    return CircuitSynthesis(
        nodes=nodes,
        L_inv=L_inv,
        R_inv=R_inv,
        C=C,
        port_loads=port_loads,
        orth_R=orth_R,
        port_reference=port_reference,
        coupled_s=coupled_s,
        eigenvalues=eigenvalues,
        files=files,
        port_names=port_names,
    )


def _read_table(path: Path | None) -> list[dict[str, str]] | None:
    """Read a tabular (non-matrix) rom CSV into row dicts."""
    if path is None:
        return None
    import csv

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [{k or "": v or "" for k, v in row.items()} for row in reader]


def _resolve_rom_files(source: str | Path | dict) -> dict[str, Path]:
    """Collect ``rom-*.csv`` files from a results dict or directory."""
    files: dict[str, Path] = {}
    if isinstance(source, dict):
        candidates = list(source.values())
    elif isinstance(source, (str, Path)):
        base = Path(source)
        candidates = sorted(base.rglob("rom-*.csv")) if base.is_dir() else [base]
    else:
        # SParams-like object carrying a files mapping
        candidates = list(getattr(source, "files", {}).values())

    for value in candidates:
        if value is None:
            continue
        path = Path(value)
        if path.is_file() and path.name.startswith("rom-") and path.suffix == ".csv":
            files.setdefault(path.name, path)
    return files


# ----------------------------------------------------------------------
# Fitting: one-pole RLC and multi-pole vector models
# ----------------------------------------------------------------------
# Resistances below this are treated as zero when seeding the optimizer.
_MIN_R = 1e-9


@dataclass(frozen=True)
class RLCFit:
    """Fitted one-pole RLC equivalent circuit.

    Attributes:
        R: Low-frequency series resistance [Ohm].
        L: Series inductance [H].
        C: Parallel capacitance [F].
        f0: Resonance frequency (1 / (2 pi sqrt(L C))) [Hz].
        Q: Quality factor at resonance (2 pi f0 L / R).
        rms_error: RMS of |Z_model - Z_data| over the fit band [Ohm].
    """

    R: float
    L: float
    C: float
    f0: float
    Q: float
    rms_error: float

    def z(self, f: NDArray) -> NDArray:
        """Model impedance [Ohm] at frequencies *f* in Hz."""
        return 1.0 / self.y(f)

    def y(self, f: NDArray) -> NDArray:
        """Model admittance [S] at frequencies *f* in Hz."""
        f = np.asarray(f, dtype=float)
        w = 2.0 * np.pi * f
        return 1.0 / (self.R + 1j * w * self.L) + 1j * w * self.C

    def to_dict(self) -> dict[str, float]:
        """Return the fitted parameters as a plain dict (SI units)."""
        return {
            "R": self.R,
            "L": self.L,
            "C": self.C,
            "f0": self.f0,
            "Q": self.Q,
            "rms_error": self.rms_error,
        }

    def __repr__(self) -> str:
        """Return concise object representation."""
        return (
            f"RLCFit(R={self.R:.4g} Ohm, L={self.L * 1e12:.3f} pH, "
            f"C={self.C * 1e15:.3f} fF, f0={self.f0 / 1e9:.3f} GHz, "
            f"Q={self.Q:.2f}, rms={self.rms_error:.2e})"
        )


def z_rlc(w_norm, Q):
    """Normalized RLC impedance ``z(w~, Q) = (1 + i w~ Q) / (1 - w~^2 + i w~/Q)``.

    ``Z(f) = R * z_rlc(f / f0, Q)``. Untyped parameters on purpose: the
    formula is used with both numpy and jax arrays.
    """
    return (1 + 1j * w_norm * Q) / (1 - w_norm**2 + 1j * w_norm / Q)


class VectorFit:
    """Multi-pole rational fit (scikit-rf ``VectorFitting``) with passivity.

    Wraps :class:`skrf.vectorFitting.VectorFitting` and adds S/Z/Y
    evaluation, stability/passivity reporting, spurious-pole detection and
    SPICE export. Stability means every pole lies in the open left half
    plane. Passivity is scikit-rf's ``is_passive`` on the fitted rational
    model (run :meth:`passivity_enforce` when it fails).
    """

    def __init__(self, vf, *, z0: float | NDArray) -> None:
        """Wrap a fitted :class:`skrf.vectorFitting.VectorFitting`."""
        self._vf = vf
        self.z0: float | NDArray = z0

    # -- underlying model --------------------------------------------------

    @property
    def raw(self):
        """The wrapped :class:`skrf.vectorFitting.VectorFitting`."""
        return self._vf

    @property
    def network(self):
        """The fitted skrf ``Network`` (measured data used for training)."""
        return self._vf.network

    @property
    def poles(self) -> NDArray:
        """Model poles [rad/s] (complex)."""
        return np.asarray(self._vf.poles)

    @property
    def residues(self) -> NDArray:
        """Model residues (one row/port pair)."""
        return np.asarray(self._vf.residues)

    @property
    def zeros(self) -> NDArray:
        """Model zeros [rad/s]."""
        return np.asarray(self._vf.zeros)

    @property
    def n_poles(self) -> int:
        """Number of poles in the fitted model."""
        return len(self._vf.poles)

    # -- quality / passivity -----------------------------------------------

    @property
    def is_stable(self) -> bool:
        """Whether all poles are in the open left half plane."""
        poles = self.poles
        return bool(np.all(poles.real < 0.0))

    def rms_error(self, **kwargs) -> float:
        """RMS error of the rational model (delegates to skrf)."""
        return float(self._vf.get_rms_error(**kwargs))

    def is_passive(self, **kwargs) -> bool:
        """Whether the fitted model passes scikit-rf's passivity test."""
        return bool(self._vf.is_passive(**kwargs))

    def passivity_test(self, **kwargs) -> NDArray:
        """Passivity metric array (delegates to skrf ``passivity_test``)."""
        return np.asarray(self._vf.passivity_test(**kwargs))

    def passivity_enforce(self, **kwargs) -> VectorFit:
        """Enforce passivity (delegates to skrf ``passivity_enforce``)."""
        self._vf.passivity_enforce(**kwargs)
        return self

    def get_spurious(self, **kwargs) -> NDArray:
        """Boolean mask of spurious poles (delegates to skrf)."""
        return np.asarray(self._vf.get_spurious(self.poles, self.residues, **kwargs))

    # -- response evaluation ------------------------------------------------

    def s(self, f: NDArray | None = None) -> NDArray:
        """Model S-parameters ``(nf, N, N)``.

        With ``f=None`` the rational model is evaluated at the *training*
        frequencies of the fit (not the training data — use ``.network.s``
        for that).
        """
        if f is None:
            freqs = np.asarray(self._vf.network.f, dtype=float)  # Hz
        else:
            freqs = np.atleast_1d(np.asarray(f, dtype=float))
        n_ports = self._vf.network.s.shape[-1]
        out = np.empty((len(freqs), n_ports, n_ports), dtype=complex)
        for i in range(n_ports):
            for j in range(n_ports):
                out[:, i, j] = self._vf.get_model_response(i, j, freqs)
        return out

    def z(self, f: NDArray | None = None) -> NDArray:
        """Model impedance matrices ``(nf, N, N)`` [Ohm].

        Uses the symmetric power-wave form for a real diagonal reference
        impedance: ``Z = sqrt(Z0) (I - S)^-1 (I + S) sqrt(Z0)``, which is
        exact when the ports carry different reference impedances.
        """
        s_model = self.s(f)
        n_freq, n_ports = s_model.shape[0], s_model.shape[-1]
        eye = _eye(n_ports)
        sqrt_z0, _ = _z0_sqrt_matrices(self.z0, n_freq, n_ports)
        return sqrt_z0 @ (np.linalg.inv(eye - s_model) @ (eye + s_model)) @ sqrt_z0

    def y(self, f: NDArray | None = None) -> NDArray:
        """Model admittance matrices ``(nf, N, N)`` [S].

        Uses the symmetric form ``Y = sqrt(Z0)^-1 (I + S)^-1 (I - S)
        sqrt(Z0)^-1``; computing ``1/Z`` of :meth:`z` is not equivalent
        when the ports carry different reference impedances.
        """
        s_model = self.s(f)
        n_freq, n_ports = s_model.shape[0], s_model.shape[-1]
        eye = _eye(n_ports)
        _, inv_z0 = _z0_sqrt_matrices(self.z0, n_freq, n_ports)
        return inv_z0 @ (np.linalg.inv(eye + s_model) @ (eye - s_model)) @ inv_z0

    # -- export --------------------------------------------------------------

    def write_spice(self, path: str, **kwargs) -> None:
        """Export the rational model as a SPICE subcircuit (skrf)."""
        self._vf.write_spice_subcircuit_s(str(path), **kwargs)

    def __repr__(self) -> str:
        """Return concise object representation."""
        return (
            f"VectorFit(n_poles={self.n_poles}, ports={self._vf.network.s.shape[-1]}, "
            f"stable={self.is_stable}, passive={self.is_passive()}, "
            f"rms={self.rms_error():.2e})"
        )


def differential_impedance(z: NDArray) -> NDArray:
    """Differential impedance ``Z11 - Z12 - Z21 + Z22`` of a Z-matrix.

    Args:
        z: Impedance matrices with shape ``(nf, N, N)`` (e.g. from
            ``skrf.Network.z`` or :meth:`CircuitSynthesis.port_impedance`).
            For a single-port network (``N == 1``) the driving-point
            impedance ``Z11`` is returned directly — it is already the
            differential quantity.

    Returns:
        Array of shape ``(nf,)`` — the impedance seen between the two
        ports of a differential pair under differential excitation.
    """
    z = np.asarray(z)
    if z.ndim != 3 or z.shape[-1] != z.shape[-2] or z.shape[-1] < 1:
        raise ValueError(
            f"expected (nf, N, N) impedance matrix with N >= 1, got {z.shape}"
        )
    if z.shape[-1] == 1:
        return z[:, 0, 0]
    return z[:, 0, 0] - z[:, 0, 1] - z[:, 1, 0] + z[:, 1, 1]


def initial_guess_rlc(f: NDArray, z: NDArray) -> tuple[float, float, float]:
    """Estimate ``(f0, Q, R)`` directly from impedance data.

    - ``f0`` is the frequency where |Z| peaks,
    - ``R`` is the low-frequency resistance Re(Z),
    - ``Q`` comes from the -3 dB bandwidth of the |Z| peak.
    """
    f = np.asarray(f, dtype=float)
    z = np.asarray(z)
    abs_z = np.abs(z)
    f0 = float(f[int(np.argmax(abs_z))])
    r = max(float(np.real(z[0])), _MIN_R)
    mask = abs_z > abs_z.max() / np.sqrt(2)
    q = f0 / float(np.ptp(f[mask])) if mask.sum() > 1 else 5.0
    return f0, q, r


def fit_rlc(
    f: NDArray,
    z: NDArray | None = None,
    *,
    s: NDArray | None = None,
    model: Literal["rlc1p", "vector_fit"] = "rlc1p",
    z0: float | NDArray = 50.0,
    solver: Solver = "auto",
    steps: int = 1000,
    learning_rate: float = 0.05,
    n_poles_real: int | None = 3,
    n_poles_cmplx: int = 3,
    enforce_passivity: bool = False,
    target_error: float = 1e-2,
) -> RLCFit | VectorFit:
    """Fit a circuit model to (S- or Z-parameter) data.

    Two models are available:

    - ``model="rlc1p"`` — the one-pole equivalent circuit
      (series R-L branch in parallel with C), returned as
      :class:`RLCFit`. Fits impedance data ``z`` with the
      normalized ``(f0, Q, R)`` parameterization (JAX/Adam in log
      space when the optional JAX dependencies are installed, else a
      scipy least-squares fit).
    - ``model="vector_fit"`` — scikit-rf's rational (vector-fitting)
      model with multiple poles, returned as :class:`VectorFit`.
      Fits S-parameter data ``s`` (or ``z``, converted first, and for
      one-port data ``z`` may be a scalar series). Supports a built-in
      passivity test and enforcement (``passivity_test`` /
      ``passivity_enforce``), and SPICE export
      (``write_spice_subcircuit_s``). Requires the optional ``skrf``
      dependency.

    Args:
        f: Frequencies in Hz.
        z: Complex impedance data [Ohm]: either a scalar series
            ``(nf,)`` or a full impedance matrix ``(nf, N, N)`` (the
            one-pole model uses the differential impedance
            ``Z11 - Z12 - Z21 + Z22`` of the matrix). One of ``z``/``s``
            must be given.
        s: Complex S-parameter data ``(nf, N, N)``; used by
            ``model="vector_fit"`` directly.
        model: ``"rlc1p"`` or ``"vector_fit"``.
        z0: Reference impedance [Ohm] used by ``model="vector_fit"``
            when converting impedance data to S-parameters (scalar,
            or per-port array of length N).
        steps: Optimizer iterations (one-pole, JAX solver only).
        learning_rate: Adam learning rate (one-pole, JAX solver only).
        solver: One-pole solver: ``"auto"`` (JAX if installed, else
            scipy), ``"jax"`` or ``"scipy"``.
        n_poles_real: Number of real poles for the vector fit. Pass
            ``None`` to run scikit-rf's ``auto_fit`` pole-adding loop
            with ``target_error``.
        n_poles_cmplx: Number of complex pole pairs for the vector fit.
        enforce_passivity: For ``model="vector_fit"``, run
            ``passivity_enforce`` after fitting when the model fails the
            passivity test.
        target_error: Target RMS error for the ``auto_fit`` loop
            (vector fit with ``n_poles_real=None``).

    Returns:
        :class:`RLCFit` (one-pole) or :class:`VectorFit` (vector fit).
    """
    if model == "vector_fit":
        return _fit_vector_skrf(
            f,
            z=z,
            s=s,
            z0=z0,
            n_poles_real=n_poles_real,
            n_poles_cmplx=n_poles_cmplx,
            enforce_passivity=enforce_passivity,
            target_error=target_error,
        )
    if model != "rlc1p":
        raise ValueError(
            f"unknown fit model {model!r}; expected 'rlc1p' or 'vector_fit'"
        )
    if s is not None:
        if z is not None:
            raise ValueError("provide either z or s, not both")
        raise ValueError(
            "model='rlc1p' fits impedance data: provide z "
            "(s is only used by model='vector_fit')"
        )
    if z is None:
        raise ValueError("provide impedance data z (or use model='vector_fit' with s)")
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=complex)
    if z.ndim == 3:
        z = differential_impedance(z)
    if solver == "auto":
        solver = "jax" if _jax_available() else "scipy"

    if solver == "jax":
        f0, q, r = _fit_rlc_jax(f, z, steps=steps, learning_rate=learning_rate)
    else:
        f0, q, r = _fit_rlc_scipy(f, z)
    return _finalize_fit(f, z, f0, q, r)


def _fit_vector_skrf(
    f: NDArray,
    *,
    z: NDArray | None,
    s: NDArray | None,
    z0: float | NDArray,
    n_poles_real: int | None,
    n_poles_cmplx: int,
    enforce_passivity: bool,
    target_error: float,
) -> VectorFit:
    """Fit a scikit-rf rational model to S- (or converted Z-) data."""
    try:
        import skrf as rf
        from skrf.vectorFitting import VectorFitting as _SkrfVectorFitting
    except ImportError as exc:
        raise ImportError(_SKRF_HINT) from exc

    f = np.asarray(f, dtype=float)
    s_data = _to_s_data(f, z, s, z0)
    z0_net = _network_z0(z0, s_data.shape[1], len(f))
    network = rf.Network(
        frequency=rf.Frequency.from_f(f, unit="Hz"),
        s=s_data,
        z0=z0_net,
        name="fit_rlc",
    )
    vf = _SkrfVectorFitting(network)
    if n_poles_real is None:
        vf.auto_fit(target_error=target_error)
    else:
        vf.vector_fit(n_poles_real=n_poles_real, n_poles_cmplx=n_poles_cmplx)
    result = VectorFit(vf, z0=z0_net)
    if enforce_passivity and not result.is_passive():
        result.passivity_enforce(f_max=float(np.max(f)))
    return result


def _to_s_data(
    f: NDArray, z: NDArray | None, s: NDArray | None, z0: float | NDArray
) -> NDArray:
    """Normalize the ``z``/``s`` inputs into an ``(nf, N, N)`` S array."""
    del f
    if z is not None and s is not None:
        raise ValueError("provide either z or s, not both")
    if s is not None:
        s = np.asarray(s, dtype=complex)
        if s.ndim != 3:
            raise ValueError(f"s must have shape (nf, N, N), got {np.asarray(s).shape}")
        return s
    if z is None:
        raise ValueError("provide impedance data z (or S-parameter data s)")
    z = np.asarray(z, dtype=complex)
    if z.ndim == 1:
        if not np.isscalar(z0):
            raise ValueError("1-port z data requires a scalar z0")
        z0_value = complex(float(np.asarray(z0, dtype=float)))
        s_data = (z - z0_value) / (z + z0_value)
        return s_data[:, None, None]
    if z.ndim != 3 or z.shape[-1] != z.shape[-2]:
        raise ValueError(f"z must have shape (nf,) or (nf, N, N), got {z.shape}")
    # Single-ended power waves with a real diagonal reference impedance:
    # S = (W - I)(W + I)^-1 with W = sqrt(Z0)^-1 Z sqrt(Z0)^-1. This is exact
    # for per-port z0 (S = (I - Z0 Y)(I + Z0 Y)^-1 is not - it mixes rows).
    eye = _eye(z.shape[-1])
    _, inv_z0 = _z0_sqrt_matrices(z0, z.shape[0], z.shape[-1])
    w = inv_z0 @ z @ inv_z0
    return (w - eye) @ np.linalg.inv(w + eye)


def _network_z0(z0: float | NDArray, n_ports: int, n_freq: int) -> float | NDArray:
    """Broadcast a per-port z0 to skrf's expected ``(nf, N)`` layout."""
    arr = np.asarray(z0, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape == (n_ports,):
        return np.tile(arr, (n_freq, 1))
    if arr.shape == (n_freq, n_ports):
        return arr
    raise ValueError(f"z0 must be scalar, (N,) or (nf, N), got {arr.shape}")


def _finalize_fit(f: NDArray, z: NDArray, f0: float, q: float, r: float) -> RLCFit:
    """Build an :class:`RLCFit` from (f0, Q, R) and score it against the data."""
    f0 = max(float(f0), 1.0)
    q = max(float(q), 1e-6)
    r = max(float(r), _MIN_R)
    w0 = 2.0 * np.pi * f0
    inductance = q * r / w0
    capacitance = 1.0 / (inductance * w0**2)
    model = r * z_rlc(f / f0, q)
    rms = float(np.sqrt(np.mean(np.abs(model - z) ** 2)))
    return RLCFit(R=r, L=inductance, C=capacitance, f0=f0, Q=q, rms_error=rms)


def _jax_available() -> bool:
    """Return whether the optional JAX dependencies can be imported."""
    try:
        import jax  # noqa: F401
        import optax  # noqa: F401
    except ImportError:
        return False
    return True


def _fit_rlc_jax(
    f: NDArray,
    z: NDArray,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[float, float, float]:
    """Adam + autodiff fit on the normalized (f0, Q, R) parameterization.

    Optimization runs in log space: the three parameters span decades of
    magnitude (f0 ~ 1e11 Hz vs R ~ Ohm), so a shared learning rate is only
    scale-free after this reparameterization.
    """
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)

    f_j = jnp.asarray(f, dtype=jnp.float64)
    z_t = jnp.asarray(z, dtype=jnp.complex128)

    @jax.jit
    def loss_fn(log_param):
        f0, q, r = jnp.exp(log_param[0]), jnp.exp(log_param[1]), jnp.exp(log_param[2])
        z_fit = r * z_rlc(f_j / f0, q)
        z_err = z_t - z_fit
        return jnp.real(jnp.sum(z_err * jnp.conj(z_err)))

    f0_ini, q_ini, r_ini = initial_guess_rlc(f, z)
    par = jnp.log(jnp.array([f0_ini, q_ini, r_ini]))
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(par)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(steps):
        _, grads = value_and_grad(par)
        updates, opt_state = optimizer.update(grads, opt_state)
        par = optax.apply_updates(par, updates)
    f0, q, r = (float(x) for x in np.exp(np.asarray(par)))
    return f0, q, r


def _fit_rlc_scipy(f: NDArray, z: NDArray) -> tuple[float, float, float]:
    """Least-squares fit without the optional JAX dependencies."""
    from scipy.optimize import least_squares

    def residuals(param):
        model = param[2] * z_rlc(f / param[0], param[1])
        return np.r_[model.real - z.real, model.imag - z.imag]

    f0_ini, q_ini, r_ini = initial_guess_rlc(f, z)
    result = least_squares(
        residuals,
        x0=[f0_ini, q_ini, r_ini],
        bounds=([np.min(f), 1e-6, 0.0], [np.max(f) * 10, 1e4, np.inf]),
        xtol=1e-15,
        ftol=1e-15,
    )
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


# ----------------------------------------------------------------------
# Batched S <-> Z <-> Y parameter conversions
# ----------------------------------------------------------------------
def _as_matrix(x: NDArray, name: str) -> NDArray:
    """Validate a parameter matrix is the complete (nf, N, N) stack."""
    arr = np.asarray(x, dtype=complex)
    if arr.ndim != 3 or arr.shape[-1] != arr.shape[-2] or arr.shape[-1] < 1:
        raise ValueError(
            f"{name} must be a complete (nf, N, N) matrix stack, got {arr.shape}"
        )
    return arr


def _eye(n_ports: int) -> NDArray:
    """Identity matrix alias with complex dtype."""
    return np.eye(n_ports, dtype=complex)


def _z0_sqrt_matrices(
    z0: float | NDArray, n_freq: int, n_ports: int
) -> tuple[NDArray, NDArray]:
    """Return ``(sqrt(Z0), 1/sqrt(Z0))`` as (nf, N, N) diagonal matrices.

    The power-wave conversions for a *real diagonal* reference impedance
    (the standard Palace / 50-Ohm case, with scalar or per-port values)
    distribute sqrt(Z0) symmetrically on both sides::

        Z = sqrt(Z0) (I - S)^-1 (I + S) sqrt(Z0)
        S = (W - I)(W + I)^-1,   W = Z0^-1/2 Z Z0^-1/2
        Y = sqrt(Z0)^-1 (I + S)^-1 (I - S) sqrt(Z0)^-1
        S = (I + N)^-1 (I - N),  N = sqrt(Z0) Y sqrt(Z0)

    Multiplying Z0 on only one side is only correct when all ports share
    the same reference impedance; with per-port values it distorts the
    result (verified against scikit-rf's s2z/z2s).
    """
    arr = np.asarray(z0, dtype=float)
    accepted = ((), (n_ports,), (n_freq, n_ports))
    if arr.ndim > 2 or arr.shape not in accepted:
        raise ValueError(f"z0 must be scalar, length-N or (nf, N); got {arr.shape}")
    if np.any(arr <= 0):
        raise ValueError("z0 entries must be positive for sqrt-form conversions")
    if arr.ndim == 0:
        sqrt = np.eye(n_ports) * float(arr) ** 0.5
        inv = np.eye(n_ports) / float(arr) ** 0.5
        return np.tile(sqrt, (n_freq, 1, 1)), np.tile(inv, (n_freq, 1, 1))
    if arr.ndim == 2:  # (nf, N): one row per frequency
        sqrt = np.stack([np.diag(np.sqrt(row)) for row in arr])
        inv = np.stack([np.diag(1.0 / np.sqrt(row)) for row in arr])
        return sqrt, inv
    sqrt = np.tile(np.diag(np.sqrt(arr)), (n_freq, 1, 1))
    inv = np.tile(np.diag(1.0 / np.sqrt(arr)), (n_freq, 1, 1))
    return sqrt, inv


def s_to_z(s: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert scattering to impedance parameters, ``(nf, N, N)`` [Ohm]."""
    s = _as_matrix(s, "s")
    n_freq, n_ports = s.shape[0], s.shape[-1]
    eye = _eye(n_ports)
    sqrt_z0, _ = _z0_sqrt_matrices(z0, n_freq, n_ports)
    return sqrt_z0 @ (np.linalg.inv(eye - s) @ (eye + s)) @ sqrt_z0


def z_to_s(z: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert impedance to scattering parameters, ``(nf, N, N)``."""
    z = _as_matrix(z, "z")
    n_freq, n_ports = z.shape[0], z.shape[-1]
    eye = _eye(n_ports)
    _, inv_z0 = _z0_sqrt_matrices(z0, n_freq, n_ports)
    w = inv_z0 @ z @ inv_z0
    return (w - eye) @ np.linalg.inv(w + eye)


def s_to_y(s: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert scattering to admittance parameters, ``(nf, N, N)`` [S]."""
    s = _as_matrix(s, "s")
    n_freq, n_ports = s.shape[0], s.shape[-1]
    eye = _eye(n_ports)
    _, inv_z0 = _z0_sqrt_matrices(z0, n_freq, n_ports)
    return inv_z0 @ (np.linalg.inv(eye + s) @ (eye - s)) @ inv_z0


def y_to_s(y: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert admittance to scattering parameters, ``(nf, N, N)``."""
    y = _as_matrix(y, "y")
    n_freq, n_ports = y.shape[0], y.shape[-1]
    eye = _eye(n_ports)
    sqrt_z0, _ = _z0_sqrt_matrices(z0, n_freq, n_ports)
    n = sqrt_z0 @ y @ sqrt_z0
    return np.linalg.solve(n + eye, eye - n)


def y_to_z(y: NDArray) -> NDArray:
    """Convert admittance to impedance parameters, ``(nf, N, N)`` [Ohm]."""
    return np.linalg.inv(_as_matrix(y, "y"))


def z_to_y(z: NDArray) -> NDArray:
    """Convert impedance to admittance parameters, ``(nf, N, N)`` [S]."""
    return np.linalg.inv(_as_matrix(z, "z"))


def is_complete(x: NDArray) -> bool:
    """Whether *x* is a full, finite ``(nf, N, N)`` parameter stack."""
    try:
        arr = _as_matrix(x, "x")
    except ValueError:
        return False
    return bool(np.all(np.isfinite(arr.view(float))))


__all__ = [
    "MIN_PALACE_VERSION",
    "CircuitSynthesis",
    "RLCFit",
    "VectorFit",
    "differential_impedance",
    "fit_rlc",
    "initial_guess_rlc",
    "is_complete",
    "load_circuit_synthesis",
    "s_to_y",
    "s_to_z",
    "y_to_s",
    "y_to_z",
    "z_rlc",
    "z_to_s",
    "z_to_y",
]
