"""Compact-model export of a uniform line (pure circuit functions).

The Traveling-wave electrode leaves gsim as a two-port: the solved line
parameters ``gamma(f)`` and ``Z0(f)`` plus a chosen length become the
uniform line's S-matrix (:func:`line_smatrix`), written to a Touchstone
``.s2p`` file (:func:`write_touchstone`) or wrapped as a callable
following the SAX model convention (:func:`sax_line_model`) — a plain
function over numpy arrays returning a dict of S-matrix entries, with no
sax import anywhere. This is gsim's half of the compact-model handoff to
a circuit simulator such as circulax: the physics is solved here, the
assembly happens there.

The junction's compact model travels the same road: the series-RC shunt
branch fitted per bias point is written to a tabular JSON file
(:func:`write_junction_model`, whose docstring is the format reference)
and read back with nothing but the stdlib and numpy
(:func:`read_junction_model`). Touchstone files are written and read
through scikit-rf (:func:`write_touchstone`, :func:`read_touchstone`),
which handles every unit, format and option-line variant of the
standard. Round-trip validation of both artifacts uses those readers
and the driven-line responses (:func:`terminated_response` from an S-matrix,
:func:`line_driven_response` from the solved line parameters).

The S-matrix referenced to a real ``Z_ref`` follows the standard
telegrapher's two-port (e.g. Pozar, *Microwave Engineering*, ch. 4)::

    S11 = S22 = (Zc^2 - Zr^2) sinh(gl) / D
    S21 = S12 = 2 Zc Zr / D
    D = 2 Zc Zr cosh(gl) + (Zc^2 + Zr^2) sinh(gl)

with ``Zc`` the line's characteristic impedance, ``Zr`` the reference,
and ``gl = gamma * length`` the complex electrical length. Sign
convention matches the rest of gsim: ``gamma = alpha + j beta`` with
``alpha >= 0`` for a lossy line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gsim.common.transmission_line import line_params_from_gamma, section_abcd

__all__ = [
    "JUNCTION_MODEL_FORMAT",
    "JUNCTION_MODEL_VERSION",
    "JunctionModel",
    "SaxLineModel",
    "TwoPort",
    "line_driven_response",
    "line_smatrix",
    "read_junction_model",
    "read_touchstone",
    "sax_line_model",
    "terminated_response",
    "write_junction_model",
    "write_touchstone",
]

#: The ``format`` field naming a junction compact-model file.
JUNCTION_MODEL_FORMAT: str = "gsim-junction-model"

#: Schema version written by :func:`write_junction_model`.
JUNCTION_MODEL_VERSION: int = 1


class SaxLineModel(Protocol):
    """A SAX-convention model: keyword arguments in, S-dict out."""

    def __call__(
        self, *, f: ArrayLike | None = None
    ) -> dict[tuple[str, str], NDArray[np.complex128]]:
        """Evaluate the S-matrix entries at frequencies ``f`` (Hz)."""
        ...


def _require_positive_length(length_m: float) -> None:
    """Refuse a line of zero or negative length."""
    if length_m <= 0.0:
        raise ValueError("length_m must be a positive line length in meters.")


def _require_ascending(freq_hz: NDArray[np.float64], where: str) -> None:
    """Refuse a frequency axis ``np.interp`` (or Touchstone) would misread.

    Args:
        freq_hz: The frequency axis (1D).
        where: Argument name for the error message.
    """
    if freq_hz.ndim != 1 or freq_hz.size == 0:
        raise ValueError(f"{where} must be a non-empty 1D array.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise ValueError(
            f"{where} must be strictly ascending: a descending or repeated "
            "frequency axis interpolates and reads back silently wrong."
        )


def _broadcast_line_params(
    gamma_per_m: ArrayLike, z0_ohm: ArrayLike
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """The two line parameters on one frequency axis, or a clear refusal.

    Args:
        gamma_per_m: Complex propagation constant per frequency (1/m).
        z0_ohm: Complex characteristic impedance per frequency (ohm); a
            scalar broadcasts over ``gamma_per_m``.

    Returns:
        The pair, broadcast against each other.

    Raises:
        ValueError: When the two do not share one frequency axis.
    """
    gamma = np.asarray(gamma_per_m, dtype=np.complex128)
    z_c = np.asarray(z0_ohm, dtype=np.complex128)
    try:
        wide_gamma, wide_z_c = np.broadcast_arrays(gamma, z_c)
    except ValueError as error:
        raise ValueError(
            f"gamma_per_m (shape {np.shape(gamma_per_m)}) and z0_ohm (shape "
            f"{np.shape(z0_ohm)}) must share one frequency axis."
        ) from error
    return (
        np.asarray(wide_gamma, dtype=np.complex128),
        np.asarray(wide_z_c, dtype=np.complex128),
    )


def line_smatrix(
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
    *,
    length_m: float,
    z_ref_ohm: complex = 50.0,
) -> NDArray[np.complex128]:
    """The two-port S-matrix of a uniform line, referenced to ``z_ref_ohm``.

    Args:
        gamma_per_m: Complex propagation constant ``alpha + j beta`` per
            frequency (1/m).
        z0_ohm: Complex characteristic impedance per frequency (ohm); a
            scalar broadcasts over ``gamma_per_m``.
        length_m: Line length in meters (> 0).
        z_ref_ohm: Port reference impedance (ohm).

    Returns:
        The S-matrix, shaped ``gamma_per_m.shape + (2, 2)``.

    Raises:
        ValueError: When the length is not positive, or the two line
            parameter arrays do not broadcast against each other.
    """
    _require_positive_length(length_m)
    gamma, z_c = _broadcast_line_params(gamma_per_m, z0_ohm)
    z_r = complex(z_ref_ohm)

    gl = gamma * length_m
    sinh, cosh = np.sinh(gl), np.cosh(gl)
    denom = 2.0 * z_c * z_r * cosh + (z_c**2 + z_r**2) * sinh
    s11 = (z_c**2 - z_r**2) * sinh / denom
    s21 = 2.0 * z_c * z_r / denom

    s = np.empty((*gamma.shape, 2, 2), dtype=np.complex128)
    s[..., 0, 0] = s11
    s[..., 1, 1] = s11
    s[..., 0, 1] = s21
    s[..., 1, 0] = s21
    return s


def write_touchstone(
    path: str | Path,
    *,
    freq_hz: ArrayLike,
    s: ArrayLike,
    z_ref_ohm: float = 50.0,
    comments: list[str] | None = None,
) -> Path:
    """Write a two-port S-matrix as a Touchstone v1 ``.s2p`` file.

    scikit-rf writes the file: a ``# Hz S RI R <z_ref>`` option line and
    one row per frequency in real and imaginary columns, which ADS,
    scikit-rf or any Touchstone consumer reads back without conversion.

    Args:
        path: Output file; the ``.s2p`` suffix is added when missing.
        freq_hz: Frequencies in Hz (ascending, 1D).
        s: The S-matrix, shaped ``(len(freq_hz), 2, 2)``.
        z_ref_ohm: Port reference impedance (ohm). Touchstone references
            are real, so an imaginary part is refused rather than
            silently dropped.
        comments: Extra provenance lines written as ``!`` comments.

    Returns:
        The written path.

    Raises:
        ValueError: On a complex reference impedance, a non-ascending
            frequency axis, or mismatched array shapes.
    """
    # complex() rather than trusting the annotation: a complex reference
    # passed at runtime must be refused, not truncated.
    z_r = complex(z_ref_ohm)
    if z_r.imag != 0.0 or z_r.real <= 0.0:
        raise ValueError(
            "A Touchstone reference impedance is a positive real number; got "
            f"{z_r}. Renormalize the S-matrix to a real reference instead."
        )
    freq = np.asarray(freq_hz, dtype=np.float64)
    _require_ascending(freq, "freq_hz")
    matrix = np.asarray(s, dtype=np.complex128)
    if matrix.shape != (freq.size, 2, 2):
        raise ValueError(
            f"s (shape {matrix.shape}) must be (len(freq_hz), 2, 2) with "
            f"freq_hz 1D (shape {freq.shape})."
        )

    target = Path(path)
    if target.suffix.lower() != ".s2p":
        # Append rather than with_suffix: a dotted stem like "line.v2"
        # must not lose its last segment.
        target = target.with_name(target.name + ".s2p")

    import skrf

    network = skrf.Network(
        frequency=skrf.Frequency.from_f(freq, unit="Hz"),
        s=matrix,
        z0=z_r.real,
        name=target.stem,
        comments="\n".join(("gsim line two-port", *(comments or []))),
    )
    # Rendered to a string and written here so the path is exactly the
    # target, whatever scikit-rf would derive from the network's name.
    text = network.write_touchstone(return_string=True, form="ri", skrf_comment=False)
    target.write_text(cast("str", text))
    return target


def sax_line_model(
    freq_hz: ArrayLike,
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
    *,
    length_m: float,
    z_ref_ohm: complex = 50.0,
) -> SaxLineModel:
    """Wrap the solved line as a SAX-convention S-model over frequency.

    The returned callable is a plain function needing only numpy: called
    with ``f`` (Hz, scalar or array; the solved frequencies when
    omitted), it linearly interpolates ``gamma`` and ``Z0`` onto ``f``
    — held at the end values outside the solved range — and returns the
    dict of S-matrix entries keyed by port pairs ``("o1", "o1")`` ...
    ``("o2", "o2")``, the sdict convention SAX composes circuits from.
    The closure copies its inputs, so it stays valid after the arrays it
    was built from are mutated or garbage collected.

    Args:
        freq_hz: Frequencies the line was solved at (Hz, ascending, 1D).
        gamma_per_m: Complex propagation constant per frequency (1/m).
        z0_ohm: Complex characteristic impedance per frequency (ohm).
        length_m: Line length in meters (> 0).
        z_ref_ohm: Port reference impedance (ohm).

    Returns:
        The model callable.

    Raises:
        ValueError: When the length is not positive, the frequency axis
            is not ascending, or the arrays do not share it.
    """
    _require_positive_length(length_m)
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64)).copy()
    _require_ascending(freq, "freq_hz")
    # The record copies its inputs and owns the resampling, so this model
    # interpolates the way the line Stage's response grid does.
    line = line_params_from_gamma(
        freq,
        np.broadcast_to(np.asarray(gamma_per_m, dtype=np.complex128), freq.shape),
        z0_ohm=np.broadcast_to(np.asarray(z0_ohm, dtype=np.complex128), freq.shape),
    )

    def model(
        *, f: ArrayLike | None = None
    ) -> dict[tuple[str, str], NDArray[np.complex128]]:
        """The line's S-matrix entries at frequencies ``f`` (Hz)."""
        grid = None if f is None else np.asarray(f, dtype=np.float64)
        at = line if grid is None else line.resampled(grid)
        s = line_smatrix(
            at.gamma_per_m, at.z0_ohm, length_m=length_m, z_ref_ohm=z_ref_ohm
        )
        if grid is not None and grid.ndim == 0:
            s = s[0]
        return {
            ("o1", "o1"): s[..., 0, 0],
            ("o1", "o2"): s[..., 0, 1],
            ("o2", "o1"): s[..., 1, 0],
            ("o2", "o2"): s[..., 1, 1],
        }

    return model


# ----------------------------------------------------------------------
# Junction compact-model file
# ----------------------------------------------------------------------


class JunctionModel(NamedTuple):
    """A junction compact model read back from its file.

    The series-RC shunt branch of the junction per meter of
    Traveling-wave electrode, tabulated over the solved bias sweep, plus
    the context a circuit tool needs to use it: the swept contact, the
    frequency the branch was fitted at, and free-form provenance.

    Attributes:
        bias_v: Applied biases on the swept contact (V), in sweep order.
        r_s_ohm_m: Series resistance at each bias (ohm*m).
        c_j_f_per_m: Junction capacitance at each bias (F/m).
        contact: Name of the swept contact.
        freq_hz: Frequency the series-RC fit's small-signal admittance
            was measured at (Hz).
        provenance: Solve settings and generator info recorded at export.
    """

    bias_v: NDArray[np.float64]
    r_s_ohm_m: NDArray[np.float64]
    c_j_f_per_m: NDArray[np.float64]
    contact: str
    freq_hz: float
    provenance: dict[str, Any]


def write_junction_model(
    path: str | Path,
    *,
    bias_v: ArrayLike,
    r_s_ohm_m: ArrayLike,
    c_j_f_per_m: ArrayLike,
    contact: str,
    freq_hz: float,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write the junction's series-RC branch per bias as a model file.

    This is the machine-readable half of the junction handoff to a
    circuit tool such as circulax: the devsim-derived C_j(V) and R_s(V)
    per meter of Traveling-wave electrode leave gsim as one JSON file no
    consumer needs gsim (or anything beyond the stdlib) to read.

    **File format** (the interface to build a consumer against): a JSON
    object with the keys

    - ``format``: the literal ``"gsim-junction-model"``;
    - ``version``: integer schema version, currently ``1``;
    - ``units``: the unit of every tabulated column, spelled out;
    - ``contact``: name of the swept contact the biases are applied to;
    - ``freq_hz``: frequency (Hz) of the small-signal admittance the
      series RC was fitted to;
    - ``provenance``: free-form solve settings and generator info;
    - ``bias_v``, ``r_s_ohm_m``, ``c_j_f_per_m``: equal-length arrays in
      sweep order — bias (V), series resistance (ohm*m) and junction
      capacitance (F/m) per meter of electrode.

    Values round-trip exactly: JSON carries the shortest representation
    that reads back to the identical float.

    Args:
        path: Output file; the ``.json`` suffix is added when missing.
        bias_v: Applied biases (V), one per point.
        r_s_ohm_m: Series resistance per bias (ohm*m).
        c_j_f_per_m: Junction capacitance per bias (F/m).
        contact: Name of the swept contact.
        freq_hz: Frequency of the admittance fit (Hz, > 0).
        provenance: Extra solve settings recorded alongside the values;
            must be JSON-serializable.

    Returns:
        The written path.

    Raises:
        ValueError: When the arrays are not equal-length non-empty 1D,
            the fit frequency is not positive, or the contact is empty.
    """
    bias = np.asarray(bias_v, dtype=np.float64)
    r_s = np.asarray(r_s_ohm_m, dtype=np.float64)
    c_j = np.asarray(c_j_f_per_m, dtype=np.float64)
    if bias.ndim != 1 or bias.size == 0:
        raise ValueError("bias_v must be a non-empty 1D array.")
    if r_s.shape != bias.shape or c_j.shape != bias.shape:
        raise ValueError(
            f"bias_v (shape {bias.shape}), r_s_ohm_m (shape {r_s.shape}) and "
            f"c_j_f_per_m (shape {c_j.shape}) must be one value per bias point."
        )
    if freq_hz <= 0:
        raise ValueError("freq_hz must be the positive fit frequency in Hz.")
    if not contact:
        raise ValueError("contact must name the swept contact.")

    target = Path(path)
    if target.suffix.lower() != ".json":
        # Append rather than with_suffix: a dotted stem like
        # "sweep.2026-09" must not lose its last segment.
        target = target.with_name(target.name + ".json")
    payload = {
        "format": JUNCTION_MODEL_FORMAT,
        "version": JUNCTION_MODEL_VERSION,
        "units": {
            "bias_v": "V",
            "r_s_ohm_m": "ohm*m",
            "c_j_f_per_m": "F/m",
            "freq_hz": "Hz",
        },
        "contact": contact,
        "freq_hz": float(freq_hz),
        "provenance": provenance or {},
        "bias_v": [float(v) for v in bias],
        "r_s_ohm_m": [float(v) for v in r_s],
        "c_j_f_per_m": [float(v) for v in c_j],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def read_junction_model(path: str | Path) -> JunctionModel:
    """Read a junction compact-model file back.

    The stdlib/numpy counterpart of :func:`write_junction_model`, and
    the reference for how a consumer reads the file — see that
    function's docstring for the format.

    Args:
        path: The model file.

    Returns:
        The tabulated junction model.

    Raises:
        ValueError: When the file does not carry the
            ``gsim-junction-model`` format, its version is newer than
            this reader, or its arrays are inconsistent.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != JUNCTION_MODEL_FORMAT:
        raise ValueError(
            f"{path} is not a {JUNCTION_MODEL_FORMAT} file (format field: "
            f"{payload.get('format')!r})."
        )
    version = payload.get("version")
    if not isinstance(version, int) or version > JUNCTION_MODEL_VERSION:
        raise ValueError(
            f"{path} has schema version {version!r}; this reader understands "
            f"versions up to {JUNCTION_MODEL_VERSION}."
        )
    bias = np.asarray(payload["bias_v"], dtype=np.float64)
    r_s = np.asarray(payload["r_s_ohm_m"], dtype=np.float64)
    c_j = np.asarray(payload["c_j_f_per_m"], dtype=np.float64)
    if bias.ndim != 1 or r_s.shape != bias.shape or c_j.shape != bias.shape:
        raise ValueError(
            f"{path} carries inconsistent columns: bias_v {bias.shape}, "
            f"r_s_ohm_m {r_s.shape}, c_j_f_per_m {c_j.shape}."
        )
    return JunctionModel(
        bias_v=bias,
        r_s_ohm_m=r_s,
        c_j_f_per_m=c_j,
        contact=str(payload["contact"]),
        freq_hz=float(payload["freq_hz"]),
        provenance=dict(payload.get("provenance", {})),
    )


# ----------------------------------------------------------------------
# Touchstone reading and terminated-line responses
# ----------------------------------------------------------------------


class TwoPort(NamedTuple):
    """A two-port read back from a Touchstone file.

    Attributes:
        freq_hz: Frequencies (Hz), in file order.
        s: The S-matrix, shaped ``(len(freq_hz), 2, 2)``.
        z_ref_ohm: The real port reference impedance (ohm).
        comments: The file's ``!`` comment lines, markers stripped.
    """

    freq_hz: NDArray[np.float64]
    s: NDArray[np.complex128]
    z_ref_ohm: float
    comments: list[str]


def read_touchstone(path: str | Path) -> TwoPort:
    """Read a two-port Touchstone ``.s2p`` file through scikit-rf.

    Any unit (Hz to GHz) and format (RI, MA, DB) the standard allows is
    read, so the file need not come from :func:`write_touchstone`.

    Args:
        path: The ``.s2p`` file.

    Returns:
        The two-port.

    Raises:
        ValueError: When the file cannot be parsed, is not a two-port,
            or its ports do not share one real reference impedance.
    """
    import skrf

    try:
        network = skrf.Network(str(path))
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"{path} is not a readable Touchstone file: {error}"
        ) from error
    if network.nports != 2:
        raise ValueError(f"{path} holds a {network.nports}-port; expected a two-port.")
    z0 = np.asarray(network.z0)
    if np.any(z0.imag != 0.0) or np.any(z0 != z0.flat[0]):
        raise ValueError(
            f"{path} references its ports to {np.unique(z0)}; a two-port with "
            "one real reference impedance is expected."
        )
    return TwoPort(
        freq_hz=np.asarray(network.f, dtype=np.float64),
        s=np.asarray(network.s, dtype=np.complex128),
        z_ref_ohm=float(z0.flat[0].real),
        comments=[
            line.strip() for line in network.comments.splitlines() if line.strip()
        ],
    )


def _abcd_response(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
    c: NDArray[np.complex128],
    d: NDArray[np.complex128],
    *,
    z_gen_ohm: complex,
    z_load_ohm: complex,
) -> NDArray[np.complex128]:
    """``V_load / V_gen`` of an ABCD two-port between generator and load."""
    z_l = complex(z_load_ohm)
    z_g = complex(z_gen_ohm)
    return np.asarray(z_l / (a * z_l + b + z_g * (c * z_l + d)), dtype=np.complex128)


def terminated_response(
    s: ArrayLike,
    *,
    z_ref_ohm: float = 50.0,
    z_gen_ohm: complex = 50.0,
    z_load_ohm: complex = 50.0,
) -> NDArray[np.complex128]:
    """``V_load / V_gen`` of a two-port between an ideal generator and a load.

    The plain network math a circuit tool applies to an exported
    two-port: the S-matrix (referenced to the real ``z_ref_ohm``) is
    converted to its ABCD chain matrix, the load closes port 2, and the
    generator — an ideal source behind ``z_gen_ohm`` — drives port 1.

    Args:
        s: The S-matrix, shaped ``(..., 2, 2)``.
        z_ref_ohm: The real reference impedance the S-parameters are
            normalized to (ohm), as the Touchstone option line records
            it.
        z_gen_ohm: Generator impedance (ohm); complex accepted.
        z_load_ohm: Load impedance (ohm); complex accepted.

    Returns:
        The complex voltage transfer per frequency, shaped ``s.shape[:-2]``.

    Raises:
        ValueError: When ``s`` is not shaped ``(..., 2, 2)``, the
            reference impedance is not a positive real number, or the
            two-port transmits nothing (``S21 = 0``) somewhere, which
            has no ABCD form.
    """
    matrix = np.asarray(s, dtype=np.complex128)
    if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"s (shape {matrix.shape}) must be shaped (..., 2, 2).")
    z_r = complex(z_ref_ohm)
    if z_r.imag != 0.0 or z_r.real <= 0.0:
        raise ValueError(
            f"z_ref_ohm must be a positive real reference impedance; got {z_r}."
        )
    if np.any(matrix[..., 1, 0] == 0.0):
        raise ValueError(
            "The two-port has S21 = 0 at some frequency; a zero-transmission "
            "network has no ABCD form, so its terminated response is undefined."
        )
    from skrf.network import s2a

    # scikit-rf converts a stack of matrices; flatten any leading axes.
    abcd = s2a(matrix.reshape(-1, 2, 2), np.asarray(z_r.real)).reshape(matrix.shape)
    return _abcd_response(
        abcd[..., 0, 0],
        abcd[..., 0, 1],
        abcd[..., 1, 0],
        abcd[..., 1, 1],
        z_gen_ohm=z_gen_ohm,
        z_load_ohm=z_load_ohm,
    )


def line_driven_response(
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
    *,
    length_m: float,
    z_gen_ohm: complex = 50.0,
    z_load_ohm: complex = 50.0,
) -> NDArray[np.complex128]:
    """``V_load / V_gen`` of a uniform line between generator and load.

    The same driven response :func:`terminated_response` computes from
    an exported S-matrix, but written directly from the solved line
    parameters via the telegrapher ABCD matrix — the independent
    reference the export round-trip is validated against.

    Args:
        gamma_per_m: Complex propagation constant per frequency (1/m).
        z0_ohm: Complex characteristic impedance per frequency (ohm); a
            scalar broadcasts over ``gamma_per_m``.
        length_m: Line length in meters (> 0).
        z_gen_ohm: Generator impedance (ohm); complex accepted.
        z_load_ohm: Load impedance (ohm); complex accepted.

    Returns:
        The complex voltage transfer per frequency.

    Raises:
        ValueError: When the length is not positive, or the two line
            parameter arrays do not broadcast against each other.
    """
    _require_positive_length(length_m)
    gamma, z_c = _broadcast_line_params(gamma_per_m, z0_ohm)
    abcd = section_abcd(gamma * length_m, z_c)
    return _abcd_response(
        abcd[..., 0, 0],
        abcd[..., 0, 1],
        abcd[..., 1, 0],
        abcd[..., 1, 1],
        z_gen_ohm=z_gen_ohm,
        z_load_ohm=z_load_ohm,
    )
