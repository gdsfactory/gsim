"""Transmission-line theory of the Traveling-wave electrode (pure functions).

The line the modulator's RF drive travels down, described on its own
terms and with no modulator in sight: what a mode solve says about it
(:class:`RFLineParams`, built by :func:`line_params_from_neff` or
:func:`line_params_from_gamma`), what its per-unit-length circuit is
(:func:`rlgc_from_line_params`, the Marks-Williams relations), how a
lumped shunt branch loads it (:func:`series_rc_from_admittance`,
:func:`loaded_line_params`), and what a periodically loaded electrode
behaves as — one period's two-port (:func:`segmented_period_abcd`), its
Bloch constants (:func:`segmented_line_params`, :func:`segmented_line`)
and how near the period sits to the Bragg condition
(:func:`bragg_fraction`).

:func:`section_abcd` is the telegrapher two-port of one uniform section
that the cascades here and the compact-model export of
:mod:`gsim.common.circuit` are both assembled from.

Sign convention is the one gsim uses throughout: ``gamma = alpha + j
beta`` with ``alpha >= 0`` on a lossy line, and fields following
``exp(+i omega t)``.
"""

from __future__ import annotations

from typing import NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.constants import speed_of_light as C0  # noqa: N812

__all__ = [
    "JunctionBranch",
    "RFLineParams",
    "bragg_fraction",
    "line_params_from_gamma",
    "line_params_from_neff",
    "loaded_line_params",
    "rlgc_from_line_params",
    "section_abcd",
    "segmented_line",
    "segmented_line_params",
    "segmented_period_abcd",
    "series_rc_from_admittance",
]


class JunctionBranch(NamedTuple):
    """The series-RC shunt branch per meter of Traveling-wave electrode.

    The lumped junction model the standard loaded-line workflow inserts
    per unit length: the junction capacitance behind the series
    resistance of the doped slab. A named pair, so the two numbers that
    always travel together do so under their own names; both are floats
    from a single Bias point's fit and same-shape arrays from a sweep's.

    Attributes:
        r_s_ohm_m: Series resistance (ohm*m).
        c_j_f_per_m: Junction capacitance (F/m).
    """

    r_s_ohm_m: float | NDArray[np.float64]
    c_j_f_per_m: float | NDArray[np.float64]


def series_rc_from_admittance(
    y_s_per_m: ArrayLike,
    *,
    freq_hz: float,
) -> JunctionBranch:
    """Fit a series-RC shunt branch to a small-signal admittance.

    Inverts ``Y = 1 / (R_s + 1/(j omega C_j))``: the branch impedance is
    ``Z = 1/Y = R_s - j/(omega C_j)``, so ``R_s = Re(Z)`` and
    ``C_j = -1/(omega Im(Z))``.

    Args:
        y_s_per_m: Complex shunt admittance per meter of Traveling-wave
            electrode (S/m);
            scalar or array, fit element by element.
        freq_hz: Frequency the admittance was measured at (Hz, > 0).

    Returns:
        The fitted :class:`JunctionBranch`, its fields the same shape as
        ``y_s_per_m``.

    Raises:
        ValueError: When the frequency is not positive, or the admittance
            is not one a series RC can represent (negative conductance,
            or a non-capacitive susceptance).
    """
    if freq_hz <= 0:
        raise ValueError("The fit frequency must be positive.")
    y = np.asarray(y_s_per_m, dtype=np.complex128)
    if np.any(y.real < 0):
        raise ValueError(
            "The admittance has negative conductance, which no series RC "
            "branch can represent."
        )
    if np.any(y.imag <= 0):
        raise ValueError(
            "The admittance is not capacitive (Im(Y) <= 0), so a series-RC "
            "junction branch cannot represent it."
        )
    omega = 2.0 * np.pi * freq_hz
    z = 1.0 / y
    return JunctionBranch(
        r_s_ohm_m=np.asarray(z.real, dtype=np.float64),
        c_j_f_per_m=np.asarray(-1.0 / (omega * z.imag), dtype=np.float64),
    )


def rlgc_from_line_params(
    freq_hz: ArrayLike,
    *,
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """RLGC per-unit-length parameters from ``gamma`` and ``Z_0``.

    Uses the telegrapher relations ``R + j omega L = gamma Z_0`` and
    ``G + j omega C = gamma / Z_0`` (Marks & Williams).

    Args:
        freq_hz: Frequencies in Hz.
        gamma_per_m: Complex propagation constant ``alpha + j beta`` in 1/m.
        z0_ohm: Complex characteristic impedance in ohms.

    Returns:
        Dict with arrays ``R`` (ohm/m), ``L`` (H/m), ``G`` (S/m), ``C`` (F/m).
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    if np.any(freq <= 0):
        raise ValueError("Frequencies must be positive.")
    gamma = np.broadcast_to(np.asarray(gamma_per_m, dtype=np.complex128), freq.shape)
    z0 = np.broadcast_to(np.asarray(z0_ohm, dtype=np.complex128), freq.shape)
    omega = 2.0 * np.pi * freq
    series = gamma * z0
    shunt = gamma / z0
    return {
        "R": series.real.copy(),
        "L": series.imag / omega,
        "G": shunt.real.copy(),
        "C": shunt.imag / omega,
    }


def loaded_line_params(
    freq_hz: ArrayLike,
    *,
    rlgc: dict[str, NDArray[np.float64]],
    junction: JunctionBranch | tuple[float, float],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Load a line's shunt admittance with a series-RC junction branch.

    The classic loaded-line assembly: the unloaded line's series
    impedance ``R + j omega L`` is unchanged, its shunt admittance
    ``G + j omega C`` gains the junction branch
    ``j omega C_j / (1 + j omega R_s C_j)``, and the loaded propagation
    constant and characteristic impedance follow from the telegrapher
    relations ``gamma = sqrt(ZY)``, ``Z_0 = sqrt(Z/Y)``.

    Args:
        freq_hz: Frequencies in Hz (> 0).
        rlgc: Unloaded per-unit-length parameters — arrays ``R`` (ohm/m),
            ``L`` (H/m), ``G`` (S/m), ``C`` (F/m) on the frequency axis,
            as :func:`rlgc_from_line_params` returns them.
        junction: The series-RC branch to insert, one (scalar) fit — as
            :meth:`gsim.tcad.results.BiasPoint.junction_branch` returns
            it.

    Returns:
        ``(gamma_per_m, z0_ohm)`` of the loaded line, per frequency.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    if np.any(freq <= 0):
        raise ValueError("Frequencies must be positive.")
    for name in ("R", "L", "G", "C"):
        if np.asarray(rlgc[name]).shape != freq.shape:
            raise ValueError(f"rlgc[{name!r}] must have the same shape as freq_hz.")
    r_s_ohm_m, c_j_f_per_m = junction
    omega = 2.0 * np.pi * freq
    z_series = rlgc["R"] + 1j * omega * rlgc["L"]
    y_junction = 1j * omega * c_j_f_per_m / (1.0 + 1j * omega * r_s_ohm_m * c_j_f_per_m)
    y_shunt = rlgc["G"] + 1j * omega * rlgc["C"] + y_junction
    # The principal square root keeps Re >= 0, the passive-line branch of
    # both quantities.
    gamma = np.sqrt(z_series * y_shunt)
    z0 = np.sqrt(z_series / y_shunt)
    return (
        np.asarray(gamma, dtype=np.complex128),
        np.asarray(z0, dtype=np.complex128),
    )


class RFLineParams(BaseModel):
    """RF transmission-line parameters versus frequency at one bias.

    Attributes:
        freq_hz: RF frequencies in Hz (ascending).
        n_rf: RF effective (phase) index per frequency.
        alpha_rf_np_m: RF amplitude loss in Np/m per frequency.
        z0_ohm: Complex characteristic impedance in ohms per frequency.
        unloaded: Whether these are the bare electrode's parameters —
            the cross-section solved with every carrier switched off —
            rather than a Bias point's answer.
        bias_v: The Bias the Cross-section was built at (V), when it was
            built from a Bias point; ``None`` for parameters that came
            from nowhere in particular (a hand-assembled line).
        signal_contact: Name of the Contact the RF drive is applied to,
            which names the conductor the impedance was read over;
            ``None`` when no Contact was involved.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    freq_hz: NDArray[np.float64]
    n_rf: NDArray[np.float64]
    alpha_rf_np_m: NDArray[np.float64]
    z0_ohm: NDArray[np.complex128]
    unloaded: bool = False
    bias_v: float | None = None
    signal_contact: str | None = None

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        """All arrays share the frequency axis; frequencies are positive."""
        shape = self.freq_hz.shape
        for name in ("n_rf", "alpha_rf_np_m", "z0_ohm"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have the same shape as freq_hz.")
        if self.freq_hz.ndim != 1 or self.freq_hz.size == 0:
            raise ValueError("freq_hz must be a non-empty 1D array.")
        if np.any(self.freq_hz <= 0):
            raise ValueError("Frequencies must be positive.")
        return self

    @property
    def gamma_per_m(self) -> NDArray[np.complex128]:
        """Complex propagation constant ``alpha + j beta`` in 1/m."""
        omega = 2.0 * np.pi * self.freq_hz
        return np.asarray(
            self.alpha_rf_np_m + 1j * omega * self.n_rf / C0, dtype=np.complex128
        )

    @property
    def rlgc(self) -> dict[str, NDArray[np.float64]]:
        """RLGC per-unit-length parameters (Marks-Williams relations)."""
        return rlgc_from_line_params(
            self.freq_hz, gamma_per_m=self.gamma_per_m, z0_ohm=self.z0_ohm
        )

    def resampled(self, freq_hz: ArrayLike) -> RFLineParams:
        """These parameters on another frequency grid.

        Linear interpolation of the index, the loss and the complex
        impedance — its real and imaginary parts separately — onto
        ``freq_hz``, with the end values held outside the solved range
        rather than extrapolated. What the line Stage's response grid and
        the SAX line model both read, so a frequency axis interpolates
        one way everywhere.

        Args:
            freq_hz: Frequencies to sample at (Hz, ascending, 1D).

        Returns:
            The same line on the new grid, carrying the same Bias, signal
            Contact and loaded/unloaded flag.

        Raises:
            ValueError: When the grid is not ascending.
        """
        grid = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
        if grid.ndim != 1 or np.any(np.diff(grid) < 0.0):
            raise ValueError("freq_hz must be a 1D ascending frequency grid.")
        return RFLineParams(
            freq_hz=grid,
            n_rf=np.interp(grid, self.freq_hz, self.n_rf),
            alpha_rf_np_m=np.interp(grid, self.freq_hz, self.alpha_rf_np_m),
            z0_ohm=np.asarray(
                np.interp(grid, self.freq_hz, self.z0_ohm.real)
                + 1j * np.interp(grid, self.freq_hz, self.z0_ohm.imag),
                dtype=np.complex128,
            ),
            unloaded=self.unloaded,
            bias_v=self.bias_v,
            signal_contact=self.signal_contact,
        )


def line_params_from_neff(
    freq_hz: ArrayLike,
    n_eff: ArrayLike,
    *,
    z0_ohm: ArrayLike,
    unloaded: bool = False,
    bias_v: float | None = None,
    signal_contact: str | None = None,
) -> RFLineParams:
    """Build :class:`RFLineParams` from complex mode effective indices.

    This is the extraction step shared by both solver routes: Palace
    BoundaryMode (``PalaceTextResults.modes[m]["n_eff"]`` per frequency)
    and the femwell adapter (``mode.n_eff``) both deliver a complex
    effective index; ``beta = omega Re(n_eff) / c0`` and
    ``alpha = omega |Im(n_eff)| / c0``.

    Args:
        freq_hz: RF frequencies in Hz.
        n_eff: Complex effective index per frequency (either sign
            convention for the imaginary part).
        z0_ohm: Characteristic impedance per frequency (complex allowed),
            e.g. from Palace's impedance postprocessing or a
            Marks-Williams extraction.
        unloaded: Flag the result as the bare electrode's — solved with
            the carriers switched off — rather than a Bias point's.
        bias_v: The Bias the Cross-section was built at (V), if any.
        signal_contact: The Contact the impedance was read over, if any.

    Returns:
        The RF line parameters.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    n_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(n_eff, dtype=np.complex128)), freq.shape
    )
    z0 = np.broadcast_to(
        np.atleast_1d(np.asarray(z0_ohm, dtype=np.complex128)), freq.shape
    )
    omega = 2.0 * np.pi * freq
    return RFLineParams(
        freq_hz=freq.copy(),
        n_rf=np.array(n_arr.real, dtype=np.float64),
        alpha_rf_np_m=np.asarray(np.abs(n_arr.imag) * omega / C0, dtype=np.float64),
        z0_ohm=np.array(z0, dtype=np.complex128),
        unloaded=unloaded,
        bias_v=bias_v,
        signal_contact=signal_contact,
    )


def line_params_from_gamma(
    freq_hz: ArrayLike,
    gamma_per_m: ArrayLike,
    *,
    z0_ohm: ArrayLike,
    unloaded: bool = False,
    bias_v: float | None = None,
    signal_contact: str | None = None,
) -> RFLineParams:
    """Build :class:`RFLineParams` from complex propagation constants.

    The inverse of :attr:`RFLineParams.gamma_per_m`, for routes that
    produce ``gamma`` directly — the loaded-line assembly
    (:func:`loaded_line_params`) rather than a mode
    solve: ``n_RF = |Im(gamma)| c0 / omega`` and
    ``alpha = |Re(gamma)|``, so either sign convention is read as loss.

    Args:
        freq_hz: RF frequencies in Hz.
        gamma_per_m: Complex propagation constant per frequency (1/m).
        z0_ohm: Characteristic impedance per frequency (complex allowed).
        unloaded: Flag the result as the bare electrode's.
        bias_v: The Bias the Cross-section was built at (V), if any.
        signal_contact: The Contact the impedance was read over, if any.

    Returns:
        The RF line parameters.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    gamma = np.broadcast_to(
        np.atleast_1d(np.asarray(gamma_per_m, dtype=np.complex128)), freq.shape
    )
    z0 = np.broadcast_to(
        np.atleast_1d(np.asarray(z0_ohm, dtype=np.complex128)), freq.shape
    )
    omega = 2.0 * np.pi * freq
    return RFLineParams(
        freq_hz=freq.copy(),
        n_rf=np.asarray(np.abs(gamma.imag) * C0 / omega, dtype=np.float64),
        alpha_rf_np_m=np.array(np.abs(gamma.real), dtype=np.float64),
        z0_ohm=np.array(z0, dtype=np.complex128),
        unloaded=unloaded,
        bias_v=bias_v,
        signal_contact=signal_contact,
    )


def _require_segmentation(fill_factor: float, period_m: float) -> None:
    """A fill factor is a fraction of a period; a period is a length."""
    if not 0.0 <= fill_factor <= 1.0:
        raise ValueError("fill_factor must lie between 0 and 1.")
    if period_m <= 0:
        raise ValueError("period_m must be positive.")


def section_abcd(theta: ArrayLike, z0_ohm: ArrayLike) -> NDArray[np.complex128]:
    """Telegrapher ABCD matrix of one uniform section of line.

    The two-port ``[[cosh(theta), Z0 sinh(theta)], [sinh(theta) / Z0,
    cosh(theta)]]`` of a section whose complex electrical length is
    ``theta = gamma l``. One section on its own is a uniform line; the
    cascades of this module are products of several.

    Args:
        theta: Complex electrical length ``gamma * l`` of the section;
            scalar or per-frequency.
        z0_ohm: Characteristic impedance of the section (ohm), broadcast
            against ``theta``.

    Returns:
        The matrices, shape ``(..., 2, 2)`` over the broadcast inputs.
    """
    theta = np.asarray(theta, dtype=np.complex128)
    z0_ohm = np.asarray(z0_ohm, dtype=np.complex128)
    cosh, sinh = np.cosh(theta), np.sinh(theta)
    return np.stack(
        [
            np.stack([cosh, z0_ohm * sinh], axis=-1),
            np.stack([sinh / z0_ohm, cosh], axis=-1),
        ],
        axis=-2,
    )


def segmented_period_abcd(
    *,
    gamma_loaded_per_m: ArrayLike,
    z0_loaded_ohm: ArrayLike,
    gamma_unloaded_per_m: ArrayLike,
    z0_unloaded_ohm: ArrayLike,
    fill_factor: float,
    period_m: float,
) -> NDArray[np.complex128]:
    """ABCD matrix of one period of a segmented Traveling-wave electrode.

    A segmented electrode is loaded by the Junction only part of the way:
    a loaded section of length ``fill_factor * period_m`` alternates with
    an unloaded one along the propagation axis. One period is cut here
    through the middle of the loaded section — half a loaded section, the
    unloaded section, the other loaded half — so the two-port is symmetric
    (``A = D``) and the periodic line has one Bloch impedance rather than
    one per direction.

    Args:
        gamma_loaded_per_m: Propagation constant of the loaded line (1/m);
            scalar or per-frequency.
        z0_loaded_ohm: Characteristic impedance of the loaded line (ohm).
        gamma_unloaded_per_m: Propagation constant of the unloaded line
            (1/m).
        z0_unloaded_ohm: Characteristic impedance of the unloaded line
            (ohm).
        fill_factor: Loaded fraction of the period, 0 to 1.
        period_m: Length of one period (m, > 0).

    Returns:
        The matrices, shape ``(..., 2, 2)`` over the broadcast inputs.
    """
    _require_segmentation(fill_factor, period_m)
    gamma_l, z0_l, gamma_u, z0_u = np.broadcast_arrays(
        np.asarray(gamma_loaded_per_m, dtype=np.complex128),
        np.asarray(z0_loaded_ohm, dtype=np.complex128),
        np.asarray(gamma_unloaded_per_m, dtype=np.complex128),
        np.asarray(z0_unloaded_ohm, dtype=np.complex128),
    )
    half_loaded = section_abcd(gamma_l * (0.5 * fill_factor * period_m), z0_l)
    unloaded = section_abcd(gamma_u * ((1.0 - fill_factor) * period_m), z0_u)
    return np.asarray(half_loaded @ unloaded @ half_loaded, dtype=np.complex128)


def segmented_line_params(
    *,
    gamma_loaded_per_m: ArrayLike,
    z0_loaded_ohm: ArrayLike,
    gamma_unloaded_per_m: ArrayLike,
    z0_unloaded_ohm: ArrayLike,
    fill_factor: float,
    period_m: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Bloch propagation constant and impedance of a segmented electrode.

    The periodic line of :func:`segmented_period_abcd` carries Bloch waves
    ``exp(-Gamma n)`` from one period to the next, with
    ``cosh(Gamma) = (A + D) / 2``. That is solved here in the form

    ``sinh^2(Gamma/2) = sinh^2((t_l + t_u)/2)
    + (Z_l - Z_u)^2 / (4 Z_l Z_u) sinh(t_l) sinh(t_u)``

    (``t = gamma l`` of each section), which is the same equation without
    the cancellation ``cosh(Gamma) - 1`` suffers for a period far below
    the wavelength — exactly where a segmented electrode operates. The
    passive branch is taken: ``Re(Gamma) >= 0``, and ``Im(Gamma) >= 0``
    on a lossless line, so the wave decays the way it travels; the Bloch
    impedance is that wave's own, ``Z_B = B / sinh(Gamma)``, whose real
    part is positive on a passive line.

    A fill factor of one returns the loaded line and a fill factor of
    zero the unloaded one, to rounding. For a period far below the
    wavelength the result tends to the line whose series impedance and
    shunt admittance are the length-weighted averages of the two lines';
    the difference is second order in the phase advance per period — below
    1e-3 relative while :func:`bragg_fraction` stays under 0.05.

    Args:
        gamma_loaded_per_m: Propagation constant of the loaded line (1/m);
            scalar or per-frequency.
        z0_loaded_ohm: Characteristic impedance of the loaded line (ohm).
        gamma_unloaded_per_m: Propagation constant of the unloaded line
            (1/m).
        z0_unloaded_ohm: Characteristic impedance of the unloaded line
            (ohm).
        fill_factor: Loaded fraction of the period, 0 to 1.
        period_m: Length of one period (m, > 0).

    Returns:
        ``(gamma_per_m, z0_ohm)`` of the periodic line: the Bloch
        propagation constant per meter and the Bloch impedance.
    """
    abcd = segmented_period_abcd(
        gamma_loaded_per_m=gamma_loaded_per_m,
        z0_loaded_ohm=z0_loaded_ohm,
        gamma_unloaded_per_m=gamma_unloaded_per_m,
        z0_unloaded_ohm=z0_unloaded_ohm,
        fill_factor=fill_factor,
        period_m=period_m,
    )
    gamma_l, z0_l, gamma_u, z0_u = np.broadcast_arrays(
        np.asarray(gamma_loaded_per_m, dtype=np.complex128),
        np.asarray(z0_loaded_ohm, dtype=np.complex128),
        np.asarray(gamma_unloaded_per_m, dtype=np.complex128),
        np.asarray(z0_unloaded_ohm, dtype=np.complex128),
    )
    theta_l = gamma_l * (fill_factor * period_m)
    theta_u = gamma_u * ((1.0 - fill_factor) * period_m)
    contrast = (z0_l - z0_u) ** 2 / (4.0 * z0_l * z0_u)
    bloch = 2.0 * np.arcsinh(
        np.sqrt(
            np.sinh(0.5 * (theta_l + theta_u)) ** 2
            + contrast * np.sinh(theta_l) * np.sinh(theta_u)
        )
    )
    # arcsinh is odd, so the other root of the square is -bloch: keep the
    # wave that decays as it travels, and the forward one when lossless.
    backward = (bloch.real < 0) | ((bloch.real == 0) & (bloch.imag < 0))
    bloch = np.where(backward, -bloch, bloch)
    z_bloch = abcd[..., 0, 1] / np.sinh(bloch)
    return (
        np.asarray(bloch / period_m, dtype=np.complex128),
        np.asarray(z_bloch, dtype=np.complex128),
    )


def bragg_fraction(
    *,
    gamma_loaded_per_m: ArrayLike,
    gamma_unloaded_per_m: ArrayLike,
    fill_factor: float,
    period_m: float,
) -> NDArray[np.float64]:
    """How near a segmented electrode's period sits to the Bragg condition.

    The RF phase advance across one period — the loaded section's plus
    the unloaded one's — as a fraction of ``pi``, where the reflections
    of successive periods add in phase and the periodic line stops
    propagating. Read off the sections rather than the Bloch constant, so
    it keeps rising past the Bragg condition instead of folding back into
    the first Brillouin zone. Far below one, the segmented electrode is
    the averaged line; approaching one, it is a filter.

    Args:
        gamma_loaded_per_m: Propagation constant of the loaded line (1/m).
        gamma_unloaded_per_m: Propagation constant of the unloaded line
            (1/m).
        fill_factor: Loaded fraction of the period, 0 to 1.
        period_m: Length of one period (m, > 0).

    Returns:
        The fraction, per frequency.
    """
    _require_segmentation(fill_factor, period_m)
    beta_l = np.abs(np.asarray(gamma_loaded_per_m, dtype=np.complex128).imag)
    beta_u = np.abs(np.asarray(gamma_unloaded_per_m, dtype=np.complex128).imag)
    phase = (beta_l * fill_factor + beta_u * (1.0 - fill_factor)) * period_m
    return np.asarray(phase / np.pi, dtype=np.float64)


def segmented_line(
    loaded: RFLineParams,
    unloaded: RFLineParams,
    *,
    fill_factor: float,
    period_m: float,
) -> RFLineParams:
    """The periodic line a segmented Traveling-wave electrode behaves as.

    Between period boundaries the segmented electrode is exactly a
    uniform line with the Bloch propagation constant and impedance
    (:func:`segmented_line_params`), so these parameters
    stand wherever a uniform line's do: in the report, and in the
    two-port of a whole number of periods.

    Args:
        loaded: The loaded line's parameters.
        unloaded: The unloaded line's, on the same frequencies.
        fill_factor: Loaded fraction of the period, 0 to 1.
        period_m: Length of one period (m, > 0).

    Returns:
        The Bloch propagation constant and impedance as line parameters,
        on the loaded line's frequencies and with its provenance.

    Raises:
        ValueError: When the two lines are not on one frequency axis.
    """
    if loaded.freq_hz.shape != unloaded.freq_hz.shape or np.any(
        loaded.freq_hz != unloaded.freq_hz
    ):
        raise ValueError("The loaded and unloaded lines must share one freq_hz axis.")
    gamma, z_bloch = segmented_line_params(
        gamma_loaded_per_m=loaded.gamma_per_m,
        z0_loaded_ohm=loaded.z0_ohm,
        gamma_unloaded_per_m=unloaded.gamma_per_m,
        z0_unloaded_ohm=unloaded.z0_ohm,
        fill_factor=fill_factor,
        period_m=period_m,
    )
    return line_params_from_gamma(
        loaded.freq_hz,
        gamma,
        z0_ohm=z_bloch,
        bias_v=loaded.bias_v,
        signal_contact=loaded.signal_contact,
    )
