"""Traveling-wave Mach-Zehnder modulator physics (pure analysis functions).

Combines the RF transmission-line parameters of
:mod:`gsim.common.transmission_line` — effective RF index, loss, and
characteristic impedance versus frequency — with the optical phase
response into the standard traveling-wave modulator figures of merit:

- small-signal electro-optic frequency response including velocity mismatch,
  RF loss, and impedance mismatch with source/load reflections
  (:func:`eo_response`, :func:`eo_bandwidth`);
- the analytic walk-off-limited bandwidth of a lossless matched line
  (:func:`walkoff_bandwidth`), and the same limit for a line whose RF index
  moves with frequency (:func:`walkoff_bandwidth_dispersive`);
- modulation efficiency ``V_pi L`` from a bias sweep of the effective-index
  shift (:func:`vpi_length_vcm`);
- the static Mach-Zehnder intensity transfer of two such Phase shifters, for
  a single-drive and a push-pull configuration (:func:`mzm_transfer`), the
  datasheet figures read off it — ``V_pi``, insertion loss, extinction ratio
  (:func:`mzm_transfer_figures`) — and the small-signal chirp parameter per
  Bias point (:func:`mzm_chirp`);
- the EO response of a segmented Traveling-wave electrode, in which only the
  loaded sections modulate the light (:func:`segmented_eo_response`), over
  the periodic line that
  :func:`gsim.common.transmission_line.segmented_line_params` describes.

The response model follows the classic single-drive analysis (e.g. Ghione,
*Semiconductor Devices for High-Speed Optoelectronics*, ch. 6): the voltage
wave on a lossy line of length L terminated in ``Z_L`` and driven through
``Z_g`` is averaged over the co-propagating optical group delay.
"""

from __future__ import annotations

import itertools
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import speed_of_light as C0  # noqa: N812
from scipy.optimize import brentq

from gsim.common.transmission_line import segmented_line_params

__all__ = [
    "QUADRATURE_RAD",
    "SINC_3DB_ARGUMENT",
    "DriveConfiguration",
    "MZMTransferFigures",
    "eo_bandwidth",
    "eo_response",
    "mzm_chirp",
    "mzm_drive_range",
    "mzm_transfer",
    "mzm_transfer_figures",
    "segmented_eo_response",
    "vpi_length_vcm",
    "walkoff_bandwidth",
    "walkoff_bandwidth_dispersive",
]

#: Argument where ``|sin(u)/u|`` falls to 1/sqrt(2) (walk-off 3 dB point).
SINC_3DB_ARGUMENT: float = 1.3915573782515105


def _f_avg(u: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Evaluate ``F(u) = (exp(u) - 1) / u`` with the ``F(0) = 1`` limit."""
    out = np.ones_like(u)
    mask = np.abs(u) > 1e-12
    out[mask] = np.expm1(u[mask]) / u[mask]
    return out


def _forward_wave_amplitudes(
    theta: ArrayLike,
    z0_ohm: ArrayLike,
    *,
    z_load_ohm: complex,
    z_gen_ohm: complex,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """The forward voltage wave at the input, and the round-trip factor.

    A line of complex electrical length ``theta`` and characteristic
    impedance ``z0_ohm``, terminated in ``z_load_ohm`` and driven by an
    ideal source behind ``z_gen_ohm``: the load reflection seen back at
    the input after a round trip, and the forward wave's amplitude there
    per volt of generator amplitude. The uniform electrode and the
    segmented one's Bloch line are the same two-port here, so they share
    this.

    Args:
        theta: ``gamma * length`` of the whole line, per frequency.
        z0_ohm: Characteristic impedance (ohm), per frequency.
        z_load_ohm: Termination impedance in ohms.
        z_gen_ohm: Generator impedance in ohms.

    Returns:
        ``(v_forward, round_trip)``, both per frequency.
    """
    theta = np.asarray(theta, dtype=np.complex128)
    z0_ohm = np.asarray(z0_ohm, dtype=np.complex128)
    reflect_load = (z_load_ohm - z0_ohm) / (z_load_ohm + z0_ohm)
    round_trip = reflect_load * np.exp(-2.0 * theta)
    tanh_theta = np.tanh(theta)
    z_in = (
        z0_ohm * (z_load_ohm + z0_ohm * tanh_theta) / (z0_ohm + z_load_ohm * tanh_theta)
    )
    v_forward = z_in / (z_in + z_gen_ohm) / (1.0 + round_trip)
    return (
        np.asarray(v_forward, dtype=np.complex128),
        np.asarray(round_trip, dtype=np.complex128),
    )


def _effective_voltage(
    freq_hz: NDArray[np.float64],
    *,
    length_m: float,
    n_rf: NDArray[np.float64],
    n_opt: float,
    alpha_rf_np_m: NDArray[np.float64],
    z0_ohm: NDArray[np.complex128],
    z_load_ohm: complex,
    z_gen_ohm: complex,
) -> NDArray[np.complex128]:
    """Optically averaged line voltage per volt of generator amplitude."""
    omega = 2.0 * np.pi * freq_hz
    gamma = alpha_rf_np_m + 1j * omega * n_rf / C0
    beta_opt = omega * n_opt / C0

    v_forward, round_trip = _forward_wave_amplitudes(
        gamma * length_m,
        z0_ohm,
        z_load_ohm=z_load_ohm,
        z_gen_ohm=z_gen_ohm,
    )

    u_forward = (1j * beta_opt - gamma) * length_m
    u_backward = (1j * beta_opt + gamma) * length_m
    averaged = v_forward * (_f_avg(u_forward) + round_trip * _f_avg(u_backward))
    return np.asarray(averaged, dtype=np.complex128)


def eo_response(
    freq_hz: ArrayLike,
    *,
    length_m: float,
    n_rf: ArrayLike,
    n_opt: float,
    alpha_rf_np_m: ArrayLike,
    z0_ohm: ArrayLike,
    z_load_ohm: complex,
    z_gen_ohm: complex,
    normalize: bool = True,
) -> NDArray[np.complex128]:
    """Small-signal electro-optic frequency response of a TW modulator.

    Averages the RF voltage wave — including load/source reflections — over
    the optical group propagation, capturing velocity mismatch, RF loss, and
    impedance mismatch simultaneously.

    Args:
        freq_hz: RF frequencies in Hz (array).
        length_m: Electrode length in meters (> 0).
        n_rf: RF effective (phase) index; scalar or per-frequency array.
        n_opt: Optical group index.
        alpha_rf_np_m: RF amplitude loss in Np/m; scalar or per-frequency.
        z0_ohm: Characteristic impedance in ohms (complex allowed); scalar
            or per-frequency.
        z_load_ohm: Termination impedance in ohms.
        z_gen_ohm: Generator impedance in ohms.
        normalize: Divide by the DC value so the response tends to 1 at low
            frequency (uses the lowest-frequency line parameters at f = 0).

    Returns:
        Complex response, same shape as ``freq_hz``.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    if np.any(freq <= 0):
        raise ValueError("Frequencies must be positive (use normalize for DC).")

    n_rf_arr = np.broadcast_to(np.asarray(n_rf, dtype=np.float64), freq.shape)
    alpha_arr = np.broadcast_to(np.asarray(alpha_rf_np_m, dtype=np.float64), freq.shape)
    z0_arr = np.broadcast_to(np.asarray(z0_ohm, dtype=np.complex128), freq.shape)

    response = _effective_voltage(
        freq,
        length_m=length_m,
        n_rf=n_rf_arr,
        n_opt=n_opt,
        alpha_rf_np_m=alpha_arr,
        z0_ohm=z0_arr,
        z_load_ohm=complex(z_load_ohm),
        z_gen_ohm=complex(z_gen_ohm),
    )

    if normalize:
        dc = _effective_voltage(
            np.array([0.0]),
            length_m=length_m,
            n_rf=n_rf_arr[:1],
            n_opt=n_opt,
            alpha_rf_np_m=alpha_arr[:1],
            z0_ohm=z0_arr[:1],
            z_load_ohm=complex(z_load_ohm),
            z_gen_ohm=complex(z_gen_ohm),
        )[0]
        response = response / dc
    return response


def eo_bandwidth(
    freq_hz: ArrayLike,
    response: ArrayLike,
    *,
    threshold: float = 1.0 / np.sqrt(2.0),
) -> float | None:
    """First frequency where the normalized |response| crosses *threshold*.

    Args:
        freq_hz: Frequencies in Hz, ascending.
        response: Complex (or magnitude) response, normalized to 1 at DC.
        threshold: Magnitude threshold; the default ``1/sqrt(2)`` is the
            3 dB electro-optic point.

    Returns:
        Linearly interpolated crossing frequency in Hz, or ``None`` when the
        response never falls below the threshold.
    """
    freq = np.asarray(freq_hz, dtype=np.float64)
    mag = np.abs(np.asarray(response))
    if freq.shape != mag.shape:
        raise ValueError("freq_hz and response must have the same shape.")
    below = np.nonzero(mag < threshold)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(freq[0])
    f0, f1 = freq[i - 1], freq[i]
    m0, m1 = mag[i - 1], mag[i]
    return float(f0 + (threshold - m0) * (f1 - f0) / (m1 - m0))


def walkoff_bandwidth(*, length_m: float, n_rf: float, n_opt: float) -> float:
    """Walk-off-limited 3 dB bandwidth of a lossless matched line in Hz.

    Solves ``|sin(u)/u| = 1/sqrt(2)`` with ``u = pi f L |n_rf - n_opt| / c``.

    Args:
        length_m: Electrode length in meters (> 0).
        n_rf: RF effective index.
        n_opt: Optical group index (different from ``n_rf``).

    Returns:
        3 dB frequency in Hz.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    mismatch = abs(n_rf - n_opt)
    if mismatch == 0:
        raise ValueError("n_rf equals n_opt: walk-off bandwidth is unbounded.")
    return SINC_3DB_ARGUMENT * C0 / (np.pi * length_m * mismatch)


def walkoff_bandwidth_dispersive(
    freq_hz: ArrayLike,
    n_rf: ArrayLike,
    *,
    length_m: float,
    n_opt: float,
) -> float | None:
    """Walk-off-limited 3 dB bandwidth of a line with a dispersive RF index.

    A loaded line's RF index falls with frequency, so there is no single
    mismatch to put in :func:`walkoff_bandwidth` — and the mean over the band
    is the wrong one, vanishing when the index crosses the optical group
    index. The limit is instead the lowest frequency satisfying the walk-off
    condition with the mismatch the line has *at that frequency*:
    ``pi f L |n_rf(f) - n_opt| / c = 1.39``.

    The index is interpolated linearly between the solved frequencies and
    held at its end values outside them, so a flat index recovers
    :func:`walkoff_bandwidth` exactly.

    Args:
        freq_hz: Solved frequencies in Hz, ascending.
        n_rf: RF effective index per frequency.
        length_m: Electrode length in meters (> 0).
        n_opt: Optical group index.

    Returns:
        3 dB frequency in Hz, or ``None`` when no frequency satisfies the
        condition — the line is velocity matched wherever it would.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    index = np.broadcast_to(np.asarray(n_rf, dtype=np.float64), freq.shape)
    mismatch = np.abs(index - n_opt)
    # The walk-off condition, as f * |mismatch(f)| = target.
    target = SINC_3DB_ARGUMENT * C0 / (np.pi * length_m)

    # Below the solved range the first index is held.
    if mismatch[0] > 0 and target / mismatch[0] <= freq[0]:
        return float(target / mismatch[0])
    # Inside it, the first solved interval the condition is met across.

    def excess(f: float) -> float:
        return float(f * abs(np.interp(f, freq, index) - n_opt) - target)

    met = freq * mismatch >= target
    for i in np.nonzero(~met[:-1] & met[1:])[0]:
        return float(brentq(excess, freq[i], freq[i + 1], xtol=1.0))
    # Past it the last index is held.
    if mismatch[-1] > 0:
        return float(max(target / mismatch[-1], freq[-1]))
    return None


def vpi_length_vcm(
    voltages: ArrayLike,
    dn_eff: ArrayLike,
    *,
    wavelength_um: float,
) -> NDArray[np.float64]:
    """Modulation efficiency ``V_pi L`` in V*cm along a bias sweep.

    Uses the local slope of the effective-index shift:
    ``V_pi L = lambda / (2 |d(dn_eff)/dV|)``.

    Args:
        voltages: Bias voltages in volts (>= 2 points, ascending).
        dn_eff: Effective-index shift at each bias.
        wavelength_um: Vacuum wavelength in um.

    Returns:
        ``V_pi L`` in V*cm at each bias point.
    """
    v = np.asarray(voltages, dtype=np.float64)
    dn = np.asarray(dn_eff, dtype=np.float64)
    if v.shape != dn.shape or v.size < 2:
        raise ValueError("voltages and dn_eff must be equal-length with >= 2 points.")
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    slope = np.gradient(dn, v)
    if np.any(slope == 0):
        raise ValueError("dn_eff slope vanishes; V_pi L is unbounded there.")
    # lambda[um] / (2 |slope|) is in V*um; 1 V*cm = 1e4 V*um.
    return wavelength_um / (2.0 * np.abs(slope)) / 1e4


# ----------------------------------------------------------------------
# Mach-Zehnder transfer and chirp
# ----------------------------------------------------------------------

#: How the two arms of the Mach-Zehnder share the drive voltage.
DriveConfiguration = Literal["single-drive", "push-pull"]

#: Static phase offset of the quadrature point — the transfer's half-power
#: point and steepest slope (rad).
QUADRATURE_RAD: float = float(np.pi / 2.0)

#: Points of the drive grid the transfer's peak and null are located on.
_MZM_GRID_POINTS = 2049

#: A Bias sweep as validated arrays: voltages, index shift, loss (dB/cm).
_Sweep = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


class MZMTransferFigures(NamedTuple):
    """The datasheet figures read off a static Mach-Zehnder transfer.

    All three figures come from one adjacent peak and null of the transfer,
    so they are reported together or not at all: a Bias sweep that does not
    drive the modulator from a peak to a null leaves them ``None`` and says
    why in ``message``, rather than extrapolating the sweep.

    Attributes:
        v_pi_v: Drive swing between the peak and the null (V).
        insertion_loss_db: Loss at the peak, ``-10 log10(T_peak)`` (dB).
        extinction_ratio_db: ``10 log10(T_peak / T_null)`` (dB); infinite
            when the arms cancel exactly.
        peak_v: Drive voltage of the peak (V).
        null_v: Drive voltage of the null (V).
        message: Why the figures are absent, when they are.
    """

    v_pi_v: float | None
    insertion_loss_db: float | None
    extinction_ratio_db: float | None
    peak_v: float | None
    null_v: float | None
    message: str | None


def _mzm_sweep(
    voltages: ArrayLike,
    dn_eff: ArrayLike,
    alpha_opt_db_cm: ArrayLike | None,
) -> _Sweep:
    """The Bias sweep as validated arrays, a missing loss sweep as no loss."""
    v = np.asarray(voltages, dtype=np.float64)
    dn = np.asarray(dn_eff, dtype=np.float64)
    if v.ndim != 1 or v.shape != dn.shape or v.size < 2:
        raise ValueError("voltages and dn_eff must be equal-length with >= 2 points.")
    if np.any(np.diff(v) <= 0.0):
        raise ValueError("voltages must be strictly ascending.")
    alpha = (
        np.zeros_like(v)
        if alpha_opt_db_cm is None
        else np.asarray(alpha_opt_db_cm, dtype=np.float64)
    )
    if alpha.shape != v.shape:
        raise ValueError("alpha_opt_db_cm must have the same shape as voltages.")
    return v, dn, alpha


def _unknown_drive(drive: str) -> ValueError:
    """The error for a drive configuration that is neither of the two."""
    return ValueError(
        f"Unknown drive configuration {drive!r}; use 'single-drive' or 'push-pull'."
    )


def _arm_power_fraction(arm_imbalance_db: float) -> float:
    """Fraction of the input power the splitter sends down arm 1."""
    return float(1.0 / (1.0 + 10.0 ** (-arm_imbalance_db / 10.0)))


def _arm_bias(voltages: NDArray[np.float64], bias_v: float | None) -> float:
    """The bias both arms rest at: the middle of the sweep unless given."""
    if bias_v is None:
        return float(0.5 * (voltages[0] + voltages[-1]))
    return float(bias_v)


def mzm_drive_range(
    voltages: ArrayLike,
    *,
    drive: DriveConfiguration = "push-pull",
    bias_v: float | None = None,
) -> tuple[float, float]:
    """Drive voltages a Bias sweep can answer for without extrapolation.

    The drive voltage is the voltage between the two arms,
    ``v = V_1 - V_2``, in both configurations. Both arms rest at the bias
    ``V_b``; a single-drive modulator puts the whole drive on arm 1
    (``V_b + v`` against ``V_b``), a push-pull one splits it evenly and
    oppositely (``V_b + v/2`` against ``V_b - v/2``). The reachable drive
    is whatever keeps both arms inside the sweep.

    Args:
        voltages: Bias voltages of the sweep in volts (ascending).
        drive: The drive configuration.
        bias_v: The bias both arms rest at (V); the middle of the sweep
            when omitted.

    Returns:
        ``(lowest, highest)`` drive voltage in volts.

    Raises:
        ValueError: When the bias lies outside the sweep, or a push-pull
            bias sits on an end of it and leaves one arm no room to move.
    """
    v = np.asarray(voltages, dtype=np.float64)
    low, high = float(v[0]), float(v[-1])
    bias = _arm_bias(v, bias_v)
    if not low <= bias <= high:
        raise ValueError(
            f"The arm bias {bias:g} V lies outside the bias sweep "
            f"({low:g} to {high:g} V)."
        )
    if drive == "single-drive":
        return low - bias, high - bias
    if drive == "push-pull":
        room = min(bias - low, high - bias)
        if room <= 0.0:
            raise ValueError(
                "A push-pull drive moves the arms either side of the bias, and "
                f"{bias:g} V is an end of the bias sweep ({low:g} to {high:g} V)."
            )
        return -2.0 * room, 2.0 * room
    raise _unknown_drive(drive)


def _mzm_arms(
    drive_v: NDArray[np.float64],
    sweep: _Sweep,
    *,
    length_m: float,
    wavelength_um: float,
    drive: DriveConfiguration,
    bias_v: float | None,
    arm_imbalance_db: float,
    phase_offset_rad: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Both arms' field amplitudes at the combiner, and their phase difference."""
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    voltages, dn, alpha_db_cm = sweep
    low, high = mzm_drive_range(voltages, drive=drive, bias_v=bias_v)
    slack = 1e-9 * (high - low)
    if np.any(drive_v < low - slack) or np.any(drive_v > high + slack):
        raise ValueError(
            f"The {drive} drive reaches {drive_v.min():g} to {drive_v.max():g} V, "
            f"outside the {low:g} to {high:g} V the bias sweep covers; the "
            "transfer is not extrapolated."
        )
    bias = _arm_bias(voltages, bias_v)
    share = 1.0 if drive == "single-drive" else 0.5
    arm_v = (bias + share * drive_v, bias - (1.0 - share) * drive_v)
    amplitude = [
        10.0 ** (-np.interp(v, voltages, alpha_db_cm) * (length_m * 1e2) / 20.0)
        for v in arm_v
    ]
    phase = [
        2.0 * np.pi * np.interp(v, voltages, dn) * length_m / (wavelength_um * 1e-6)
        for v in arm_v
    ]
    fraction = _arm_power_fraction(arm_imbalance_db)
    return (
        np.asarray(np.sqrt(fraction) * amplitude[0], dtype=np.float64),
        np.asarray(np.sqrt(1.0 - fraction) * amplitude[1], dtype=np.float64),
        np.asarray(phase[0] - phase[1] + phase_offset_rad, dtype=np.float64),
    )


def mzm_transfer(
    drive_v: ArrayLike,
    voltages: ArrayLike,
    dn_eff: ArrayLike,
    alpha_opt_db_cm: ArrayLike | None = None,
    *,
    length_m: float,
    wavelength_um: float,
    drive: DriveConfiguration = "push-pull",
    bias_v: float | None = None,
    arm_imbalance_db: float = 0.0,
    phase_offset_rad: float = QUADRATURE_RAD,
) -> NDArray[np.float64]:
    """Static intensity transfer ``T(v)`` of a Mach-Zehnder modulator.

    The same Phase shifter sits in both arms. A splitter sends the
    fraction ``r`` of the input power down arm 1 and the rest down arm 2,
    each arm attenuates and delays its field by what the Bias sweep says
    at that arm's voltage, and an ideal 3 dB combiner adds them::

        T = (r a_1^2 + (1 - r) a_2^2 + 2 sqrt(r (1 - r)) a_1 a_2 cos(dphi)) / 2
        a_k = 10^(-alpha(V_k) L / 20)
        dphi = 2 pi L (dn(V_1) - dn(V_2)) / lambda + phi_0

    See :func:`mzm_drive_range` for how the drive voltage ``v`` sets the
    arm voltages ``V_1``, ``V_2`` in each configuration. The index shift
    and the loss are interpolated linearly between the Bias points and
    never extrapolated.

    Args:
        drive_v: Drive voltages ``V_1 - V_2`` to evaluate at (V).
        voltages: Bias voltages of the sweep in volts (ascending).
        dn_eff: Effective-index shift at each bias.
        alpha_opt_db_cm: Optical loss at each bias (dB/cm); no loss when
            omitted.
        length_m: Phase shifter length in each arm, in meters (> 0).
        wavelength_um: Vacuum wavelength in um.
        drive: The drive configuration.
        bias_v: The bias both arms rest at (V); the middle of the sweep
            when omitted.
        arm_imbalance_db: Power the splitter sends down arm 1 over the
            power it sends down arm 2, ``10 log10(r / (1 - r))`` (dB);
            0 is a balanced interferometer.
        phase_offset_rad: Static phase delay ``phi_0`` added to arm 1
            (rad) — the heater. The default is the quadrature point,
            where a balanced modulator rests at half its peak.

    Returns:
        ``T`` at each drive voltage, as a fraction of the input power.

    Raises:
        ValueError: When a drive voltage takes an arm outside the sweep.
    """
    field_1, field_2, dphi = _mzm_arms(
        np.asarray(drive_v, dtype=np.float64),
        _mzm_sweep(voltages, dn_eff, alpha_opt_db_cm),
        length_m=length_m,
        wavelength_um=wavelength_um,
        drive=drive,
        bias_v=bias_v,
        arm_imbalance_db=arm_imbalance_db,
        phase_offset_rad=phase_offset_rad,
    )
    return np.asarray(
        (field_1**2 + field_2**2 + 2.0 * field_1 * field_2 * np.cos(dphi)) / 2.0,
        dtype=np.float64,
    )


def _phase_crossings(
    drive_v: NDArray[np.float64], dphi: NDArray[np.float64]
) -> list[tuple[float, int]]:
    """Drive voltages where the arm phase difference is a multiple of pi.

    Returns ``(voltage, multiple)`` pairs ordered by voltage. The phase
    difference is piecewise linear on the grid, so each crossing is
    interpolated linearly.
    """
    crossings: list[tuple[float, int]] = []
    first = int(np.ceil(dphi.min() / np.pi))
    last = int(np.floor(dphi.max() / np.pi))
    for multiple in range(first, last + 1):
        excess = dphi - multiple * np.pi
        crossings.extend(
            (float(drive_v[i]), multiple) for i in np.nonzero(excess == 0.0)[0]
        )
        for i in np.nonzero(excess[:-1] * excess[1:] < 0.0)[0]:
            step = excess[i] / (excess[i] - excess[i + 1])
            crossings.append(
                (float(drive_v[i] + step * (drive_v[i + 1] - drive_v[i])), multiple)
            )
    return sorted(crossings)


def mzm_transfer_figures(
    voltages: ArrayLike,
    dn_eff: ArrayLike,
    alpha_opt_db_cm: ArrayLike | None = None,
    *,
    length_m: float,
    wavelength_um: float,
    drive: DriveConfiguration = "push-pull",
    bias_v: float | None = None,
    arm_imbalance_db: float = 0.0,
    phase_offset_rad: float = QUADRATURE_RAD,
) -> MZMTransferFigures:
    """``V_pi``, insertion loss and extinction ratio of the static transfer.

    Read off :func:`mzm_transfer` the way a bench measurement reads them:
    the peak and the null are the adjacent drive voltages where the arm
    phase difference is an even and an odd multiple of pi — the pair
    nearest zero drive; ``V_pi`` is the swing between them, the insertion
    loss is the transfer at the peak and the extinction ratio the peak
    over the null. Where the index shift is linear in bias ``V_pi`` is the
    Modulation efficiency over the length, in either configuration, the
    drive voltage being the voltage between the arms in both. The
    extinction is limited by what unbalances the two fields at the null:
    the splitter's imbalance, and the loss differing between two arms that
    sit at different voltages. A loss both arms share lowers the peak and
    the null alike.

    Args:
        voltages: Bias voltages of the sweep in volts (ascending).
        dn_eff: Effective-index shift at each bias.
        alpha_opt_db_cm: Optical loss at each bias (dB/cm); no loss when
            omitted.
        length_m: Phase shifter length in each arm, in meters (> 0).
        wavelength_um: Vacuum wavelength in um.
        drive: The drive configuration.
        bias_v: The bias both arms rest at (V); the middle of the sweep
            when omitted.
        arm_imbalance_db: Splitter power imbalance, arm 1 over arm 2 (dB).
        phase_offset_rad: Static phase delay added to arm 1 (rad).

    Returns:
        The figures, or all ``None`` with a message when the sweep does
        not drive the modulator from a peak to a null.
    """
    sweep = _mzm_sweep(voltages, dn_eff, alpha_opt_db_cm)

    def arms(
        drive_v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return _mzm_arms(
            drive_v,
            sweep,
            length_m=length_m,
            wavelength_um=wavelength_um,
            drive=drive,
            bias_v=bias_v,
            arm_imbalance_db=arm_imbalance_db,
            phase_offset_rad=phase_offset_rad,
        )

    low, high = mzm_drive_range(sweep[0], drive=drive, bias_v=bias_v)
    # The phase difference bends where an arm passes a Bias point, so those
    # drive voltages join the grid and the linear interpolation between
    # grid points is exact.
    reach = (1.0 if drive == "single-drive" else 2.0) * (
        sweep[0] - _arm_bias(sweep[0], bias_v)
    )
    bends = np.concatenate([reach, -reach])
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(low, high, _MZM_GRID_POINTS),
                bends[(bends > low) & (bends < high)],
            ]
        )
    )
    dphi = arms(grid)[2]

    pairs = [
        (one, other)
        for one, other in itertools.pairwise(_phase_crossings(grid, dphi))
        if (one[1] - other[1]) % 2
    ]
    if not pairs:
        swing = (dphi.max() - dphi.min()) / np.pi
        return MZMTransferFigures(
            v_pi_v=None,
            insertion_loss_db=None,
            extinction_ratio_db=None,
            peak_v=None,
            null_v=None,
            message=(
                f"The bias sweep is too short to reach V_pi: its {drive} span of "
                f"{low:g} to {high:g} V moves the arm phase difference by "
                f"{swing:.2f} pi and does not take the transfer from a peak to "
                "a null, so V_pi, the insertion loss and the extinction ratio "
                "are not reported rather than extrapolated. Widen the bias "
                "sweep or lengthen the electrode."
            ),
        )
    one, other = min(pairs, key=lambda pair: abs(pair[0][0] + pair[1][0]))
    peak_v, null_v = (one[0], other[0]) if one[1] % 2 == 0 else (other[0], one[0])
    field_1, field_2, _ = arms(np.array([peak_v, null_v]))
    # At the peak the two fields add and at the null they subtract, exactly.
    t_peak = float((field_1[0] + field_2[0]) ** 2 / 2.0)
    t_null = float((field_1[1] - field_2[1]) ** 2 / 2.0)
    return MZMTransferFigures(
        v_pi_v=abs(peak_v - null_v),
        insertion_loss_db=float(-10.0 * np.log10(t_peak)),
        extinction_ratio_db=(
            float(10.0 * np.log10(t_peak / t_null)) if t_null > 0.0 else float("inf")
        ),
        peak_v=peak_v,
        null_v=null_v,
        message=None,
    )


def mzm_chirp(
    voltages: ArrayLike,
    dn_eff: ArrayLike,
    alpha_opt_db_cm: ArrayLike | None = None,
    *,
    wavelength_um: float,
    drive: DriveConfiguration = "push-pull",
    arm_imbalance_db: float = 0.0,
    phase_offset_rad: float = QUADRATURE_RAD,
) -> NDArray[np.float64]:
    """Small-signal chirp parameter of the modulator at each Bias point.

    The standard (Koyama-Iga) definition, for both arms resting at the
    Bias point and a small drive ``v`` on top of it::

        chirp = (d Phi / dv) / ((1 / 2I) dI / dv)

    with ``I`` the output intensity and ``Phi`` the output field's phase.
    Each arm's field ``t`` follows its voltage through the local slopes of
    the sweep, ``d ln(t) / dV = -(L/2) d(alpha)/dV - j (2 pi L / lambda)
    d(n_eff)/dV`` with ``alpha`` the power attenuation, so the chirp
    depends on the ratio of the absorption and index slopes,
    ``rho = lambda alpha' / (4 pi n')``, on the drive configuration, the
    splitter imbalance and the static phase offset — and not on the
    length, which scales both slopes alike.

    Sign convention: fields follow ``exp(+j omega t)`` as everywhere in
    gsim, so an arm contributes ``exp(-j 2 pi n L / lambda)``, and ``Phi``
    is the phase of the output field in that convention. The instantaneous
    optical frequency is then ``omega + d Phi / dt``, and a positive chirp
    is a frequency that rises (a blue shift) while the intensity rises.

    A single-drive modulator with no absorption slope has chirp ``+1`` at
    the default quadrature point ``+pi/2`` and ``-1`` at the other one,
    ``-pi/2``, whichever way the index moves with bias; the absorption
    slope pulls it to ``(1 - rho) / (1 + rho)``. A balanced push-pull
    modulator has chirp exactly 0 where the absorption does not move with
    bias — the ideal push-pull drive. A real Phase shifter leaves the
    residue ``-rho``, the two arms' losses moving oppositely, and a
    splitter imbalance adds ``(r - 1/2) / (sqrt(r (1 - r)) sin(phi_0))``.

    Args:
        voltages: Bias voltages of the sweep in volts (ascending).
        dn_eff: Effective-index shift at each bias.
        alpha_opt_db_cm: Optical loss at each bias (dB/cm); no loss when
            omitted.
        wavelength_um: Vacuum wavelength in um.
        drive: The drive configuration.
        arm_imbalance_db: Splitter power imbalance, arm 1 over arm 2 (dB).
        phase_offset_rad: Static phase delay added to arm 1 (rad).

    Returns:
        The chirp parameter at each Bias point; NaN or infinite where the
        drive does not modulate the intensity (resting on a peak or a null).
    """
    v, dn, alpha_db_cm = _mzm_sweep(voltages, dn_eff, alpha_opt_db_cm)
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    # d ln(t)/dV per meter of arm: the real part from the power attenuation
    # (dB/cm to 1/m), the imaginary part from the index.
    gain_real = -0.5 * np.gradient(alpha_db_cm, v) * 1e2 * np.log(10.0) / 10.0
    gain_imag = -2.0 * np.pi * np.gradient(dn, v) / (wavelength_um * 1e-6)

    # w = (c_1 s_1 + c_2 s_2) / (c_1 + c_2): the resting arm fields c_k
    # weighted by the share s_k of the drive each arm takes. Written out in
    # real arithmetic so a balanced push-pull drive has Re(w) = 0 exactly;
    # the denominator |c_1 + c_2|^2 is common to both parts and cancels.
    fraction = _arm_power_fraction(arm_imbalance_db)
    cross = np.sqrt(fraction * (1.0 - fraction))
    if drive == "single-drive":
        weight_real = fraction + cross * np.cos(phase_offset_rad)
    elif drive == "push-pull":
        weight_real = fraction - 0.5
    else:
        raise _unknown_drive(drive)
    weight_imag = -cross * np.sin(phase_offset_rad)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            (gain_real * weight_imag + gain_imag * weight_real)
            / (gain_real * weight_real - gain_imag * weight_imag),
            dtype=np.float64,
        )


def _segmented_effective_voltage(
    freq_hz: NDArray[np.float64],
    *,
    n_periods: int,
    period_m: float,
    fill_factor: float,
    n_opt: float,
    n_rf_loaded: NDArray[np.float64],
    alpha_loaded_np_m: NDArray[np.float64],
    z0_loaded_ohm: NDArray[np.complex128],
    n_rf_unloaded: NDArray[np.float64],
    alpha_unloaded_np_m: NDArray[np.float64],
    z0_unloaded_ohm: NDArray[np.complex128],
    z_load_ohm: complex,
    z_gen_ohm: complex,
) -> NDArray[np.complex128]:
    """Line voltage averaged over the loaded sections, per generator volt.

    The integral of ``V(z) exp(j beta_opt z)`` over the loaded sections
    only, divided by the whole electrode length — so an electrode loaded
    half of the way modulates half as much.
    """
    omega = 2.0 * np.pi * freq_hz
    gamma_l = alpha_loaded_np_m + 1j * omega * n_rf_loaded / C0
    gamma_u = alpha_unloaded_np_m + 1j * omega * n_rf_unloaded / C0
    beta_opt = omega * n_opt / C0

    # A lossless line at DC has no Bloch impedance (0/0), and needs none:
    # the voltage is uniform whatever stands in for it.
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_b, z_b = segmented_line_params(
            gamma_loaded_per_m=gamma_l,
            z0_loaded_ohm=z0_loaded_ohm,
            gamma_unloaded_per_m=gamma_u,
            z0_unloaded_ohm=z0_unloaded_ohm,
            fill_factor=fill_factor,
            period_m=period_m,
        )
    bloch = gamma_b * period_m
    z_b = np.where(bloch == 0, z0_loaded_ohm, z_b)

    # At the period boundaries the periodic line is a uniform one with the
    # Bloch constant and impedance: the same terminated-line amplitudes.
    v_forward, round_trip = _forward_wave_amplitudes(
        bloch * n_periods,
        z_b,
        z_load_ohm=z_load_ohm,
        z_gen_ohm=z_gen_ohm,
    )

    # Inside a period each loaded half carries its own forward and backward
    # telegrapher waves, fixed by the Bloch wave's V and I at the boundary
    # it touches; ``ratio`` is Z_loaded / Z_Bloch, signed by direction.
    half_m = 0.5 * fill_factor * period_m
    opt = 1j * beta_opt * half_m
    rf = gamma_l * half_m

    def per_period(
        ratio: NDArray[np.complex128], step: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        first = (1.0 - ratio) * _f_avg(opt + rf) + (1.0 + ratio) * _f_avg(opt - rf)
        last = (1.0 + ratio) * _f_avg(rf - opt) + (1.0 - ratio) * _f_avg(-rf - opt)
        return 0.25 * fill_factor * (first + np.exp(step) * last)

    ratio = z0_loaded_ohm / z_b
    u_forward = 1j * beta_opt * period_m - bloch
    u_backward = 1j * beta_opt * period_m + bloch
    forward = per_period(ratio, u_forward) * (
        _f_avg(u_forward * n_periods) / _f_avg(u_forward)
    )
    backward = per_period(-ratio, u_backward) * (
        _f_avg(u_backward * n_periods) / _f_avg(u_backward)
    )
    averaged = v_forward * (forward + round_trip * backward)
    return np.asarray(averaged, dtype=np.complex128)


def segmented_eo_response(
    freq_hz: ArrayLike,
    *,
    n_periods: int,
    period_m: float,
    fill_factor: float,
    n_opt: float,
    n_rf_loaded: ArrayLike,
    alpha_loaded_np_m: ArrayLike,
    z0_loaded_ohm: ArrayLike,
    n_rf_unloaded: ArrayLike,
    alpha_unloaded_np_m: ArrayLike,
    z0_unloaded_ohm: ArrayLike,
    z_load_ohm: complex,
    z_gen_ohm: complex,
    normalize: bool = True,
) -> NDArray[np.complex128]:
    """Small-signal EO response of a segmented Traveling-wave electrode.

    :func:`eo_response` for the periodic line of
    :func:`segmented_period_abcd`: the voltage wave — Bloch waves between
    the terminations, telegrapher waves inside each section — is averaged
    over the co-propagating light across the loaded sections only, since
    an unloaded section holds no Junction to modulate. The electrode is
    ``n_periods`` whole periods long.

    Args:
        freq_hz: RF frequencies in Hz (array).
        n_periods: Number of periods along the electrode (>= 1).
        period_m: Length of one period (m, > 0).
        fill_factor: Loaded fraction of the period; above 0 — an electrode
            loaded nowhere has no response — and at most 1.
        n_opt: Optical group index.
        n_rf_loaded: RF index of the loaded line; scalar or per-frequency.
        alpha_loaded_np_m: RF amplitude loss of the loaded line (Np/m).
        z0_loaded_ohm: Characteristic impedance of the loaded line (ohm).
        n_rf_unloaded: RF index of the unloaded line.
        alpha_unloaded_np_m: RF amplitude loss of the unloaded line (Np/m).
        z0_unloaded_ohm: Characteristic impedance of the unloaded line
            (ohm).
        z_load_ohm: Termination impedance in ohms.
        z_gen_ohm: Generator impedance in ohms.
        normalize: Divide by the DC value so the response tends to 1 at
            low frequency, as :func:`eo_response` does. Unnormalized, the
            response carries the fill factor: the drive voltage acts over
            the loaded fraction of the length alone.

    Returns:
        Complex response, same shape as ``freq_hz``.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    # The fill factor's range and the period's sign are checked where the
    # periodic line is assembled, in segmented_period_abcd.
    if fill_factor == 0.0:
        raise ValueError(
            "fill_factor must be above 0: an electrode loaded nowhere "
            "does not modulate."
        )
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1.")
    if np.any(freq <= 0):
        raise ValueError("Frequencies must be positive (use normalize for DC).")

    def on_axis(values: ArrayLike, dtype: type) -> NDArray:
        return np.broadcast_to(np.asarray(values, dtype=dtype), freq.shape)

    lines = {
        "n_rf_loaded": on_axis(n_rf_loaded, np.float64),
        "alpha_loaded_np_m": on_axis(alpha_loaded_np_m, np.float64),
        "z0_loaded_ohm": on_axis(z0_loaded_ohm, np.complex128),
        "n_rf_unloaded": on_axis(n_rf_unloaded, np.float64),
        "alpha_unloaded_np_m": on_axis(alpha_unloaded_np_m, np.float64),
        "z0_unloaded_ohm": on_axis(z0_unloaded_ohm, np.complex128),
    }

    def averaged(
        at_hz: NDArray[np.float64], sampled: dict[str, NDArray]
    ) -> NDArray[np.complex128]:
        return _segmented_effective_voltage(
            at_hz,
            n_periods=int(n_periods),
            period_m=period_m,
            fill_factor=fill_factor,
            n_opt=n_opt,
            z_load_ohm=complex(z_load_ohm),
            z_gen_ohm=complex(z_gen_ohm),
            **sampled,
        )

    response = averaged(freq, lines)
    if normalize:
        # DC holds the lowest-frequency line parameters, as eo_response does.
        first = {name: values[:1] for name, values in lines.items()}
        response = response / averaged(np.array([0.0]), first)[0]
    return response
