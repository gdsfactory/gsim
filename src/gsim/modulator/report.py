"""What ``Study.report()`` returns: the whole-device figures of merit.

The mode solvers (Palace BoundaryMode or the femwell adapter) produce
complex effective indices and characteristic impedances versus frequency
and bias; the charge/optics side produces the effective-index shift along
the bias sweep. This module packages those into typed containers and one
entry point, :func:`twmzm_figures_of_merit`, that returns the full device
report: EO frequency response and 3 dB bandwidth, velocity mismatch and
the analytic walk-off limit, RLGC line parameters, V_pi·L, and the
Mach-Zehnder built from that Phase shifter — its static transfer, V_pi,
insertion loss, extinction ratio and chirp — all computed with the pure
analysis functions of :mod:`gsim.modulator.twmzm` over the line
:mod:`gsim.common.transmission_line` describes.

:class:`LoadedLineComparison` is the other container the Study hands
back: the two loaded-line routes side by side at one Bias point, and the
gate that fails where they disagree.

Sign convention for complex effective indices is ``exp(+i omega t)``
(lossy: ``Im(n_eff) < 0``); the extraction accepts either sign of the
imaginary part and treats its magnitude as loss.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.common.transmission_line import RFLineParams, segmented_line
from gsim.modulator.twmzm import (
    QUADRATURE_RAD,
    DriveConfiguration,
    eo_bandwidth,
    eo_response,
    mzm_chirp,
    mzm_drive_range,
    mzm_transfer,
    mzm_transfer_figures,
    segmented_eo_response,
    vpi_length_vcm,
    walkoff_bandwidth_dispersive,
)

__all__ = [
    "SILICON_ONLY_RTOL",
    "LoadedLineComparison",
    "OpticalPhaseSweep",
    "TWMZMReport",
    "twmzm_figures_of_merit",
]

#: The :meth:`LoadedLineComparison.check` tolerances a silicon-only charge
#: solve (``study.charge(oxide=False)``) needs: just outside the gap
#: measured without the fringing field in C_j, at worst 25 % on n_RF, 59 %
#: on the loss and 42 % on |Z0| across the two demo devices.
SILICON_ONLY_RTOL: dict[str, float] = {
    "rtol_n_rf": 0.3,
    "rtol_alpha": 0.7,
    "rtol_z0": 0.5,
}


class LoadedLineComparison(BaseModel):
    """The two loaded-line routes side by side, at one Bias point.

    The direct route solves the carrier-loaded Staircase as one
    cross-section; the assembled route combines the unloaded (bare
    electrode) RLGC with the charge solve's series-RC junction branch per
    unit length. Both are the same compact model assembled two ways, so
    their n_RF, loss and Z0 should agree — up to what the lumped branch
    cannot capture of the distributed junction.

    What is measured, at 20 and 30 GHz on the two demo devices. The two
    routes agree on the series R and L to 1 %; the gap is in the shunt
    branch. With Poisson solved in the doped silicon alone
    (``study.charge(oxide=False)``) the assembled route reads 16-25 % low
    on n_RF, 46-59 % low on the loss and 25-42 % high on |Z0|: the direct
    solve finds 115-150 pF/m more shunt capacitance on the rib Phase
    shifter. Most of that is the field fringing through the oxide around
    the Junction, between the two conducting halves of the silicon, which
    a silicon-only charge solve cannot see. With the oxide in the charge
    solve, the default, C_j carries it: on the rib Phase shifter the
    shunt-C gap falls to 10-40 pF/m and the routes agree to 2-5 % on n_RF
    and 10 % on |Z0|; on the abrupt demo device, whose electrodes stand
    0.3 um from the Junction, they narrow to 15 % and 23 %.

    What is not explained. The loss gap stays at 33-41 % with the
    capacitance matched, and the direct solve's shunt conductance is two
    to three times the assembly's on the rib Phase shifter (ten times on
    the abrupt demo device), so the gap is not the missing capacitance
    felt squared. The direct solve is the reference.

    The default tolerances of :meth:`check` are set just outside the gap
    measured with the oxide in the charge solve; a route bug (a dropped
    conductivity, a wrong-branch mode, a unit slip) overshoots them by
    multiples. A silicon-only charge solve needs
    :data:`SILICON_ONLY_RTOL`.

    Attributes:
        direct: The direct loaded solve's line parameters.
        assembled: The loaded-line assembly's parameters, from the
            unloaded RLGC plus the junction branch.
    """

    direct: RFLineParams
    assembled: RFLineParams

    @model_validator(mode="after")
    def validate_axes(self) -> Self:
        """Both routes answer on the same frequency axis."""
        if self.direct.freq_hz.shape != self.assembled.freq_hz.shape or np.any(
            self.direct.freq_hz != self.assembled.freq_hz
        ):
            raise ValueError("The two routes must share one freq_hz axis.")
        return self

    @property
    def freq_hz(self) -> NDArray[np.float64]:
        """The shared frequency axis (Hz)."""
        return self.direct.freq_hz

    @property
    def delta_n_rf(self) -> NDArray[np.float64]:
        """Relative n_RF difference, assembled against direct."""
        return np.asarray(
            (self.assembled.n_rf - self.direct.n_rf) / self.direct.n_rf,
            dtype=np.float64,
        )

    @property
    def delta_alpha(self) -> NDArray[np.float64]:
        """Relative RF-loss difference, assembled against direct."""
        return np.asarray(
            (self.assembled.alpha_rf_np_m - self.direct.alpha_rf_np_m)
            / self.direct.alpha_rf_np_m,
            dtype=np.float64,
        )

    @property
    def delta_z0(self) -> NDArray[np.float64]:
        """Relative |Z0| deviation, assembled against direct."""
        return np.asarray(
            np.abs(self.assembled.z0_ohm - self.direct.z0_ohm)
            / np.abs(self.direct.z0_ohm),
            dtype=np.float64,
        )

    def check(
        self,
        *,
        rtol_n_rf: float = 0.2,
        rtol_alpha: float = 0.5,
        rtol_z0: float = 0.3,
    ) -> None:
        """Fail loudly where the two routes disagree.

        The defaults sit just outside the systematic gap the class
        docstring describes (at worst 15 % on n_RF, 41 % on the loss and
        23 % on |Z0| across the two demo devices, the oxide in the charge
        solve), so they pass an honest assembly and fail a broken route,
        which misses by multiples. Pass ``**SILICON_ONLY_RTOL`` for a
        charge solve run with ``oxide=False``.

        Args:
            rtol_n_rf: Relative tolerance on n_RF.
            rtol_alpha: Relative tolerance on the RF loss.
            rtol_z0: Relative tolerance on |Z0|.

        Raises:
            ValueError: Naming the quantity that diverged and the
                frequency it diverged at.
        """
        for name, delta, rtol in (
            ("n_RF", self.delta_n_rf, rtol_n_rf),
            ("the RF loss", self.delta_alpha, rtol_alpha),
            ("Z0", self.delta_z0, rtol_z0),
        ):
            excess = np.abs(delta) > rtol
            if np.any(excess):
                where = int(np.argmax(np.abs(np.where(excess, delta, 0.0))))
                raise ValueError(
                    f"The two loaded-line routes disagree on {name}: "
                    f"{delta[where]:+.1%} relative at "
                    f"{self.freq_hz[where] / 1e9:g} GHz "
                    f"(tolerance {rtol:.0%}). Direct "
                    f"{self.direct.n_rf[where]:g}/"
                    f"{self.direct.alpha_rf_np_m[where]:g}/"
                    f"{self.direct.z0_ohm[where]:.3g} vs assembled "
                    f"{self.assembled.n_rf[where]:g}/"
                    f"{self.assembled.alpha_rf_np_m[where]:g}/"
                    f"{self.assembled.z0_ohm[where]:.3g} "
                    "(n_RF/loss Np/m/Z0 ohm)."
                )


class OpticalPhaseSweep(BaseModel):
    """Optical response of the phase shifter along a bias sweep.

    Attributes:
        voltages_v: Bias voltages in volts (>= 2 points).
        dn_eff: Effective-index shift at each bias.
        alpha_opt_db_cm: Optional optical loss in dB/cm at each bias.
        wavelength_um: Vacuum wavelength in um.
        n_group: Optical group index used for the velocity-mismatch
            analysis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    voltages_v: NDArray[np.float64]
    dn_eff: NDArray[np.float64]
    alpha_opt_db_cm: NDArray[np.float64] | None = None
    wavelength_um: float = Field(gt=0.0)
    n_group: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        """Bias sweep arrays share one axis with at least two points."""
        if self.voltages_v.ndim != 1 or self.voltages_v.size < 2:
            raise ValueError("voltages_v must be 1D with >= 2 points.")
        if self.dn_eff.shape != self.voltages_v.shape:
            raise ValueError("dn_eff must have the same shape as voltages_v.")
        if (
            self.alpha_opt_db_cm is not None
            and self.alpha_opt_db_cm.shape != self.voltages_v.shape
        ):
            raise ValueError("alpha_opt_db_cm must match voltages_v.")
        return self


class TWMZMReport(BaseModel):
    """Full traveling-wave modulator device report.

    Attributes:
        freq_hz: RF frequencies of the response (Hz).
        response: Normalized complex EO response per frequency.
        bandwidth_3db_hz: 3 dB EO bandwidth, or None when the response
            stays above the threshold over the swept range.
        walkoff_bandwidth_hz: Walk-off-limited bandwidth of a lossless
            matched line, with the Velocity mismatch the line has at that
            frequency (the last solved index held past the solved range),
            or None when velocity matched.
        velocity_mismatch: ``n_rf - n_group`` per frequency.
        z0_ohm: Characteristic impedance per frequency.
        z_load_ohm: Termination impedance used.
        z_gen_ohm: Generator impedance used.
        rlgc: RLGC per-unit-length parameters per frequency.
        vpi_l_vcm: V_pi·L in V*cm at each bias point.
        voltages_v: Bias voltages of the V_pi·L sweep.
        length_m: Electrode length (m); for a segmented electrode, the
            whole number of periods nearest the length asked for.
        drive: Drive configuration of the Mach-Zehnder the transfer, its
            figures and the chirp are reported for.
        arm_bias_v: The bias both arms rest at (V).
        drive_v: Drive voltages of the transfer — the voltage between the
            two arms — spanning what the Bias sweep covers (V).
        transfer: Static intensity transfer ``T`` per drive voltage, as a
            fraction of the input power.
        v_pi_v: Drive swing from the transfer's peak to its null at this
            length (V) — the loaded fraction of it, for a segmented
            electrode — or None when the Bias sweep does not reach both.
        insertion_loss_db: Loss at the transfer's peak (dB), or None with
            ``v_pi_v``.
        extinction_ratio_db: Peak over null of the transfer (dB), infinite
            for arms that cancel exactly, or None with ``v_pi_v``.
        transfer_message: Why ``v_pi_v``, ``insertion_loss_db`` and
            ``extinction_ratio_db`` are absent, when they are.
        chirp: Small-signal chirp parameter at each bias point, in the sign
            convention of :func:`gsim.modulator.twmzm.mzm_chirp`.
        fill_factor: Fraction of the Traveling-wave electrode the Junction
            loads. One is an electrode loaded all the way; below one the
            electrode is segmented, and every line quantity here —
            ``z0_ohm``, ``n_rf``, ``alpha_rf_np_m``, ``velocity_mismatch``,
            ``rlgc`` — is the periodic line's Bloch one, the response
            counts the loaded sections only, and ``vpi_l_vcm`` is the
            device's, the Phase shifter's divided by the fill factor.
        period_m: Period of the segmentation (m); ``None`` when the
            electrode is not segmented.
        n_rf: RF effective index per frequency.
        alpha_rf_np_m: RF amplitude loss per frequency (Np/m).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    freq_hz: NDArray[np.float64]
    response: NDArray[np.complex128]
    bandwidth_3db_hz: float | None
    walkoff_bandwidth_hz: float | None
    velocity_mismatch: NDArray[np.float64]
    z0_ohm: NDArray[np.complex128]
    z_load_ohm: complex
    z_gen_ohm: complex
    rlgc: dict[str, NDArray[np.float64]]
    vpi_l_vcm: NDArray[np.float64]
    voltages_v: NDArray[np.float64]
    length_m: float
    drive: DriveConfiguration
    arm_bias_v: float
    drive_v: NDArray[np.float64]
    transfer: NDArray[np.float64]
    v_pi_v: float | None
    insertion_loss_db: float | None
    extinction_ratio_db: float | None
    transfer_message: str | None
    chirp: NDArray[np.float64]
    n_rf: NDArray[np.float64]
    alpha_rf_np_m: NDArray[np.float64]
    fill_factor: float = 1.0
    period_m: float | None = None


#: Drive voltages the report's Mach-Zehnder transfer is sampled on.
_TRANSFER_POINTS = 401


def twmzm_figures_of_merit(
    rf: RFLineParams,
    optical: OpticalPhaseSweep,
    *,
    length_m: float,
    z_load_ohm: complex = 50.0,
    z_gen_ohm: complex = 50.0,
    drive: DriveConfiguration = "push-pull",
    arm_bias_v: float | None = None,
    arm_imbalance_db: float = 0.0,
    phase_offset_rad: float = QUADRATURE_RAD,
    unloaded: RFLineParams | None = None,
    fill_factor: float = 1.0,
    period_m: float | None = None,
) -> TWMZMReport:
    """Combine RF line parameters and the optical sweep into the report.

    A fill factor below one reports a segmented Traveling-wave electrode:
    loaded sections of ``fill_factor * period_m`` alternating with
    unloaded ones, which reaches toward 50 ohm and a lower RF index at the
    price of modulating over the loaded fraction only. Its line behaviour
    is cascaded from the two Cross-section solves
    (:func:`gsim.common.transmission_line.segmented_line_params`), which is an
    approximation of the 3D structure: both sections share one electrode
    Cross-section, with no loading fins drawn, and the fields at each
    section boundary are not solved. The electrode is taken as the whole
    number of periods nearest ``length_m``.

    Args:
        rf: RF line parameters versus frequency (one bias point).
        optical: Optical phase response along the bias sweep.
        length_m: Electrode length in meters (> 0).
        z_load_ohm: Termination impedance in ohms.
        z_gen_ohm: Generator impedance in ohms.
        drive: Drive configuration of the Mach-Zehnder the Phase shifter
            is put in the arms of, ``"single-drive"`` or ``"push-pull"``.
        arm_bias_v: The bias both arms rest at (V); the middle of the
            sweep when omitted.
        arm_imbalance_db: Power the splitter sends down the first arm over
            the second (dB); 0 is a balanced interferometer.
        phase_offset_rad: Static phase offset between the arms (rad); the
            quadrature point by default.
        unloaded: The unloaded line's parameters on the same frequencies;
            needed — and read — only for a fill factor below one.
        fill_factor: Fraction of the electrode the Junction loads, above 0
            and at most 1. One is the electrode loaded all the way.
        period_m: Period of the segmentation (m, > 0); needed only for a
            fill factor below one.

    Returns:
        The assembled :class:`TWMZMReport`.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    if not 0.0 < fill_factor <= 1.0:
        raise ValueError("fill_factor must be above 0 and at most 1.")

    if fill_factor < 1.0:
        if unloaded is None:
            raise ValueError(
                "A fill factor below one needs the unloaded line's "
                "parameters: pass unloaded=."
            )
        if period_m is None:
            raise ValueError("A fill factor below one needs a period: pass period_m=.")
        # The electrode is the whole number of periods nearest the length
        # asked for, and every figure of the report is taken over it.
        n_periods = max(1, round(length_m / period_m))
        length_m = n_periods * period_m
        # From here on ``rf`` is the periodic line: every line quantity of
        # the report is the Bloch one.
        loaded = rf
        rf = segmented_line(
            loaded, unloaded, fill_factor=fill_factor, period_m=period_m
        )
        response = segmented_eo_response(
            rf.freq_hz,
            n_periods=n_periods,
            period_m=period_m,
            fill_factor=fill_factor,
            n_opt=optical.n_group,
            n_rf_loaded=loaded.n_rf,
            alpha_loaded_np_m=loaded.alpha_rf_np_m,
            z0_loaded_ohm=loaded.z0_ohm,
            n_rf_unloaded=unloaded.n_rf,
            alpha_unloaded_np_m=unloaded.alpha_rf_np_m,
            z0_unloaded_ohm=unloaded.z0_ohm,
            z_load_ohm=z_load_ohm,
            z_gen_ohm=z_gen_ohm,
        )
    else:
        period_m = None
        response = eo_response(
            rf.freq_hz,
            length_m=length_m,
            n_rf=rf.n_rf,
            n_opt=optical.n_group,
            alpha_rf_np_m=rf.alpha_rf_np_m,
            z0_ohm=rf.z0_ohm,
            z_load_ohm=z_load_ohm,
            z_gen_ohm=z_gen_ohm,
        )
    bandwidth = eo_bandwidth(rf.freq_hz, response)

    # Not read off the band's mean index: a loaded line's index falls with
    # frequency and can cross the group index, where the mean mismatch
    # vanishes and the limit it implies runs away.
    walkoff = walkoff_bandwidth_dispersive(
        rf.freq_hz, rf.n_rf, length_m=length_m, n_opt=optical.n_group
    )

    # The Phase shifter in both arms of a Mach-Zehnder: the transfer over
    # the drive the Bias sweep covers, and the figures read off it. The
    # transfer interpolates along the sweep, so it reads it in bias order.
    # Only the loaded fraction of a segmented electrode shifts the phase, so
    # that is the length of Phase shifter each arm holds.
    loaded_length_m = length_m * fill_factor
    order = np.argsort(optical.voltages_v, kind="stable")
    sweep = (
        optical.voltages_v[order],
        optical.dn_eff[order],
        None if optical.alpha_opt_db_cm is None else optical.alpha_opt_db_cm[order],
    )
    drive_v = np.linspace(
        *mzm_drive_range(sweep[0], drive=drive, bias_v=arm_bias_v),
        _TRANSFER_POINTS,
    )
    transfer = mzm_transfer(
        drive_v,
        *sweep,
        length_m=loaded_length_m,
        wavelength_um=optical.wavelength_um,
        drive=drive,
        bias_v=arm_bias_v,
        arm_imbalance_db=arm_imbalance_db,
        phase_offset_rad=phase_offset_rad,
    )
    # Per bias point, so back in the order the sweep came in.
    chirp = np.empty_like(sweep[0])
    chirp[order] = mzm_chirp(
        *sweep,
        wavelength_um=optical.wavelength_um,
        drive=drive,
        arm_imbalance_db=arm_imbalance_db,
        phase_offset_rad=phase_offset_rad,
    )
    figures = mzm_transfer_figures(
        *sweep,
        length_m=loaded_length_m,
        wavelength_um=optical.wavelength_um,
        drive=drive,
        bias_v=arm_bias_v,
        arm_imbalance_db=arm_imbalance_db,
        phase_offset_rad=phase_offset_rad,
    )

    return TWMZMReport(
        freq_hz=rf.freq_hz,
        response=np.asarray(response, dtype=np.complex128),
        bandwidth_3db_hz=bandwidth,
        walkoff_bandwidth_hz=walkoff,
        velocity_mismatch=np.asarray(rf.n_rf - optical.n_group, dtype=np.float64),
        z0_ohm=rf.z0_ohm,
        z_load_ohm=complex(z_load_ohm),
        z_gen_ohm=complex(z_gen_ohm),
        rlgc=rf.rlgc,
        vpi_l_vcm=vpi_length_vcm(
            optical.voltages_v,
            optical.dn_eff,
            wavelength_um=optical.wavelength_um,
        )
        # The light is modulated along the loaded fraction only, so the
        # device needs that much more length than its Phase shifter does.
        / fill_factor,
        voltages_v=optical.voltages_v,
        length_m=length_m,
        drive=drive,
        arm_bias_v=(
            float(arm_bias_v)
            if arm_bias_v is not None
            else float(0.5 * (sweep[0][0] + sweep[0][-1]))
        ),
        drive_v=drive_v,
        transfer=transfer,
        v_pi_v=figures.v_pi_v,
        insertion_loss_db=figures.insertion_loss_db,
        extinction_ratio_db=figures.extinction_ratio_db,
        transfer_message=figures.message,
        chirp=chirp,
        n_rf=rf.n_rf,
        alpha_rf_np_m=rf.alpha_rf_np_m,
        fill_factor=fill_factor,
        period_m=period_m,
    )
