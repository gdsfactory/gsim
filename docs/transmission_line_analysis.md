# Transmission-line analysis

Install `gsim[rf]` for the optional scikit-rf dependency. The utilities in `gsim.palace.transmission` operate on
two-port `skrf.Network` objects, including those returned by `load_sparams(...).to_skrf()`. Frequencies are in Hz,
lengths in metres, and propagation is `gamma_per_m = alpha + 1j * beta`, with alpha in Np/m and beta in rad/m. A forward
wave varies as `exp(-gamma * length)`.

## Propagation and an independent length check

Two lengths must share the same uniform reciprocal line and unchanged launch networks. The left and right launches may
differ. The transfer-matrix ratio cancels their effect on the propagation eigenvalues. This example uses a known matched
line with phase index 2.4 and attenuation 25 Np/m:

```python
import numpy as np
import skrf as rf

from gsim.palace.transmission import extract_propagation, predict_line

frequency = rf.Frequency.from_f([40e9, 60e9, 80e9], unit="hz")
gamma = 25 + 1j * 2 * np.pi * frequency.f * 2.4 / 299792458
medium = rf.media.DefinedGammaZ0(frequency=frequency, z0=50, gamma=gamma)
short = medium.line(100e-6, unit="m")
long = medium.line(300e-6, unit="m")

propagation = extract_propagation(
    short, long, length_difference_m=200e-6, maximum_phase_index=3
)
np.testing.assert_allclose(propagation.gamma_per_m, gamma)
np.testing.assert_allclose(propagation.phase_index, 2.4)

predicted = predict_line(
    short, long, length_difference_m=200e-6,
    target_difference_m=100e-6, maximum_phase_index=3,
)
np.testing.assert_allclose(predicted.s, medium.line(200e-6, unit="m").s, atol=1e-12)
```

For solver output, compare `predicted` with an independently simulated intermediate line. Passing this check supports
the uniform-line/common-launch assumption; two lengths alone cannot establish it.

The phase-index bound must come from independent knowledge. Its phase difference must remain strictly below pi at the
highest supplied frequency. The routine does not unwrap an ambiguous logarithm or infer a valid bound from wrapped data.
It preserves negative attenuation instead of enforcing passivity. Frequency grids must match exactly; S data must be
finite, with the same real positive reference normalization on both ports and all networks. Transfer matrices with
condition numbers above `1e12`, degenerate phase roots, and reciprocal eigenvalue products differing from unity by more
than 0.1% are rejected.

## Simulated TRL calibration

The TRL wrappers require explicit reference planes and equal, isolated reflect terminations. They use zero
instrument-switch terms for fixed-boundary simulated S matrices. The caller establishes that launches are common across
standards and DUTs and that both intrinsic reflect coefficients are the same; matching array shapes cannot verify these
physical assumptions.

`reference_planes_m=(left, right)` specifies distances inward from the line ends. For a 100-micrometre thru,
`(50e-6, 50e-6)` places both planes at its midpoint. A physical 200-micrometre uniform DUT then has 100 micrometres of
line between the calibrated planes. Asymmetric shifts are also supported. `reflect_offset_m` locates each equal
termination relative to its line end.

Continuing the matched-line example:

```python
from gsim.palace.transmission import calibrate_multiline_trl, calibrate_trl

termination = medium.delay_short(50e-6, unit="m")
left = medium.resistor(3)
right = medium.shunt_capacitor(30e-15)
thru_standard = left ** short ** right
line_standard = left ** long ** right
reflect = rf.two_port_reflect(left ** termination, right.flipped() ** termination)
geometry = dict(
    thru_length_m=100e-6, maximum_phase_index=3,
    reference_planes_m=(50e-6, 50e-6), reflect_offset_m=50e-6,
)
calibration = calibrate_trl(
    thru_standard, reflect, line_standard, line_length_m=300e-6, **geometry
)
raw_dut = left ** medium.line(200e-6, unit="m") ** right
corrected = calibration.apply_cal(raw_dut)
np.testing.assert_allclose(corrected.s, medium.line(100e-6, unit="m").s, atol=1e-7)
assert calibration.wave_basis == "normalized_line"

multiline = calibrate_multiline_trl(
    thru_standard, reflect,
    [line_standard, left ** medium.line(1100e-6, unit="m") ** right],
    line_lengths_m=[300e-6, 1100e-6], **geometry,
)
np.testing.assert_allclose(
    multiline.apply_cal(raw_dut).s, corrected.s, atol=1e-7
)
```

The returned network's `z0=1` is a **normalized line-wave label**, not an inferred one-ohm characteristic impedance.
Calibration does not supply the independent impedance information needed to renormalize to physical ohms or extract
RLGC. The underlying algorithms follow scikit-rf's
[TRL](https://scikit-rf.readthedocs.io/en/latest/api/calibration/generated/skrf.calibration.calibration.TRL.html) and
[NIST multiline TRL](https://scikit-rf.readthedocs.io/en/latest/api/calibration/generated/skrf.calibration.calibration.NISTMultilineTRL.html)
implementations.

Single-line TRL requires at least 20 degrees of separation from a 0/180-degree singularity by default. Multiline TRL
requires one usable line pair per frequency and a short line whose phase is bounded below pi. That line fixes the
branch; longer lines may wrap and improve conditioning. The solved multiline propagation is checked against the bounded
anchor. Use `line_conditioning` to inspect phase separation before selecting standards; it does not assess loss or mesh
accuracy.

Degenerate backend solves raise a contextual error. In particular, scikit-rf 1.9 NIST calibration can return nonfinite
coefficients for perfectly matched standards with exactly zero launch reflections, even when their phase separation is
adequate. Single-line `calibrate_trl` supports that special case.

Reference-plane correctness requires independent geometry information. The tests include unequal launches, an imperfect
unknown reflect, held-out uniform and mismatched DUTs, and a deliberately incorrect physical-length expectation that
must disagree with the calibrated result.
