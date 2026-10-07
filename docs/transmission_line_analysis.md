# Transmission-line analysis with scikit-rf

Use scikit-rf directly for TRL, multiline TRL and fixture de-embedding. gsim supplies the Palace result conversion;
calibration algorithms and propagation extraction stay in scikit-rf. This example is tested with **scikit-rf 2.1.0**:

```bash
uv pip install 'scikit-rf>=2.1'
```

Import algorithms from `skrf.calibration`, rather than the removed top-level aliases. Convert each independently
simulated standard and DUT using its actual port reference impedance:

```pycon
>>> from gsim.palace import load_sparams
>>> thru = load_sparams("thru/output").to_skrf(z0=50)
>>> reflect = load_sparams("reflect/output").to_skrf(z0=50)
>>> lines = [load_sparams(path).to_skrf(z0=50) for path in line_output_dirs]
>>> measured_dut = load_sparams("dut/output").to_skrf(z0=50)
```

Supply the full two-port matrix, including both excitations. `to_skrf()` can fill missing reciprocal entries, which
cannot replace checking the simulated port mode, normalization and reciprocity. All networks must use the same frequency
grid and wave convention. Standards and DUTs must share their launches; the two launches may differ. TRL additionally
needs equal, isolated reflect terminations. Lengths are in metres and frequencies are in Hz.

## A reproducible example

The following blocks run together without simulations. Independent telegrapher matrices generate a uniform line with
series loss, high-frequency impedance 70 ohms and phase index 2.6, between unequal launches at 50-ohm ports. A separate
700-micrometre line is withheld from calibration. Propagation is `gamma = alpha + 1j * beta`, with a forward wave
`exp(-gamma * length)`; alpha is in Np/m and beta is in rad/m.

```python
import numpy as np
import skrf as rf
from scipy.constants import c
from skrf.calibration import NISTMultilineTRL, TUGMultilineTRL
from skrf.network import two_port_reflect

frequency = rf.Frequency(5, 60, 111, "ghz")
omega = 2 * np.pi * frequency.f
capacitance_per_m = 2.6 / (70 * c)
inductance_per_m = 70 * 2.6 / c
resistance_per_m = 5000.0
gamma_exact = np.sqrt(
    (resistance_per_m + 1j * omega * inductance_per_m)
    * (1j * omega * capacitance_per_m)
)
impedance_exact = gamma_exact / (1j * omega * capacitance_per_m)


def line(length_m):
    """Create an analytical line at physical 50-ohm reference ports."""
    argument = gamma_exact * length_m
    abcd = np.zeros((len(frequency), 2, 2), dtype=complex)
    abcd[:, 0, 0] = abcd[:, 1, 1] = np.cosh(argument)
    abcd[:, 0, 1] = impedance_exact * np.sinh(argument)
    abcd[:, 1, 0] = np.sinh(argument) / impedance_exact
    return rf.Network(frequency=frequency, a=abcd, z0=50)


ports = rf.media.DefinedGammaZ0(frequency, z0=50)
left = ports.resistor(3) ** ports.shunt_capacitor(20e-15)
right = ports.inductor(40e-12) ** ports.shunt_resistor(1200)
lengths_m = np.array([100, 300, 900, 3100]) * 1e-6
measured_lines = [left ** line(length) ** right for length in lengths_m]
termination = line(50e-6) ** ports.short()
reflect = two_port_reflect(left ** termination, right.flipped() ** termination)
measured_dut = left ** line(700e-6) ** right
zero_switch = rf.Network(frequency=frequency, s=np.zeros(len(frequency)), z0=50)
```

Zero switch terms are appropriate for fixed-boundary simulations, not a substitute for measured VNA switch terms.

## Phase conditioning and physical impedance

At every frequency, at least one line pair needs phase separation away from multiples of 180 degrees. This example uses
a 20-degree minimum. The independently estimated index and short line help select the propagation branch; conditioning
alone does not prove the selected mode or branch is correct.

```python
def phase_separation_deg(beta, lengths):
    """Return the best pair's distance from a 0/180-degree singularity."""
    differences = np.array(
        [b - a for i, a in enumerate(lengths) for b in lengths[i + 1 :]]
    )
    phase = np.asarray(beta)[:, None] * differences
    return np.rad2deg(np.abs(np.angle(np.exp(2j * phase))) / 2).max(axis=1)


index_estimate = 2.6  # From independent cross-section information.
maximum_index = 3.2
short_difference = lengths_m[1] - lengths_m[0]
assert omega[-1] / c * maximum_index * short_difference < np.pi
assert np.all(phase_separation_deg(omega * index_estimate / c, lengths_m) > 20)

calibration = NISTMultilineTRL(
    measured=[measured_lines[0], reflect, *measured_lines[1:]],
    Grefls=[-1],
    l=lengths_m.tolist(),
    er_est=index_estimate**2,
    gamma_root_choice="estimate",
    refl_offset=50e-6,
    ref_plane=0,
    switch_terms=(zero_switch, zero_switch.copy()),
    c0=capacitance_per_m,
    z0_ref=50,
)
calibration.run()
if not all(np.isfinite(value).all() for value in calibration.coefs.values()):
    raise ValueError("Non-finite calibration coefficients; inspect the standards")
corrected = calibration.apply_cal(measured_dut)
assert np.isfinite(corrected.s).all()
assert np.all(phase_separation_deg(calibration.gamma.imag, lengths_m) > 20)
np.testing.assert_allclose(calibration.gamma, gamma_exact, rtol=1e-9)
np.testing.assert_allclose(calibration.z0, impedance_exact, rtol=1e-9)
np.testing.assert_allclose(corrected.s, line(700e-6).s, atol=1e-9)
np.testing.assert_allclose(corrected.z0, 50)
print(f"Zc at 60 GHz: {calibration.z0[-1]:.2f} ohms")
```

`c0` is capacitance **per length (F/m)** from the same cross-section. Under the TEM/quasi-TEM approximation and **G=0**,
scikit-rf obtains `Zc = gamma / (1j * omega * c0)` and renormalizes the DUT to `z0_ref=50`. Here the expected result is
approximately **70.00-0.76j ohms at 60 GHz**. `calibration.z0` is the inferred line impedance; `corrected.z0` is the
output network's 50-ohm reference. This does not validate dielectric loss or establish physical RLGC for a periodic
cell. If shunt conductance matters, provide independently justified `z0_line` instead of `c0`, or leave physical
impedance undetermined. Omitting both does not make the calibrated line a known 50-ohm line.

Use the extracted medium and scikit-rf's `embed` to predict an independent length, without another extraction routine:

```python
extracted_medium = rf.media.DefinedGammaZ0(
    frequency, gamma=calibration.gamma, z0=calibration.z0, z0_port=50
)
predicted = calibration.embed(extracted_medium.line(700e-6, unit="m"))
np.testing.assert_allclose(predicted.s, measured_dut.s, atol=1e-9)
```

On solver data, compare against a separately simulated length. Reconstructing the calibration standards alone cannot
test the uniform-line/common-launch assumption.

## Unequal reference planes

With the nonzero thru length supplied above, `ref_plane=0` places the calibrated planes at the physical line ends;
scikit-rf accounts for the thru internally. Positive shifts below move each plane inward from its corresponding end.

In scikit-rf 2.1.0, unequal NIST `ref_plane` values can break reciprocal transmission
([upstream issue #1444](https://github.com/scikit-rf/scikit-rf/issues/1444)). Keep `ref_plane=0` and cascade
negative-length sections **after calibration**, using the extracted line impedance and the same 50-ohm port basis:

```python
shift_left_m, shift_right_m = 20e-6, 80e-6
shifted = corrected.copy()
if shift_left_m != 0:
    shifted = extracted_medium.line(-shift_left_m, unit="m") ** shifted
if shift_right_m != 0:
    shifted = shifted ** extracted_medium.line(-shift_right_m, unit="m")
np.testing.assert_allclose(shifted.s[:, 1, 0], shifted.s[:, 0, 1], atol=1e-9)
np.testing.assert_allclose(shifted.s, line(600e-6).s, atol=1e-9)
```

In the line's own wave basis, a shift multiplies S11 by `exp(2*gamma*d1)`, S22 by `exp(2*gamma*d2)`, and **both**
transmissions by `exp(gamma*(d1+d2))`. Do not apply that elementwise formula directly to a mismatched 50-ohm network;
the cascaded sections account for the impedance transformation. Negative shifts add line; positive shifts remove line.

## Other scikit-rf methods

TUG multiline TRL offers another maintained propagation/calibration algorithm. A nonzero first line is treated as the
thru; `ref_plane=0` again restores the physical ends. Physical impedance can be supplied through `renormalize`:

```python
tug = TUGMultilineTRL(
    line_meas=measured_lines,
    line_lengths=lengths_m.tolist(),
    er_est=index_estimate**2,
    reflect_meas=[reflect],
    reflect_est=[-1],
    reflect_offset=50e-6,
    ref_plane=0,
    switch_terms=(zero_switch, zero_switch.copy()),
)
tug.run()
np.testing.assert_allclose(tug.gamma, gamma_exact, rtol=1e-9)
tug.renormalize(tug.gamma / (1j * omega * capacitance_per_m), 50)
np.testing.assert_allclose(tug.apply_cal(measured_dut).s, corrected.s, atol=1e-9)
```

TUG passes the tested exactly matched case on 2.1.0. NIST recovers its propagation constant but can still produce
non-finite calibration coefficients when every launch/line reflection is exactly zero. Check coefficients and the
corrected DUT, not just gamma; do not reject all matched lines or assume this limitation ended with 1.9. For a single
usable line use `skrf.calibration.TRL`. For fixture removal with a suitable 2x-thru, see
[`IEEEP370_SE_NZC_2xThru`](https://scikit-rf.readthedocs.io/en/latest/api/calibration/generated/skrf.calibration.deembedding.IEEEP370_SE_NZC_2xThru.html).
P370 removes fixtures under its own bandwidth/fixture assumptions; it does not independently determine a uniform line's
characteristic impedance or replace checking TRL standards.

References: scikit-rf's
[multiline TRL example](https://scikit-rf.readthedocs.io/en/latest/examples/metrology/Multiline%20TRL.html),
[NIST parameters](https://scikit-rf.readthedocs.io/en/latest/api/calibration/generated/skrf.calibration.calibration.NISTMultilineTRL.__init__.html)
and
[TUG parameters](https://scikit-rf.readthedocs.io/en/latest/api/calibration/generated/skrf.calibration.calibration.TUGMultilineTRL.__init__.html).
