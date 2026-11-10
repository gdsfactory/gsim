"""Independent phase and power checks on an actual solver export."""

from pathlib import Path

import numpy as np
import pytest

from gsim import fdtd

DATA = Path(__file__).parent / "data" / "plane_fields"
SPEED_OF_LIGHT_M_S = 299792458


def test_real_non_square_plane_preserves_array_axes_and_s_parameters():
    result = fdtd.FDTDResult.from_file(DATA / "result.json")
    assert result.convergence["converged"]
    assert np.count_nonzero(result.valid) >= 3
    assert np.all(np.abs(result.s_parameters[("o2", "o1")].values[result.valid]) > 0.99)
    for monitor in result.monitors.values():
        assert monitor.shape == (32, 25)
        assert monitor.field_coordinates is not None
        assert monitor.field_coordinates.u_um.shape == (32,)
        assert monitor.field_coordinates.v_um.shape == (25,)
        assert monitor.heatmap(wavelength_um=1.55).load().shape == (25, 32)
        for sample in monitor.fields:
            assert sample.shape == (25, 32)
            raw = np.load(sample.real_file) + 1j * np.load(sample.imag_file)
            np.testing.assert_array_equal(sample.load(), raw)


def test_real_forward_fields_and_s_parameters_use_positive_time_phasors():
    result = fdtd.FDTDResult.from_file(DATA / "result.json")
    near = result.monitors["near"].field("ey", wavelength_um=1.55)
    far = result.monitors["far"].field("ey", wavelength_um=1.55)
    distance = far.coordinates.normal_position_um - near.coordinates.normal_position_um
    assert distance == pytest.approx(0.2)
    phase = np.angle(np.vdot(near.load(), far.load()))
    # For exp(+i*omega*t), forward propagation has exp(-i*beta*x).
    # This short separation cannot wrap even at the core's index of 3.47.
    effective_index = -phase * near.wavelength_um / (2 * np.pi * distance)
    assert 1.44 < effective_index < 3.47
    order = np.argsort(result.frequency_hz[result.valid])
    transmission = result.s_parameters[("o2", "o1")].values[result.valid][order]
    assert np.all(np.diff(np.unwrap(np.angle(transmission))) < 0)


@pytest.mark.parametrize("name", ["near", "far"])
def test_real_colocated_field_power_agrees_with_independent_flux(name):
    monitor = fdtd.FDTDResult.from_file(DATA / "result.json").monitors[name]
    fields = {sample.component: sample.load() for sample in monitor.fields}
    sample = monitor.fields[0]
    # The backend flux is in solver units: recover the solver length unit from
    # its c=1 time unit, then integrate the independent colocated field arrays.
    length_unit_m = sample.metadata["solver_time_unit_seconds"] * SPEED_OF_LIGHT_M_S
    du = np.diff(sample.coordinates.u_um).mean() * 1e-6 / length_unit_m
    dv = np.diff(sample.coordinates.v_um).mean() * 1e-6 / length_unit_m
    cross_product = (
        fields["ey"] * fields["hz"].conj() - fields["ez"] * fields["hy"].conj()
    )
    power = 0.5 * np.real(cross_product).sum() * du * dv
    index = np.argmin(abs(monitor.wavelength_um - sample.wavelength_um))
    assert monitor.flux is not None
    # The native flux retains transverse Yee staggering; the exported arrays
    # average onto common primal points. This 50 nm grid differs by about 1.1%.
    assert power == pytest.approx(monitor.flux[index], rel=0.015)
