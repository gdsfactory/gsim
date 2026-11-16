"""Read the complex mode-Z.csv format introduced by awslabs/palace#1119."""

import pytest

from gsim.palace.line_impedance import native_line_impedance
from gsim.palace.results import load_text_results


def write_complex_results(tmp_path, *, real="50", imaginary="-10"):
    """Write the component names and units emitted by upstream Palace."""
    (tmp_path / "mode-Z.csv").write_text(
        " m, Re{Z_PV[3]} (Ohm), Im{Z_PV[3]} (Ohm),"
        " L_PV[3] (H/m), C_PV[3] (F/m),"
        " Re{Z_VI[3]} (Ohm), Im{Z_VI[3]} (Ohm),"
        " Re{Z_PV[5]} (Ohm), Im{Z_PV[5]} (Ohm)\n"
        f"2,{real},{imaginary},1e-7,2e-10,20,30,75,-5\n"
        "4,80,2,3e-7,4e-10,45,-3,90,0\n"
    )
    return load_text_results(tmp_path)


def test_complex_impedances_keep_modes_indices_and_phase(tmp_path):
    results = write_complex_results(tmp_path)

    assert results.complex_characteristic_impedance(index=3, mode=2) == 50 - 10j
    assert (
        results.complex_characteristic_impedance(index=3, mode=2, quantity="Z_VI")
        == 20 + 30j
    )
    assert results.complex_characteristic_impedance(index=5, mode=2) == 75 - 5j
    assert results.complex_characteristic_impedance(index=3, mode=4) == 80 + 2j
    assert results.complex_characteristic_impedance(index=5, mode=4) == 90 + 0j
    assert results.complex_characteristic_impedance(index=1, mode=2) is None
    assert results.complex_characteristic_impedance(index=3, mode=1) is None
    assert (
        results.complex_characteristic_impedance(index=5, mode=2, quantity="Z_VI")
        is None
    )


def test_scalar_access_keeps_real_values_and_lc_columns(tmp_path):
    results = write_complex_results(tmp_path)

    assert results.characteristic_impedance(index=3, mode=2) == 50.0
    assert isinstance(results.characteristic_impedance(index=3, mode=2), float)
    assert results.mode_impedance[3][2]["Z_PV"] == 50.0
    assert results.mode_impedance[3][2]["Im{Z_PV}"] == -10.0
    assert results.characteristic_impedance(
        index=3, mode=2, quantity="L_PV"
    ) == pytest.approx(1e-7)
    assert results.characteristic_impedance(
        index=3, mode=2, quantity="C_PV"
    ) == pytest.approx(2e-10)


def test_legacy_scalar_files_still_load_without_changing_return_type(tmp_path):
    (tmp_path / "mode-Z.csv").write_text("m,Z_PV[1] (Ohm),Z_VI[1] (Ohm)\n1,50,45\n")
    results = load_text_results(tmp_path)

    assert results.characteristic_impedance() == 50.0
    assert isinstance(results.characteristic_impedance(), float)
    assert results.complex_characteristic_impedance() == 50 + 0j
    assert results.complex_characteristic_impedance(quantity="Z_VI") == 45 + 0j
    assert results.mode_impedance == {1: {1: {"Z_PV": 50.0, "Z_VI": 45.0}}}


@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf", "", "unavailable"])
@pytest.mark.parametrize("component", ["real", "imaginary"])
def test_invalid_component_does_not_fabricate_a_complex_impedance(
    tmp_path, invalid, component
):
    results = write_complex_results(tmp_path, **{component: invalid})

    assert results.complex_characteristic_impedance(index=3, mode=2) is None
    assert results.complex_characteristic_impedance(index=3, mode=4) == 80 + 2j
    assert (
        results.complex_characteristic_impedance(index=3, mode=2, quantity="Z_VI")
        == 20 + 30j
    )
    assert native_line_impedance(results, index=3, mode_id=2, n_eff=2.0) is None


@pytest.mark.parametrize("component", ["Re", "Im"])
def test_incomplete_component_pair_is_unavailable(tmp_path, component):
    (tmp_path / "mode-Z.csv").write_text(f"m,{component}{{Z_PV[1]}} (Ohm)\n1,50\n")
    results = load_text_results(tmp_path)

    assert results.complex_characteristic_impedance() is None


@pytest.mark.parametrize("legacy_first", [True, False])
def test_component_pair_takes_precedence_over_a_scalar_alias(tmp_path, legacy_first):
    columns = ["Re{Z_PV[1]} (Ohm)", "Im{Z_PV[1]} (Ohm)"]
    values = ["50", "-10"]
    if legacy_first:
        columns.insert(0, "Z_PV[1] (Ohm)")
        values.insert(0, "999")
    else:
        columns.append("Z_PV[1] (Ohm)")
        values.append("999")
    (tmp_path / "mode-Z.csv").write_text(
        "m," + ",".join(columns) + "\n1," + ",".join(values) + "\n"
    )
    results = load_text_results(tmp_path)

    assert results.characteristic_impedance() == 50.0
    assert results.complex_characteristic_impedance() == 50 - 10j


def test_power_current_impedance_uses_magnitude_and_conjugation(tmp_path):
    results = write_complex_results(tmp_path)

    reading = native_line_impedance(results, index=3, mode_id=2, n_eff=2.0)

    # |20 + 30j|^2 / conj(50 - 10j) = 1300 / (50 + 10j) = 25 - 5j.
    # Squaring Z_VI, dropping phase, or omitting the conjugate changes this result.
    assert reading is not None
    assert reading.z0_ohm == pytest.approx(25 - 5j)
    assert reading.wall_mode is False


def test_invalid_component_pair_does_not_fall_back_to_a_scalar_alias(tmp_path):
    (tmp_path / "mode-Z.csv").write_text(
        "m,Z_PV[1] (Ohm),Re{Z_PV[1]} (Ohm),Im{Z_PV[1]} (Ohm)\n1,999,nan,inf\n"
    )
    results = load_text_results(tmp_path)

    assert results.complex_characteristic_impedance() is None
    assert results.characteristic_impedance() is None


def test_complex_current_impedance_sign_does_not_discard_forward_power(tmp_path):
    (tmp_path / "mode-Z.csv").write_text(
        "m,Re{Z_PV[1]} (Ohm),Im{Z_PV[1]} (Ohm),"
        "Re{Z_VI[1]} (Ohm),Im{Z_VI[1]} (Ohm)\n"
        "1,10,50,-20,30\n"
    )
    results = load_text_results(tmp_path)

    reading = native_line_impedance(results, index=1, mode_id=1, n_eff=2.0)

    # 1300 / (10 - 50j) = 5 + 25j has positive forward active power.
    assert reading is not None
    assert reading.z0_ohm == pytest.approx(5 + 25j)
    assert reading.wall_mode is False


def test_complex_impedance_keeps_the_wall_mode_diagnostic(tmp_path):
    results = write_complex_results(tmp_path, real="1e-6", imaginary="-1e-7")

    reading = native_line_impedance(results, index=3, mode_id=2, n_eff=2.0)

    assert reading is not None
    assert reading.wall_mode is True
    assert "window wall" in reading.diagnostic
