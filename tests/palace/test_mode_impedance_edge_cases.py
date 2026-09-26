"""Invalid CSV values must not contaminate valid modal impedance results."""

import pytest

from gsim.palace.results import load_text_results


@pytest.mark.parametrize("invalid_value", ["nan", "inf", "-inf", "", "unavailable"])
def test_nonfinite_mode_impedances_are_omitted(tmp_path, invalid_value):
    (tmp_path / "mode-Z.csv").write_text(
        "m,Z_PV[1] (Ohm),Z_VI[1] (Ohm),Z_PV[2] (Ohm)\n"
        f"1,{invalid_value},50,{invalid_value}\n"
        "2,75,80,90\n"
    )
    results = load_text_results(tmp_path)
    assert results.characteristic_impedance(index=1, mode=1) is None
    assert results.characteristic_impedance(index=2, mode=1) is None
    assert results.characteristic_impedance(index=1, mode=1, quantity="Z_VI") == 50
    assert results.characteristic_impedance(index=1, mode=2) == 75
    assert results.characteristic_impedance(index=2, mode=2) == 90
