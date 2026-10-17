"""Check mode-field reconstruction and incomplete runner outputs."""

import json

import numpy as np
import pytest

from gsim.meep.results import ModeSweepResult


@pytest.mark.parametrize("plane", ["xz", "yz", None])
def test_mode_directory_restores_complex_fields_and_metadata(tmp_path, plane):
    metadata = dict(
        n_eff=2.4,
        wavelength=1.55,
        frequency=1 / 1.55,
        n_group=4.1,
        band_num=3,
        wl_idx=2,
        cross_section_plane=plane,
        parity="EVEN_Y",
    )
    (tmp_path / "mode_results.json").write_text(json.dumps({"results": [metadata]}))
    horizontal, vertical = np.array([-1, 0, 1]), np.array([-0.5, 0.5])
    np.save(tmp_path / "mode_h_grid.npy", horizontal)
    np.save(tmp_path / "mode_z_grid.npy", vertical)
    fields = {
        name: np.arange(6).reshape(2, 3) + (index + 1) * 1j
        for index, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    }
    for name, values in fields.items():
        np.save(tmp_path / f"mode_f_band3_wl2_{name}.npy", values)
    sweep = ModeSweepResult.from_directory(tmp_path)
    mode = sweep.at(1.55).band(3)
    assert mode is not None
    assert (mode.n_eff, mode.n_group, mode.parity) == (2.4, 4.1, "EVEN_Y")
    np.testing.assert_array_equal(mode.z_grid, vertical)
    if plane is not None:
        np.testing.assert_array_equal(
            mode.x_grid if plane == "xz" else mode.y_grid, horizontal
        )
    assert (mode.y_grid if plane == "xz" else mode.x_grid) is None
    for name, values in fields.items():
        np.testing.assert_array_equal(mode.fields[name], values)
    assert sweep.to_dict()["wavelength"] == [1.55]
    assert sweep.at(1.6).results == []
    assert sweep.band(4) is None


@pytest.mark.parametrize(
    "filename", ["mode_z_grid.npy", "mode_h_grid.npy", "mode_f_band1_wl0_Ex.npy"]
)
def test_corrupt_optional_arrays_preserve_valid_fields(tmp_path, filename):
    (tmp_path / "mode_results.json").write_text(
        json.dumps(
            {
                "results": [
                    dict(
                        n_eff=2.4,
                        wavelength=1.55,
                        frequency=1 / 1.55,
                        cross_section_plane="xz",
                    )
                ]
            }
        )
    )
    (tmp_path / filename).write_bytes(b"incomplete numpy data")
    field = np.array([1 + 2j, 3 - 4j])
    np.save(tmp_path / "mode_f_band1_wl0_Ey.npy", field)
    mode = ModeSweepResult.from_directory(tmp_path).band(1)
    assert mode is not None
    np.testing.assert_array_equal(mode.fields["Ey"], field)
    assert "Ex" not in mode.fields
    assert mode.x_grid is mode.z_grid is None


@pytest.mark.parametrize("contents", [None, "incomplete json", "{}"])
def test_missing_malformed_and_empty_mode_metadata(tmp_path, contents):
    if contents is not None:
        (tmp_path / "mode_results.json").write_text(contents)
    error = FileNotFoundError if contents is None else json.JSONDecodeError
    if contents == "{}":
        assert ModeSweepResult.from_directory(tmp_path).results == []
    else:
        with pytest.raises(error):
            ModeSweepResult.from_directory(tmp_path)
