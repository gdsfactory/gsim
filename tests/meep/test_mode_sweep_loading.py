"""Check mode metadata and complex field reconstruction from runner outputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gsim.meep.results import ModeSweepResult


def _write_metadata(directory, **overrides):
    metadata = {"n_eff": 2.4, "wavelength": 1.55, "frequency": 1 / 1.55}
    metadata.update(overrides)
    (directory / "mode_results.json").write_text(json.dumps({"results": [metadata]}))


@pytest.mark.parametrize("plane", ["xz", "yz", None])
def test_loader_restores_complex_fields_grids_and_metadata(tmp_path, plane):
    _write_metadata(
        tmp_path,
        cross_section_plane=plane,
        wl_idx=2,
        band_num=3,
        n_group=4.1,
        kdom=[1.5, 0, 0],
        parity="EVEN_Y",
    )
    horizontal_grid = np.array([-1, 0, 1])
    z_grid = np.array([-0.5, 0.5])
    np.save(tmp_path / "mode_h_grid.npy", horizontal_grid)
    np.save(tmp_path / "mode_z_grid.npy", z_grid)
    fields = {}
    for index, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        fields[component] = np.arange(6).reshape(2, 3) + (index + 1) * 1j
        np.save(tmp_path / f"mode_f_band3_wl2_{component}.npy", fields[component])

    sweep = ModeSweepResult.from_directory(str(tmp_path))
    mode = sweep.at(1.55).band(3)

    assert mode is not None
    assert mode.n_eff == 2.4
    assert mode.frequency == pytest.approx(1 / 1.55)
    assert mode.n_group == 4.1
    assert mode.parity == "EVEN_Y"
    assert mode.kdom == [1.5, 0, 0]
    assert mode.cross_section_plane == plane
    np.testing.assert_array_equal(mode.z_grid, z_grid)
    if plane == "xz":
        np.testing.assert_array_equal(mode.x_grid, horizontal_grid)
        assert mode.y_grid is None
    elif plane == "yz":
        np.testing.assert_array_equal(mode.y_grid, horizontal_grid)
        assert mode.x_grid is None
    else:
        assert mode.x_grid is None
        assert mode.y_grid is None
    assert mode.fields.keys() == fields.keys()
    for component, values in fields.items():
        np.testing.assert_array_equal(mode.fields[component], values)
    assert sweep.at(1.6).results == []
    assert sweep.band(4) is None


def test_loader_defaults_when_optional_metadata_and_arrays_are_absent(tmp_path):
    _write_metadata(tmp_path)
    mode = ModeSweepResult.from_directory(tmp_path).band(1)
    assert mode is not None
    assert mode.fields == {}
    assert mode.kdom == []
    assert mode.parity == "NO_PARITY"
    assert mode.n_group is None
    assert mode.z_grid is None
    assert mode.x_grid is None
    assert mode.y_grid is None


@pytest.mark.parametrize(
    "corrupt_file", ["mode_z_grid.npy", "mode_h_grid.npy", "mode_f_band1_wl0_Ex.npy"]
)
def test_corrupt_optional_arrays_do_not_discard_other_fields(tmp_path, corrupt_file):
    _write_metadata(tmp_path, cross_section_plane="xz")
    (tmp_path / corrupt_file).write_bytes(b"incomplete numpy data")
    valid_field = np.array([1 + 2j, 3 - 4j])
    np.save(tmp_path / "mode_f_band1_wl0_Ey.npy", valid_field)

    mode = ModeSweepResult.from_directory(tmp_path).band(1)

    assert mode is not None
    np.testing.assert_array_equal(mode.fields["Ey"], valid_field)
    assert "Ex" not in mode.fields
    assert mode.x_grid is None
    assert mode.z_grid is None


def test_missing_mode_metadata_reports_directory(tmp_path):
    with pytest.raises(
        FileNotFoundError, match=r"mode_results\.json not found"
    ) as error:
        ModeSweepResult.from_directory(tmp_path)
    assert str(tmp_path) in str(error.value)


def test_malformed_metadata_is_not_silently_ignored(tmp_path):
    (tmp_path / "mode_results.json").write_text("incomplete json")
    with pytest.raises(json.JSONDecodeError):
        ModeSweepResult.from_directory(tmp_path)


@pytest.mark.parametrize("contents", [{}, {"results": []}])
def test_empty_sweep_produces_empty_dataframe_columns(tmp_path, contents):
    (tmp_path / "mode_results.json").write_text(json.dumps(contents))
    sweep = ModeSweepResult.from_directory(tmp_path)
    assert sweep.results == []
    assert sweep.band(1) is None
    assert sweep.to_dict() == {
        key: []
        for key in [
            "wavelength",
            "frequency",
            "band_num",
            "n_eff",
            "n_group",
            "parity",
            "kdom",
            "fields",
        ]
    }


def test_dataframe_columns_preserve_sample_order_and_fields(tmp_path):
    _write_metadata(tmp_path, n_group=4.1, parity="EVEN_Y", kdom=[1, 0, 0])
    field = np.array([1 + 2j, 3 - 4j])
    np.save(tmp_path / "mode_f_band1_wl0_Ex.npy", field)
    sweep = ModeSweepResult.from_directory(tmp_path)

    columns = sweep.to_dict()

    assert columns["wavelength"] == [1.55]
    assert columns["frequency"] == [1 / 1.55]
    assert columns["n_eff"] == [2.4]
    assert columns["n_group"] == [4.1]
    assert columns["band_num"] == [1]
    assert columns["parity"] == ["EVEN_Y"]
    assert columns["kdom"] == [[1, 0, 0]]
    np.testing.assert_array_equal(columns["fields"][0]["Ex"], field)
