"""Complex source-mode arrays, physical plot coordinates, and dry-run parsing."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsim.fdtd import FDTDResult


def mode_document(tmp_path, *, dry_run=True):
    fields = {}
    expected = {}
    for index, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        values = np.arange(6).reshape(2, 3) + index + 1j * (index + 0.5)
        expected[component] = values
        real_name = f"{component}_real.npy"
        imag_name = f"{component}_imag.npy"
        np.save(tmp_path / real_name, values.real)
        np.save(tmp_path / imag_name, values.imag)
        fields[component] = {"real_file": real_name, "imag_file": imag_name}
    return {
        "dry_run": dry_run,
        "excitation_type": "eigenmode",
        "excited_port": "o1",
        "ports": ["o1"],
        "frequencies": {"wavelength_nm": [], "hz": []},
        "convergence": {"timesteps_run": 0 if dry_run else 123},
        "source_mode": {
            "port": "o1",
            "wavelength_nm": 1550,
            "n_eff": 2.4,
            "beta_per_um": 2 * np.pi * 2.4 / 1.55,
            "normal_axis": "x",
            "normal_sign": 1,
            "u_axis": "y",
            "v_axis": "z",
            "plane_position_nm": 120,
            "u_extent_nm": [-900, 900],
            "v_extent_nm": [-400, 600],
            "shape": [2, 3],
            "fields": fields,
        },
    }, expected


@pytest.mark.parametrize("dry_run", [True, False])
def test_mode_from_dry_or_full_result(tmp_path, dry_run):
    document, expected = mode_document(tmp_path, dry_run=dry_run)
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    result = FDTDResult.from_file(path)
    mode = result.source_mode
    assert mode is not None
    assert result.dry_run == dry_run
    assert mode.n_eff == 2.4
    assert mode.wavelength_um == 1.55
    assert mode.beta_per_um * mode.wavelength_um / (2 * np.pi) == pytest.approx(
        mode.n_eff
    )
    for component, values in expected.items():
        np.testing.assert_array_equal(mode.load(component), values)
    figure, axes = mode.plot()
    np.testing.assert_allclose(
        axes.images[0].get_array(),
        np.sqrt(
            sum(np.abs(expected[component]) ** 2 for component in ("Ex", "Ey", "Ez"))
        ),
    )
    assert axes.get_xlabel() == "y (µm)"
    assert axes.get_ylabel() == "z (µm)"
    assert axes.images[0].get_extent() == [-0.9, 0.9, -0.4, 0.6]
    assert "2.40000" in axes.get_title()
    plt.close(figure)


def test_mode_arrays_are_lazy_and_shape_checked(tmp_path):
    document, _ = mode_document(tmp_path)
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    (tmp_path / "Ex_real.npy").unlink()
    mode = FDTDResult.from_file(path).source_mode
    assert mode is not None
    assert mode.n_eff == 2.4
    with pytest.raises(FileNotFoundError):
        mode.load("Ex")
    np.save(tmp_path / "Ex_real.npy", np.zeros((3, 2)))
    with pytest.raises(ValueError, match="shape"):
        mode.load("Ex")
    with pytest.raises(ValueError, match="component"):
        mode.load("wrong")
    with pytest.raises(ValueError, match="part"):
        mode.plot("Ey", part="wrong")


@pytest.mark.parametrize(
    ("component", "part"),
    [("Ey", "abs"), ("Ey", "real"), ("Ey", "imag"), ("abs_h", "abs")],
)
def test_mode_plot_selects_field_part_on_existing_axes(tmp_path, component, part):
    document, expected = mode_document(tmp_path)
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    mode = FDTDResult.from_file(path).source_mode
    assert mode is not None
    if component == "abs_h":
        expected_image = np.sqrt(
            sum(np.abs(expected[name]) ** 2 for name in ("Hx", "Hy", "Hz"))
        )
    else:
        values = expected[component]
        expected_image = {
            "abs": np.abs(values),
            "real": values.real,
            "imag": values.imag,
        }[part]
    figure, axes = plt.subplots()
    try:
        assert mode.plot(component, part=part, ax=axes, cmap="magma") == (figure, axes)
        np.testing.assert_allclose(
            np.asarray(axes.images[0].get_array()), expected_image
        )
        assert axes.images[0].get_cmap().name == "magma"
    finally:
        plt.close(figure)


def test_mode_loads_downloaded_sidecars_from_file_map(tmp_path):
    document, expected = mode_document(tmp_path)
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    download_directory = tmp_path / "downloaded"
    download_directory.mkdir()
    files = {
        filename: (tmp_path / filename).rename(download_directory / filename)
        for filename in ("Ex_real.npy", "Ex_imag.npy")
    }
    mode = FDTDResult.from_file(path, files=files).source_mode
    assert mode is not None
    np.testing.assert_array_equal(mode.load("Ex"), expected["Ex"])


def test_dry_run_cannot_report_s_parameters(tmp_path):
    document, _ = mode_document(tmp_path)
    document["s_parameters"] = {"S(o1,o1)": [{"re": 0, "im": 0}]}
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="dry run cannot contain"):
        FDTDResult.from_file(path)
