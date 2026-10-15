"""Check numerical S-parameters, diagnostic discovery, and plot data."""

from __future__ import annotations

import json
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsim.meep.models.results import SParameterResult


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "s_parameters.csv"
    path.write_text(
        "wavelength,S11_mag,S11_phase,S21_mag,S21_phase\n"
        "1.5,0.1,180,0.5,90\n"
        "1.6,0,0,1,-90\n"
    )
    return path


def test_csv_preserves_complex_phase_wavelengths_and_port_names(csv_path):
    result = SParameterResult.from_directory(csv_path.parent)

    assert result.wavelengths == [1.5, 1.6]
    assert result.port_names == ["port_1", "port_2"]
    np.testing.assert_allclose(result.s_params["S11"], [-0.1, 0], atol=1e-15)
    np.testing.assert_allclose(result.s_params["S21"], [0.5j, -1j], atol=1e-15)


@pytest.mark.parametrize("contents", ["", "wavelength,S11_mag,S11_phase\n"])
def test_csv_without_rows_has_no_samples(tmp_path, contents):
    path = tmp_path / "s_parameters.csv"
    path.write_text(contents)
    result = SParameterResult.from_csv(path)
    assert result.wavelengths == []
    assert all(values == [] for values in result.s_params.values())


@pytest.mark.parametrize("value", ["invalid", ""])
def test_csv_rejects_invalid_numeric_samples(tmp_path, value):
    path = tmp_path / "s_parameters.csv"
    path.write_text(f"wavelength,S11_mag,S11_phase\n1.5,{value},0\n")
    with pytest.raises(ValueError):
        SParameterResult.from_csv(path)


@pytest.mark.parametrize("with_csv", [False, True])
@pytest.mark.parametrize("debug_contents", ['{"resolution": 20}', "broken json"])
def test_result_directory_discovers_debug_and_diagnostics(
    tmp_path, csv_path, with_csv, debug_contents
):
    if not with_csv:
        csv_path.unlink()
    (tmp_path / "meep_debug.json").write_text(debug_contents)
    files = {
        "geometry_xy": "meep_geometry_xy.png",
        "geometry_xz": "meep_geometry_xz.png",
        "geometry_yz": "meep_geometry_yz.png",
        "fields_xy": "meep_fields_xy.png",
        "animation": "meep_animation.mp4",
    }
    for filename in [*files.values(), "meep_frame_0001.png"]:
        (tmp_path / filename).touch()

    result = SParameterResult.from_directory(tmp_path)

    expected_debug = (
        json.loads(debug_contents) if debug_contents.startswith("{") else {}
    )
    assert result.debug_info == expected_debug
    assert result.diagnostic_images == {
        **{key: str(tmp_path / filename) for key, filename in files.items()},
        "animation_frames": str(tmp_path),
    }
    assert bool(result.wavelengths) == with_csv


def test_empty_directory_returns_empty_preview(tmp_path):
    assert SParameterResult.from_directory(tmp_path) == SParameterResult()


@pytest.fixture
def sample_result():
    return SParameterResult(
        wavelengths=[1.5, 1.6],
        s_params={"S11": [0j, 0.1j], "S21": [1 + 0j, -1j]},
    )


@pytest.mark.parametrize(("db", "expected"), [(True, [-100, -20]), (False, [0, 0.1])])
def test_matplotlib_plot_uses_correct_magnitude_and_case_insensitive_keys(
    sample_result, db, expected
):
    figure = sample_result.plot(db=db, keys=["s11"], color="red")
    try:
        axes = figure.axes[0]
        assert len(axes.lines) == 1
        line = axes.lines[0]
        np.testing.assert_allclose(line.get_xdata(), [1.5, 1.6])
        np.testing.assert_allclose(line.get_ydata(), expected)
        assert line.get_label() == "S11"
        assert line.get_color() == "red"
        assert axes.get_ylabel() == ("|S| (dB)" if db else "|S|")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("keys", [None, ["s21"]])
def test_plotly_plot_contains_magnitudes_and_phases(sample_result, keys):
    figure = sample_result.plot_plotly(keys=keys)
    expected_names = ["S11", "S11", "S21", "S21"] if keys is None else ["S21", "S21"]
    assert [trace.name for trace in figure.data] == expected_names
    np.testing.assert_allclose(figure.data[-2].y, [0, 0])
    np.testing.assert_allclose(figure.data[-1].y, [0, -90])
    if keys is None:
        np.testing.assert_allclose(figure.data[0].y, [-100, -20])
        np.testing.assert_allclose(figure.data[1].y, [0, 90])


@pytest.mark.parametrize("phase", [False, True])
def test_interactive_plot_hides_reflections_and_limits_visible_transmissions(phase):
    result = SParameterResult(
        wavelengths=[1.55],
        s_params={name: [0.1j] for name in ["S11", "S21", "S31", "S41", "S51", "S61"]},
    )
    figure = result.plot_interactive(phase=phase)

    assert [trace.name for trace in figure.data] == [
        "S21",
        "S31",
        "S41",
        "S51",
        "S61",
        "S11",
    ]
    assert [trace.visible for trace in figure.data] == [True] * 4 + ["legendonly"] * 2
    for trace in figure.data:
        np.testing.assert_allclose(trace.y, [90 if phase else -20])


def test_diagnostics_display_images_and_skip_videos_and_directories(monkeypatch):
    display = Mock()
    image = Mock()
    monkeypatch.setattr("IPython.display.display", display)
    monkeypatch.setattr("IPython.display.Image", image)
    result = SParameterResult(
        diagnostic_images={
            "geometry_xy": "geometry.png",
            "animation": "movie.mp4",
            "animation_frames": "frames",
        }
    )

    result.show_diagnostics()

    image.assert_called_once_with(filename="geometry.png")
    display.assert_called_once_with(image.return_value)


def test_animation_display_embeds_video(monkeypatch):
    display = Mock()
    video = Mock()
    monkeypatch.setattr("IPython.display.display", display)
    monkeypatch.setattr("IPython.display.Video", video)

    SParameterResult(diagnostic_images={"animation": "movie.mp4"}).show_animation()

    video.assert_called_once_with("movie.mp4", embed=True, mimetype="video/mp4")
    display.assert_called_once_with(video.return_value)


def test_missing_diagnostics_do_not_display_anything(monkeypatch):
    display = Mock()
    monkeypatch.setattr("IPython.display.display", display)
    result = SParameterResult()
    result.show_diagnostics()
    result.show_animation()
    display.assert_not_called()
