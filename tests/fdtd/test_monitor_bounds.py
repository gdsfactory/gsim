"""Submitted plane monitors respect the physical domain in every orientation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from gsim import fdtd


def _crop_warnings(caught):
    return [
        warning
        for warning in caught
        if "cropped to the physical (non-PML) domain" in str(warning.message)
    ]


def _simulation(pdk, domain_kind="automatic"):
    simulation = fdtd.Simulation(pdk=pdk)
    simulation.materials(background="SiO2")
    simulation.geometry("straight")
    simulation.source(port="o1", wavelength_span_um=0, num_wavelengths=1)
    if domain_kind == "explicit":
        simulation.domain(x_bounds=(0, 2), y_bounds=(-1.25, 1.25), z_bounds=(-1, 1))
    elif domain_kind == "mixed":
        simulation.domain(y_bounds=(-1.25, 1.25))
    return simulation


@pytest.mark.parametrize("normal", ["+x", "-x", "+y", "-y", "+z", "-z"])
@pytest.mark.parametrize("domain_kind", ["automatic", "explicit", "mixed"])
def test_crops_submitted_plane_and_warns_once(
    tmp_path, fdtd_pdk_module, normal, domain_kind
):
    simulation = _simulation(fdtd_pdk_module, domain_kind)
    size = [3.0, 4.0, 4.0]
    normal_axis = "xyz".index(normal[-1])
    size[normal_axis] = 0
    monitor = simulation.monitors.add_plane(
        "image",
        center_um=(1, 0, 0),
        size_um=size,
        normal=normal,
        flux=False,
        heatmap=fdtd.Heatmap(wavelengths_um=[1.55]),
    )
    requested = monitor.model_dump()
    with pytest.warns(UserWarning, match="Monitor 'image' cropped") as caught:
        artifacts = simulation.write(tmp_path)
    cropped = _crop_warnings(caught)
    assert len(cropped) == 1
    assert Path(cropped[0].filename) == Path(__file__)
    document = json.loads(artifacts.config_path.read_text())
    assert document["grid"]["non_pml_bounds"] == {
        "min": [0, -1250, -1000],
        "max": [2000, 1250, 1000],
    }
    expected_min = [0, -1250, -1000]
    expected_max = [2000, 1250, 1000]
    expected_min[normal_axis] = expected_max[normal_axis] = [1000, 0, 0][normal_axis]
    submitted = document["monitors"][0]
    assert submitted["region_min"] == expected_min
    assert submitted["region_max"] == expected_max
    assert submitted["normal"] == normal
    assert submitted["heatmap"]["wavelengths"] == [1550]
    assert monitor.model_dump() == requested
    for axis in "xyz".replace(normal[-1], ""):
        assert f"{axis} extent" in str(cropped[0].message)
    assert "um" in str(cropped[0].message)
    assert "measurement aperture" not in str(cropped[0].message)


@pytest.mark.parametrize("domain_kind", ["automatic", "explicit", "mixed"])
@pytest.mark.parametrize(
    ("normal", "center", "size", "error"),
    [
        ("+x", (-1, 0, 0), (0, 1, 1), "x-position"),
        ("-x", (3, 0, 0), (0, 1, 1), "x-position"),
        ("+y", (1, -2, 0), (1, 0, 1), "y-position"),
        ("-y", (1, 2, 0), (1, 0, 1), "y-position"),
        ("+z", (1, 0, -2), (1, 1, 0), "z-position"),
        ("-z", (1, 0, 2), (1, 1, 0), "z-position"),
        ("+x", (1, 0, -2), (0, 1, 1), "no area"),
        ("+y", (4, 0, 0), (1, 0, 1), "no area"),
        ("+z", (1, 3, 0), (1, 1, 0), "no area"),
        ("+z", (3, 0, 0), (2, 1, 0), "no area"),
        ("+z", (3, 0, 0), (0, 1, 0), "no area"),
    ],
)
def test_unusable_monitor_fails_before_meshing(
    tmp_path, fdtd_pdk_module, domain_kind, normal, center, size, error
):
    simulation = _simulation(fdtd_pdk_module, domain_kind)
    simulation.monitors.add_plane(
        "outside", center_um=center, size_um=size, normal=normal
    )
    with pytest.raises(fdtd.FDTDConfigError, match=f"Monitor 'outside'.*{error}"):
        simulation.write(tmp_path)
    assert not (tmp_path / "mesh.msh").exists()
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("normal", ["+x", "-x", "+y", "-y", "+z", "-z"])
def test_boundary_plane_is_unchanged_without_warning(tmp_path, fdtd_pdk_module, normal):
    simulation = _simulation(fdtd_pdk_module)
    normal_axis = "xyz".index(normal[-1])
    center = [1, 0, 0]
    size = [2, 2.5, 2]
    center[normal_axis] = ([0, -1.25, -1] if normal[0] == "-" else [2, 1.25, 1])[
        normal_axis
    ]
    size[normal_axis] = 0
    simulation.monitors.add_plane(
        "boundary", center_um=center, size_um=size, normal=normal
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = simulation.write(tmp_path)
    assert not _crop_warnings(caught)
    submitted = json.loads(artifacts.config_path.read_text())["monitors"][0]
    for axis in range(3):
        assert submitted["region_min"][axis] == center[axis] * 1000 - size[axis] * 500
        assert submitted["region_max"][axis] == center[axis] * 1000 + size[axis] * 500


def test_crop_warning_describes_aperture_change_and_preserves_fiber(
    tmp_path, fdtd_pdk_module
):
    simulation = _simulation(fdtd_pdk_module)
    simulation.monitors.add_plane(
        "fiber",
        center_um=(1, 0, 0.8),
        size_um=(4, 4, 0),
        normal="-z",
        wavelengths_um=[1.55],
        fiber_mode=fdtd.FiberMode(
            propagation_direction=(0, 0, -1),
            e_polarization=(0, 1, 0),
            focal_point_um=(1, 0, 0.11),
            waist_radius_um=0.5,
        ),
    )
    with pytest.warns(UserWarning, match="Cropping changes the measurement aperture"):
        artifacts = simulation.write(tmp_path)
    monitor = json.loads(artifacts.config_path.read_text())["monitors"][0]
    assert monitor["region_min"] == [0, -1250, 800]
    assert monitor["region_max"] == [2000, 1250, 800]
    assert monitor["flux"] is True
    assert monitor["wavelengths"] == [1550]
    assert monitor["fiber_mode"]["focal_point"] == [1000, 0, 110]
    assert monitor["fiber_mode"]["waist_radius"] == 500


def test_layout_only_shapes_crop_the_field_image(tmp_path, fdtd_pdk_module):
    simulation = _simulation(fdtd_pdk_module)
    component = simulation.resolved.component.copy()
    component.add_polygon([(10, 10), (12, 10), (12, 12), (10, 12)], layer=(999, 0))
    simulation.geometry(component)
    simulation.add_monitors(xy_name="field_xy")
    with pytest.warns(UserWarning, match="Monitor 'field_xy' cropped"):
        artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text())
    field = next(
        monitor for monitor in document["monitors"] if monitor["name"] == "field_xy"
    )
    assert field["region_min"] == [0, -250, 110]
    assert field["region_max"] == [2000, 1250, 110]


def test_oversized_port_images_are_cropped_in_yz_and_xz(tmp_path, fdtd_pdk_module):
    for rotated in (False, True):
        simulation = _simulation(fdtd_pdk_module)
        if rotated:
            component = simulation.resolved.component.copy()
            component.rotate(90)
            simulation.geometry(component)
        simulation.add_monitors(xy=False, margin_um=2)
        with pytest.warns(UserWarning, match="Monitor 'port_o[12]' cropped") as caught:
            artifacts = simulation.write(tmp_path / str(rotated))
        assert len(_crop_warnings(caught)) == 2
        document = json.loads(artifacts.config_path.read_text())
        for monitor in document["monitors"]:
            assert monitor["region_min"][2] == -1000
            assert monitor["region_max"][2] == 1000
            for axis in range(3):
                assert (
                    monitor["region_min"][axis]
                    >= document["grid"]["non_pml_bounds"]["min"][axis]
                )
                assert (
                    monitor["region_max"][axis]
                    <= document["grid"]["non_pml_bounds"]["max"][axis]
                )


def test_boundary_roundoff_is_snapped_without_warning(tmp_path, fdtd_pdk_module):
    simulation = _simulation(fdtd_pdk_module)
    simulation.monitors.add_plane(
        "roundoff", center_um=(-1e-14, 0, 0), size_um=(0, 1, 1), normal="+x"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = simulation.write(tmp_path)
    assert not _crop_warnings(caught)
    submitted = json.loads(artifacts.config_path.read_text())["monitors"][0]
    assert submitted["region_min"][0] == submitted["region_max"][0] == 0


def test_larger_domain_restores_original_requested_aperture(tmp_path, fdtd_pdk_module):
    simulation = _simulation(fdtd_pdk_module)
    simulation.monitors.add_plane(
        "image", center_um=(1, 0, 0), size_um=(0, 4, 4), normal="+x"
    )
    with pytest.warns(UserWarning, match="cropped"):
        simulation.write(tmp_path / "small")
    simulation.domain(y_bounds=(-3, 3), z_bounds=(-3, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = simulation.write(tmp_path / "large")
    assert not _crop_warnings(caught)
    monitor = json.loads(artifacts.config_path.read_text())["monitors"][0]
    assert monitor["region_min"] == [1000, -2000, -2000]
    assert monitor["region_max"] == [1000, 2000, 2000]


@pytest.mark.parametrize("normal", ["+x", "-x", "+y", "-y", "+z", "-z"])
@pytest.mark.parametrize("point_probe", [False, True])
def test_contained_zero_width_probes_keep_their_requested_extent(
    tmp_path, fdtd_pdk_module, normal, point_probe
):
    simulation = _simulation(fdtd_pdk_module)
    normal_axis = "xyz".index(normal[-1])
    size = [0, 0, 0] if point_probe else [2, 2, 2]
    size[normal_axis] = 0
    size[(normal_axis + 1) % 3] = 0
    center = (1, 0, 0)
    simulation.monitors.add_plane(
        "probe",
        center_um=center,
        size_um=size,
        normal=normal,
        flux=False,
        heatmap=fdtd.Heatmap(wavelengths_um=[1.55]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = simulation.write(tmp_path)
    assert not _crop_warnings(caught)
    monitor = json.loads(artifacts.config_path.read_text())["monitors"][0]
    for axis in range(3):
        assert monitor["region_min"][axis] == center[axis] * 1000 - size[axis] * 500
        assert monitor["region_max"][axis] == center[axis] * 1000 + size[axis] * 500
