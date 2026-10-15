"""Check cross-section index profiles against simple analytical geometries."""

from __future__ import annotations

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.meep import mode_solver
from gsim.meep.models.config import MaterialData
from gsim.meep.models.results import ModeResult


@pytest.fixture
def index_context(monkeypatch):
    # Fix material indices so expectations depend only on geometry, not on a
    # material database or dispersion fit. GDS slicing and plotting remain real.
    materials = {
        "core": MaterialData(epsilon_diag=[4, 4, 4]),
        "clad": MaterialData(epsilon_diag=[2.25, 2.25, 2.25]),
    }
    monkeypatch.setattr(
        mode_solver, "_resolve_materials_cached", lambda *_args, **_kwargs: materials
    )
    stack = LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=layer,
                zmin=zmin,
                zmax=zmax,
                thickness=zmax - zmin,
                material=name,
                layer_type="dielectric",
            )
            for name, layer, zmin, zmax in [
                ("clad", (2, 0), -1, 0),
                ("core", (1, 0), 0, 1),
            ]
        },
        dielectrics=[{"zmin": -1, "zmax": 0, "material": "clad"}],
    )
    component = gf.Component()
    component.add_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], layer=(1, 0))
    return stack, component


def _add_cut_port(component, orientation, center=(0, 0)):
    component.add_port(
        name="cut", center=center, orientation=orientation, width=0.5, layer=(1, 0)
    )


@pytest.mark.parametrize("orientation", [0, 90, 180, 270])
def test_cross_section_profile_matches_patterned_core_and_full_span_cladding(
    index_context, orientation
):
    stack, component = index_context
    _add_cut_port(component, orientation)
    grid = np.array([-1.5, -0.75, 0, 0.75, 1.5])
    kwargs = {"y_grid" if orientation % 180 == 0 else "x_grid": grid}

    profile = mode_solver.refractive_index_profile(
        stack,
        1.55,
        z_grid=np.array([-1.5, -0.5, 0.25, 0.75, 1.5]),
        component=component,
        port="cut",
        **kwargs,
    )

    np.testing.assert_allclose(
        profile,
        [
            [1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ],
    )


@pytest.mark.parametrize("orientation", [0, 90])
def test_sidewall_taper_narrows_core_with_height(index_context, orientation):
    stack, component = index_context
    stack.layers["core"].sidewall_angle = 45
    _add_cut_port(component, orientation)
    kwargs = {"y_grid" if orientation == 0 else "x_grid": np.array([-0.6, 0, 0.6])}

    profile = mode_solver.refractive_index_profile(
        stack,
        1.55,
        z_grid=np.array([0.25, 0.75]),
        component=component,
        port="cut",
        **kwargs,
    )

    np.testing.assert_allclose(profile, [[2, 2, 2], [1, 2, 1]])


@pytest.mark.parametrize("orientation", [0, 90])
def test_slice_outside_drawn_core_does_not_expand_it_to_full_span(
    index_context, orientation
):
    stack, component = index_context
    _add_cut_port(component, orientation, center=(3, 3))
    kwargs = {"y_grid" if orientation == 0 else "x_grid": np.array([-0.5, 0.5])}

    profile = mode_solver.refractive_index_profile(
        stack,
        1.55,
        z_grid=np.array([-0.5, 0.5]),
        component=component,
        port="cut",
        **kwargs,
    )

    np.testing.assert_allclose(profile, [[1.5, 1.5], [1, 1]])


def test_slab_profile_respects_layer_interfaces_and_background(index_context):
    stack, _ = index_context
    profile = mode_solver.refractive_index_profile(
        stack,
        1.55,
        z_grid=np.array([-2.0, -1, 0, 0.5, 1, 2]),
        background_material="clad",
    )
    np.testing.assert_allclose(profile, [1.5, 1.5, 2, 2, 1.5, 1.5])


@pytest.mark.parametrize(
    ("orientation", "grid_name", "message"),
    [
        (0, "x_grid", "y_grid is required"),
        (90, "y_grid", "x_grid is required"),
    ],
)
def test_cross_section_requires_matching_horizontal_axis(
    index_context, orientation, grid_name, message
):
    stack, component = index_context
    _add_cut_port(component, orientation)
    with pytest.raises(ValueError, match=message):
        mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.5]),
            component=component,
            port="cut",
            **{grid_name: np.array([0.0])},
        )


@pytest.mark.parametrize("orientation", [0, 90])
def test_cross_section_rejects_ambiguous_horizontal_axes(index_context, orientation):
    stack, component = index_context
    _add_cut_port(component, orientation)
    with pytest.raises(ValueError, match="not valid"):
        mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.5]),
            component=component,
            port="cut",
            x_grid=np.array([0.0]),
            y_grid=np.array([0.0]),
        )


@pytest.mark.parametrize("grid_name", ["x_grid", "y_grid"])
def test_cross_section_requires_a_port(index_context, grid_name):
    stack, component = index_context
    with pytest.raises(ValueError, match="port is required"):
        mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.5]),
            component=component,
            **{grid_name: np.array([0.0])},
        )


def test_cross_section_reports_missing_port_and_available_names(index_context):
    stack, component = index_context
    _add_cut_port(component, 0)
    with pytest.raises(ValueError, match="Port 'missing' not found") as error:
        mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.5]),
            y_grid=np.array([0.0]),
            component=component,
            port="missing",
        )
    assert "cut" in str(error.value)


@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize("existing_axes", [False, True])
def test_mode_index_plot_contains_correct_grid_values(
    index_context, orientation, existing_axes
):
    stack, component = index_context
    _add_cut_port(component, orientation)
    grid_kwargs = {"y_grid" if orientation == 0 else "x_grid": np.array([-1.5, 0, 1.5])}
    result = ModeResult(
        n_eff=1.8,
        wavelength=1.55,
        frequency=1 / 1.55,
        z_grid=np.array([-0.5, 0.5, 1.5]),
        stack=stack,
        component=component,
        port_or_position="cut",
        cross_section_plane="yz" if orientation == 0 else "xz",
        **grid_kwargs,
    )
    axes = plt.subplots()[1] if existing_axes else None

    figure, axes = result.plot_index(ax=axes, show=False)
    try:
        np.testing.assert_allclose(
            axes.collections[0].get_array().reshape(3, 3),
            [
                [1.5, 1.5, 1.5],
                [1, 2, 1],
                [1, 1, 1],
            ],
        )
        assert axes.get_xlabel() == ("y (µm)" if orientation == 0 else "x (µm)")
        assert axes.get_ylabel() == "z (µm)"
        assert len(figure.axes) == 2  # Main plot plus colorbar.
    finally:
        plt.close(figure)


@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize("sidewall_angle", [0, 30])
def test_mode_field_overlay_preserves_field_values_and_geometry(
    index_context, orientation, sidewall_angle
):
    stack, component = index_context
    stack.layers["core"].sidewall_angle = sidewall_angle
    _add_cut_port(component, orientation)
    grid_kwargs = {"y_grid" if orientation == 0 else "x_grid": np.array([-1.5, 0, 1.5])}
    fields = np.arange(9).reshape(3, 3) + 1j
    result = ModeResult(
        n_eff=1.8,
        wavelength=1.55,
        frequency=1 / 1.55,
        fields={"Ey": fields},
        z_grid=np.array([-0.5, 0.5, 1.5]),
        stack=stack,
        component=component,
        port_or_position="cut",
        cross_section_plane="yz" if orientation == 0 else "xz",
        **grid_kwargs,
    )

    figure, axes = result.plot_mode(
        index=True,
        geometry=True,
        norm="real",
        show=False,
        geom_kwargs={"color": "red"},
    )
    try:
        np.testing.assert_allclose(
            axes.collections[0].get_array().reshape(3, 3), fields.real
        )
        assert len(axes.collections) == 2  # Field and index underlay.
        core_patch = next(
            patch for patch in axes.patches if patch.get_label() == "core"
        )
        np.testing.assert_allclose(core_patch.get_edgecolor(), [1, 0, 0, 1])
        assert axes.get_xlim()[0] <= -1
        assert axes.get_xlim()[1] >= 1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("existing_axes", [False, True])
def test_slab_index_plot_contains_exact_layer_values(index_context, existing_axes):
    stack, _ = index_context
    z_grid = np.array([-0.5, 0.5, 1.5])
    result = ModeResult(
        n_eff=1.8,
        wavelength=1.55,
        frequency=1 / 1.55,
        z_grid=z_grid,
        stack=stack,
    )
    axes = plt.subplots()[1] if existing_axes else None

    figure, axes = result.plot_index(ax=axes, show=False)
    try:
        np.testing.assert_allclose(axes.lines[0].get_xdata(), z_grid)
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.5, 2, 1])
        assert axes.get_ylabel() == "n"
    finally:
        plt.close(figure)


def test_slab_field_plot_adds_index_and_unique_dielectric_interfaces(index_context):
    stack, _ = index_context
    fields = np.array([1j, 2j, 3j])
    result = ModeResult(
        n_eff=1.8,
        wavelength=1.55,
        frequency=1 / 1.55,
        fields={"Ey": fields},
        z_grid=np.array([-0.5, 0.5, 1.5]),
        stack=stack,
    )

    figure, axes = result.plot_mode(index=True, geometry=True, norm="imag", show=False)
    try:
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [1, 2, 3])
        assert [list(line.get_ydata()) for line in axes.lines[1:]] == [[-1, -1], [0, 0]]
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [1.5, 2, 1])
    finally:
        plt.close(figure)
