"""Check analytical index profiles and their field/geometry overlays."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsim.meep import mode_solver
from gsim.meep.models.results import ModeResult


@pytest.fixture(params=["xz", "yz", None])
def index_result(index_context, request):
    stack, component = index_context
    plane = request.param
    component.add_port(
        "cut",
        center=(0, 0),
        orientation=0 if plane == "yz" else 90,
        width=0.5,
        layer=(1, 0),
    )
    grid = (
        {}
        if plane is None
        else {"y_grid" if plane == "yz" else "x_grid": np.array([-1.5, 0, 1.5])}
    )
    fields = np.arange(3 if plane is None else 9, dtype=float) + 1j
    return ModeResult(
        n_eff=1.8,
        wavelength=1.55,
        frequency=1 / 1.55,
        stack=stack,
        component=component,
        port_or_position="cut",
        cross_section_plane=plane,
        z_grid=np.array([-0.5, 0.5, 1.5]),
        fields={"Ey": fields if plane is None else fields.reshape(3, 3)},
        **grid,
    )


@pytest.mark.parametrize("existing_axes", [False, True])
def test_index_plot_contains_analytical_values(index_result, existing_axes):
    axes = plt.subplots()[1] if existing_axes else None
    figure, axes = index_result.plot_index(ax=axes, show=False)
    try:
        if index_result.cross_section_plane is None:
            np.testing.assert_allclose(axes.lines[0].get_xdata(), [-0.5, 0.5, 1.5])
            np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.5, 2, 1])
        else:
            np.testing.assert_allclose(
                axes.collections[0].get_array().reshape(3, 3),
                [
                    [1.5, 1.5, 1.5],
                    [1, 2, 1],
                    [1, 1, 1],
                ],
            )
            assert len(figure.axes) == 2
    finally:
        plt.close(figure)


@pytest.mark.parametrize("norm", ["real", "imag"])
@pytest.mark.parametrize("sidewall_angle", [0, 30])
def test_field_and_geometry_overlays_preserve_data(index_result, norm, sidewall_angle):
    index_result.stack.layers["core"].sidewall_angle = sidewall_angle
    figure, axes = index_result.plot_mode(
        index=True,
        geometry=True,
        norm=norm,
        show=False,
        geom_kwargs={"color": "red"},
    )
    try:
        field = index_result.fields["Ey"]
        expected = field.real if norm == "real" else field.imag
        if index_result.cross_section_plane is None:
            np.testing.assert_allclose(axes.lines[0].get_ydata(), expected)
            assert [list(line.get_ydata()) for line in axes.lines[1:]] == [
                [-1, -1],
                [0, 0],
            ]
            np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [1.5, 2, 1])
        else:
            np.testing.assert_allclose(
                axes.collections[0].get_array().reshape(3, 3), expected
            )
            assert len(axes.collections) == 2
            core = next(patch for patch in axes.patches if patch.get_label() == "core")
            np.testing.assert_allclose(core.get_edgecolor(), [1, 0, 0, 1])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("orientation", [0, 90])
def test_taper_and_off_axis_slice_match_physical_core(index_context, orientation):
    stack, component = index_context
    stack.layers["core"].sidewall_angle = 45
    component.add_port(
        "cut", center=(0, 0), orientation=orientation, width=0.5, layer=(1, 0)
    )
    component.add_port(
        "outside", center=(3, 3), orientation=orientation, width=0.5, layer=(1, 0)
    )
    grid = {"y_grid" if orientation == 0 else "x_grid": np.array([-0.6, 0, 0.6])}
    for port, expected in [
        ("cut", [[2, 2, 2], [1, 2, 1]]),
        ("outside", [[1, 1, 1], [1, 1, 1]]),
    ]:
        profile = mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.25, 0.75]),
            component=component,
            port=port,
            **grid,
        )
        np.testing.assert_allclose(profile, expected)


def test_slab_interfaces_and_background_index(index_context):
    profile = mode_solver.refractive_index_profile(
        index_context[0],
        1.55,
        z_grid=np.array([-2.0, -1, 0, 0.5, 1, 2]),
        background_material="clad",
    )
    np.testing.assert_allclose(profile, [1.5, 1.5, 2, 2, 1.5, 1.5])


@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize(
    "invalid", ["missing-port", "no-port", "wrong-axis", "both-axes"]
)
def test_profile_rejects_incomplete_or_ambiguous_geometry(
    index_context, orientation, invalid
):
    stack, component = index_context
    component.add_port(
        "cut", center=(0, 0), orientation=orientation, width=0.5, layer=(1, 0)
    )
    kwargs = {"y_grid" if orientation == 0 else "x_grid": np.array([0.0])}
    port = (
        "missing"
        if invalid == "missing-port"
        else None
        if invalid == "no-port"
        else "cut"
    )
    if invalid == "wrong-axis":
        kwargs = {"x_grid" if orientation == 0 else "y_grid": np.array([0.0])}
    elif invalid == "both-axes":
        kwargs = {"x_grid": np.array([0.0]), "y_grid": np.array([0.0])}
    with pytest.raises(ValueError):
        mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.5]),
            component=component,
            port=port,
            **kwargs,
        )
