"""Tests for thermo-optic coupling (gsim.thermal.optical)."""

from __future__ import annotations

import warnings

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.femwell.adapter import epsilon_by_region, solve_modes

pytest.importorskip("skfem")

from gsim.thermal.mesh import build_cross_section_mesh
from gsim.thermal.optical import (
    element_centroids,
    sample_temperature,
    solve_thermo_optic_shift,
    thermo_optic_permittivity,
)

CORE_W = 0.5
CORE_T = 0.22
WAVELENGTH_UM = 1.55
DN_DT = {"core": 1.86e-4, "SiO2": 0.95e-5}  # Si and SiO2 at 1550 nm (1/K)


@pytest.fixture(scope="module")
def stack() -> LayerStack:
    """SOI stack: silicon substrate, BOX, oxide cladding and a Si core."""
    layers = {
        "core": Layer(
            name="core",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=CORE_T,
            thickness=CORE_T,
            material="silicon",
            layer_type="dielectric",
        ),
    }
    materials = {m: MATERIALS_DB[m].to_dict() for m in ("silicon", "SiO2")}
    dielectrics = [
        {"name": "substrate", "zmin": -12.0, "zmax": -2.0, "material": "silicon"},
        {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
        {"name": "clad", "zmin": 0.0, "zmax": 3.0, "material": "SiO2"},
    ]
    return LayerStack(
        pdk_name="test", layers=layers, materials=materials, dielectrics=dielectrics
    )


@pytest.fixture(scope="module")
def strip() -> gf.Component:
    """A 0.5 um Si strip running along x."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    c.add_polygon(
        [(-5, -CORE_W / 2), (5, -CORE_W / 2), (5, CORE_W / 2), (-5, CORE_W / 2)],
        layer=(1, 0),
    )
    return c


@pytest.fixture(scope="module")
def thermal_mesh(strip, stack, tmp_path_factory):
    """Thermal Window: 10 um wide, substrate cut to 3 um deep."""
    return build_cross_section_mesh(
        strip,
        stack,
        "x=0",
        window=(-5.0, 5.0),
        depth=3.0,
        output=tmp_path_factory.mktemp("thermal") / "thermal.msh",
    )


def test_one_kelvin_shifts_the_index_by_exactly_dn_dt_per_material():
    # Si core, and a slightly lossy oxide (exp(+i omega t): Im(n) < 0).
    n0 = np.array([3.476, 1.444 - 0.001j])
    dn_dt = np.array([1.86e-4, 0.95e-5])

    eps = thermo_optic_permittivity(n0, dn_dt, np.array([1.0, 1.0]))

    np.testing.assert_allclose(
        np.sqrt(eps), [3.476186, 1.4440095 - 0.001j], rtol=0, atol=1e-12
    )


def test_no_temperature_rise_leaves_the_permittivity_at_n0_squared():
    eps = thermo_optic_permittivity(np.array([3.476]), np.array([1.86e-4]), [0.0])
    np.testing.assert_allclose(eps, [12.082576], rtol=0, atol=1e-12)


def test_a_linear_rise_is_sampled_exactly_anywhere_on_the_thermal_mesh(thermal_mesh):
    x, z = thermal_mesh.mesh.p
    rise = 2.5 * x - 1.5 * z + 7.0
    rng = np.random.default_rng(0)
    (h0, h1), (v0, v1) = thermal_mesh.window, thermal_mesh.z_range
    points = np.vstack([rng.uniform(h0, h1, 200), rng.uniform(v0, v1, 200)])
    # Window corners and a node too: edges of the mesh are inside it.
    points = np.hstack([points, [[h0, h1], [v0, v1]], thermal_mesh.mesh.p[:, :1]])

    values = sample_temperature(thermal_mesh, rise, points)

    np.testing.assert_allclose(
        values, 2.5 * points[0] - 1.5 * points[1] + 7.0, rtol=0, atol=1e-10
    )


def test_a_point_outside_the_thermal_mesh_is_an_error_naming_it(thermal_mesh):
    rise = np.zeros(thermal_mesh.mesh.p.shape[1])
    points = np.array([[0.0, 0.0, 6.5], [0.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match=r"\(6\.5, 0(\.0)?\)"):
        sample_temperature(thermal_mesh, rise, points)


def test_element_centroids_fall_inside_their_own_region(thermal_mesh):
    centroids = element_centroids(thermal_mesh)
    core = centroids[:, thermal_mesh.region_mask("core")]
    assert centroids.shape == (2, thermal_mesh.mesh.t.shape[1])
    assert np.all(np.abs(core[0]) < CORE_W / 2)
    assert np.all((core[1] > 0.0) & (core[1] < CORE_T))


def _optical_inputs(mesh, stack):
    """Per-element unperturbed epsilon and dn/dT of an optical Window."""
    eps = epsilon_by_region(mesh.msh_path, stack, wavelength_um=WAVELENGTH_UM)
    eps0 = np.array([eps[r] for r in mesh.element_region])
    dn_dt = np.array([DN_DT[r] for r in mesh.element_region])
    return eps0, dn_dt


@pytest.fixture(scope="module")
def optical_mesh(strip, stack, tmp_path_factory):
    """Optical Window: 3 x 2.22 um around the strip, oxide all round."""
    return build_cross_section_mesh(
        strip,
        stack,
        "x=0",
        window=(-1.5, 1.5),
        z_range=(-1.0, 1.22),
        max_size={"core": 0.03},
        output=tmp_path_factory.mktemp("optical") / "optical.msh",
    )


@pytest.fixture(scope="module")
def heated_10k(optical_mesh, stack):
    pytest.importorskip("femwell")
    eps0, dn_dt = _optical_inputs(optical_mesh, stack)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a contained Mode warns about nothing
        return solve_thermo_optic_shift(
            optical_mesh,
            eps0,
            dn_dt,
            np.full(eps0.size, 10.0),
            wavelength_um=WAVELENGTH_UM,
        )


def test_resolve_and_first_order_agree_for_a_uniform_10k_rise(heated_10k):
    shift = heated_10k
    assert shift.delta_neff.real == pytest.approx(
        shift.delta_neff_first_order.real, rel=0.02
    )
    # The core carries the shift. A high-contrast strip's index moves a
    # little *more* than its core's own (here ~1.08 x 10 dn/dT_Si, mesh-
    # converged): the shift weights the core's field energy by n_core,
    # so it is not a fraction-weighted mean of the two coefficients.
    assert shift.delta_neff.real > 10 * DN_DT["SiO2"]
    assert shift.delta_neff.real == pytest.approx(10 * DN_DT["core"], rel=0.15)
    assert shift.delta_neff == shift.neff - shift.neff0


def test_the_unperturbed_mode_is_the_te0_strip_mode_and_contained(heated_10k):
    shift = heated_10k
    assert 2.3 < shift.neff0.real < 2.6
    assert shift.mode0.te_fraction > 0.5
    assert shift.mode.te_fraction > 0.5
    assert shift.boundary_ratio < 1e-3


def test_the_shift_is_linear_in_the_rise(optical_mesh, stack, heated_10k):
    eps0, dn_dt = _optical_inputs(optical_mesh, stack)
    half = solve_thermo_optic_shift(
        optical_mesh,
        eps0,
        dn_dt,
        np.full(eps0.size, 5.0),
        wavelength_um=WAVELENGTH_UM,
    )
    assert heated_10k.delta_neff.real / half.delta_neff.real == pytest.approx(
        2.0, rel=0.01
    )


def test_the_unperturbed_index_matches_the_femwell_adapter_on_the_same_mesh(
    optical_mesh, stack, heated_10k
):
    # The adapter loads the msh on its own and fills epsilon by Region
    # name, so agreement means both loaders see one element order. The
    # thermal path solves with order 1 elements; the adapter defaults to 2.
    eps = epsilon_by_region(optical_mesh.msh_path, stack, wavelength_um=WAVELENGTH_UM)
    modes = solve_modes(
        optical_mesh.msh_path, epsilon=eps, wavelength_um=WAVELENGTH_UM, order=1
    )
    assert heated_10k.neff0 == pytest.approx(complex(modes[0].n_eff), rel=1e-9)


def test_an_index_guess_seeds_the_solve_without_changing_the_shift(
    optical_mesh, stack, heated_10k
):
    eps0, dn_dt = _optical_inputs(optical_mesh, stack)
    seeded = solve_thermo_optic_shift(
        optical_mesh,
        eps0,
        dn_dt,
        np.full(eps0.size, 10.0),
        wavelength_um=WAVELENGTH_UM,
        n_guess=3.4777,  # the Si core's index
    )
    assert seeded.neff0 == pytest.approx(heated_10k.neff0, rel=1e-9)
    assert seeded.delta_neff == pytest.approx(heated_10k.delta_neff, rel=1e-6)


# TM0 of a 220 nm strip reaches further into the oxide than TE0 and is
# (correctly) flagged at the edge of this Window; selection is the point.
@pytest.mark.filterwarnings("ignore:The Mode's field at the optical Window edge")
def test_tm_selection_picks_the_tm0_mode(optical_mesh, stack, heated_10k):
    eps0, dn_dt = _optical_inputs(optical_mesh, stack)
    shift = solve_thermo_optic_shift(
        optical_mesh,
        eps0,
        dn_dt,
        np.full(eps0.size, 10.0),
        wavelength_um=WAVELENGTH_UM,
        polarization="TM",
    )
    assert shift.mode0.te_fraction < 0.5
    assert shift.mode.te_fraction < 0.5
    assert shift.neff0.real < heated_10k.neff0.real
    assert shift.delta_neff.real > 0.0


def test_a_window_too_small_for_the_mode_warns(strip, stack, tmp_path):
    pytest.importorskip("femwell")
    mesh = build_cross_section_mesh(
        strip,
        stack,
        "x=0",
        window=(-0.6, 0.6),
        z_range=(-0.3, 0.5),
        output=tmp_path / "small.msh",
    )
    eps0, dn_dt = _optical_inputs(mesh, stack)
    with pytest.warns(UserWarning, match="optical Window"):
        shift = solve_thermo_optic_shift(
            mesh, eps0, dn_dt, np.zeros(eps0.size), wavelength_um=WAVELENGTH_UM
        )
    assert shift.boundary_ratio > 1e-3


def test_per_element_inputs_of_the_wrong_length_are_rejected(optical_mesh, stack):
    eps0, dn_dt = _optical_inputs(optical_mesh, stack)
    with pytest.raises(ValueError, match="element"):
        solve_thermo_optic_shift(
            optical_mesh,
            eps0,
            dn_dt,
            np.zeros(eps0.size - 1),
            wavelength_um=WAVELENGTH_UM,
        )
