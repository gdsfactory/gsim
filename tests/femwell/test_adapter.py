"""Hermetic tests for the femwell adapter's epsilon mapping (no femwell)."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.common.modes import select_line_mode
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MaterialProperties
from gsim.femwell.adapter import (
    conductor_index_bound,
    elementwise_epsilon,
    epsilon_by_region,
)

WL_UM = 1.55
F_RF = 50e9


def _two_region_mesh(tmp_path):
    """Unit square split into two triangle groups 'core' and 'clad'."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = [
        ("triangle", np.array([[0, 1, 2]])),
        ("triangle", np.array([[0, 2, 3]])),
    ]
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            "gmsh:physical": [np.array([1]), np.array([2])],
            "gmsh:geometrical": [np.array([1]), np.array([2])],
        },
        field_data={
            "core": np.array([1, 2]),
            "clad": np.array([2, 2]),
        },
    )
    path = tmp_path / "two_region.msh"
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
    return path


def _stack():
    stack = LayerStack(pdk_name="test", units="um")
    stack.layers["core"] = Layer(
        name="core",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.22,
        thickness=0.22,
        material="si",
        layer_type="dielectric",
        mesh_resolution="fine",
    )
    return stack


class TestEpsilonByRegion:
    def test_unresolvable_region_raises(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        stack = _stack()
        with pytest.raises(ValueError, match="clad"):
            epsilon_by_region(path, stack, wavelength_um=WL_UM)

    def test_resolves_with_stack_materials(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        stack = _stack()
        stack.materials["clad"] = {"permittivity": 2.085}
        eps = epsilon_by_region(path, stack, wavelength_um=WL_UM)
        # Sellmeier silicon near 1.55 um: n ~ 3.476.
        assert eps["core"].real == pytest.approx(3.476**2, rel=5e-3)
        assert eps["clad"].real == pytest.approx(2.085, rel=1e-9)

    def test_rf_conductive_region_is_lossy(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        stack = _stack()
        stack.materials["clad"] = {"permittivity": 11.9, "conductivity": 1e3}
        stack.materials["si"] = {"permittivity": 11.9}
        eps = epsilon_by_region(path, stack, frequency_hz=F_RF)
        # exp(+i omega t): lossy medium has Im(eps) < 0.
        assert eps["clad"].imag < 0.0
        assert eps["clad"].real == pytest.approx(11.9)
        assert eps["core"].imag == pytest.approx(0.0)

    def test_requires_exactly_one_target(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        with pytest.raises(ValueError, match="exactly one"):
            epsilon_by_region(path, _stack())
        with pytest.raises(ValueError, match="exactly one"):
            epsilon_by_region(path, _stack(), wavelength_um=WL_UM, frequency_hz=F_RF)


class TestElementwiseEpsilon:
    def test_linear_field_sampled_at_centroids(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        # eps(x, y) = 10 + 2x on a dense sample grid.
        xs, ys = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
        eps_samples = 10.0 + 2.0 * xs
        result = elementwise_epsilon(path, xs.ravel(), ys.ravel(), eps_samples.ravel())
        mesh = meshio.read(str(path))
        tris = np.vstack([b.data for b in mesh.cells if b.type == "triangle"])
        centroids = mesh.points[tris][:, :, :2].mean(axis=1)
        np.testing.assert_allclose(
            result.real, 10.0 + 2.0 * centroids[:, 0], rtol=1e-12
        )
        np.testing.assert_allclose(result.imag, 0.0, atol=1e-12)

    def test_outside_hull_uses_nearest_or_fill(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        # Samples only in a corner far from most centroids.
        x = np.array([2.0, 2.1, 2.0])
        y = np.array([2.0, 2.0, 2.1])
        eps = np.array([5.0, 5.0, 5.0])
        nearest = elementwise_epsilon(path, x, y, eps)
        np.testing.assert_allclose(nearest.real, 5.0)
        filled = elementwise_epsilon(path, x, y, eps, fill=1.0 + 0j)
        np.testing.assert_allclose(filled.real, 1.0)

    def test_complex_values_preserved(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        xs, ys = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
        eps = np.full(xs.size, 12.0 - 0.5j)
        result = elementwise_epsilon(path, xs.ravel(), ys.ravel(), eps)
        np.testing.assert_allclose(result, 12.0 - 0.5j)

    def test_mismatched_lengths_raise(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        with pytest.raises(ValueError, match="same length"):
            elementwise_epsilon(path, [0.0, 1.0], [0.0, 1.0], [1.0])


class TestOverrides:
    def test_user_override_wins_over_database(self, tmp_path):
        path = _two_region_mesh(tmp_path)
        stack = _stack()
        stack.materials["clad"] = {"permittivity": 2.085}
        eps = epsilon_by_region(
            path,
            stack,
            wavelength_um=WL_UM,
            overrides={"si": MaterialProperties(permittivity=9.0)},
        )
        assert eps["core"].real == pytest.approx(9.0)


class TestConductorIndexBound:
    """The |n_eff| above which a Mode lives inside a conductor (#498)."""

    def test_a_tenth_of_the_metal_index(self):
        eps_metal = 1.0 - 6.8e8j  # aluminium at 1 GHz
        bound = conductor_index_bound({"metal": eps_metal, "oxide": 3.9 + 0j})
        assert bound == pytest.approx(0.1 * abs(np.sqrt(eps_metal)))

    def test_the_least_conductive_strong_conductor_sets_it(self):
        bound = conductor_index_bound(
            {"al": 1.0 - 6.8e8j, "w": 1.0 - 3.3e8j, "oxide": 3.9 + 0j}
        )
        assert bound == pytest.approx(0.1 * abs(np.sqrt(1.0 - 3.3e8j)))

    def test_a_weak_conductor_alone_gives_no_bound(self):
        """11.7-12j conducts, but 0.1|sqrt(eps)| = 0.41 is below the light line."""
        with pytest.warns(UserWarning, match="returns no bound"):
            assert conductor_index_bound({"si": 11.7 - 12j, "oxide": 3.9 + 0j}) is None

    @pytest.mark.parametrize("eps_si", [11.7 - 12j, 11.7 - 1000j])
    def test_a_weak_conductor_next_to_metal_gives_no_bound(self, eps_si):
        """The aluminium bound alone would let a doped-silicon Mode through.

        With aluminium, the bound would be about 2608. A Mode confined to
        silicon at 11.7-1000j, near sqrt(eps_si) = 22.49-22.23j, is under
        it and slower than the 5.13-4.68j line Mode, so selection would
        pick it. Returning no bound leaves that choice to the caller.
        """
        eps = {"al": 1.0 - 6.8e8j, "si": eps_si, "oxide": 3.9 + 0j}
        aluminium_bound = 0.1 * abs(np.sqrt(1.0 - 6.8e8j))
        line, si_mode = 5.13 - 4.68j, 22.49 - 22.23j
        assert abs(si_mode) < aluminium_bound
        assert select_line_mode([line, si_mode], max_index=aluminium_bound) == si_mode
        with pytest.warns(UserWarning, match="returns no bound"):
            assert conductor_index_bound(eps) is None

    def test_the_margin_is_against_the_largest_dielectric_index(self):
        """A conductor clears oxide by 10x but not silicon is skipped."""
        eps_cond = 1.0 - 9e4j  # 0.1|sqrt(eps)| = 30
        assert conductor_index_bound({"c": eps_cond, "ox": 3.9 + 0j}) == (
            pytest.approx(30.0, rel=1e-4)
        )
        with pytest.warns(UserWarning, match="returns no bound"):
            assert (
                conductor_index_bound({"c": eps_cond, "si": 11.7 + 0j, "ox": 3.9 + 0j})
                is None
            )

    def test_per_element_values(self):
        eps = np.array([3.9, 3.9, 1.0 - 6.8e8j, 3.9], dtype=complex)
        assert conductor_index_bound(eps) == pytest.approx(
            0.1 * abs(np.sqrt(1.0 - 6.8e8j))
        )

    def test_a_lossy_dielectric_is_not_a_conductor(self):
        """Loss below the real permittivity leaves the Mode free to sit there."""
        assert conductor_index_bound({"si": 11.7 - 5.0j, "oxide": 3.9 + 0j}) is None

    def test_the_fraction_is_a_parameter(self):
        eps = {"metal": 1.0 - 6.8e8j}
        default = conductor_index_bound(eps)
        assert default is not None
        assert conductor_index_bound(eps, fraction=0.5) == pytest.approx(5 * default)

    def test_the_fraction_and_margin_must_be_positive(self):
        with pytest.raises(ValueError, match="fraction"):
            conductor_index_bound({"metal": 1.0 - 6.8e8j}, fraction=0.0)
        with pytest.raises(ValueError, match="margin"):
            conductor_index_bound({"metal": 1.0 - 6.8e8j}, margin=0.0)
