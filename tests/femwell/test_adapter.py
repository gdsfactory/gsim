"""Hermetic tests for the femwell adapter's epsilon mapping (no femwell)."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MaterialProperties
from gsim.femwell.adapter import (
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
