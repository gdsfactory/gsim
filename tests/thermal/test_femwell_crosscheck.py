"""Cross-check the heat solver against femwell's own transient solver.

femwell ships ``solve_thermal_transient`` (Crank-Nicolson, current-density
source, fixed-temperature boundaries). On the same mesh, with the same
conductivity, heat capacity and Joule source, gsim's ``solve_transient``
with ``theta=0.5`` must give the same temperature history.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

pytest.importorskip("femwell")
skfem = pytest.importorskip("skfem")
shapely_geometry = pytest.importorskip("shapely.geometry")

from gsim.thermal.solver import solve_transient, time_constant_1e  # noqa: E402

LineString = shapely_geometry.LineString
Polygon = shapely_geometry.Polygon


def _rect(x0: float, y0: float, x1: float, y1: float):
    """Axis-aligned shapely rectangle."""
    return Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])


@pytest.fixture(scope="module")
def femwell_heater():
    """femwell's TiN heater example geometry, meshed by femwell (um)."""
    from femwell.mesh import mesh_from_OrderedDict
    from skfem.io import from_meshio

    w_sim, h_clad, h_box = 16.0, 2.8, 1.0
    w_core, h_core = 0.5, 0.22
    offset, h_heater, w_heater = 2.2, 0.14, 2.0
    polygons = OrderedDict(
        bottom=LineString([(-w_sim / 2, -h_box), (w_sim / 2, -h_box)]),
        core=_rect(-w_core / 2, 0, w_core / 2, h_core),
        heater=_rect(-w_heater / 2, offset, w_heater / 2, offset + h_heater),
        clad=_rect(-w_sim / 2, 0, w_sim / 2, h_clad),
        box=_rect(-w_sim / 2, -h_box, w_sim / 2, 0),
    )
    resolutions = {
        "core": {"resolution": 0.1, "distance": 1},
        "heater": {"resolution": 0.1, "distance": 1},
        "clad": {"resolution": 1, "distance": 1},
        "box": {"resolution": 1, "distance": 1},
    }
    mesh = from_meshio(
        mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.5)
    )
    basis0 = skfem.Basis(mesh, skfem.ElementTriP0(), intorder=4)
    k_si = {"core": 148.0, "box": 1.38, "clad": 1.38, "heater": 28.0}
    rho_cp = {
        "core": 711.0 * 2330.0,
        "box": 709.0 * 2203.0,
        "clad": 709.0 * 2203.0,
        "heater": 598.0 * 5240.0,
    }
    k = basis0.zeros()
    capacity = basis0.zeros()
    for name, value in k_si.items():
        k[basis0.get_dofs(elements=name)] = value
        capacity[basis0.get_dofs(elements=name)] = rho_cp[name]
    return mesh, basis0, k, capacity, w_heater * h_heater


def test_transient_matches_femwell_on_the_same_mesh(femwell_heater) -> None:
    """Same mesh, source and Crank-Nicolson step: identical core temperature."""
    from femwell.thermal_transient import solve_thermal_transient

    mesh, basis0, k, capacity, heater_area_um2 = femwell_heater
    sigma = 2.3e6
    current = 0.007
    dt = 0.25e-6
    steps = 120

    # femwell works on a um mesh with k scaled by 1e-12 and J in A/um^2.
    basis, temps = solve_thermal_transient(
        basis0,
        k * 1e-12,
        (k / capacity) * 1e12,
        {"heater": sigma},
        current_densities_0={"heater": 0.0},
        current_densities={"heater": lambda _t: current / heater_area_um2},
        fixed_boundaries={"bottom": 0},
        dt=dt,
        steps=steps,
    )

    @skfem.LinearForm
    def unit(v, _w):
        """Integral of the test function (core-average weights)."""
        return v

    weights = unit.assemble(basis.with_elements("core"))
    weights = weights / weights.sum()
    femwell_core = np.asarray(temps) @ weights

    heat = np.zeros(mesh.t.shape[1])
    j_si = current / (heater_area_um2 * 1e-12)
    heat[basis0.get_dofs(elements="heater")] = j_si**2 / sigma
    history = solve_transient(
        mesh,
        k,
        capacity,
        heat,
        duration_us=steps * dt * 1e6,
        steps=steps,
        theta=0.5,
        boundary_facets={"bottom": mesh.boundaries["bottom"]},
        sink="bottom",
        sample=weights,
        degree=1,  # femwell's thermal solvers are P1
    )

    np.testing.assert_allclose(history.values, femwell_core, rtol=0, atol=1e-9)
    time_us = np.arange(steps + 1) * dt * 1e6
    assert time_constant_1e(history.time_us, history.values) == pytest.approx(
        time_constant_1e(time_us, femwell_core), rel=1e-9
    )
