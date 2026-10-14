"""Tests for the thermal solver core: Joule source, steady and transient solves."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.thermal.solver import (
    drive_power_per_length,
    joule_heat_density,
)

# --- Joule source -----------------------------------------------------------


def test_joule_single_tin_strip_matches_femwell_example_resistance():
    # femwell TiN heater: 2 x 0.14 um, sigma = 2.3e6 S/m, L = 320 um -> R ~ 497 ohm.
    sigma = np.array([2.3e6, 0.0])
    mask = np.array([True, False])
    area = np.array([0.28, 5.0])
    src = joule_heat_density(sigma, mask, area, power_per_length_W_per_m=85.0)
    assert src.resistance_per_length * 320e-6 == pytest.approx(497, rel=2e-3)
    # Heat per length integrates back to the drive; nothing outside the heater.
    assert np.sum(src.heat_density * area * 1e-12) == pytest.approx(85.0, rel=1e-12)
    assert src.heat_density[1] == 0.0
    # E^2 sigma = q in the conductor.
    assert src.heat_density[0] == pytest.approx(2.3e6 * src.field**2)


def test_joule_two_conductors_split_power_by_conductance():
    # Strip A: sigma 1e6, 1 um^2 ; strip B: sigma 3e6, 2 um^2 -> conductances 1 : 6.
    sigma = np.array([1e6, 3e6, 3e6, 1e3])
    mask = np.array([True, True, True, False])
    area = np.array([1.0, 1.0, 1.0, 4.0])
    src = joule_heat_density(sigma, mask, area, power_per_length_W_per_m=70.0)
    power_a = src.heat_density[0] * 1e-12
    power_b = np.sum(src.heat_density[1:3]) * 1e-12
    assert power_a == pytest.approx(10.0)
    assert power_b == pytest.approx(60.0)
    # Sigma*A sum = 1e6*1e-12 + 3e6*2e-12 = 7e-6 S*m -> R' = 1/7e-6 ohm/m.
    assert src.resistance_per_length == pytest.approx(1 / 7e-6)
    # Conductivity outside the heater mask carries no current.
    assert src.heat_density[3] == 0.0


def test_drive_conversions_to_power_per_length():
    r_per_len = 1e6  # ohm/m ; L = 100 um -> R = 100 ohm
    assert drive_power_per_length(
        r_per_len, power_W=0.02, length_um=100.0
    ) == pytest.approx(200.0)
    # V = 1 V over 100 ohm -> 10 mW over 100 um -> 100 W/m.
    assert drive_power_per_length(
        r_per_len, voltage_V=1.0, length_um=100.0
    ) == pytest.approx(100.0)
    # I = 10 mA: P/L = I^2 R' = 1e-4 * 1e6 = 100 W/m (no length needed).
    assert drive_power_per_length(r_per_len, current_A=0.01) == pytest.approx(100.0)
    assert drive_power_per_length(
        r_per_len, power_per_length_W_per_m=5.0
    ) == pytest.approx(5.0)


def test_drive_requires_exactly_one_drive_and_a_length_where_needed():
    with pytest.raises(ValueError, match="exactly one"):
        drive_power_per_length(1e6, power_W=1.0, current_A=1.0, length_um=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        drive_power_per_length(1e6)
    with pytest.raises(ValueError, match="length_um"):
        drive_power_per_length(1e6, voltage_V=1.0)


def test_joule_rejects_bad_inputs():
    area = np.ones(2)
    with pytest.raises(ValueError, match="shape"):
        joule_heat_density(
            np.ones(3), np.ones(2, bool), area, power_per_length_W_per_m=1.0
        )
    with pytest.raises(ValueError, match="conductance"):
        joule_heat_density(
            np.zeros(2), np.ones(2, bool), area, power_per_length_W_per_m=1.0
        )
    with pytest.raises(ValueError, match="power"):
        joule_heat_density(
            np.ones(2), np.ones(2, bool), area, power_per_length_W_per_m=-1.0
        )


# --- Steady solve -----------------------------------------------------------


def _slab(width_um: float, height_um: float, nx: int, ny: int):
    """Rectangle mesh in um with left/right/top/bottom facet groups."""
    from skfem import MeshTri

    mesh = MeshTri.init_tensor(
        np.linspace(0.0, width_um, nx + 1), np.linspace(0.0, height_um, ny + 1)
    )
    bf = mesh.boundary_facets()
    mid = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    facets = {
        "bottom": bf[np.isclose(mid[1], 0.0)],
        "top": bf[np.isclose(mid[1], height_um)],
        "left": bf[np.isclose(mid[0], 0.0)],
        "right": bf[np.isclose(mid[0], width_um)],
    }
    return mesh, facets


def _centroid_y(mesh):
    return mesh.p[1, mesh.t].mean(axis=0)


def _top_value(mesh, temperature, height_um):
    return float(np.mean(temperature[np.isclose(mesh.p[1], height_um)]))


def test_two_layer_slab_matches_series_thermal_resistance():
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_steady

    # Bottom layer 0..6 um (k = 1.4), top layer 6..8 um (k = 0.2) heated uniformly.
    t1, t2, k1, k2, q = 6.0, 2.0, 1.4, 0.2, 1e12  # q in W/m^3
    mesh, facets = _slab(2.0, t1 + t2, 2, 32)
    yc = _centroid_y(mesh)
    k = np.where(yc < t1, k1, k2)
    qe = np.where(yc > t1, q, 0.0)
    dT = solve_steady(mesh, k, qe, boundary_facets=facets)
    # Flux q*t2 through layer 1: dT_i = q t2 t1/k1 ; top adds q t2^2/(2 k2).
    um = 1e-6
    expected_top = q * t2 * um * t1 * um / k1 + q * (t2 * um) ** 2 / (2 * k2)
    assert expected_top == pytest.approx(18.571429, rel=1e-6)  # 12/1.4 + 10 K
    assert _top_value(mesh, dT, t1 + t2) == pytest.approx(expected_top, rel=5e-3)
    assert np.allclose(dT[np.isclose(mesh.p[1], 0.0)], 0.0)


def test_convective_top_matches_1d_robin_analytic():
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_steady

    # L = 10 um, k = 1, q = 1e10 W/m^3, h = 1e5 W/m^2/K on top (hL/k = 1).
    # T(y) = -q y^2/2k + C y, C = qL(1 + hL/2k)/(k + hL) -> T(L) = 0.25 K
    # (0.5 K if the top were adiabatic).
    mesh, facets = _slab(2.0, 10.0, 2, 40)
    n = mesh.t.shape[1]
    dT = solve_steady(
        mesh,
        np.ones(n),
        np.full(n, 1e10),
        boundary_facets=facets,
        convective={"top": 1e5},
    )
    assert _top_value(mesh, dT, 10.0) == pytest.approx(0.25, rel=5e-3)


def test_convective_only_without_sink():
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_steady

    # All heat qL leaves through bottom Robin: T(0) = qL/h = 1 K, top = 1.5 K.
    mesh, facets = _slab(2.0, 10.0, 2, 40)
    n = mesh.t.shape[1]
    dT = solve_steady(
        mesh,
        np.ones(n),
        np.full(n, 1e10),
        boundary_facets=facets,
        sink=(),
        convective={"bottom": 1e5},
    )
    assert float(np.mean(dT[np.isclose(mesh.p[1], 0.0)])) == pytest.approx(
        1.0, rel=5e-3
    )
    assert _top_value(mesh, dT, 10.0) == pytest.approx(1.5, rel=5e-3)


def test_operator_is_linear_in_power_and_reusable():
    pytest.importorskip("skfem")
    from gsim.thermal.solver import HeatOperator, solve_steady

    mesh, facets = _slab(4.0, 4.0, 8, 8)
    n = mesh.t.shape[1]
    yc = _centroid_y(mesh)
    xc = mesh.p[0, mesh.t].mean(axis=0)
    k = np.where(yc < 2.0, 1.4, 148.0)
    q = np.where((yc > 3.0) & (np.abs(xc - 2.0) < 1.0), 1e13, 0.0)
    op = HeatOperator(mesh, k, boundary_facets=facets, convective={"top": 10.0})
    one = op.solve(q)
    assert one.shape == (mesh.p.shape[1],)
    assert one.max() > 0
    np.testing.assert_allclose(op.solve(3.0 * q), 3.0 * one, rtol=1e-12)
    np.testing.assert_allclose(op.solve(np.zeros(n)), 0.0)
    np.testing.assert_allclose(
        solve_steady(mesh, k, q, boundary_facets=facets, convective={"top": 10.0}),
        one,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": -1.0}, "conductivity"),
        ({"k_len": 3}, "shape"),
        ({"q_len": 3}, "shape"),
        ({"sink": "floor"}, "floor"),
        ({"convective": {"lid": 10.0}}, "lid"),
        ({"convective": {"top": -1.0}}, "h"),
        ({"sink": ()}, "heat path"),
    ],
)
def test_steady_rejects_bad_inputs(kwargs, match):
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_steady

    mesh, facets = _slab(2.0, 2.0, 2, 2)
    n = mesh.t.shape[1]
    k = np.full(kwargs.get("k_len", n), kwargs.get("k", 1.0))
    q = np.zeros(kwargs.get("q_len", n))
    with pytest.raises(ValueError, match=match):
        solve_steady(
            mesh,
            k,
            q,
            boundary_facets=facets,
            sink=kwargs.get("sink", "bottom"),
            convective=kwargs.get("convective"),
        )


# --- Transient --------------------------------------------------------------

# Slab 0..L (L = 10 um), sink at y = 0, adiabatic top, uniform k and rho*c_p.
# The slowest mode is sin(pi y / 2L): tau_1 = rho c_p (2L/pi)^2 / k
# = 1.6e6 * 4e-10 / (1.4 pi^2) s = 46.318 us.
_L, _K, _RHOCP, _Q = 10.0, 1.4, 1.6e6, 1e11
_TAU1_US = 46.318


def _slab_problem():
    mesh, facets = _slab(1.0, _L, 1, 40)
    n = mesh.t.shape[1]
    top_weights = np.isclose(mesh.p[1], _L) / np.sum(np.isclose(mesh.p[1], _L))
    return mesh, facets, np.full(n, _K), np.full(n, _RHOCP), np.full(n, _Q), top_weights


def test_step_response_helpers_on_exact_exponential():
    from gsim.thermal.solver import rise_time_10_90, time_constant_1e

    t = np.linspace(0.0, 50.0, 5001)
    y = 3.0 * (1.0 - np.exp(-t / 5.0))
    assert time_constant_1e(t, y) == pytest.approx(5.0, rel=1e-3)
    assert rise_time_10_90(t, y) == pytest.approx(5.0 * np.log(9.0), rel=1e-3)


def test_transient_converges_to_steady_state():
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_steady, solve_transient

    mesh, facets, k, rc, q, _ = _slab_problem()
    hist = solve_transient(
        mesh, k, rc, q, duration_us=10 * _TAU1_US, steps=400, boundary_facets=facets
    )
    steady = solve_steady(mesh, k, q, boundary_facets=facets)
    assert hist.time_us.shape == (401,)
    assert hist.time_us[0] == 0.0
    assert hist.time_us[-1] == pytest.approx(10 * _TAU1_US)
    assert hist.values.shape == (401, mesh.p.shape[1])
    np.testing.assert_allclose(hist.values[0], 0.0)
    np.testing.assert_allclose(hist.values[-1], steady, rtol=5e-3, atol=1e-6)


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_slab_late_decay_matches_first_mode_time_constant(theta):
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_steady, solve_transient

    mesh, facets, k, rc, q, top = _slab_problem()
    hist = solve_transient(
        mesh,
        k,
        rc,
        q,
        duration_us=5 * _TAU1_US,
        steps=500,
        theta=theta,
        boundary_facets=facets,
        sample=top,
    )
    final = float(top @ solve_steady(mesh, k, q, boundary_facets=facets))
    deficit = final - hist.values
    i1, i2 = 200, 400  # t = 2 tau, 4 tau: higher modes decayed by e^-18
    tau = (hist.time_us[i2] - hist.time_us[i1]) / np.log(deficit[i1] / deficit[i2])
    assert tau == pytest.approx(_TAU1_US, rel=0.01)


def test_rise_time_is_stable_under_time_step_refinement():
    pytest.importorskip("skfem")
    from gsim.thermal.solver import rise_time_10_90, solve_transient

    mesh, facets, k, rc, q, top = _slab_problem()

    def rise(steps):
        hist = solve_transient(
            mesh,
            k,
            rc,
            q,
            duration_us=6 * _TAU1_US,
            steps=steps,
            boundary_facets=facets,
            sample=lambda dT: top @ dT,
        )
        return rise_time_10_90(hist.time_us, hist.values)

    coarse, fine = rise(200), rise(800)
    assert coarse == pytest.approx(fine, rel=0.02)
    # The top response is dominated by the first mode (higher-mode weights
    # fall as 1/(2n+1)^3), so the rise is close to tau_1 ln 9 = 101.8 us.
    assert fine == pytest.approx(101.8, rel=0.02)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rc": -1.0}, "heat_capacity"),
        ({"steps": 0}, "steps"),
        ({"duration_us": 0.0}, "duration_us"),
        ({"theta": 0.2}, "theta"),
        ({"sample": np.ones(3)}, "sample"),
    ],
)
def test_transient_rejects_bad_inputs(kwargs, match):
    pytest.importorskip("skfem")
    from gsim.thermal.solver import solve_transient

    mesh, facets = _slab(2.0, 2.0, 2, 2)
    n = mesh.t.shape[1]
    with pytest.raises(ValueError, match=match):
        solve_transient(
            mesh,
            np.ones(n),
            np.full(n, kwargs.get("rc", 1e6)),
            np.zeros(n),
            duration_us=kwargs.get("duration_us", 1.0),
            steps=kwargs.get("steps", 10),
            theta=kwargs.get("theta", 1.0),
            boundary_facets=facets,
            sample=kwargs.get("sample"),
        )
