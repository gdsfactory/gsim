"""Superconducting layers: the London surface inductance written as ``Impedance``.

Parts, in order:

1. The closed forms of ``gsim.palace._superconductor`` against their limits and
   against a finite-difference solve of ``H'' = H / lambda^2`` across a film.
2. A quasi-static coplanar-line reference: a partial-inductance solve of the
   London equation over the rectangular cells of the conductors, with the
   sheet and the hollow-shell surface models compared with it. These are
   physics checks of the choice of formula for each conductor model. They are
   not checks of Palace; the cloud runs are listed in the docs.
3. The ``Boundaries.Impedance`` entries from hand-made mesh groups.
4. The same on real coarse meshes (generic PDK, no solver, nothing submitted).
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import mu_0
from scipy.linalg import solve_banded
from scipy.special import ellipk

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.hashing import compute_input_hash
from gsim.palace import BoundaryModeSim, DrivenSim, EigenmodeSim
from gsim.palace import _superconductor as sc
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.mesh.validation import check_lumped_port_contact
from gsim.palace.models import DrivenConfig
from gsim.palace.models.ports import ImpedanceBoundaryConfig
from gsim.palace.ports.config import PalacePort, PortType
from tests.palace.test_boundarymode_conductor_pec import meshed_sim as meshed_2d_sim
from tests.palace.test_conductor_crack import _fake_gsim_runtime
from tests.palace.test_mesh_integration import (
    _make_qpdk_component,
    _make_qpdk_sim,
    _make_qpdk_stack,
)
from tests.palace.test_run_local import _mesh_result, _setup_local_palace, _setup_sim
from tests.palace.test_symmetry_mesh import _mesh_sim as _symmetry_mesh_sim
from tests.palace.test_workflow import _make_cpw_component

# ---------------------------------------------------------------------------
# 1. Closed forms
# ---------------------------------------------------------------------------

UM = 1e-6


def coth(x: float) -> float:
    """Hyperbolic cotangent."""
    return 1.0 / math.tanh(x)


class TestFormulaLimits:
    def test_thin_film_limit_of_both_values(self):
        lam, t = 0.2, 2e-4  # t = 1e-3 lambda
        kinetic = mu_0 * (lam * UM) ** 2 / (t * UM)
        assert sc.sheet_inductance(lam, t) == pytest.approx(kinetic, rel=1e-3)
        assert sc.face_inductance(lam, t) == pytest.approx(2 * kinetic, rel=1e-3)
        assert sc.one_sided_inductance(lam, t) == pytest.approx(kinetic, rel=1e-3)

    def test_thick_film_limit(self):
        lam, t = 0.1, 5.0  # t = 50 lambda
        assert sc.face_inductance(lam, t) == pytest.approx(mu_0 * lam * UM, rel=1e-12)
        assert sc.sheet_inductance(lam, t) == pytest.approx(
            mu_0 * lam * UM / 2, rel=1e-12
        )
        assert sc.one_sided_inductance(lam, t) == pytest.approx(
            mu_0 * lam * UM, rel=1e-12
        )

    @pytest.mark.parametrize("t", [1e-3, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0])
    def test_face_is_twice_the_sheet_at_every_thickness(self, t):
        assert sc.face_inductance(0.15, t) == pytest.approx(
            2 * sc.sheet_inductance(0.15, t), rel=1e-12
        )

    def test_face_and_one_sided_differ_by_two_only_for_a_thin_film(self):
        lam = 0.2
        thin = sc.face_inductance(lam, 2e-3) / sc.one_sided_inductance(lam, 2e-3)
        thick = sc.face_inductance(lam, 20.0) / sc.one_sided_inductance(lam, 20.0)
        assert thin == pytest.approx(2.0, rel=1e-3)
        assert thick == pytest.approx(1.0, rel=1e-12)

    def test_the_sheet_value_by_a_second_form(self):
        # lambda = 0.2 um, t = 0.1 um: mu0 (lambda / 2) cosh(x) / sinh(x), x = 0.25
        expected = mu_0 * 0.1e-6 * math.cosh(0.25) / math.sinh(0.25)
        assert sc.sheet_inductance(0.2, 0.1) == pytest.approx(expected, rel=1e-12)

    def test_known_value(self):
        # lambda = 0.1 um, t = 0.1 um: mu0 * 0.1e-6 / 2 / tanh(0.5), by hand
        expected = 4e-7 * math.pi * 1e-7 / 2 / math.tanh(0.5)
        assert sc.sheet_inductance(0.1, 0.1) == pytest.approx(expected, rel=1e-6)


def slab_e(lam: float, t: float, h_bottom: float, h_top: float, n: int = 4001):
    """E / (i w) at both faces of a film ``0 < z < t``, and the total sheet current.

    Finite differences for ``H'' = H / lambda^2`` with the outside fields given
    at both faces; ``J = -H'`` and ``E = i w mu0 lambda^2 J``. Lengths in m.
    """
    z = np.linspace(0.0, t, n)
    h = z[1] - z[0]
    ab = np.zeros((3, n))
    ab[1, :] = -2.0 / h**2 - 1.0 / lam**2
    ab[0, 1:] = 1.0 / h**2
    ab[2, :-1] = 1.0 / h**2
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 1] = 0.0
    ab[2, -2] = 0.0
    rhs = np.zeros(n)
    rhs[0], rhs[-1] = h_bottom, h_top
    field = solve_banded((1, 1), ab, rhs)
    current = -np.gradient(field, h, edge_order=2)
    return mu_0 * lam**2 * current[0], mu_0 * lam**2 * current[-1], h_bottom - h_top


class TestFiniteDifferenceReference:
    CASES = ((0.1, 0.15), (0.2, 0.1), (0.05, 0.3), (0.1, 0.01), (0.1, 0.8))

    @pytest.mark.parametrize(("lam", "t"), CASES)
    def test_antisymmetric_field_gives_the_sheet_and_face_values(self, lam, t):
        e_bottom, e_top, total = slab_e(lam * UM, t * UM, 1.0, -1.0)
        assert e_bottom == pytest.approx(e_top, rel=1e-9)  # equal on both faces
        assert e_bottom / total == pytest.approx(sc.sheet_inductance(lam, t), rel=1e-5)
        assert e_bottom / (total / 2) == pytest.approx(
            sc.face_inductance(lam, t), rel=1e-5
        )

    @pytest.mark.parametrize(("lam", "t"), CASES)
    def test_field_on_one_side_gives_the_textbook_value(self, lam, t):
        e_bottom, e_top, total = slab_e(lam * UM, t * UM, 1.0, 0.0)
        assert e_bottom / total == pytest.approx(
            sc.one_sided_inductance(lam, t), rel=1e-5
        )
        # the far face sees mu0 lambda csch(t / lambda)
        assert e_top / total == pytest.approx(
            mu_0 * lam * UM / math.sinh(t / lam), rel=1e-4
        )

    @pytest.mark.parametrize(("lam", "t"), CASES)
    def test_thin_film_current_split_does_not_matter(self, lam, t):
        """Whatever the field split, E / K equals mu0 lambda coth ... to first order."""
        u = t / lam
        for h_top in (-1.0, 0.0, 0.5):
            e_bottom, e_top, total = slab_e(lam * UM, t * UM, 1.0, h_top)
            k_b, k_t = 1.0, -h_top
            expected_b = mu_0 * lam * UM * (coth(u) * k_b + k_t / math.sinh(u))
            expected_t = mu_0 * lam * UM * (k_b / math.sinh(u) + coth(u) * k_t)
            assert e_bottom == pytest.approx(expected_b, rel=1e-5)
            assert e_top == pytest.approx(expected_t, rel=1e-5)
            assert total == pytest.approx(k_b + k_t)


def three_layer_e(lam: float, t: float, h_bottom: float, h_top: float) -> float:
    """E / (i w) at the bottom sheet of the hollow-shell model.

    Two sheets, each with surface inductance ``mu0 lambda coth(t / 2 lambda)``
    at ``z = 0`` and ``z = t``, and a field-transparent interior of thickness t
    and permeability mu0 between them. Unknowns: the interior field ``H_m`` and
    the two sheet currents, from the jump conditions ``K_1 = H_b - H_m``,
    ``K_2 = H_m - H_t`` and Faraday's law across the interior,
    ``E_2 - E_1 = -i w mu0 t H_m`` with ``E_i = i w Ls K_i``. Solved as a
    linear system, independently of the closed form in the docs.
    """
    ls = mu_0 * lam / math.tanh(t / (2 * lam))
    # unknowns x = (H_m, K_1, K_2); equations as rows
    a = np.array(
        [
            [1.0, 1.0, 0.0],  # K_1 + H_m = H_b
            [-1.0, 0.0, 1.0],  # K_2 - H_m = -H_t
            [mu_0 * t, -ls, ls],  # ls (K_2 - K_1) + mu0 t H_m = 0
        ]
    )
    b = np.array([h_bottom, -h_top, 0.0])
    _, k_1, _ = np.linalg.solve(a, b)
    return ls * k_1


class TestHollowShellModel:
    @pytest.mark.parametrize("u", [0.01, 1.0, 2.0, 4.0, 10.0, 100.0])
    def test_closed_form_against_the_linear_system(self, u):
        lam, t = 0.1 * UM, 0.1 * u * UM
        c = coth(u / 2)
        ratio_closed = c * (c + u) / (2 * c + u) / coth(u)
        model = three_layer_e(lam, t, 1.0, 0.0)
        exact = mu_0 * lam * coth(u)  # one-sided film, closed form
        assert model / exact == pytest.approx(ratio_closed, rel=1e-9)
        if u <= 10.0:  # the grid resolves lambda: finite differences agree
            fd = slab_e(lam, t, 1.0, 0.0, n=2001)[0]
            assert fd == pytest.approx(exact, rel=1e-4)

    @pytest.mark.parametrize(
        ("u", "error"),
        [(1.0, -0.021), (2.0, -0.093), (4.0, -0.140), (10.0, -0.083)],
    )
    def test_the_quoted_one_sided_errors(self, u, error):
        c = coth(u / 2)
        assert c * (c + u) / (2 * c + u) / coth(u) - 1 == pytest.approx(error, abs=1e-3)

    @pytest.mark.parametrize("u", [0.01, 0.5, 3.0, 10.0])
    def test_antisymmetric_field_is_exact_at_every_thickness(self, u):
        lam, t = 0.1 * UM, 0.1 * u * UM
        exact = slab_e(lam, t, 1.0, -1.0, n=2001)[0]
        assert three_layer_e(lam, t, 1.0, -1.0) == pytest.approx(exact, rel=1e-4)

    def test_both_limits_are_exact_for_a_one_sided_field(self):
        for u in (1e-3, 200.0):
            lam, t = 0.1 * UM, 0.1 * u * UM
            exact = lam * mu_0 * coth(u)
            assert three_layer_e(lam, t, 1.0, 0.0) == pytest.approx(exact, rel=2e-3)


class TestSettingsModel:
    def test_values_are_stored(self):
        cfg = sc.SuperconductorConfig(london_depth=0.1, thickness=0.2)
        assert (cfg.london_depth, cfg.thickness) == (0.1, 0.2)
        assert sc.SuperconductorConfig(london_depth=0.1).thickness is None

    @pytest.mark.parametrize("value", [0.0, -0.1, -1e-7])
    def test_non_positive_values_are_refused(self, value):
        with pytest.raises(ValueError, match="greater than 0"):
            sc.SuperconductorConfig(london_depth=value)
        with pytest.raises(ValueError, match="greater than 0"):
            sc.SuperconductorConfig(london_depth=0.1, thickness=value)

    @pytest.mark.parametrize("value", [1e-7, 2e-8, 9e-5])
    def test_meter_sized_values_are_refused(self, value):
        with pytest.raises(ValueError, match="um, not m"):
            sc.SuperconductorConfig(london_depth=value)
        with pytest.raises(ValueError, match="um, not m"):
            sc.SuperconductorConfig(london_depth=0.1, thickness=value)

    def test_assignment_is_validated(self):
        cfg = sc.SuperconductorConfig(london_depth=0.1)
        with pytest.raises(ValueError, match="um, not m"):
            cfg.london_depth = 1e-7

    def test_unknown_fields_are_refused(self):
        with pytest.raises(ValueError, match="Extra inputs"):
            sc.SuperconductorConfig(london_depth=0.1, sigma=1.0)  # ty: ignore[unknown-argument]

    def test_a_huge_depth_warns(self, caplog):
        with caplog.at_level("WARNING", logger="gsim.palace._superconductor"):
            sc.SuperconductorConfig(london_depth=50.0)
        assert "above 10 um" in caplog.text

    def test_formulas_refuse_non_positive_lengths(self):
        for function in (
            sc.sheet_inductance,
            sc.face_inductance,
            sc.one_sided_inductance,
        ):
            with pytest.raises(ValueError, match="positive"):
                function(0.0, 0.1)
            with pytest.raises(ValueError, match="positive"):
                function(0.1, -0.1)


# ---------------------------------------------------------------------------
# 2. Quasi-static coplanar-line reference
# ---------------------------------------------------------------------------


def graded(x0: float, x1: float, hmin: float, growth: float, hmax: float):
    """Cell edges on ``[x0, x1]``, fine at both ends."""
    half = (x1 - x0) / 2
    edges = [0.0]
    h = hmin
    while edges[-1] + h < half:
        edges.append(edges[-1] + h)
        h = min(h * growth, hmax)
    edges.append(half)
    e = np.array(edges)
    return np.unique(np.concatenate([x0 + e, x1 - e[::-1]]))


def gmd_rect_ln(a, b):
    """Logarithm of the geometric mean distance of an a x b rectangle (Grover)."""
    d = np.hypot(a, b)
    return (
        np.log(d)
        - (a / b) ** 2 / 12 * np.log(1 + (b / a) ** 2)
        - (b / a) ** 2 / 12 * np.log(1 + (a / b) ** 2)
        + 2 / 3 * (a / b) * np.arctan(b / a)
        + 2 / 3 * (b / a) * np.arctan(a / b)
        - 25 / 12
    )


S, GAP, G = 10e-6, 6e-6, 50e-6  # centre strip, gap, ground width (m)


def _conductors():
    return [
        (-S / 2, S / 2, 0),
        (S / 2 + GAP, S / 2 + GAP + G, 1),
        (-S / 2 - GAP - G, -S / 2 - GAP, 1),
    ]


def solid_cells(t: float, hx: float, hz: float) -> np.ndarray:
    """Rows ``(xc, zc, wx, wz, conductor)`` over the three solid rectangles."""
    cells = []
    z_edges = graded(0.0, t, hz, 1.25, t / 2)
    for x0, x1, cid in _conductors():
        x_edges = graded(x0, x1, hx, 1.12, 0.5e-6)
        cells.extend(
            (
                0.5 * (x_edges[i] + x_edges[i + 1]),
                0.5 * (z_edges[k] + z_edges[k + 1]),
                x_edges[i + 1] - x_edges[i],
                z_edges[k + 1] - z_edges[k],
                cid,
            )
            for i in range(len(x_edges) - 1)
            for k in range(len(z_edges) - 1)
        )
    return np.array(cells)


def shell_cells(t: float, hx: float, d: float) -> np.ndarray:
    """Cells of depth ``d`` on the four faces of each rectangle (a hollow shell)."""
    cells = []
    for x0, x1, cid in _conductors():
        x_edges = graded(x0, x1, hx, 1.12, 0.5e-6)
        for i in range(len(x_edges) - 1):
            xc, wx = 0.5 * (x_edges[i] + x_edges[i + 1]), x_edges[i + 1] - x_edges[i]
            cells.append((xc, d / 2, wx, d, cid))
            cells.append((xc, t - d / 2, wx, d, cid))
        z_edges = graded(d, t - d, max(d, hx), 1.12, 0.5e-6)
        for k in range(len(z_edges) - 1):
            zc, wz = 0.5 * (z_edges[k] + z_edges[k + 1]), z_edges[k + 1] - z_edges[k]
            cells.append((x0 + d / 2, zc, d, wz, cid))
            cells.append((x1 - d / 2, zc, d, wz, cid))
    return np.array(cells)


def solve_inductance(cells: np.ndarray, london: float) -> float:
    """Inductance per length [H/m] of the line, with ``E = i w london J`` per cell.

    Partial-inductance solve: currents in the cells of the centre strip sum to 1
    A and those of the grounds to -1 A, with a common potential on each.
    """
    xc, zc, wx, wz, cid = cells.T
    n = len(xc)
    r2 = (xc[:, None] - xc[None, :]) ** 2 + (zc[:, None] - zc[None, :]) ** 2
    np.fill_diagonal(r2, 1.0)
    log_r = 0.5 * np.log(r2)
    np.fill_diagonal(log_r, gmd_rect_ln(wx, wz))
    m = -mu_0 / (2 * np.pi) * log_r
    m[np.diag_indices(n)] += london / (wx * wz)
    centre = (cid == 0).astype(float)
    ground = 1.0 - centre
    a = np.zeros((n + 2, n + 2))
    a[:n, :n] = m
    a[:n, n] = -centre
    a[:n, n + 1] = -ground
    a[n, :n] = centre
    a[n + 1, :n] = ground
    rhs = np.zeros(n + 2)
    rhs[n], rhs[n + 1] = 1.0, -1.0
    sol = np.linalg.solve(a, rhs)
    return sol[n] - sol[n + 1]


@pytest.fixture(scope="module")
def cpw_reference() -> dict[str, float]:
    """lambda = 0.2 um, t = 0.1 um on the coarse grid, all models."""
    lam, t = 0.2e-6, 0.1e-6
    h = min(lam, t) / 4
    solid = solid_cells(t, h, h)
    l_pec = solve_inductance(solid, 0.0)
    alpha_full = solve_inductance(solid, mu_0 * lam**2) / l_pec - 1

    one_row = solid_cells(t, h, t)  # hz = t: one row of cells across the film
    ls_sheet = sc.sheet_inductance(lam / UM, t / UM)
    l_row_pec = solve_inductance(one_row, 0.0)
    alpha_sheet = solve_inductance(one_row, ls_sheet * t) / l_row_pec - 1

    d = min(lam, t) / 20
    shell = shell_cells(t, h, d)
    l_shell_pec = solve_inductance(shell, 0.0)
    ls_face = sc.face_inductance(lam / UM, t / UM)
    ls_face_one_sided = sc.one_sided_inductance(lam / UM, t / UM)
    return {
        "l_geo": l_pec,
        "alpha_full": alpha_full,
        "alpha_sheet": alpha_sheet,
        "alpha_shell": solve_inductance(shell, ls_face * d) / l_shell_pec - 1,
        "alpha_shell_one_sided": solve_inductance(shell, ls_face_one_sided * d)
        / l_shell_pec
        - 1,
    }


class TestQuasiStaticCoplanarReference:
    """Sheet and hollow-shell surface models against the full London solve."""

    def test_the_reference_is_a_sizeable_kinetic_fraction(self, cpw_reference):
        assert 0.15 < cpw_reference["alpha_full"] < 0.3

    def test_the_sheet_is_within_two_percent(self, cpw_reference):
        full, sheet = cpw_reference["alpha_full"], cpw_reference["alpha_sheet"]
        assert sheet == pytest.approx(full, rel=0.02)

    def test_the_hollow_shell_is_within_six_percent(self, cpw_reference):
        full, shell = cpw_reference["alpha_full"], cpw_reference["alpha_shell"]
        assert shell == pytest.approx(full, rel=0.06)

    def test_the_textbook_value_per_face_is_far_too_low(self, cpw_reference):
        full = cpw_reference["alpha_full"]
        assert cpw_reference["alpha_shell_one_sided"] < 0.7 * full

    def test_geometric_inductance_against_conformal_mapping(self, cpw_reference):
        k = S / (S + 2 * GAP)
        conformal = mu_0 * ellipk(1 - k**2) / (4 * ellipk(k**2))
        # t = 0.1 um and 50 um grounds against t = 0 and infinite grounds
        assert cpw_reference["l_geo"] == pytest.approx(conformal, rel=0.03)

    def test_geometric_inductance_against_scikit_rf(self, cpw_reference):
        skrf = pytest.importorskip("skrf")
        from skrf.media import CPW

        medium = CPW(
            skrf.Frequency(5, 5, 1, "GHz"),
            w=S,
            s=GAP,
            h=500e-6,
            ep_r=11.45,
            t=0.1e-6,
            rho=1.68e-8,
            diel="frequencyinvariant",
        )
        z_eff, eps_eff, *_ = medium.analyse_quasi_static(
            11.45, S, GAP, 500e-6, 0.1e-6, False
        )
        l_skrf = float(np.real(z_eff * np.sqrt(eps_eff))) / c_light
        assert cpw_reference["l_geo"] == pytest.approx(l_skrf, rel=0.03)


# ---------------------------------------------------------------------------
# 3. The Impedance entries from hand-made mesh groups
# ---------------------------------------------------------------------------

LAMBDA_UM = 0.1
SHELL_ATTRIBUTES = (4, 5)


def _stack(*, thickness: float = 3.0, layer_type: str = "conductor") -> LayerStack:
    """Two conductor layers (``metal`` and ``other``) and a via layer."""
    stack = LayerStack()
    for name, z0 in (("metal", 0.0), ("other", 10.0)):
        stack.layers[name] = Layer(
            name=name,
            gds_layer=(1, 0),
            zmin=z0,
            zmax=z0 + thickness,
            thickness=thickness,
            material="aluminum",
            layer_type=layer_type if name == "metal" else "conductor",
        )
    stack.layers["via"] = Layer(
        name="via",
        gds_layer=(2, 0),
        zmin=3.0,
        zmax=10.0,
        thickness=7.0,
        material="aluminum",
        layer_type="via",
    )
    stack.materials = {"aluminum": {"conductivity": 3.77e7}}
    return stack


def _groups(
    *,
    mode: str = "shell",
    wave_ports: bool = False,
    symmetry: str | None = None,
    other_layer: bool = False,
) -> dict:
    """Mesh groups: ``metal`` as a shell, a planar sheet or a native 2D boundary."""
    conductors: dict[str, dict] = {}
    pec: dict[str, dict] = {}
    if mode == "shell":
        conductors = {"metal_xy": {"phys_group": 4}, "metal_z": {"phys_group": 5}}
    elif mode == "sheet":
        pec = {"metal": {"phys_group": 4}, "pec_block_1_xy": {"phys_group": 9}}
    elif mode == "2d":
        conductors = {"metal": {"phys_group": 4}}
    elif mode == "2d-pec":
        pec = {"metal": {"phys_group": 4}}
    if other_layer:
        conductors |= {"other_xy": {"phys_group": 14}, "other_z": {"phys_group": 15}}
    boundary_surfaces: dict[str, dict] = {"absorbing": {"phys_group": [8, 12]}}
    if symmetry:
        boundary_surfaces["symmetry"] = {
            "phys_group": [30],
            "tags": [1],
            "axis": "y",
            "position": 0.0,
            "kind": symmetry,
            "keep": "positive",
        }
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": conductors,
        "pec_surfaces": pec,
        "port_surfaces": (
            {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}} if wave_ports else {}
        ),
        "boundary_surfaces": boundary_surfaces,
    }


def _wave_ports() -> list[PalacePort]:
    """Two wave ports on the metal layer."""
    return [
        PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="metal"),
        PalacePort(
            name="o2", port_type=PortType.WAVEPORT, layer="metal", excited=False
        ),
    ]


def _generate(
    tmp_path,
    simulation_type: str,
    groups: dict,
    superconductors=None,
    ports=(),
    stack: LayerStack | None = None,
    **kwargs,
) -> dict:
    """Write a config from hand-made groups and read it back."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = generate_palace_config(
        groups=groups,
        ports=list(ports),
        port_info=[],
        stack=stack or _stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type=simulation_type,
        superconductors=superconductors,
        **kwargs,
    )
    return json.loads(path.read_text())


def _metal(thickness: float | None = None) -> dict[str, sc.SuperconductorConfig]:
    """The ``metal`` layer as a superconductor."""
    return {
        "metal": sc.SuperconductorConfig(london_depth=LAMBDA_UM, thickness=thickness)
    }


class TestImpedanceEntries:
    @pytest.mark.parametrize("simulation_type", ["driven", "eigenmode"])
    def test_shell_faces_get_the_face_value_without_crack(
        self, tmp_path, simulation_type
    ):
        config = _generate(tmp_path, simulation_type, _groups(), _metal())
        face = sc.face_inductance(LAMBDA_UM, 3.0)  # the stack's 3 um
        boundaries = config["Boundaries"]
        assert boundaries["Impedance"] == [
            {"Attributes": [4], "Ls": face},
            {"Attributes": [5], "Ls": face},
        ]
        assert boundaries["Conductivity"] == []
        assert "Crack" not in json.dumps(config)
        assert "PEC" not in boundaries

    def test_explicit_thickness_sets_the_value(self, tmp_path):
        config = _generate(tmp_path, "driven", _groups(), _metal(thickness=0.1))
        face = sc.face_inductance(LAMBDA_UM, 0.1)
        assert [e["Ls"] for e in config["Boundaries"]["Impedance"]] == [face, face]

    def test_planar_sheet_gets_the_sheet_value_and_leaves_pec(self, tmp_path):
        config = _generate(tmp_path, "driven", _groups(mode="sheet"), _metal(0.1))
        boundaries = config["Boundaries"]
        assert boundaries["Impedance"] == [
            {"Attributes": [4], "Ls": sc.sheet_inductance(LAMBDA_UM, 0.1)}
        ]
        assert boundaries["PEC"] == {"Attributes": [9]}  # the PEC block stays PEC

    def test_pec_key_is_absent_when_the_sheet_was_the_only_entry(self, tmp_path):
        groups = _groups(mode="sheet")
        del groups["pec_surfaces"]["pec_block_1_xy"]
        boundaries = _generate(tmp_path, "driven", groups, _metal(0.1))["Boundaries"]
        assert "PEC" not in boundaries
        assert boundaries["Conductivity"] == []

    def test_a_pec_block_with_the_layer_name_is_matched_exactly(self, tmp_path):
        """Only the layer's own key: ``pec_block_1_xy`` is not layer ``pec_block_1``."""
        stack = _stack()
        groups = _groups(mode="sheet")
        config = _generate(tmp_path, "driven", groups, _metal(0.1), stack=stack)
        attributes = [
            a for e in config["Boundaries"]["Impedance"] for a in e["Attributes"]
        ]
        assert attributes == [4]

    def test_other_layers_keep_their_conductivity(self, tmp_path):
        config = _generate(tmp_path, "driven", _groups(other_layer=True), _metal())
        boundaries = config["Boundaries"]
        assert [e["Attributes"] for e in boundaries["Conductivity"]] == [[14], [15]]
        assert all(e["Crack"] is True for e in boundaries["Conductivity"])
        assert [e["Attributes"] for e in boundaries["Impedance"]] == [[4], [5]]

    def test_boundary_mode_gets_the_face_value(self, tmp_path):
        config = _generate(tmp_path, "boundarymode", _groups(mode="2d"), _metal(0.1))
        assert config["Boundaries"]["Impedance"] == [
            {"Attributes": [4], "Ls": sc.face_inductance(LAMBDA_UM, 0.1)}
        ]
        assert "Conductivity" not in config["Boundaries"]
        assert "Crack" not in json.dumps(config)

    def test_boundary_mode_pec_conductor_layer(self, tmp_path):
        """A layer without conductivity is a PEC curve in 2D; it takes Ls too."""
        config = _generate(
            tmp_path, "boundarymode", _groups(mode="2d-pec"), _metal(0.1)
        )
        assert config["Boundaries"]["Impedance"] == [
            {"Attributes": [4], "Ls": sc.face_inductance(LAMBDA_UM, 0.1)}
        ]
        assert "PEC" not in config["Boundaries"]

    def test_boundary_mode_pec_moves_the_layer_to_pec_and_says_so(
        self, tmp_path, caplog
    ):
        with caplog.at_level("WARNING"):
            config = _generate(
                tmp_path,
                "boundarymode",
                _groups(mode="2d"),
                _metal(0.1),
                conductor_boundary="pec",
            )
        boundaries = config["Boundaries"]
        assert boundaries["PEC"] == {"Attributes": [4]}
        assert "Impedance" not in boundaries
        assert "Conductivity" not in boundaries
        assert "['metal']" in caplog.text
        assert "perfect conductors" in caplog.text

    def test_the_values_are_not_divided_by_a_length(self, tmp_path):
        """An interface entry is divided by its curve length; this one is not."""
        groups = _groups()
        groups["interface_surfaces"] = {
            "interface_a_b": {"phys_group": 20, "length_um": 10.0}
        }
        config = _generate(
            tmp_path,
            "driven",
            groups,
            _metal(0.1),
            impedance_boundaries=[
                ImpedanceBoundaryConfig(layer_a="a", layer_b="b", inductance=1e-9)
            ],
        )
        entries = config["Boundaries"]["Impedance"]
        assert entries[0] == {"Attributes": [20], "Ls": pytest.approx(1e-9 / 10e-6)}
        face = sc.face_inductance(LAMBDA_UM, 0.1)
        assert entries[1:] == [
            {"Attributes": [4], "Ls": face},
            {"Attributes": [5], "Ls": face},
        ]

    def test_no_superconductor_leaves_the_config_unchanged(self, tmp_path):
        plain = _generate(tmp_path / "a", "driven", _groups())
        empty = _generate(tmp_path / "b", "driven", _groups(), {})
        assert plain == empty
        assert "Impedance" not in plain["Boundaries"]

    def test_entries_are_sorted_by_group_name(self, tmp_path):
        groups = _groups()
        groups["conductor_surfaces"] = {
            "metal_z": {"phys_group": 5},
            "metal_xy": {"phys_group": 4},
        }
        entries = _generate(tmp_path, "driven", groups, _metal())["Boundaries"][
            "Impedance"
        ]
        assert [e["Attributes"] for e in entries] == [[4], [5]]  # _xy before _z


class TestWavePorts:
    @pytest.mark.parametrize("policy", ["pec", "inherit"])
    def test_wave_port_pec_follows_the_policy(self, tmp_path, policy):
        config = _generate(
            tmp_path,
            "driven",
            _groups(wave_ports=True),
            _metal(),
            _wave_ports(),
            driven_config=DrivenConfig(waveport_boundary=policy),
        )
        boundaries = config["Boundaries"]
        assert [e["Attributes"] for e in boundaries["Impedance"]] == [[4], [5]]
        assert boundaries["WavePort"]
        if policy == "inherit":
            assert "WavePortPEC" not in boundaries
        else:
            assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}


class TestSymmetryAndWall:
    @pytest.mark.parametrize("kind", ["pmc", "pec"])
    def test_symmetry_plane_does_not_touch_the_impedance_entries(self, tmp_path, kind):
        base = _generate(tmp_path / "a", "driven", _groups(), _metal())
        config = _generate(tmp_path / "b", "driven", _groups(symmetry=kind), _metal())
        boundaries = config["Boundaries"]
        assert boundaries["Impedance"] == base["Boundaries"]["Impedance"]
        plane = {"PMC" if kind == "pmc" else "PEC": {"Attributes": [30]}}
        for key, section in plane.items():
            assert boundaries[key] == section
        assert 30 not in boundaries["Absorbing"]["Attributes"]

    def test_pec_plane_merges_with_pec_not_with_the_sheet(self, tmp_path):
        groups = _groups(mode="sheet", symmetry="pec")
        boundaries = _generate(tmp_path, "driven", groups, _metal(0.1))["Boundaries"]
        assert boundaries["PEC"] == {"Attributes": [9, 30]}
        assert [e["Attributes"] for e in boundaries["Impedance"]] == [[4]]

    def test_metallic_outer_wall(self, tmp_path):
        base = _generate(tmp_path / "a", "driven", _groups(), _metal())
        config = _generate(
            tmp_path / "b",
            "driven",
            _groups(),
            _metal(),
            absorbing_boundary=False,
            metallic_boundaries=True,
        )
        boundaries = config["Boundaries"]
        assert boundaries["PEC"] == {"Attributes": [8, 12]}
        assert boundaries["Impedance"] == base["Boundaries"]["Impedance"]


class TestErrors:
    def test_a_layer_that_is_not_in_the_stack(self, tmp_path):
        superconductors = {"nope": sc.SuperconductorConfig(london_depth=0.1)}
        with pytest.raises(ValueError, match="'nope' is not in the stack") as error:
            _generate(tmp_path, "driven", _groups(), superconductors)
        assert "['metal', 'other']" in str(error.value)

    def test_a_via_layer(self, tmp_path):
        superconductors = {"via": sc.SuperconductorConfig(london_depth=0.1)}
        with pytest.raises(ValueError, match="not a conductor"):
            _generate(tmp_path, "driven", _groups(), superconductors)

    def test_zero_stack_thickness_needs_an_explicit_one(self, tmp_path):
        stack = _stack(thickness=0.0)
        with pytest.raises(ValueError, match="zero thickness"):
            _generate(tmp_path, "driven", _groups(mode="sheet"), _metal(), stack=stack)
        config = _generate(
            tmp_path / "ok", "driven", _groups(mode="sheet"), _metal(0.1), stack=stack
        )
        assert config["Boundaries"]["Impedance"][0]["Ls"] == pytest.approx(
            sc.sheet_inductance(LAMBDA_UM, 0.1)
        )

    @pytest.mark.parametrize(
        "simulation_type", ["electrostatic", "electrostatics", "magnetostatic"]
    )
    def test_static_problems_are_refused(self, tmp_path, simulation_type):
        with pytest.raises(ValueError, match=r"not .*static"):
            _generate(tmp_path, simulation_type, _groups(), _metal())

    def test_a_user_impedance_entry_on_the_same_surface(self, tmp_path):
        with pytest.raises(ValueError, match=r"\[4\].*Boundaries.Impedance"):
            _generate(
                tmp_path,
                "driven",
                _groups(),
                _metal(),
                impedance_boundaries=[
                    ImpedanceBoundaryConfig(attributes=[4], resistance=1.0)
                ],
            )

    def test_a_port_on_the_same_surface(self, tmp_path):
        groups = _groups(wave_ports=True)
        groups["port_surfaces"]["P1"]["phys_group"] = 4
        with pytest.raises(ValueError, match=r"\[4\].*Boundaries.WavePort"):
            _generate(tmp_path, "driven", groups, _metal(), _wave_ports())

    def test_a_layer_with_no_surface_in_the_mesh_writes_nothing(self, tmp_path):
        config = _generate(tmp_path, "driven", _groups(mode="none"), _metal())
        assert "Impedance" not in config["Boundaries"]


# ---------------------------------------------------------------------------
# 4. Real coarse meshes (generic PDK; no solver, nothing submitted)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def superconducting(sim, layer: str, **kwargs):
    """``set_superconductor`` for the length of a block, then back to normal metal."""
    before = dict(sim.superconductors)
    sim.set_superconductor(layer, **kwargs)
    sim._superconductor_warned.clear()
    try:
        yield sim
    finally:
        sim.superconductors.clear()
        sim.superconductors.update(before)


def _boundaries(sim) -> dict:
    """Write the configuration and return its ``Boundaries``."""
    return json.loads(Path(sim.write_config()).read_text())["Boundaries"]


def _thickness(sim, layer: str) -> float:
    """The thickness of a stack layer, in um."""
    lyr = sim._resolve_stack().layers[layer]
    return float(lyr.zmax - lyr.zmin)


def _mesh_cpw_driven(folder: Path, *, planar: bool = False) -> DrivenSim:
    """The docs' gpdk CPW line with CPW ports, coarse."""
    sim = DrivenSim()
    sim.set_output_dir(str(folder))
    sim.set_geometry(_make_cpw_component())
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=50.0, z_above=100.0, z_below=100.0)
    sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
    sim.mesh(preset="coarse", planar_conductors=planar)
    return sim


def _mesh_cpw_eigen(folder: Path) -> EigenmodeSim:
    """The same line as an eigenmode simulation with the same ports."""
    sim = EigenmodeSim()
    sim.set_output_dir(str(folder))
    sim.set_geometry(_make_cpw_component())
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=50.0, z_above=100.0, z_below=100.0)
    sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_eigenmode(num_modes=3, target=50e9)
    sim.mesh(preset="coarse")
    return sim


@pytest.fixture(scope="module")
def cpw_shell(tmp_path_factory) -> DrivenSim:
    """Driven gpdk CPW line, conductors as shells."""
    return _mesh_cpw_driven(tmp_path_factory.mktemp("sc-shell") / "sim")


@pytest.fixture(scope="module")
def cpw_sheet(tmp_path_factory) -> DrivenSim:
    """Driven gpdk CPW line, conductors as planar sheets."""
    return _mesh_cpw_driven(tmp_path_factory.mktemp("sc-sheet") / "sim", planar=True)


@pytest.fixture(scope="module")
def cpw_eigen(tmp_path_factory) -> EigenmodeSim:
    """Eigenmode gpdk CPW line, conductors as shells."""
    return _mesh_cpw_eigen(tmp_path_factory.mktemp("sc-eigen") / "sim")


@pytest.fixture(scope="module")
def qpdk_planar(tmp_path_factory):
    """A zero-thickness superconductor layer (QPDK-style stack), planar sheets."""
    return _make_qpdk_sim(
        _make_qpdk_component(),
        _make_qpdk_stack(),
        tmp_path_factory.mktemp("sc-qpdk"),
    )


@pytest.fixture(scope="module")
def mode_2d(tmp_path_factory) -> BoundaryModeSim:
    """A 2D cross-section with conductors on ``metal1`` and ``metal3``."""
    return meshed_2d_sim(tmp_path_factory.mktemp("sc-2d"))


@pytest.fixture(scope="module")
def symmetric_half(tmp_path_factory) -> DrivenSim:
    """A GSG half model with wave ports and a PMC plane through the signal."""
    return _symmetry_mesh_sim(
        tmp_path_factory.mktemp("sc-sym"), gssg=False, kind="pmc", max_size=True
    )


def _group_pgs(sim, *names: str) -> list[list[int]]:
    """Attribute lists of conductor groups, by group name."""
    groups = sim.mesh_groups["conductor_surfaces"]
    return [[groups[name]["phys_group"]] for name in names]


class TestRealMeshes:
    def test_driven_shell(self, cpw_shell):
        with superconducting(cpw_shell, "metal1", london_depth=0.1):
            boundaries = _boundaries(cpw_shell)
        face = sc.face_inductance(0.1, _thickness(cpw_shell, "metal1"))
        assert boundaries["Impedance"] == [
            {"Attributes": a, "Ls": face}
            for a in _group_pgs(cpw_shell, "metal1_xy", "metal1_z")
        ]
        assert boundaries["Conductivity"] == []
        assert "PEC" not in boundaries
        assert "Crack" not in json.dumps(boundaries)

    def test_the_same_sim_is_normal_metal_again_afterwards(self, cpw_shell):
        boundaries = _boundaries(cpw_shell)
        assert "Impedance" not in boundaries
        assert all(e["Crack"] is True for e in boundaries["Conductivity"])
        assert len(boundaries["Conductivity"]) == 2

    def test_driven_planar_sheet(self, cpw_sheet):
        with superconducting(cpw_sheet, "metal1", london_depth=0.1, thickness=0.1):
            boundaries = _boundaries(cpw_sheet)
        pec = cpw_sheet.mesh_groups["pec_surfaces"]["metal1"]["phys_group"]
        assert boundaries["Impedance"] == [
            {"Attributes": [pec], "Ls": sc.sheet_inductance(0.1, 0.1)}
        ]
        assert boundaries["Conductivity"] == []
        assert "PEC" not in boundaries

    def test_qpdk_style_zero_thickness_needs_a_thickness(self, qpdk_planar):
        with (
            superconducting(qpdk_planar, "superconductor", london_depth=0.1),
            pytest.raises(ValueError, match="zero thickness"),
        ):
            qpdk_planar.write_config()

    def test_qpdk_style_all_conductors_superconducting_passes_validation(
        self, qpdk_planar
    ):
        """No Conductivity or PEC entry is left: the validation counts Impedance."""
        with superconducting(
            qpdk_planar, "superconductor", london_depth=0.1, thickness=0.1
        ):
            boundaries = _boundaries(qpdk_planar)  # validates the mesh and config
            validation = qpdk_planar.validate_mesh()
        assert "PEC" not in boundaries
        assert boundaries["Conductivity"] == []
        pec = qpdk_planar.mesh_groups["pec_surfaces"]["superconductor"]["phys_group"]
        assert boundaries["Impedance"] == [
            {"Attributes": [pec], "Ls": sc.sheet_inductance(0.1, 0.1)}
        ]
        assert validation.valid, validation.errors

    def test_the_contact_check_counts_superconducting_shell_faces(self, cpw_shell):
        """Lumped ports touch conductors that are Impedance surfaces only."""
        with superconducting(cpw_shell, "metal1", london_depth=0.1):
            _boundaries(cpw_shell)
            contacts = check_lumped_port_contact(cpw_shell.output_dir)
        assert contacts
        assert all(count > 0 for count in contacts.values())

    def test_eigenmode(self, cpw_eigen):
        with superconducting(cpw_eigen, "metal1", london_depth=0.1):
            boundaries = _boundaries(cpw_eigen)
        face = sc.face_inductance(0.1, _thickness(cpw_eigen, "metal1"))
        assert boundaries["Impedance"] == [
            {"Attributes": a, "Ls": face}
            for a in _group_pgs(cpw_eigen, "metal1_xy", "metal1_z")
        ]
        assert boundaries["Conductivity"] == []
        assert "Crack" not in json.dumps(boundaries)

    def test_boundary_mode_both_layers(self, mode_2d):
        with (
            superconducting(mode_2d, "metal1", london_depth=0.1),
            superconducting(mode_2d, "metal3", london_depth=0.2),
        ):
            boundaries = _boundaries(mode_2d)
        entries = boundaries["Impedance"]
        assert entries == [
            {
                "Attributes": [
                    mode_2d.mesh_groups["conductor_surfaces"][name]["phys_group"]
                ],
                "Ls": sc.face_inductance(lam, _thickness(mode_2d, name)),
            }
            for name, lam in (("metal1", 0.1), ("metal3", 0.2))
        ]
        assert "Conductivity" not in boundaries
        assert "Crack" not in json.dumps(boundaries)

    def test_boundary_mode_one_layer_keeps_the_other_conductivity(self, mode_2d):
        with superconducting(mode_2d, "metal3", london_depth=0.2):
            boundaries = _boundaries(mode_2d)
        groups = mode_2d.mesh_groups["conductor_surfaces"]
        assert [e["Attributes"] for e in boundaries["Conductivity"]] == [
            [groups["metal1"]["phys_group"]]
        ]
        assert [e["Attributes"] for e in boundaries["Impedance"]] == [
            [groups["metal3"]["phys_group"]]
        ]

    def test_boundary_mode_with_pec_conductors(self, mode_2d, caplog):
        mode_2d.conductor_boundary = "pec"
        try:
            with (
                superconducting(mode_2d, "metal3", london_depth=0.2),
                caplog.at_level("WARNING"),
            ):
                boundaries = _boundaries(mode_2d)
        finally:
            mode_2d.conductor_boundary = "conductivity"
        groups = mode_2d.mesh_groups["conductor_surfaces"]
        assert "Impedance" not in boundaries
        assert "Conductivity" not in boundaries
        assert set(boundaries["PEC"]["Attributes"]) == {
            groups["metal1"]["phys_group"],
            groups["metal3"]["phys_group"],
        }
        assert "perfect conductors" in caplog.text

    def test_symmetry_plane(self, symmetric_half):
        with superconducting(symmetric_half, "metal1", london_depth=0.1):
            boundaries = _boundaries(symmetric_half)
        plane = symmetric_half.mesh_groups["boundary_surfaces"]["symmetry"][
            "phys_group"
        ]
        plane = plane if isinstance(plane, list) else [plane]
        assert boundaries["PMC"] == {"Attributes": sorted(plane)}
        face = sc.face_inductance(0.1, _thickness(symmetric_half, "metal1"))
        shell = symmetric_half.mesh_groups["conductor_surfaces"]
        assert boundaries["Impedance"] == [
            {"Attributes": [shell[name]["phys_group"]], "Ls": face}
            for name in sorted(shell)
        ]
        assert boundaries["Conductivity"] == []
        assert not set(plane) & {
            a for e in boundaries["Impedance"] for a in e["Attributes"]
        }

    def test_symmetry_plane_with_wave_ports_follows_the_policy(self, symmetric_half):
        for policy in ("pec", "inherit"):
            symmetric_half.set_driven(
                fmin=1e9, fmax=100e9, num_points=5, waveport_boundary=policy
            )
            with superconducting(symmetric_half, "metal1", london_depth=0.1):
                boundaries = _boundaries(symmetric_half)
            attributes = {a for e in boundaries["Impedance"] for a in e["Attributes"]}
            plane = symmetric_half.mesh_groups["boundary_surfaces"]["symmetry"][
                "phys_group"
            ]
            plane = set(plane if isinstance(plane, list) else [plane])
            if policy == "inherit":
                assert "WavePortPEC" not in boundaries
            else:
                listed = set(boundaries["WavePortPEC"]["Attributes"])
                assert attributes <= listed
                assert not plane & listed  # the PMC plane stays natural


class TestSavedSettingsAndCache:
    def test_setting_round_trips_through_saved_settings(self, cpw_eigen):
        with superconducting(cpw_eigen, "metal1", london_depth=0.15, thickness=0.5):
            written = _boundaries(cpw_eigen)["Impedance"]
            restored = EigenmodeSim.model_validate_json(cpw_eigen.model_dump_json())
            assert restored.superconductors == cpw_eigen.superconductors
            # a restored sim has lost its private mesh state: hand it over
            restored._last_mesh_result = cpw_eigen._last_mesh_result
            restored._last_ports = cpw_eigen._last_ports
            restored._stack_kwargs = dict(cpw_eigen._stack_kwargs)
            restored._airbox_config = dict(cpw_eigen._airbox_config)
            restored._output_dir = cpw_eigen._output_dir
            assert _boundaries(restored)["Impedance"] == written

    def test_a_sim_without_a_mesh_round_trips_too(self):
        plain = DrivenSim()
        plain.set_superconductor("metal1", london_depth=0.1, thickness=0.2)
        restored = DrivenSim.model_validate_json(plain.model_dump_json())
        assert restored.superconductors == {
            "metal1": sc.SuperconductorConfig(london_depth=0.1, thickness=0.2)
        }

    def test_the_cache_key_changes_with_the_london_depth(self, cpw_eigen):
        folder = cpw_eigen.output_dir

        def key() -> str:
            _boundaries(cpw_eigen)
            return compute_input_hash(folder, "palace")

        with superconducting(cpw_eigen, "metal1", london_depth=0.1):
            first = key()
            assert key() == first  # deterministic
            cpw_eigen.superconductors["metal1"].london_depth = 0.2
            second = key()
        normal = key()
        assert len({first, second, normal}) == 3


class TestMeshReuse:
    def test_the_setting_travels_with_the_mesh(self, cpw_eigen, tmp_path):
        with superconducting(cpw_eigen, "metal1", london_depth=0.1, thickness=0.5):
            eigen_entries = _boundaries(cpw_eigen)["Impedance"]
            driven = DrivenSim(absorbing_boundary=False)
            driven.set_output_dir(str(tmp_path / "driven"))
            driven.add_cpw_port(
                "o1",
                layer="metal1",
                s_width=10,
                gap_width=6,
                length=5.0,
                excited=True,
            )
            driven.reuse_mesh(cpw_eigen)
            driven.set_driven(fmin=1e9, fmax=100e9, num_points=5)
            assert driven.superconductors == cpw_eigen.superconductors
            assert driven.superconductors is not cpw_eigen.superconductors
            assert (
                driven.superconductors["metal1"]
                is not cpw_eigen.superconductors["metal1"]
            )
            assert _boundaries(driven)["Impedance"] == eigen_entries

            # not a mesh input: changing it afterwards is no fingerprint error
            driven.superconductors["metal1"].london_depth = 0.2
            changed = _boundaries(driven)["Impedance"]
            face = sc.face_inductance(0.2, 0.5)
            assert [e["Ls"] for e in changed] == [face, face]
            assert cpw_eigen.superconductors["metal1"].london_depth == 0.1

    def test_a_different_setting_of_the_receiver_is_replaced_with_a_warning(
        self, cpw_eigen, tmp_path, caplog
    ):
        driven = DrivenSim(absorbing_boundary=False)
        driven.set_output_dir(str(tmp_path / "driven"))
        driven.add_cpw_port(
            "o1", layer="metal1", s_width=10, gap_width=6, length=5.0, excited=True
        )
        driven.set_superconductor("metal3", london_depth=0.3)
        with (
            superconducting(cpw_eigen, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            driven.reuse_mesh(cpw_eigen)
        assert set(driven.superconductors) == {"metal1"}
        assert "['metal3']" in caplog.text
        assert "replaced" in caplog.text

    def test_a_source_without_the_setting_clears_the_receivers(
        self, cpw_eigen, tmp_path
    ):
        driven = DrivenSim(absorbing_boundary=False)
        driven.set_output_dir(str(tmp_path / "driven"))
        driven.add_cpw_port(
            "o1", layer="metal1", s_width=10, gap_width=6, length=5.0, excited=True
        )
        driven.set_superconductor("metal1", london_depth=0.3)
        driven.reuse_mesh(cpw_eigen)
        assert driven.superconductors == {}
        assert "Impedance" not in _boundaries(driven)

    def test_an_identical_setting_is_not_a_warning(self, cpw_eigen, tmp_path, caplog):
        driven = DrivenSim(absorbing_boundary=False)
        driven.set_output_dir(str(tmp_path / "driven"))
        driven.add_cpw_port(
            "o1", layer="metal1", s_width=10, gap_width=6, length=5.0, excited=True
        )
        driven.set_superconductor("metal1", london_depth=0.1)
        with (
            superconducting(cpw_eigen, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            driven.reuse_mesh(cpw_eigen)
        assert "replaced" not in caplog.text


class TestWarnings:
    """Logged once per layer and setting at ``write_config``."""

    @staticmethod
    def _messages(caplog) -> list[str]:
        return [
            r.getMessage() for r in caplog.records if "layer 'metal1'" in r.getMessage()
        ]

    def test_a_thick_sheet_warns_once(self, cpw_sheet, caplog):
        # t = 0.7 um
        with (
            superconducting(cpw_sheet, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            cpw_sheet.write_config()
            cpw_sheet.write_config()
        messages = self._messages(caplog)
        assert len(messages) == 1
        assert "planar sheet with t/lambda = 7" in messages[0]

    def test_a_thin_sheet_does_not_warn(self, cpw_sheet, caplog):
        with (
            superconducting(cpw_sheet, "metal1", london_depth=0.1, thickness=0.1),
            caplog.at_level("WARNING"),
        ):
            cpw_sheet.write_config()
        assert not self._messages(caplog)

    def test_a_changed_setting_warns_again(self, cpw_sheet, caplog):
        with (
            superconducting(cpw_sheet, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            cpw_sheet.write_config()
            cpw_sheet.superconductors["metal1"].london_depth = 0.05
            cpw_sheet.write_config()
        assert len(self._messages(caplog)) == 2

    def test_a_thin_shell_warns_and_a_thick_one_does_not(self, cpw_shell, caplog):
        # t/lambda = 7
        with (
            superconducting(cpw_shell, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            cpw_shell.write_config()
        assert not self._messages(caplog)
        caplog.clear()
        # t/lambda = 0.7
        with (
            superconducting(cpw_shell, "metal1", london_depth=1.0),
            caplog.at_level("WARNING"),
        ):
            cpw_shell.write_config()
        messages = self._messages(caplog)
        assert len(messages) == 1
        assert "3D shell with t/lambda = 0.7" in messages[0]

    def test_an_explicit_thickness_that_differs_from_the_mesh_warns(
        self, cpw_shell, caplog
    ):
        with (
            superconducting(cpw_shell, "metal1", london_depth=0.1, thickness=0.5),
            caplog.at_level("WARNING"),
        ):
            cpw_shell.write_config()
        messages = self._messages(caplog)
        assert len(messages) == 1
        assert "differs from the thickness 0.7 um" in messages[0]

    def test_an_eigenmode_thin_shell_mentions_spurious_modes(self, cpw_eigen, caplog):
        with (
            superconducting(cpw_eigen, "metal1", london_depth=1.0),
            caplog.at_level("WARNING"),
        ):
            cpw_eigen.write_config()
        assert any("spurious" in m for m in self._messages(caplog))

    def test_a_layer_without_surfaces_warns(self, cpw_shell, caplog):
        with (
            superconducting(cpw_shell, "metal2", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            cpw_shell.write_config()
        messages = [r.getMessage() for r in caplog.records]
        assert any("'metal2' has no surface in the mesh" in m for m in messages)

    @pytest.mark.parametrize("policy", ["pec", "inherit"])
    def test_wave_ports_name_their_policy(self, symmetric_half, caplog, policy):
        symmetric_half.set_driven(
            fmin=1e9, fmax=100e9, num_points=5, waveport_boundary=policy
        )
        with (
            superconducting(symmetric_half, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            symmetric_half.write_config()
        text = " ".join(self._messages(caplog))
        if policy == "pec":
            assert "treats the superconducting surfaces as PEC" in text
        else:
            assert "was not verified" in text

    def test_the_mode_solve_note_is_absent_without_wave_ports(self, cpw_shell, caplog):
        with (
            superconducting(cpw_shell, "metal1", london_depth=0.1),
            caplog.at_level("WARNING"),
        ):
            cpw_shell.write_config()
        assert "port mode" not in caplog.text


class TestRunLocalVersion:
    @staticmethod
    def _run(monkeypatch, tmp_path, *, tag, answer, runtime, caplog):
        if runtime == "gsim":
            _fake_gsim_runtime(monkeypatch, tmp_path)
        else:
            _setup_local_palace(monkeypatch, tmp_path)
        monkeypatch.delenv("PALACE_SIF", raising=False)
        monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
        monkeypatch.setenv("PALACETOOLKIT_PALACE_CPU_TAG", tag)
        output_dir = tmp_path / "sim"
        sim = DrivenSim()
        sim._last_mesh_result = _mesh_result(1_000, 1_000)
        _setup_sim(sim, output_dir)
        sim.set_superconductor("metal", london_depth=0.1)
        config = {"Boundaries": {"Impedance": [{"Attributes": [4], "Ls": 1e-13}]}}
        (output_dir / "config.json").write_text(json.dumps(config))

        def fake_run(cmd, **_kwargs):
            if cmd[-1] == "--version":
                return SimpleNamespace(
                    stdout=answer or "", stderr="", returncode=0 if answer else 1
                )
            return SimpleNamespace(stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        with caplog.at_level("WARNING"):
            sim.run_local(num_processes=1, verbose=False)
        return caplog.text

    @pytest.mark.parametrize(
        ("runtime", "tag", "answer", "warned"),
        [
            ("gsim", "0.18.0", None, "Palace 0.18"),
            ("gsim", "0.19.0", None, None),
            ("user", "0.18.0", "Palace v0.19.0", None),
            ("user", "0.19.0", "Palace v0.18.0", "Palace 0.18"),
            ("user", "0.18.0", None, "an unknown Palace version"),
        ],
    )
    def test_old_or_unknown_palace_is_named(
        self, monkeypatch, tmp_path, caplog, runtime, tag, answer, warned
    ):
        text = self._run(
            monkeypatch,
            tmp_path,
            tag=tag,
            answer=answer,
            runtime=runtime,
            caplog=caplog,
        )
        if warned is None:
            assert "superconducting Impedance entries" not in text
        else:
            assert "superconducting Impedance entries" in text
            assert warned in text


# ---------------------------------------------------------------------------
# 5. The docs stay true
# ---------------------------------------------------------------------------

DOCS = Path(__file__).resolve().parents[2] / "docs"


class TestDocs:
    def test_the_guide_example_writes_two_face_entries(self, tmp_path, monkeypatch):
        """The example of docs/api/palace.md runs as written, up to write_config()."""
        import re

        text = (DOCS / "api" / "palace.md").read_text(encoding="utf-8")
        _, found, rest = text.partition("## Superconducting layers")
        assert found
        match = re.search(r"```python\n(.*?)```", rest, flags=re.DOTALL)
        assert match is not None
        block = match.group(1)
        assert ".run(" not in block
        monkeypatch.chdir(tmp_path)
        namespace: dict = {"__name__": "docs_superconductor"}
        exec(compile(block, "docs/api/palace.md", "exec"), namespace)  # noqa: S102
        entries = namespace["config"]["Boundaries"]["Impedance"]
        face = sc.face_inductance(0.1, 0.7)  # gpdk metal1 is 0.7 um thick
        assert [e["Ls"] for e in entries] == [pytest.approx(face)] * 2
        assert namespace["sim"].superconductors["metal1"].london_depth == 0.1

    def test_the_conductor_page_quotes_the_hollow_shell_errors(self):
        import re

        text = (DOCS / "palace_conductors.md").read_text(encoding="utf-8")
        row = re.search(
            r"^\| Hollow shell against the exact one-sided film \| (.+) \|$",
            text,
            flags=re.MULTILINE,
        )
        assert row, "the table of the hollow-shell errors is missing"
        quoted = [float(v.strip().rstrip("%")) for v in row.group(1).split("|")]
        computed = []
        for u in (1.0, 2.0, 4.0, 10.0):
            c = coth(u / 2)
            computed.append(100 * (c * (c + u) / (2 * c + u) / coth(u) - 1))
        assert quoted == pytest.approx(computed, abs=0.06)

    def test_the_conductor_page_lists_the_cloud_runs(self):
        text = (DOCS / "palace_conductors.md").read_text(encoding="utf-8")
        assert "### Cloud runs" in text
        assert "No Palace run has been made" not in text
        for row in (
            "| 2D boundary mode | 0.2 | 0.212 | 0.209 |",
            "| 2D boundary mode | 0.1 | 0.069 | 0.067 |",
            "| 3D resonator, planar sheet | 0.2 | 0.211 | 0.214 |",
            "| 3D resonator, shell | 0.2 | 0.202 | 0.209 |",
        ):
            assert row in text

    def test_the_page_uses_text_fences_for_the_new_section(self):
        """Only the two executed snippets are python blocks (test_conductor_docs)."""
        text = (DOCS / "palace_conductors.md").read_text(encoding="utf-8")
        section = text.split("## Superconducting layers (London penetration depth)")[1]
        section = section.split("\n## ")[0]
        assert "```python" not in section
        assert "```text" in section
