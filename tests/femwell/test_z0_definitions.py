"""Power-current, power-voltage and voltage-current impedance of femwell RF modes.

Three definitions of a line's characteristic impedance, from the same
mode: ``Z_PI = 2 P / |I|^2``, ``Z_PV = |V|^2 / (2 P*)`` and
``Z_VI = V / I`` (``exp(+i omega t)``; ``V = -integral E_t . dl`` along a
path from the signal conductor to the return).

The three are tied by an algebraic identity,

    Z_PI * conj(Z_PV) = (2 P / |I|^2) * (|V|^2 / (2 P)) = |V|^2 / |I|^2 = |Z_VI|^2,

which holds for any ``P``, ``V`` and ``I`` that enter the three the same
way. It checks that the code reads one ``P``, one ``V`` and one ``I``; it
cannot check the fields. What depends on the fields and the path is
whether the three *agree*, which is ``P = V I* / 2`` and is reported as
``2 P / (V I*) - 1`` rather than asserted to be zero.

The analytic reference is a dielectric-filled coaxial line,
``Z_0 = eta_0 ln(b/a) / (2 pi sqrt(eps_r))``, with a perfect inner conductor
(current by Ampere's contour) and with a finite-conductivity copper one
(current by conduction, a lossy line).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import mu_0 as MU0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import (
    Conductor,
    z0_power_current,
    z0_power_voltage,
    z0_voltage_current,
)

R_INNER = 1.0
R_OUTER = 3.0
EPS_DIELECTRIC = 2.25
SIGMA_COPPER = 5.8e7
FREQ_HZ = 100e9
Z0_ANALYTIC = (
    MU0 * C0 / (2 * np.pi * np.sqrt(EPS_DIELECTRIC)) * np.log(R_OUTER / R_INNER)
)

INNER_PEC = Conductor("conductor", ((-R_INNER, R_INNER), (-R_INNER, R_INNER)), "pec")
INNER_VOLUME = Conductor("conductor", INNER_PEC.extent, "volume")

#: Radial path from the inner conductor's surface to the outer wall along
#: +x. Both ends are vertices of the circular outlines the mesher draws,
#: which are inscribed polygons: a path at another angle would end where
#: a polygon chord has left the circle, outside the mesh.
RADIAL_PATH = ((R_INNER, 0.0), (R_OUTER, 0.0))


class TestCommonDefinitions:
    """The shared formulas, with no mode behind them."""

    def test_power_voltage_of_a_matched_line(self):
        # |V|^2 / (2 P*) = 100 / 2 = 50 ohm for P = 1 W-equivalent, V = 10.
        assert z0_power_voltage(1.0, 10.0) == pytest.approx(50.0)

    def test_power_voltage_conjugates_the_power(self):
        # 100 / (2 (1 - 1j)) = 25 + 25j
        assert z0_power_voltage(1.0 + 1.0j, 10.0) == pytest.approx(25.0 + 25.0j)

    def test_a_lossy_line_has_negative_reactance(self):
        """exp(+i omega t): P = Z |I|^2 / 2 with Im(Z) < 0 gives Im(Z_PV) < 0."""
        z = 50.0 - 2.0j
        current = 0.3 * np.exp(0.4j)
        power = 0.5 * z * abs(current) ** 2
        voltage = z * current
        assert z0_power_voltage(power, voltage) == pytest.approx(z)
        assert z0_power_voltage(power, voltage).imag < 0.0

    def test_power_voltage_ignores_the_phase_and_sign_of_the_voltage(self):
        reference = z0_power_voltage(1.0 - 0.1j, 10.0)
        assert z0_power_voltage(1.0 - 0.1j, -10.0) == reference
        assert z0_power_voltage(1.0 - 0.1j, 10.0j) == reference
        assert z0_power_voltage(1.0 - 0.1j, 10.0 * np.exp(0.7j)) == pytest.approx(
            reference, rel=1e-15
        )

    def test_power_voltage_flips_back_a_mode_travelling_the_other_way(self):
        assert z0_power_voltage(-(1.0 - 0.1j), 10.0) == z0_power_voltage(
            1.0 - 0.1j, 10.0
        )

    def test_power_voltage_refuses_zeros(self):
        with pytest.raises(ValueError, match="no power"):
            z0_power_voltage(0.0, 10.0)
        with pytest.raises(ValueError, match="no voltage"):
            z0_power_voltage(1.0, 0.0)

    def test_voltage_current_is_the_ratio(self):
        assert z0_voltage_current(10.0, 0.2) == pytest.approx(50.0)
        assert z0_voltage_current(10.0 - 1.0j, 0.2) == pytest.approx(
            (10.0 - 1.0j) / 0.2
        )

    def test_voltage_current_ignores_a_common_phase(self):
        reference = z0_voltage_current(10.0 - 1.0j, 0.2)
        phase = np.exp(1.3j)
        assert z0_voltage_current((10.0 - 1.0j) * phase, 0.2 * phase) == pytest.approx(
            reference, rel=1e-15
        )

    def test_voltage_current_has_a_positive_real_part_either_way_round(self):
        """The raw ratio flips with the path, the current convention and
        the propagation direction, none of which is physical."""
        forward = z0_voltage_current(10.0 - 1.0j, 0.2)
        assert z0_voltage_current(-(10.0 - 1.0j), 0.2) == forward
        assert z0_voltage_current(10.0 - 1.0j, -0.2) == forward
        assert forward.real > 0.0

    def test_voltage_current_refuses_zeros(self):
        with pytest.raises(ValueError, match="no current"):
            z0_voltage_current(10.0, 0.0)
        with pytest.raises(ValueError, match="no voltage"):
            z0_voltage_current(0.0, 0.2)

    def test_the_identity_between_the_three_holds_for_any_integrals(self):
        """``Z_PI conj(Z_PV) = |Z_VI|^2``, whatever P, V and I are.

        Nothing here makes the three agree: P is not V I* / 2.
        """
        rng = np.random.default_rng(7)
        for _ in range(50):
            power, voltage, current = (complex(*rng.normal(size=2)) for _ in range(3))
            z_pi = z0_power_current(power, current)
            z_pv = z0_power_voltage(power, voltage)
            z_vi = z0_voltage_current(voltage, current)
            assert z_pi * np.conj(z_pv) == pytest.approx(abs(z_vi) ** 2, rel=1e-12)


def _mesh_pec_coax(path, size):
    """Annulus between R_INNER and R_OUTER; the inner disk is a hole."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        inner = occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        annulus, _ = occ.cut([(2, outer)], [(2, inner)])
        occ.synchronize()
        group = gmsh.model.addPhysicalGroup(2, [tag for _dim, tag in annulus])
        gmsh.model.setPhysicalName(2, group, "dielectric")
        # Pin the size sources so the mesh does not depend on a developer
        # ~/.gmsh-options; gsim's own mesh paths pin these.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def _solve_pec_coax(tmp_path_factory, size):
    pytest.importorskip("femwell")
    pytest.importorskip("skfem")
    pytest.importorskip("gmsh")
    from gsim.femwell.adapter import solve_modes

    path = tmp_path_factory.mktemp("femwell-z0-defs") / f"pec_coax_{size}.msh"
    _mesh_pec_coax(path, size)
    modes = solve_modes(
        path,
        epsilon={"dielectric": EPS_DIELECTRIC + 0j},
        wavelength_um=C0 / FREQ_HZ * 1e6,
        num_modes=1,
        order=2,
        metallic_boundaries=True,
        n_guess=np.sqrt(EPS_DIELECTRIC),
    )
    return modes[0], path


@pytest.fixture(scope="module")
def pec_coax(tmp_path_factory):
    return _solve_pec_coax(tmp_path_factory, 0.12)


@pytest.fixture(scope="module")
def pec_coax_ladder(tmp_path_factory):
    return [
        (size, *_solve_pec_coax(tmp_path_factory, size)) for size in (0.4, 0.2, 0.1)
    ]


@pytest.fixture(scope="module")
def copper_coax(tmp_path_factory):
    """The same coax with a finite-conductivity copper inner conductor."""
    pytest.importorskip("femwell")
    pytest.importorskip("skfem")
    gmsh = pytest.importorskip("gmsh")
    from gsim.femwell.adapter import solve_modes

    path = tmp_path_factory.mktemp("femwell-z0-defs-cu") / "coax.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        inner = occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        occ.fragment([(2, outer)], [(2, inner)])
        occ.synchronize()
        by_area = sorted(
            (occ.getMass(2, tag), tag) for _dim, tag in gmsh.model.getEntities(2)
        )
        core = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, core, "conductor")
        diel = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, diel, "dielectric")
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.35)
        field = gmsh.model.mesh.field
        ball = field.add("Ball")
        field.setNumber(ball, "Radius", R_INNER + 0.15)
        field.setNumber(ball, "VIn", 0.07)
        field.setNumber(ball, "VOut", 0.35)
        field.setAsBackgroundMesh(ball)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    omega = 2 * np.pi * FREQ_HZ
    modes = solve_modes(
        path,
        epsilon={
            "conductor": 1.0 - 1j * SIGMA_COPPER / (omega * EPS0),
            "dielectric": EPS_DIELECTRIC + 0j,
        },
        wavelength_um=C0 / FREQ_HZ * 1e6,
        num_modes=1,
        metallic_boundaries=True,
    )
    return modes[0], path


def _three(mode, mesh, conductor, path=RADIAL_PATH):
    from gsim.femwell.adapter import (
        z0_power_current as fw_z_pi,
    )
    from gsim.femwell.adapter import (
        z0_power_voltage as fw_z_pv,
    )
    from gsim.femwell.adapter import (
        z0_voltage_current as fw_z_vi,
    )

    z_pi = fw_z_pi(mode, frequency_hz=FREQ_HZ, conductor=conductor, mesh=mesh)
    z_pv = fw_z_pv(mode, voltage_path=path)
    z_vi = fw_z_vi(
        mode, frequency_hz=FREQ_HZ, conductor=conductor, mesh=mesh, voltage_path=path
    )
    return z_pi, z_pv, z_vi


class TestAnalyticCoax:
    def test_all_three_definitions_reach_the_analytic_impedance(self, pec_coax):
        """Lossless coax, order-2 solve on a 0.12 um mesh.

        Measured relative errors on this mesh: Z_PI 5.6e-3 (it reads H on
        the domain boundary, the same figure the existing contour test
        bounds at 1e-2), Z_PV 9e-4 and Z_VI 2.4e-3. All three within 1e-2.
        """
        mode, mesh = pec_coax
        for z in _three(mode, mesh, INNER_PEC):
            assert z.real == pytest.approx(Z0_ANALYTIC, rel=1e-2)
            assert abs(z.imag) < 1e-2 * Z0_ANALYTIC

    def test_the_power_voltage_impedance_converges_with_the_mesh(self, pec_coax_ladder):
        """Measured on the 0.4, 0.2, 0.1 um meshes: 9.6e-3, 2.5e-3 and
        6.6e-4, falling at each refinement."""
        errors = []
        for _size, mode, _mesh in pec_coax_ladder:
            from gsim.femwell.adapter import z0_power_voltage as fw_z_pv

            z = fw_z_pv(mode, voltage_path=RADIAL_PATH)
            errors.append(abs(z.real / Z0_ANALYTIC - 1.0))
        assert errors[0] > errors[1] > errors[2]
        assert errors[0] < 2e-2
        assert errors[2] < 2e-3

    def test_the_voltage_path_integral_is_exact_for_the_discrete_field(self, pec_coax):
        """Gauss-Legendre on each piece between mesh edges is exact from
        two points up, so more points change nothing but rounding."""
        from gsim.femwell.adapter import path_voltage

        mode, _mesh = pec_coax
        reference = path_voltage(mode, *RADIAL_PATH, points_per_element=8)
        for points in (1, 2, 3, 5):
            assert path_voltage(
                mode, *RADIAL_PATH, points_per_element=points
            ) == pytest.approx(reference, rel=1e-9)

    def test_the_identity_holds_on_real_fields_and_the_mismatch_is_small(
        self, pec_coax, record_property
    ):
        """``Z_PI conj(Z_PV) = |Z_VI|^2`` to rounding; the three agree to the
        size of ``2 P / (V I*) - 1``, which is reported, not asserted zero.

        Measured on this mesh: 3.3e-3 (Z_PI over Z_VI as ``2 P / (V I*)``).
        """
        from gsim.femwell.adapter import (
            _poynting_power,
            electrode_current,
            path_voltage,
        )

        mode, mesh = pec_coax
        z_pi, z_pv, z_vi = _three(mode, mesh, INNER_PEC)
        assert z_pi * np.conj(z_pv) == pytest.approx(abs(z_vi) ** 2, rel=1e-12)

        power = _poynting_power(mode)
        current = electrode_current(
            mode, frequency_hz=FREQ_HZ, conductor=INNER_PEC, mesh=mesh
        )
        voltage = path_voltage(mode, *RADIAL_PATH)
        mismatch = abs(2.0 * power / (voltage * np.conj(current)) - 1.0)
        record_property("two_p_over_v_i_conj_minus_one", float(mismatch))
        assert mismatch < 1e-2

    def test_normalization_and_phase_of_the_mode_cancel(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, mesh = pec_coax
        factor = 3.7 * np.exp(1j * 0.61)
        scaled = replace(mode, E=mode.E * factor, H=mode.H * factor)
        for before, after in zip(
            _three(mode, mesh, INNER_PEC), _three(scaled, mesh, INNER_PEC), strict=True
        ):
            assert after == pytest.approx(before, rel=1e-12)
        assert path_voltage(scaled, *RADIAL_PATH) == pytest.approx(
            factor * path_voltage(mode, *RADIAL_PATH), rel=1e-12
        )

    def test_the_sign_of_the_path_does_not_matter(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, mesh = pec_coax
        start, end = RADIAL_PATH
        assert path_voltage(mode, end, start) == pytest.approx(
            -path_voltage(mode, start, end), rel=1e-12
        )
        for forward, backward in zip(
            _three(mode, mesh, INNER_PEC),
            _three(mode, mesh, INNER_PEC, path=(end, start)),
            strict=True,
        ):
            assert backward == pytest.approx(forward, rel=1e-12)

    def test_the_voltage_is_in_the_scale_of_the_current(self, pec_coax):
        """A um mesh puts V and I both at 1e6 times SI: V / I is in ohms."""
        from gsim.femwell.adapter import electrode_current, path_voltage

        mode, mesh = pec_coax
        voltage = path_voltage(mode, *RADIAL_PATH)
        current = electrode_current(
            mode, frequency_hz=FREQ_HZ, conductor=INNER_PEC, mesh=mesh
        )
        assert abs(voltage / current) == pytest.approx(Z0_ANALYTIC, rel=1e-2)


class TestVoltagePathOnTheMesh:
    def test_a_path_through_the_perfect_conductor_hole_is_refused(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, _mesh = pec_coax
        with pytest.raises(ValueError, match="leaves the mesh"):
            path_voltage(mode, (-2.0, 0.0), (2.0, 0.0))

    def test_a_path_that_starts_in_the_hole_is_refused(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, _mesh = pec_coax
        with pytest.raises(ValueError, match="leaves the mesh"):
            path_voltage(mode, (0.0, 0.0), (2.0, 0.0))

    def test_a_path_past_the_outer_wall_is_refused(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, _mesh = pec_coax
        with pytest.raises(ValueError, match="leaves the mesh"):
            path_voltage(mode, (1.5, 0.0), (3.5, 0.0))

    def test_a_path_with_no_length_is_refused(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, _mesh = pec_coax
        with pytest.raises(ValueError, match="zero length"):
            path_voltage(mode, (2.0, 0.0), (2.0, 0.0))

    def test_a_quadrature_without_points_is_refused(self, pec_coax):
        from gsim.femwell.adapter import path_voltage

        mode, _mesh = pec_coax
        with pytest.raises(ValueError, match="points_per_element"):
            path_voltage(mode, *RADIAL_PATH, points_per_element=0)


class TestLossyLine:
    def test_a_lossy_line_has_negative_reactance_in_all_three(self, copper_coax):
        """Copper inner conductor at 100 GHz, exp(+i omega t): Im(Z) < 0.

        Measured: Z_PV = 45.96 - 2.219j ohm; the three agree to about 1e-4.
        """
        mode, mesh = copper_coax
        z_pi, z_pv, z_vi = _three(mode, mesh, INNER_VOLUME)
        for z in (z_pi, z_pv, z_vi):
            assert z.real > 0.0
            assert z.imag < 0.0
        # The same finite-conductivity correction the power-current test
        # bounds: real part a few percent above the lossless value.
        assert abs(z_pv.real - Z0_ANALYTIC) < 0.08 * Z0_ANALYTIC
        assert abs(z_pv - z_pi) < 1e-3 * abs(z_pi)
        assert abs(z_vi - z_pi) < 1e-3 * abs(z_pi)

    def test_the_voltage_does_not_depend_on_where_in_the_metal_the_path_starts(
        self, copper_coax
    ):
        """The metal is an equipotential to the skin-depth correction, so a
        path from the surface and one from the axis read the same."""
        from gsim.femwell.adapter import z0_power_voltage as fw_z_pv

        mode, _mesh = copper_coax
        from_surface = fw_z_pv(mode, voltage_path=((R_INNER, 0.0), (R_OUTER, 0.0)))
        from_axis = fw_z_pv(mode, voltage_path=((0.0, 0.0), (R_OUTER, 0.0)))
        assert from_axis == pytest.approx(from_surface, rel=1e-3)


@pytest.fixture(scope="module")
def square_mode():
    """A unit square with a square hole cut out, carrying a linear E_t."""
    skfem = pytest.importorskip("skfem")
    full = skfem.MeshTri.init_sqsymmetric().refined(3)
    centroids = full.p[:, full.t].mean(axis=1)
    inside_hole = (
        (centroids[0] > 0.4)
        & (centroids[0] < 0.6)
        & (centroids[1] > 0.4)
        & (centroids[1] < 0.6)
    )
    mesh = full.remove_elements(np.flatnonzero(inside_hole))
    basis = skfem.Basis(mesh, skfem.ElementTriN2() * skfem.ElementTriP2())
    (_, tangential), _ = basis.split(np.zeros(basis.N, dtype=complex))
    field = np.zeros(basis.N, dtype=complex)
    # Linear tangential field E_t = (x + 2i, y / 2 - 1), in the space.
    field[basis.split_indices()[0]] = tangential.project(
        lambda x: np.stack([x[0] + 2j, 0.5 * x[1] - 1.0 + 0j * x[0]]),
        dtype=complex,
    )
    return SimpleNamespace(basis=basis, E=field)


class TestPathPiecesOnAHoleyMesh:
    """The mesh check with no solver: a unit square with elements cut out."""

    def test_a_linear_field_is_integrated_exactly_along_a_slanted_path(
        self, square_mode
    ):
        from gsim.femwell.adapter import path_voltage

        a, b = np.array([0.1, 0.1]), np.array([0.9, 0.25])
        exact = -(
            0.5 * (b[0] ** 2 - a[0] ** 2)
            + 2j * (b[0] - a[0])
            + 0.25 * (b[1] ** 2 - a[1] ** 2)
            - (b[1] - a[1])
        )
        for points in (1, 3, 6):
            voltage = path_voltage(
                square_mode, tuple(a), tuple(b), points_per_element=points
            )
            assert voltage == pytest.approx(exact, abs=1e-12)

    def test_a_path_along_mesh_edges_is_integrated_exactly(self, square_mode):
        from gsim.femwell.adapter import path_voltage

        # The square's diagonal is a line of mesh edges.
        voltage = path_voltage(square_mode, (0.0, 0.0), (0.25, 0.25))
        exact = -(0.5 * 0.25**2 + 2j * 0.25 + 0.25 * 0.25**2 - 0.25)
        assert voltage == pytest.approx(exact, abs=1e-12)

    def test_a_path_across_the_hole_is_refused(self, square_mode):
        from gsim.femwell.adapter import path_voltage

        with pytest.raises(ValueError, match="leaves the mesh"):
            path_voltage(square_mode, (0.1, 0.5), (0.9, 0.5))

    def test_a_path_around_the_hole_is_not(self, square_mode):
        from gsim.femwell.adapter import path_voltage

        assert np.isfinite(path_voltage(square_mode, (0.1, 0.2), (0.9, 0.2)))

    def test_a_path_leaving_the_square_is_refused(self, square_mode):
        from gsim.femwell.adapter import path_voltage

        with pytest.raises(ValueError, match="leaves the mesh"):
            path_voltage(square_mode, (0.5, 0.1), (1.5, 0.1))
