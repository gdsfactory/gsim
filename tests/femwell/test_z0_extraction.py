"""Marks-Williams power-current Z_0 extraction from femwell RF modes.

Validated against the analytic impedance of a dielectric-filled coaxial
line whose inner conductor is a finite-conductivity copper disk: the
power-current definition ``Z_0 = 2P / |I|^2`` must land on
``(eta_0 / (2 pi sqrt(eps_r))) ln(b/a)`` to within the good-conductor
corrections, and it must be invariant to the arbitrary normalization of
the mode fields.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import mu_0 as MU0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import Conductor
from gsim.femwell.adapter import solve_modes, z0_power_current

pytest.importorskip("femwell")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")

# Coax geometry (um) and analysis frequency.
R_INNER = 1.0
R_OUTER = 3.0
EPS_DIELECTRIC = 2.25
SIGMA_COPPER = 5.8e7
FREQ_HZ = 100e9
ETA0 = MU0 * C0
Z0_ANALYTIC = ETA0 / (2 * np.pi * np.sqrt(EPS_DIELECTRIC)) * np.log(R_OUTER / R_INNER)


#: The inner conductor, named to the current integral by its Region and
#: bounding box; the model says which integral reads it.
INNER = Conductor("conductor", ((-R_INNER, R_INNER), (-R_INNER, R_INNER)), "volume")
INNER_PEC = Conductor("conductor", INNER.extent, "pec")


@pytest.fixture(scope="module")
def coax_mode(tmp_path_factory):
    import gmsh

    path = tmp_path_factory.mktemp("femwell-z0") / "coax.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        inner = occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        occ.fragment([(2, outer)], [(2, inner)])
        occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        by_area = sorted(
            (gmsh.model.occ.getMass(2, tag), tag) for _dim, tag in surfaces
        )
        pg_core = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, pg_core, "conductor")
        pg_diel = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, pg_diel, "dielectric")
        # Pin the size sources so the mesh does not depend on a
        # developer ~/.gmsh-options; gsim's own mesh paths pin these.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        # Resolve the ~0.2 um skin depth at the conductor surface.
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
    eps_conductor = 1.0 - 1j * SIGMA_COPPER / (omega * EPS0)
    modes = solve_modes(
        path,
        epsilon={
            "conductor": eps_conductor,
            "dielectric": EPS_DIELECTRIC + 0j,
        },
        wavelength_um=C0 / FREQ_HZ * 1e6,
        num_modes=1,
        metallic_boundaries=True,
    )
    return modes[0], path


class TestZ0PowerCurrent:
    def test_coax_matches_analytic(self, coax_mode):
        mode, mesh = coax_mode
        z0 = z0_power_current(mode, frequency_hz=FREQ_HZ, conductor=INNER, mesh=mesh)
        # Measured 3.2% high with a small negative reactance — the expected
        # finite-conductivity correction at 100 GHz; the margin covers mesh
        # variance across gmsh versions.
        assert abs(z0.real - Z0_ANALYTIC) < 0.08 * Z0_ANALYTIC
        assert abs(z0.imag) < 0.1 * Z0_ANALYTIC

    def test_normalization_invariant(self, coax_mode):
        from dataclasses import replace

        mode, mesh = coax_mode
        factor = 3.7 * np.exp(1j * 0.61)
        scaled = replace(mode, E=mode.E * factor, H=mode.H * factor)
        z0 = z0_power_current(mode, frequency_hz=FREQ_HZ, conductor=INNER, mesh=mesh)
        z0_scaled = z0_power_current(
            scaled, frequency_hz=FREQ_HZ, conductor=INNER, mesh=mesh
        )
        assert z0_scaled == pytest.approx(z0, rel=1e-12)

    def test_a_conductor_that_is_not_on_the_mesh_is_reported(self, coax_mode):
        mode, mesh = coax_mode
        with pytest.raises(ValueError, match="'signal' not found"):
            z0_power_current(
                mode,
                frequency_hz=FREQ_HZ,
                conductor=Conductor("signal", INNER.extent, "volume"),
                mesh=mesh,
            )

    def test_a_volume_conductor_is_not_a_hole(self, coax_mode):
        """Naming a meshed conductor as perfect finds no outline to loop."""
        mode, mesh = coax_mode
        with pytest.raises(ValueError, match="not a hole"):
            z0_power_current(mode, frequency_hz=FREQ_HZ, conductor=INNER_PEC, mesh=mesh)


# The same coax with a perfect inner conductor: the disk is left out of the
# meshed domain, so the mode has no conduction current at all and the only
# current there is Ampere's contour integral of H around the hole. A PEC
# coax also has no finite-conductivity correction, so it must land on the
# analytic impedance far more tightly than the copper one does.
#
# Solved at order 2 on purpose. The contour integral reads H on the domain
# boundary, and femwell derives H from the curl of E, so an order-1 solve
# leaves it piecewise constant exactly where the integral is taken: the
# same three meshes come out 29%, 17% and 8% high at order 1 and within
# 0.4% at order 2, at every one of them.
@pytest.fixture(scope="module")
def pec_coax_mode(tmp_path_factory):
    import gmsh

    path = tmp_path_factory.mktemp("femwell-z0-pec") / "pec_coax.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        inner = occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        annulus, _ = occ.cut([(2, outer)], [(2, inner)])
        occ.synchronize()
        pg = gmsh.model.addPhysicalGroup(2, [tag for _dim, tag in annulus])
        gmsh.model.setPhysicalName(2, pg, "dielectric")
        # Pin the size sources so the mesh does not depend on a
        # developer ~/.gmsh-options; gsim's own mesh paths pin these.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.12)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

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


class TestContourCurrent:
    def test_the_pec_coax_carries_the_tem_mode(self, pec_coax_mode):
        """The domain is homogeneous, so the TEM index is exactly sqrt(eps)."""
        mode, _mesh = pec_coax_mode
        n_eff = complex(mode.n_eff)
        assert n_eff.real == pytest.approx(np.sqrt(EPS_DIELECTRIC), rel=2e-3)
        assert abs(n_eff.imag) < 1e-6

    def test_the_contour_current_reaches_the_analytic_impedance(self, pec_coax_mode):
        """A perfect conductor has no volume current; Ampere's law has one.

        The round inner conductor is a hole in the mesh, found by its
        bounding box: every domain-boundary facet inside it is its
        outline, since the outer wall lies well beyond.
        """
        mode, mesh = pec_coax_mode
        z0 = z0_power_current(
            mode, frequency_hz=FREQ_HZ, conductor=INNER_PEC, mesh=mesh
        )
        assert z0.real == pytest.approx(Z0_ANALYTIC, rel=0.01)
        assert abs(z0.imag) < 0.01 * Z0_ANALYTIC

    def test_the_contour_current_is_normalization_invariant(self, pec_coax_mode):
        from dataclasses import replace

        mode, mesh = pec_coax_mode
        factor = 3.7 * np.exp(1j * 0.61)
        scaled = replace(mode, E=mode.E * factor, H=mode.H * factor)
        z0 = z0_power_current(
            mode, frequency_hz=FREQ_HZ, conductor=INNER_PEC, mesh=mesh
        )
        z0_scaled = z0_power_current(
            scaled, frequency_hz=FREQ_HZ, conductor=INNER_PEC, mesh=mesh
        )
        assert z0_scaled == pytest.approx(z0, rel=1e-12)

    def test_a_perfect_conductor_has_no_region_to_integrate_over(self, pec_coax_mode):
        """Naming the hole as a meshed Region finds nothing on the mesh."""
        mode, mesh = pec_coax_mode
        with pytest.raises(ValueError, match="not found"):
            z0_power_current(mode, frequency_hz=FREQ_HZ, conductor=INNER, mesh=mesh)
