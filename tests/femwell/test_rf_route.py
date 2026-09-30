"""Carrier-derived sigma(x, y) solved to an RF mode and into the report.

The end-to-end RF contract of the TCAD TW-MZM workflow: a conductive
(carrier-loaded) cross-section is solved with femwell, the solver's
complex n_eff supplies gamma, the Marks-Williams power-current integral
supplies Z_0, and both feed ``twmzm_figures_of_merit`` — no analytic
loaded-line stand-in anywhere in the chain.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import Conductor
from gsim.common.transmission_line import line_params_from_neff
from gsim.femwell.adapter import solve_modes, z0_power_current
from gsim.modulator.report import OpticalPhaseSweep, twmzm_figures_of_merit

pytest.importorskip("femwell")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")

SIGMA_S_PER_M = 2e4  # depleted-junction-scale silicon conductivity
EPS_SI = 11.9
EPS_OX = 3.9
FREQS_HZ = np.array([10e9, 30e9])


@pytest.fixture(scope="module")
def doped_strip_mesh(tmp_path_factory):
    import gmsh

    path = tmp_path_factory.mktemp("femwell-rf") / "doped.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        clad = occ.addRectangle(-4.0, -2.0, 0.0, 8.0, 4.0)
        core = occ.addRectangle(-2.0, -0.11, 0.0, 4.0, 0.22)
        occ.fragment([(2, clad)], [(2, core)])
        occ.synchronize()
        by_area = sorted(
            (gmsh.model.occ.getMass(2, tag), tag)
            for _dim, tag in gmsh.model.getEntities(2)
        )
        pg_core = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, pg_core, "si")
        pg_clad = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, pg_clad, "oxide")
        # Pin the size sources so the mesh does not depend on a
        # developer ~/.gmsh-options; gsim's own mesh paths pin these.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.35)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _solve_line_params(mesh_path):
    n_eff = []
    z0 = []
    for freq in FREQS_HZ:
        omega = 2 * np.pi * freq
        eps_si = EPS_SI - 1j * SIGMA_S_PER_M / (omega * EPS0)
        modes = solve_modes(
            mesh_path,
            epsilon={"si": eps_si, "oxide": EPS_OX + 0j},
            wavelength_um=C0 / freq * 1e6,
            num_modes=1,
            metallic_boundaries=True,
        )
        n_eff.append(complex(modes[0].n_eff))
        z0.append(
            z0_power_current(
                modes[0],
                frequency_hz=freq,
                conductor=Conductor("si", ((-2.0, 2.0), (-0.11, 0.11)), "volume"),
                mesh=mesh_path,
            )
        )
    return line_params_from_neff(FREQS_HZ, n_eff, z0_ohm=z0)


class TestRFRouteEndToEnd:
    def test_solver_derived_gamma_and_z0_reach_the_report(self, doped_strip_mesh):
        rf = _solve_line_params(doped_strip_mesh)

        # The conductive load must show up as loss and slow-wave behavior.
        assert np.all(rf.alpha_rf_np_m > 0)
        assert np.all(rf.n_rf > np.sqrt(EPS_OX) * 0.5)
        assert np.all(rf.z0_ohm.real > 0)

        optical = OpticalPhaseSweep(
            voltages_v=np.array([0.0, 1.0, 2.0]),
            dn_eff=np.array([0.0, -1e-4, -1.8e-4]),
            wavelength_um=1.55,
            n_group=3.8,
        )
        report = twmzm_figures_of_merit(rf, optical, length_m=2e-3)
        assert report.response.shape == FREQS_HZ.shape
        assert np.isfinite(report.response).all()
        assert np.all(np.isfinite(report.rlgc["C"]))
        assert report.rlgc["C"][0] > 0
        assert np.all(report.vpi_l_vcm > 0)
