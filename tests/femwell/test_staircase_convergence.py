"""n_eff convergence in strip count N on actual mode solves.

The staircase route bins a continuous carrier-induced index profile into N
piecewise-constant strips (the representation Palace can express). The
hermetic staircase tests show the *material* profile converges with N; this
suite closes the loop on the mode solve itself: femwell solves the same
mesh once with the continuous per-element epsilon (the reference) and once
per strip count, and n_eff must converge to the reference as N grows.
femwell stands in for Palace here because the shipped cross-validation
gate (tests/femwell/test_cross_validation.py) pins both solvers to the
same n_eff on identical piecewise-constant materials.

What it does not cover: the mesh is fixed and only the *materials* vary,
so nothing here can see a staircase whose geometry is not the device.
That is tests/modulator/test_representation_gate.py.
"""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.femwell.adapter import solve_modes
from gsim.modulator.staircase import staircase_profile

pytest.importorskip("femwell")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")

WL_UM = 1.55
N_SI = 3.48
N_CLAD = 1.444
CORE_Y0, CORE_Y1 = 0.0, 0.22
# Carrier-depletion-like index dip, strongly y-dependent inside the core.
DN_PEAK = -0.08


def _dn_profile(y: np.ndarray) -> np.ndarray:
    """Continuous index change across the core thickness."""
    t = np.clip((y - CORE_Y0) / (CORE_Y1 - CORE_Y0), 0.0, 1.0)
    return DN_PEAK * np.exp(-(((t - 0.35) / 0.25) ** 2))


@pytest.fixture(scope="module")
def strip_mesh(tmp_path_factory):
    import gmsh

    path = tmp_path_factory.mktemp("femwell-staircase") / "strip.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        clad = occ.addRectangle(-1.5, -1.0, 0.0, 3.0, 2.22)
        core = occ.addRectangle(-0.25, CORE_Y0, 0.0, 0.5, CORE_Y1 - CORE_Y0)
        occ.fragment([(2, clad)], [(2, core)])
        occ.synchronize()
        by_area = sorted(
            (gmsh.model.occ.getMass(2, tag), tag)
            for _dim, tag in gmsh.model.getEntities(2)
        )
        pg_core = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, pg_core, "core")
        pg_clad = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, pg_clad, "clad")
        # Pin the size sources rather than inherit them: a developer
        # ~/.gmsh-options can otherwise drive element size from geometry
        # points and extend it from the boundary, which grades the core
        # and costs the staircase its convergence. gsim's own mesh paths
        # pin the same two options.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.06)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _element_epsilon(mesh_path, dn_of_y) -> np.ndarray:
    """Per-element epsilon: core follows n(y) = N_SI + dn(y), clad constant."""
    mio = meshio.read(str(mesh_path))
    tri_blocks = [
        (block.data, np.asarray(phys))
        for block, phys in zip(
            mio.cells, mio.cell_data.get("gmsh:physical", []), strict=False
        )
        if block.type == "triangle"
    ]
    tris = np.vstack([data for data, _p in tri_blocks])
    phys = np.concatenate([p for _d, p in tri_blocks])
    group_tags = {
        str(name): int(np.asarray(data)[0])
        for name, data in mio.field_data.items()
        if int(np.asarray(data)[1]) == 2
    }
    centroids = mio.points[tris][:, :, :2].mean(axis=1)
    eps = np.full(tris.shape[0], N_CLAD**2, dtype=np.complex128)
    in_core = phys == group_tags["core"]
    eps[in_core] = (N_SI + dn_of_y(centroids[in_core, 1])) ** 2
    return eps


def _solve(mesh_path, eps) -> complex:
    modes = solve_modes(mesh_path, epsilon=eps, wavelength_um=WL_UM, num_modes=1)
    return complex(modes[0].n_eff)


class TestStaircaseModeConvergence:
    def test_n_eff_converges_to_continuous_reference(self, strip_mesh):
        n_ref = _solve(strip_mesh, _element_epsilon(strip_mesh, _dn_profile))

        y_samples = np.linspace(CORE_Y0, CORE_Y1, 201)
        errors: dict[int, float] = {}
        for n_bins in (1, 4, 16):
            edges, means = staircase_profile(
                y_samples, _dn_profile(y_samples), n_bins=n_bins
            )

            def dn_staircased(y, edges=edges, means=means):
                idx = np.clip(np.searchsorted(edges, y) - 1, 0, len(means) - 1)
                return means[idx]

            n_eff = _solve(strip_mesh, _element_epsilon(strip_mesh, dn_staircased))
            errors[n_bins] = abs(n_eff.real - n_ref.real)

        # The reference problem is meaningful: the dip moves n_eff by much
        # more than the finest staircase error.
        n_undoped = _solve(strip_mesh, _element_epsilon(strip_mesh, np.zeros_like))
        assert abs(n_ref.real - n_undoped.real) > 10 * errors[16]
        # Monotone convergence toward the continuous profile.
        assert errors[4] < errors[1]
        assert errors[16] <= errors[4]
        assert errors[16] < 5e-4
