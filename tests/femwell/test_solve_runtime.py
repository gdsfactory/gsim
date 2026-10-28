"""femwell solve tests, gated on the femwell/skfem runtime being installed."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.femwell.adapter import solve_modes

pytest.importorskip("femwell")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")


def build_strip_mesh(path, *, clad_width=3.0, clad_height=2.22):
    """Triangle mesh of a strip waveguide in a clad box of a given size (um)."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        clad = occ.addRectangle(
            -clad_width / 2, -(clad_height - 0.22) / 2, 0.0, clad_width, clad_height
        )
        core = occ.addRectangle(-0.25, 0.0, 0.0, 0.5, 0.22)
        occ.fragment([(2, clad)], [(2, core)])
        occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        # The core rectangle is by far the smallest fragment.
        by_area = sorted(
            (gmsh.model.occ.getMass(2, tag), tag) for _dim, tag in surfaces
        )
        core_tags = [by_area[0][1]]
        clad_tags = [tag for _area, tag in by_area[1:]]
        pg_core = gmsh.model.addPhysicalGroup(2, core_tags)
        gmsh.model.setPhysicalName(2, pg_core, "core")
        pg_clad = gmsh.model.addPhysicalGroup(2, clad_tags)
        gmsh.model.setPhysicalName(2, pg_clad, "clad")
        # Pin the size sources so the mesh does not depend on a
        # developer ~/.gmsh-options; gsim's own mesh paths pin these.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.08)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="module")
def strip_mesh(tmp_path_factory):
    """Structured triangle mesh of a strip waveguide cross-section (um)."""
    return build_strip_mesh(tmp_path_factory.mktemp("femwell-runtime") / "strip.msh")


class TestPiecewiseConstantSolve:
    def test_strip_waveguide_fundamental_mode(self, strip_mesh):
        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
            num_modes=1,
        )
        n_eff = np.real(modes[0].n_eff)
        # Fundamental mode of a 500x220 nm silicon strip is well guided.
        assert 1.444 < n_eff < 3.48
        assert n_eff > 2.0

    def test_unknown_region_raises(self, strip_mesh):
        with pytest.raises(ValueError, match="no_such_region"):
            solve_modes(
                strip_mesh,
                epsilon={"no_such_region": 12.0 + 0j},
                wavelength_um=1.55,
            )

    def test_region_missing_from_epsilon_raises(self, strip_mesh):
        # A region left out of the map would otherwise be solved at eps = 0.
        with pytest.raises(ValueError, match="clad"):
            solve_modes(
                strip_mesh,
                epsilon={"core": 3.48**2 + 0j},
                wavelength_um=1.55,
            )


class TestContinuousSolve:
    def test_elementwise_epsilon_matches_piecewise(self, strip_mesh):
        from gsim.femwell.adapter import elementwise_epsilon

        mesh = meshio.read(str(strip_mesh))
        tris = np.vstack([b.data for b in mesh.cells if b.type == "triangle"])
        centroids = mesh.points[tris][:, :, :2].mean(axis=1)
        # Build continuous samples reproducing the piecewise structure.
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        eps_samples = np.where(
            (np.abs(xs) < 0.25) & (ys > 0.0) & (ys < 0.22),
            3.48**2,
            1.444**2,
        ).astype(complex)
        eps_elements = elementwise_epsilon(strip_mesh, xs, ys, eps_samples)

        modes_pc = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )
        modes_cont = solve_modes(strip_mesh, epsilon=eps_elements, wavelength_um=1.55)
        assert np.real(modes_cont[0].n_eff) == pytest.approx(
            np.real(modes_pc[0].n_eff), rel=5e-3
        )


class TestBoundaryFieldRatio:
    def test_a_roomy_box_contains_the_mode(self, strip_mesh):
        from gsim.femwell.adapter import boundary_field_ratio

        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        assert boundary_field_ratio(modes[0]) < 1e-2

    def test_a_box_barely_wider_than_the_core_does_not(self, tmp_path):
        from gsim.femwell.adapter import boundary_field_ratio

        clipped = build_strip_mesh(
            tmp_path / "narrow.msh", clad_width=0.7, clad_height=0.4
        )
        modes = solve_modes(
            clipped,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        assert boundary_field_ratio(modes[0]) > 1e-2


class TestFieldFractionOutside:
    """How much of a Mode sits off a region that carries the physics.

    A Staircase carries its carrier response only where the strips are,
    so the fraction of the mode outside them is what says whether the
    strip extent was chosen for this mode or for the geometry alone.
    """

    def test_an_interval_holding_the_core_holds_most_of_the_mode(self, strip_mesh):
        from gsim.femwell.adapter import field_fraction_outside

        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        # The core spans -0.25..0.25; a micron either side of it holds the
        # evanescent tails too.
        assert field_fraction_outside(modes[0], (-1.0, 1.0)) < 0.05

    def test_an_interval_beside_the_core_holds_almost_none_of_it(self, strip_mesh):
        from gsim.femwell.adapter import field_fraction_outside

        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        assert field_fraction_outside(modes[0], (1.0, 1.4)) > 0.9

    def test_narrowing_the_interval_can_only_raise_the_fraction(self, strip_mesh):
        from gsim.femwell.adapter import field_fraction_outside

        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        wide = field_fraction_outside(modes[0], (-1.0, 1.0))
        narrow = field_fraction_outside(modes[0], (-0.25, 0.25))
        assert narrow > wide

    def test_the_vertical_axis_is_selectable(self, strip_mesh):
        from gsim.femwell.adapter import field_fraction_outside

        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        # The core spans z = 0..0.22 and the mode is bound to it.
        assert field_fraction_outside(modes[0], (-1.0, 1.0), axis=1) < 0.05

    def test_a_descending_interval_is_reported(self, strip_mesh):
        from gsim.femwell.adapter import field_fraction_outside

        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )

        with pytest.raises(ValueError, match="ascending"):
            field_fraction_outside(modes[0], (1.0, -1.0))
