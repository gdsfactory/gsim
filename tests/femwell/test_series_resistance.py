"""Series resistance of a lossy metal line at the default element order.

An aluminium strip in oxide inside a perfectly conducting box, solved at
1 GHz. The strip is 0.5 um thick against a 2.6 um skin depth, so its
current is close to uniform and the line's series resistance must be the
strip's DC resistance, ``1 / (sigma W t)``: the box wall returns the
current with no loss of its own. The resistance is read the way a line
solve reads it, ``R' = Re(gamma Z0)`` from the Mode's index and its
power-current impedance.

The mesh is fine at the strip (50 nm) and coarse in the oxide (2 um),
like a typical cross-section mesh. At element order 1 this reads R'
8.0 % low; at order 2 the same mesh lands within 5e-4. The error comes
from the coarse oxide, not from the strip. Refining only around the
strip does not fix it: at 25 nm, order 1 reads 12.6 % low and order 2
3.2e-3 low. Capping the oxide at 0.5 um does: order 1 then reads 1.0 %
low and order 2 7e-5 high. So order 2 reduces this error by one to two
orders of magnitude on a coarse oxide but does not remove it. The test
holds the default order to the tolerance it reaches on this mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import Conductor, select_line_mode
from gsim.common.transmission_line import line_params_from_neff
from gsim.femwell import conductor_index_bound, solve_modes, z0_power_current

pytest.importorskip("femwell")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")

WIDTH_UM = 2.0
THICKNESS_UM = 0.5
SIGMA_AL = 3.8e7
EPS_OXIDE = 3.9
FREQ_HZ = 1e9
BOX_UM = (12.0, 6.0)
R_DC_OHM_PER_M = 1.0 / (SIGMA_AL * WIDTH_UM * THICKNESS_UM * 1e-12)

STRIP = Conductor(
    "metal",
    ((-WIDTH_UM / 2, WIDTH_UM / 2), (-THICKNESS_UM / 2, THICKNESS_UM / 2)),
    "volume",
)


def _write_strip_mesh(path, *, near_um: float, far_um: float):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        box = occ.addRectangle(-BOX_UM[0] / 2, -BOX_UM[1] / 2, 0.0, *BOX_UM)
        strip = occ.addRectangle(
            -WIDTH_UM / 2, -THICKNESS_UM / 2, 0.0, WIDTH_UM, THICKNESS_UM
        )
        occ.fragment([(2, box)], [(2, strip)])
        occ.synchronize()
        by_area = sorted(
            (gmsh.model.occ.getMass(2, tag), tag)
            for _dim, tag in gmsh.model.getEntities(2)
        )
        pg_metal = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, pg_metal, "metal")
        pg_oxide = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, pg_oxide, "oxide")
        # Pin the size sources so the mesh does not depend on a
        # developer ~/.gmsh-options; gsim's own mesh paths pin these.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", far_um)
        # near_um elements in and around the strip, far_um in the oxide.
        field = gmsh.model.mesh.field
        near = field.add("Box")
        field.setNumber(near, "VIn", near_um)
        field.setNumber(near, "VOut", far_um)
        field.setNumber(near, "XMin", -WIDTH_UM / 2 - 0.1)
        field.setNumber(near, "XMax", WIDTH_UM / 2 + 0.1)
        field.setNumber(near, "YMin", -THICKNESS_UM / 2 - 0.1)
        field.setNumber(near, "YMax", THICKNESS_UM / 2 + 0.1)
        field.setNumber(near, "Thickness", 1.0)
        field.setAsBackgroundMesh(near)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="module")
def strip_mesh(tmp_path_factory):
    """50 nm at the strip, 2 um in the far oxide."""
    path = tmp_path_factory.mktemp("femwell-series") / "strip.msh"
    return _write_strip_mesh(path, near_um=0.05, far_um=2.0)


@pytest.fixture(scope="module")
def capped_strip_mesh(tmp_path_factory):
    """25 nm at the strip, oxide capped at 0.5 um: the mesh of #498."""
    path = tmp_path_factory.mktemp("femwell-series-capped") / "strip.msh"
    return _write_strip_mesh(path, near_um=0.025, far_um=0.5)


def _epsilon() -> dict[str, complex]:
    omega = 2 * np.pi * FREQ_HZ
    return {
        "metal": 1.0 - 1j * SIGMA_AL / (omega * EPS0),
        "oxide": EPS_OXIDE + 0j,
    }


def _series_resistance(
    mesh_path, *, num_modes: int = 4, max_index=None, **solve_kwargs
) -> float:
    modes = solve_modes(
        mesh_path,
        epsilon=_epsilon(),
        wavelength_um=C0 / FREQ_HZ * 1e6,
        num_modes=num_modes,
        metallic_boundaries=True,
        n_guess=3.0,
        **solve_kwargs,
    )
    mode = select_line_mode(modes, max_index=max_index)
    z0 = z0_power_current(mode, frequency_hz=FREQ_HZ, conductor=STRIP, mesh=mesh_path)
    params = line_params_from_neff(
        np.array([FREQ_HZ]), [complex(mode.n_eff)], z0_ohm=[z0]
    )
    return float(params.rlgc["R"][0])


class TestSeriesResistance:
    def test_default_order_reaches_the_dc_resistance(self, strip_mesh):
        r_series = _series_resistance(strip_mesh)
        assert r_series == pytest.approx(R_DC_OHM_PER_M, rel=3e-3)

    def test_order_one_is_still_available_and_reads_low(self, strip_mesh):
        """The first-order solve stays callable; on this mesh it is percent low."""
        r_series = _series_resistance(strip_mesh, order=1)
        assert r_series < (1 - 3e-2) * R_DC_OHM_PER_M


class TestManyModes:
    """More Modes than the line needs also bring Modes inside the metal (#498).

    On the capped mesh at order 2, a 12-Mode solve returns Modes with
    ``|n_eff|`` of 1e4 to 5e4 confined to the strip, some inside the loss
    bound; the slowest-Mode rule picked one and read R' about 1e9 off. The
    conductor index bound rejects them and the line Mode is read again.
    """

    def test_the_bound_selects_the_line_mode(self, capped_strip_mesh):
        r_series = _series_resistance(
            capped_strip_mesh,
            num_modes=12,
            max_index=conductor_index_bound(_epsilon()),
        )
        assert r_series == pytest.approx(R_DC_OHM_PER_M, rel=3e-3)
