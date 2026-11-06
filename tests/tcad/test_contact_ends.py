"""A Contact whose end lies on an Interface holds its potential there.

A four-Region diode slab (``n_pad | n_rib | p_rib | p_pad``) whose two
electrodes are exactly as wide as the pads, so each Contact ends where
its pad meets the rib, on a declared Interface, and, when the oxide is
solved, on the oxide too. The other Regions' nodes at those end points
touch the same metal.

The mesh is a structured grid of right triangles, on which DEVSIM's
finite-box discretisation of the Laplacian equals the P1 finite-element
one. The silicon is barely doped and the small-signal solve runs far
above its dielectric relaxation rate, so the carriers do not follow the
AC signal and the capacitance is that of the dielectric alone. It is
compared with an independent P1 Laplace solve (scikit-fem) on the same
grid, every node of each Contact held at its electrode's potential.

Requires DEVSIM: run with ``pytest -m tcad_local``.
"""

from __future__ import annotations

from itertools import pairwise

import meshio
import numpy as np
import pytest

from gsim.tcad import Contact, Interface, Materials, Solver, StepDoping
from gsim.tcad.api import TCADMesh
from gsim.tcad.device import Device, check_layout
from gsim.tcad.mesh import UM_TO_CM
from tests._helpers import skip_without_devsim

skip_without_devsim()
skfem = pytest.importorskip("skfem")

pytestmark = pytest.mark.tcad_local

H = 0.05  # um, grid step
SPANS = {  # y extent (um) of each doped Region
    "n_pad": (0.0, 0.3),
    "n_rib": (0.3, 0.6),
    "p_rib": (0.6, 0.9),
    "p_pad": (0.9, 1.2),
}
DOPANT = {"n_pad": "donor", "n_rib": "donor", "p_rib": "acceptor", "p_pad": "acceptor"}
SLAB = (0.0, 0.2)  # z extent (um) of the silicon
OXIDE = (0.3, 0.55)  # y extent (um) of the oxide over the rib
# 1e11 cm^-3 relaxes in about 5 ns; at 100 GHz the carriers stay put.
DOPING_CM3 = 1e11
FREQ_HZ = 1e11
EPS_SI = 11.1 * 8.85e-12  # F/m, DEVSIM's simple_physics silicon
EPS_OX = 3.9 * 8.8541878128e-12  # F/m, gsim's insulator permittivity


def _cells(oxide: bool) -> dict[str, list[tuple[float, float]]]:
    """Lower-left corner of every grid cell (um), by Region.

    The oxide lies on the n side of the rib, from the cathode to short of
    the Junction, so it meets no Interface but at the Contact's end. The
    electrodes themselves are not meshed.
    """
    cells: dict[str, list[tuple[float, float]]] = {}
    ys = np.arange(0.0, 1.2 - 1e-9, H)
    for y in ys:
        mid = y + H / 2
        region = next(r for r, (y0, y1) in SPANS.items() if y0 < mid < y1)
        for z in np.arange(SLAB[0], SLAB[1] - 1e-9, H):
            cells.setdefault(region, []).append((y, z))
        if oxide and OXIDE[0] < mid < OXIDE[1]:
            for z in np.arange(0.2, 0.4 - 1e-9, H):
                cells.setdefault("sio2", []).append((y, z))
    return cells


def _write_mesh(directory, *, oxide: bool):
    """Write the grid as gmsh 2.2 in um and in cm; return it for the reference."""
    index: dict[tuple[int, int], int] = {}

    def node(y: float, z: float) -> int:
        key = (round(y / H), round(z / H))
        return index.setdefault(key, len(index))

    triangles, regions = [], []
    names = list(SPANS) + (["sio2"] if oxide else [])
    for region, corners in _cells(oxide).items():
        for y, z in corners:
            a, b = node(y, z), node(y + H, z)
            c, d = node(y + H, z + H), node(y, z + H)
            triangles += [(a, b, c), (a, c, d)]
            regions += [names.index(region) + 1] * 2

    def segments(y0, z0, y1, z1):
        steps = round(max(abs(y1 - y0), abs(z1 - z0)) / H)
        along = [
            (y0 + (y1 - y0) * i / steps, z0 + (z1 - z0) * i / steps)
            for i in range(steps + 1)
        ]
        return [(node(*p), node(*q)) for p, q in pairwise(along)]

    top = SLAB[1]
    lines = {
        "cathode": segments(0.0, top, 0.3, top),
        "anode": segments(0.9, top, 1.2, top),
        "n_link": segments(0.3, 0.0, 0.3, top),
        "junction": segments(0.6, 0.0, 0.6, top),
        "p_link": segments(0.9, 0.0, 0.9, top),
    }
    if oxide:
        lines["interface_n_rib_sio2"] = segments(OXIDE[0], top, OXIDE[1], top)

    points = np.zeros((len(index), 3))
    for (iy, iz), i in index.items():
        points[i, :2] = iy * H, iz * H
    tags = {name: i + 1 for i, name in enumerate(names)}
    tags |= {name: len(names) + i + 1 for i, name in enumerate(lines)}
    line_cells = [pair for curve in lines.values() for pair in curve]
    line_tags = [tags[name] for name, curve in lines.items() for _ in curve]
    physical = [np.array(line_tags), np.array(regions)]

    def write(path, scale):
        meshio.write(
            str(path),
            meshio.Mesh(
                points * scale,
                [("line", np.array(line_cells)), ("triangle", np.array(triangles))],
                cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical},
                field_data={
                    name: np.array([tag, 2 if name in names else 1])
                    for name, tag in tags.items()
                },
            ),
            file_format="gmsh22",
            binary=False,
        )
        return path

    groups = {
        "volumes": dict.fromkeys(names),
        "pec_surfaces": dict.fromkeys(("cathode_metal", "anode_metal")),
        "contact_lines": dict.fromkeys(("cathode", "anode")),
        "interface_lines": dict.fromkeys(("n_link", "junction", "p_link")),
        "interface_surfaces": dict.fromkeys(
            name for name in lines if name.startswith("interface_")
        ),
    }
    mesh = TCADMesh(
        mesh_path=write(directory / "grid.msh", 1.0),
        devsim_mesh_path=write(directory / "grid_cm.msh", UM_TO_CM),
        groups=groups,
    )
    permittivity = [EPS_SI] * len(SPANS) + [EPS_OX]
    eps = np.array([permittivity[r - 1] for r in regions])
    return mesh, points[:, :2] * 1e-6, np.array(triangles), eps, lines


def _p1_capacitance_f_per_m(points, triangles, eps, lines) -> float:
    """Cathode-to-anode capacitance per meter by scikit-fem P1."""
    from skfem.helpers import dot, grad

    basis = skfem.Basis(skfem.MeshTri(points.T, triangles.T), skfem.ElementTriP1())

    @skfem.BilinearForm
    def laplace(u, v, w):
        return w["eps"] * dot(grad(u), grad(v))

    stiffness = laplace.assemble(
        basis, eps=np.repeat(eps[:, None], basis.X.shape[-1], axis=1)
    )
    hot = np.unique(lines["cathode"])
    cold = np.unique(lines["anode"])
    u = np.zeros(len(points))
    u[hot] = 1.0
    u = skfem.solve(
        *skfem.condense(stiffness, np.zeros(len(points)), x=u, D=np.r_[hot, cold])
    )
    return float(u @ (stiffness @ u))


def _device(
    mesh: TCADMesh, *, oxide: bool, doping_cm3: dict[str, float] | None = None
) -> Device:
    doping = [
        StepDoping(
            region=region,
            dopant_type=kind,
            concentration_cm3=(doping_cm3 or {}).get(region, DOPING_CM3),
        )
        for region, kind in DOPANT.items()
    ]
    contacts = [
        Contact(name="cathode", semiconductor="n_pad", electrode="cathode_metal"),
        Contact(name="anode", semiconductor="p_pad", electrode="anode_metal"),
    ]
    interfaces = [
        Interface(name="n_link", region_a="n_pad", region_b="n_rib"),
        Interface(name="junction", region_a="n_rib", region_b="p_rib"),
        Interface(name="p_link", region_a="p_rib", region_b="p_pad"),
    ]
    materials = Materials(insulators={"sio2": 3.9} if oxide else {})
    return Device(
        mesh=mesh,
        materials=materials,
        doping=doping,
        contacts=contacts,
        interfaces=interfaces,
        solver=Solver(small_signal_freq_hz=FREQ_HZ, junction_freq_hz=FREQ_HZ),
        layout=check_layout(
            mesh.groups,
            doping=doping,
            insulators=materials.insulators,
            contacts=contacts,
            interfaces=interfaces,
        ),
    )


@pytest.mark.parametrize("oxide", [False, True], ids=["silicon", "with-oxide"])
def test_the_capacitance_holds_every_node_of_the_contact(tmp_path, oxide):
    mesh, points, triangles, eps, lines = _write_mesh(tmp_path, oxide=oxide)
    device = _device(mesh, oxide=oxide)
    try:
        point = device.solve_point(0.0, "cathode")
    finally:
        device.release()
    # The carriers barely respond: the admittance is all but lossless.
    assert abs(point.admittance_s_per_m.real) < 1e-3 * point.admittance_s_per_m.imag
    assert point.capacitance_f_per_m == pytest.approx(
        _p1_capacitance_f_per_m(points, triangles, eps, lines), rel=1e-5, abs=0.0
    )


def _potential_at(device: Device, region: str, point_um) -> float:
    import devsim

    values = {
        name: np.asarray(
            devsim.get_node_model_values(device=device.name, region=region, name=name)
        )
        for name in ("x", "y", "Potential")
    }
    at = np.hypot(
        values["x"] - point_um[0] * UM_TO_CM, values["y"] - point_um[1] * UM_TO_CM
    )
    assert at.min() < 1e-12
    return float(values["Potential"][np.argmin(at)])


def test_a_doping_step_at_a_contact_end_holds_each_side_neutral(tmp_path):
    # The cathode's pad is ten times more doped than the rib it ends on: a
    # high-low step at the Contact's end, as a contact implant beside a
    # plus implant.
    import devsim

    doping = {"n_pad": 1e18, "n_rib": 1e17, "p_rib": 1e17, "p_pad": 1e17}
    mesh, *_ = _write_mesh(tmp_path, oxide=False)
    device = _device(mesh, oxide=False, doping_cm3=doping)
    try:
        point = device.solve_point(1.0, "cathode")
        v_t = float(
            devsim.get_parameter(device=device.name, region="n_pad", name="V_t")
        )
        n_i = float(
            devsim.get_parameter(device=device.name, region="n_pad", name="n_i")
        )
        end = (SPANS["n_pad"][1], SLAB[1])
        # Each side of the step at the end is an ohmic contact of its own
        # doping, as every node of a Contact is: neutral, at the metal's
        # Fermi level. The potential steps by the high-low built-in voltage
        # there, V_t ln(10), as it does across the step a Debye length in.
        for region in ("n_pad", "n_rib"):
            assert _potential_at(device, region, end) == pytest.approx(
                1.0 + v_t * np.arcsinh(doping[region] / (2 * n_i)), abs=1e-9
            )
    finally:
        device.release()
    # The end's carrier currents are the cathode's: what leaves the anode
    # enters the cathode.
    currents = point.currents_a_per_cm
    assert abs(currents["cathode"]) > 0
    assert currents["cathode"] + currents["anode"] == pytest.approx(
        0.0, abs=1e-6 * abs(currents["cathode"])
    )
