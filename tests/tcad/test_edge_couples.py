"""Finite-box edge couples and node volumes on an unstructured mesh.

A carriers-off Laplace problem on a small mesh with obtuse triangles in
both of its dielectric Regions, solved by DEVSIM and by an independent P1
finite-element assembly (scikit-fem). Requires DEVSIM: run with
``pytest -m tcad_local``.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from tests._helpers import skip_without_devsim

skip_without_devsim()
skfem = pytest.importorskip("skfem")

pytestmark = pytest.mark.tcad_local

_UM = 1e-4  # cm
_EPS0 = 8.8541878128e-14  # F/cm
_PERMITTIVITY = {"lower": 3.9, "upper": 11.9}
_NX, _NY = 10, 10  # cells across 2 um by 1 um; the interface at row _NY // 2
_NAMES = ["lower", "upper", "bot", "top", "mid"]
_NAME_IDS = itertools.count()


def _mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points (um), triangles and each triangle's Region index.

    A 2 um by 1 um grid of 0.2 um by 0.1 um cells, each split along a
    random diagonal, whose interior nodes are jittered so that obtuse
    triangles appear in both Regions. Nodes on the walls, the contact
    sides and the interface row stay on their lines.
    """
    rng = np.random.default_rng(7)
    xs = np.linspace(0.0, 2.0, _NX + 1)
    ys = np.linspace(0.0, 1.0, _NY + 1)
    hx, hy = xs[1], ys[1]
    pts = np.array([(x, y) for y in ys for x in xs])
    for j in range(1, _NY):
        if j == _NY // 2:
            continue
        for i in range(1, _NX):
            pts[j * (_NX + 1) + i] += rng.uniform(-0.3, 0.3, 2) * (hx, hy)
    tris, labels = [], []
    for j in range(_NY):
        for i in range(_NX):
            a = j * (_NX + 1) + i
            b, c, d = a + 1, a + _NX + 1, a + _NX + 2
            pair = (
                [(a, b, d), (a, d, c)] if rng.random() < 0.5 else [(a, b, c), (b, d, c)]
            )
            tris += pair
            labels += [0 if j < _NY // 2 else 1] * 2
    tris = np.array(tris)
    u, v = pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]]
    assert np.all(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] > 0.0)
    return pts, tris, np.array(labels)


def _max_angle_deg(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    angles = []
    for k in range(3):
        o, i, j = tris[:, k], tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u, v = pts[i] - pts[o], pts[j] - pts[o]
        cos = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        angles.append(np.degrees(np.arccos(cos)))
    return np.max(angles, axis=0)


def _contact_edges(pts: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """Bottom contact on x <= 0.8 um, top contact on x >= 1.2 um, interface."""
    row = _NX + 1
    bottom = [(i, i + 1) for i in range(_NX) if pts[i + 1, 0] <= 0.8 + 1e-12]
    top0 = _NY * row
    top = [
        (top0 + i, top0 + i + 1) for i in range(_NX) if pts[top0 + i, 0] >= 1.2 - 1e-12
    ]
    mid0 = (_NY // 2) * row
    mid = [(mid0 + i, mid0 + i + 1) for i in range(_NX)]
    return {"bot": bottom, "top": top, "mid": mid}


@pytest.fixture(scope="module")
def mesh():
    return _mesh()


@pytest.fixture(scope="module")
def p1_capacitance(mesh) -> float:
    """Capacitance per cm of depth from scikit-fem P1, 1 V between contacts."""
    from skfem.helpers import dot, grad

    pts, tris, labels = mesh
    m = skfem.MeshTri(pts.T * _UM, tris.T)
    basis = skfem.Basis(m, skfem.ElementTriP1())
    eps = np.array([_PERMITTIVITY[_NAMES[k]] for k in labels]) * _EPS0

    @skfem.BilinearForm
    def laplace(u, v, w):
        return w["eps"] * dot(grad(u), grad(v))

    eps_q = np.repeat(eps[:, None], basis.X.shape[-1], axis=1)
    stiffness = laplace.assemble(basis, eps=eps_q)
    edges = _contact_edges(pts)
    bottom = np.unique(edges["bot"])
    top = np.unique(edges["top"])
    u = np.zeros(len(pts))
    u[top] = 1.0
    u = skfem.solve(
        *skfem.condense(stiffness, np.zeros(len(pts)), x=u, D=np.r_[bottom, top])
    )
    return float(u @ (stiffness @ u))


def _devsim_device(mesh, kind: str) -> tuple[Any, str]:
    """Carriers-off DEVSIM device on the mesh.

    ``kind`` is ``"signed"`` for gsim's couples, ``"builtin"`` for DEVSIM's own.
    """
    from devsim.python_packages import simple_physics as sp

    from gsim.tcad.device import _install_signed_couples
    from gsim.tcad.runtime import require_devsim

    devsim = require_devsim()
    pts, tris, labels = mesh
    tag = next(_NAME_IDS)
    mesh_name, device = f"couples_mesh_{tag}", f"couples_device_{tag}"
    elements: list[float] = []
    for tri, label in zip(tris, labels, strict=True):
        elements += [2, int(label), *map(int, tri)]
    for name, edges in _contact_edges(pts).items():
        for a, b in edges:
            elements += [1, _NAMES.index(name), a, b]
    coordinates = np.column_stack([pts * _UM, np.zeros(len(pts))]).ravel()
    devsim.create_gmsh_mesh(
        mesh=mesh_name,
        coordinates=list(coordinates),
        elements=elements,
        physical_names=_NAMES,
    )
    for region in ("lower", "upper"):
        devsim.add_gmsh_region(
            mesh=mesh_name, gmsh_name=region, region=region, material="Oxide"
        )
    devsim.add_gmsh_contact(
        mesh=mesh_name, gmsh_name="bot", region="lower", material="metal", name="bot"
    )
    devsim.add_gmsh_contact(
        mesh=mesh_name, gmsh_name="top", region="upper", material="metal", name="top"
    )
    devsim.add_gmsh_interface(
        mesh=mesh_name, gmsh_name="mid", region0="lower", region1="upper", name="mid"
    )
    devsim.finalize_mesh(mesh=mesh_name)
    devsim.create_device(mesh=mesh_name, device=device)
    if kind == "signed":
        _install_signed_couples(devsim, device, ["lower", "upper"])
    for region, permittivity in _PERMITTIVITY.items():
        devsim.set_parameter(
            device=device,
            region=region,
            name="Permittivity",
            value=permittivity * _EPS0,
        )
        sp.CreateOxidePotentialOnly(device, region)
    sp.CreateOxideContact(device, "lower", "bot")
    sp.CreateOxideContact(device, "upper", "top")
    sp.CreateSiliconOxideInterface(device, "mid")
    devsim.set_parameter(device=device, name=sp.GetContactBiasName("bot"), value=0.0)
    devsim.set_parameter(device=device, name=sp.GetContactBiasName("top"), value=1.0)
    return devsim, device


def _devsim_capacitance(mesh, kind: str) -> float:
    devsim, device = _devsim_device(mesh, kind)
    try:
        devsim.solve(
            type="dc", absolute_error=1e-15, relative_error=1e-12, maximum_iterations=5
        )
        return abs(
            devsim.get_contact_charge(
                device=device, contact="top", equation="PotentialEquation"
            )
        )
    finally:
        devsim.delete_device(device=device)


def _node_volume_sums(mesh, kind: str) -> dict[str, float]:
    """Sum of the node volumes DEVSIM integrates with, per Region (um^2)."""
    devsim, device = _devsim_device(mesh, kind)
    try:
        model = devsim.get_parameter(device=device, name="node_volume_model")
        return {
            region: float(
                np.sum(
                    devsim.get_node_model_values(
                        device=device, region=region, name=model
                    )
                )
            )
            / _UM**2
            for region in ("lower", "upper")
        }
    finally:
        devsim.delete_device(device=device)


def test_the_mesh_has_obtuse_triangles_in_both_regions(mesh):
    pts, tris, labels = mesh
    obtuse = _max_angle_deg(pts, tris) > 90.0 + 1e-6
    assert obtuse[labels == 0].sum() >= 5
    assert obtuse[labels == 1].sum() >= 5


def test_signed_couples_match_an_independent_p1_solve(mesh, p1_capacitance):
    capacitance = _devsim_capacitance(mesh, "signed")
    assert capacitance == pytest.approx(p1_capacitance, rel=1e-9)


def test_builtin_couples_overestimate_on_obtuse_triangles(mesh, p1_capacitance):
    # DEVSIM's own couples take |cot| of each angle, which adds coupling
    # across every obtuse triangle.
    capacitance = _devsim_capacitance(mesh, "builtin")
    assert capacitance > p1_capacitance * (1.0 + 1e-4)


@pytest.mark.parametrize(
    ("kind", "exact"), [("signed", True), ("builtin", False)], ids=["signed", "builtin"]
)
def test_node_volumes_and_region_area(mesh, kind, exact):
    # Each Region is 2 um by 0.5 um.
    sums = _node_volume_sums(mesh, kind)
    for region in ("lower", "upper"):
        if exact:
            assert sums[region] == pytest.approx(1.0, rel=1e-12)
        else:
            assert sums[region] > 1.0 + 1e-3
