"""Drift-diffusion with negative edge couples: a PN junction on an obtuse mesh.

A 1e17/1e17 cm^-3 abrupt silicon junction, 0.8 um long and 0.08 um tall,
solved by DEVSIM's Scharfetter-Gummel drift-diffusion on two meshes: a
right-angle grid, where no couple is negative, and the same grid with its
interior nodes jittered, where many triangles are obtuse and some edges
carry a negative couple. The depletion width must not depend on the mesh.
Requires DEVSIM: run with ``pytest -m tcad_local``.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from tests._helpers import skip_without_devsim

skip_without_devsim()
scipy_interpolate = pytest.importorskip("scipy.interpolate")

pytestmark = pytest.mark.tcad_local

_UM = 1e-4  # cm
_LENGTH, _HEIGHT = 0.8, 0.08  # um
_NX, _NY = 80, 4
_DOPING = 1e17  # cm^-3 on each side
_BIASES = (0.0, -1.0, -2.0)  # V on the P contact
_NAMES = ["si", "left", "right"]
_NAME_IDS = itertools.count()


def _mesh(jitter: float) -> tuple[np.ndarray, np.ndarray]:
    """Return points (um) and triangles of the strip.

    ``jitter == 0`` gives right triangles (couples zero on the diagonals,
    positive elsewhere). Otherwise interior nodes move by up to ``jitter``
    cells and each cell is split along a random diagonal.
    """
    rng = np.random.default_rng(3)
    xs = np.linspace(0.0, _LENGTH, _NX + 1)
    ys = np.linspace(0.0, _HEIGHT, _NY + 1)
    pts = np.array([(x, y) for y in ys for x in xs])
    if jitter:
        for j in range(1, _NY):
            for i in range(1, _NX):
                pts[j * (_NX + 1) + i] += rng.uniform(-jitter, jitter, 2) * (
                    xs[1],
                    ys[1],
                )
    tris = []
    for j in range(_NY):
        for i in range(_NX):
            a = j * (_NX + 1) + i
            b, c, d = a + 1, a + _NX + 1, a + _NX + 2
            if not jitter or rng.random() < 0.5:
                tris += [(a, b, d), (a, d, c)]
            else:
                tris += [(a, b, c), (b, d, c)]
    return pts, np.array(tris)


def _solve(devsim: Any) -> None:
    devsim.solve(
        type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=40
    )


def _depletion_width_um(devsim: Any, device: str) -> float:
    """Integrated space charge over the doping, along the strip's mid-line."""
    values = {
        name: np.asarray(
            devsim.get_node_model_values(device=device, region="si", name=name)
        )
        for name in ("x", "y", "Electrons", "Holes", "NetDoping")
    }
    xy = np.column_stack([values["x"], values["y"]]) / _UM
    rho = values["Holes"] - values["Electrons"] + values["NetDoping"]
    xs = np.linspace(0.0, _LENGTH, 4001)
    line = np.column_stack([xs, np.full_like(xs, _HEIGHT / 2)])
    sampled = scipy_interpolate.LinearNDInterpolator(xy, rho)(line)
    return float(np.trapezoid(np.abs(sampled), xs) / _DOPING)


def _sweep(jitter: float, kind: str) -> dict[str, Any]:
    """Depletion widths at ``_BIASES`` and the couples/volumes counts.

    ``kind`` is ``"signed"`` for gsim's couples, ``"builtin"`` for DEVSIM's own.
    """
    from devsim.python_packages import model_create as mc
    from devsim.python_packages import simple_physics as sp

    from gsim.tcad.device import (
        SIGNED_EDGE_COUPLE,
        SIGNED_NODE_VOLUME,
        _install_signed_couples,
    )
    from gsim.tcad.runtime import require_devsim

    devsim = require_devsim()
    pts, tris = _mesh(jitter)
    tag = next(_NAME_IDS)
    mesh_name, device = f"obtuse_pn_mesh_{tag}", f"obtuse_pn_device_{tag}"
    row = _NX + 1
    elements: list[float] = []
    for tri in tris:
        elements += [2, 0, *map(int, tri)]
    for j in range(_NY):
        elements += [1, 1, j * row, (j + 1) * row]
        elements += [1, 2, j * row + _NX, (j + 1) * row + _NX]
    coordinates = np.column_stack([pts * _UM, np.zeros(len(pts))]).ravel()
    devsim.create_gmsh_mesh(
        mesh=mesh_name,
        coordinates=list(coordinates),
        elements=elements,
        physical_names=_NAMES,
    )
    devsim.add_gmsh_region(
        mesh=mesh_name, gmsh_name="si", region="si", material="Silicon"
    )
    for contact in ("left", "right"):
        devsim.add_gmsh_contact(
            mesh=mesh_name,
            gmsh_name=contact,
            region="si",
            material="metal",
            name=contact,
        )
    devsim.finalize_mesh(mesh=mesh_name)
    devsim.create_device(mesh=mesh_name, device=device)
    try:
        result: dict[str, Any] = {}
        if kind == "signed":
            _install_signed_couples(devsim, device, ["si"])
            couples = devsim.get_edge_model_values(
                device=device, region="si", name=SIGNED_EDGE_COUPLE
            )
            volumes = devsim.get_node_model_values(
                device=device, region="si", name=SIGNED_NODE_VOLUME
            )
            result["negative_couples"] = int(np.sum(np.asarray(couples) < 0.0))
            result["negative_volumes"] = int(np.sum(np.asarray(volumes) < 0.0))
        sp.SetSiliconParameters(device, "si", 300)
        junction = _LENGTH / 2 * _UM
        mc.CreateNodeModel(device, "si", "Acceptors", f"{_DOPING}*step({junction}-x)")
        mc.CreateNodeModel(device, "si", "Donors", f"{_DOPING}*step(x-{junction})")
        mc.CreateNodeModel(device, "si", "NetDoping", "Donors-Acceptors")
        sp.CreateSolution(device, "si", "Potential")
        sp.CreateSiliconPotentialOnly(device, "si")
        for contact in ("left", "right"):
            sp.CreateSiliconPotentialOnlyContact(device, "si", contact)
            devsim.set_parameter(
                device=device, name=sp.GetContactBiasName(contact), value=0.0
            )
        _solve(devsim)
        for carrier, initial in (
            ("Electrons", "IntrinsicElectrons"),
            ("Holes", "IntrinsicHoles"),
        ):
            sp.CreateSolution(device, "si", carrier)
            devsim.set_node_values(
                device=device, region="si", name=carrier, init_from=initial
            )
        sp.CreateSiliconDriftDiffusion(device, "si")
        for contact in ("left", "right"):
            sp.CreateSiliconDriftDiffusionAtContact(device, "si", contact)
        _solve(devsim)
        widths, bias = [], 0.0
        for target in _BIASES:
            for step in np.linspace(bias, target, 5)[1:]:
                devsim.set_parameter(
                    device=device, name=sp.GetContactBiasName("left"), value=float(step)
                )
                _solve(devsim)
            bias = target
            widths.append(_depletion_width_um(devsim, device))
        result["widths_um"] = np.array(widths)
        return result
    finally:
        devsim.delete_device(device=device)
        devsim.delete_mesh(mesh=mesh_name)


@pytest.fixture(scope="module")
def right_angle() -> dict[str, Any]:
    return _sweep(0.0, "signed")


@pytest.fixture(scope="module")
def obtuse_signed() -> dict[str, Any]:
    return _sweep(0.35, "signed")


def test_the_obtuse_mesh_has_negative_couples(right_angle, obtuse_signed):
    assert right_angle["negative_couples"] == 0
    assert obtuse_signed["negative_couples"] >= 100


def test_depletion_width_does_not_depend_on_the_mesh(right_angle, obtuse_signed):
    # The right-angle mesh reads 0.129, 0.199 and 0.251 um.
    assert np.all(np.diff(right_angle["widths_um"]) > 0.0)
    np.testing.assert_allclose(
        obtuse_signed["widths_um"], right_angle["widths_um"], rtol=1e-2
    )


def test_builtin_couples_shrink_the_depletion_width(right_angle):
    # DEVSIM's own |cot| couples add coupling across every obtuse triangle;
    # on this mesh they read the width 20 % to 25 % low.
    builtin = _sweep(0.35, "builtin")
    assert np.all(builtin["widths_um"] < 0.9 * right_angle["widths_um"])
