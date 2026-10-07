"""Analytic fixtures shared across test packages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray


def skip_without_devsim() -> None:
    """Skip the caller when DEVSIM cannot be imported.

    Goes through :func:`gsim.tcad.runtime.require_devsim` rather than a
    bare import, so the first import picks up devsim-openblas the way a
    user's does: DEVSIM loads its BLAS/LAPACK library only on that first
    import, and cannot initialise again in the same process after a
    failure. ``pytest.importorskip("devsim")`` would bypass both.
    """
    from gsim.tcad.runtime import require_devsim

    try:
        require_devsim()
    except ImportError as err:
        pytest.skip(f"DEVSIM is unavailable: {err}", allow_module_level=True)


def series_rc_admittance(
    freq_hz: ArrayLike, r_ohm_m: float | ArrayLike, c_f_per_m: float | ArrayLike
) -> NDArray[np.complex128]:
    """Admittance of a series-RC branch, straight from Z = R + 1/(jwC)."""
    omega = 2 * np.pi * np.asarray(freq_hz, dtype=np.float64)
    return np.asarray(
        1.0 / (np.asarray(r_ohm_m) + 1.0 / (1j * omega * np.asarray(c_f_per_m))),
        dtype=np.complex128,
    )


def fake_coupling(n_cm3: ArrayLike, p_cm3: ArrayLike) -> SimpleNamespace:
    """A stand-in carrier coupling: linear in the concentrations, no physics.

    Shaped like the carriers Stage's response — index shift, absorption
    and conductivity per sample — so a Staircase test can inject it in
    place of a plasma-dispersion model. Electrons conduct twice as well
    as holes, so a profile's asymmetry survives into the conductivity.
    """
    n = np.asarray(n_cm3, dtype=np.float64)
    p = np.asarray(p_cm3, dtype=np.float64)
    return SimpleNamespace(
        index_shift=-1e-20 * (n + p),
        absorption_cm=1e-17 * (n + p),
        conductivity_s_per_m=2e-15 * n + 1e-15 * p,
    )


def draw_pn_rib(
    *,
    center_y: float = -20.0,
    rib_width: float = 0.4,
    zmax: float = 0.22,
    sigma_s_per_m: float = 1.6e3,
):
    """A rib whose two halves are the touching Regions ``n_rib`` | ``p_rib``.

    The smallest device with a metallurgical Junction on it, drawn by
    hand on a doped cross-section stack: the n half on the low side of
    ``center_y``, the p half on the high side, on a 90 nm slab.

    Returns:
        ``(component, stack)``.
    """
    import gdsfactory as gf

    from gsim.common.cross_section import build_doped_cross_section
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import make_doped_materials

    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, rib_width), centered=True, layer=(1, 0))
    wg.y = center_y
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    halves = {
        "n_rib": ((20, 0), (center_y - rib_width / 2, center_y)),
        "p_rib": ((21, 0), (center_y, center_y + rib_width / 2)),
    }
    layer_specs = {}
    for name, (gds_layer, (y0, y1)) in halves.items():
        comp.add_polygon(
            [(-5.0, y0), (5.0, y0), (5.0, y1), (-5.0, y1)], layer=gds_layer
        )
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=zmax,
            thickness=zmax,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    materials = make_doped_materials(
        [(name, sigma_s_per_m) for name in halves], permittivity=11.9
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )
    return comp, stack


def longest_edge_in_box(
    mesh_path, h: tuple[float, float], z: tuple[float, float]
) -> float:
    """Longest edge (um) of the triangles lying wholly inside a box.

    A triangle with a corner outside the box straddles its border, where the
    size field steps from the box's size to the surrounding one; such an
    element measures the transition, not the box.
    """
    import meshio

    mesh = meshio.read(str(mesh_path))
    points = mesh.points[:, :2]
    triangles = np.vstack([c.data for c in mesh.cells if c.type == "triangle"])
    corners = points[triangles]
    inside = (
        (corners[..., 0] > h[0])
        & (corners[..., 0] < h[1])
        & (corners[..., 1] > z[0])
        & (corners[..., 1] < z[1])
    ).all(axis=1)
    corners = corners[inside]
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=-1)
    return float(edges.max())
