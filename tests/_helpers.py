"""Analytic fixtures shared across test packages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray


def skip_without_devsim() -> None:
    """Skip the caller when DEVSIM cannot be imported.

    Goes through :func:`gsim.tcad.runtime.require_devsim` rather than a
    bare import, so the first import picks up scipy-openblas32 the way a
    user's does: DEVSIM loads its BLAS/LAPACK library only on that first
    import, and cannot initialise again in the same process after a
    failure. ``pytest.importorskip("devsim")`` would bypass both.
    """
    from gsim.tcad.runtime import require_devsim

    try:
        require_devsim()
    except ImportError as err:
        pytest.skip(f"DEVSIM is unavailable: {err}", allow_module_level=True)


def restore_generic_pdk() -> None:
    """Activate the generic PDK and put its layers back at their enum slots.

    All PDKs share one ``kfactory`` layout. Activating a PDK that has a layer
    enum (IHP) deletes the empty layers of the previous one (gdsfactory's
    ``_prune_foreign_layers``), so the generic slots 0, 3, 4, ... are free until
    the generic PDK is activated again. A raw ``(layer, datatype)`` registered
    meanwhile takes the lowest free slot, say ``(2, 0)`` at slot 0. When the
    generic PDK returns, gdsfactory moves that layer to ``SLAB150``'s slot 3
    with ``Layout.move_layer`` and ``delete_layer``. Both raise ``RuntimeError``
    once any cached (locked) cell exists, and gdsfactory logs and swallows it.
    Slot 3 is then re-inserted without the layer name and the layout holds two
    ``(2, 0)`` layers: ``LAYER.SLAB150`` writes to one while a ``(2, 0)`` lookup
    reads the other, and the region comes back empty.

    This activates the generic PDK, moves a foreign layer out of any generic
    slot, names each slot as the enum does and merges duplicate ``(layer,
    datatype)`` layers into the enum's slot, unlocking cells for the moment it
    takes. Shapes are moved, never dropped.
    """
    try:
        import gdsfactory as gf
        import kfactory as kf
        from kfactory import kdb
    except ImportError:
        return

    gf.gpdk.PDK.activate()
    layout = kf.kcl.layout

    def tuple_of(index: int) -> tuple[int, int]:
        info = layout.get_info(index)
        return info.layer, info.datatype

    locked = [cell for cell in layout.each_cell() if cell.is_locked()]
    for cell in locked:
        cell.locked = False
    changed = False
    try:
        for layer in gf.gpdk.LAYER:  # ty: ignore[not-iterable]
            slot = int(layer)
            key = (layer.layer, layer.datatype)
            wanted = kdb.LayerInfo(layer.layer, layer.datatype, layer.name)
            if layout.is_valid_layer(slot) and tuple_of(slot) != key:
                held = tuple_of(slot)
                twins = [
                    index
                    for index in layout.layer_indexes()
                    if index != slot and tuple_of(index) == held
                ]
                if twins:
                    new_home = twins[0]
                else:
                    new_home = layout.insert_layer(kdb.LayerInfo(*held))
                layout.move_layer(slot, new_home)
                layout.delete_layer(slot)
                changed = True
            if not layout.is_valid_layer(slot):
                layout.insert_layer_at(slot, wanted)
                changed = True
            elif layout.get_info(slot) != wanted:
                layout.set_info(slot, wanted)
                changed = True
            for other in list(layout.layer_indexes()):
                if other != slot and tuple_of(other) == key:
                    layout.move_layer(other, slot)
                    layout.delete_layer(other)
                    changed = True
    finally:
        for cell in locked:
            cell.locked = True
    if changed:
        gf.gpdk.PDK.activate(force=True)


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

    Shaped like a carrier coupling's response — index shift, absorption
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


def draw_rectangle(comp, layer, *, width: float, height: float, x0: float, y: float):
    """Draw a ``width`` x ``height`` rectangle into ``comp``, centred on ``y``.

    Drawn directly rather than through a cached ``gf.c.rectangle`` cell: a
    cell cached before another test registers a PDK that names these GDS
    numbers (IHP names ``(30, *)`` its Metal3 layers) keeps its shapes on a
    layer index that ``(30, 0)`` no longer resolves to, and the region reads
    as empty.
    """
    comp.add_polygon(
        [
            (x0, y - height / 2),
            (x0 + width, y - height / 2),
            (x0 + width, y + height / 2),
            (x0, y + height / 2),
        ],
        layer=layer,
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
    draw_rectangle(comp, (1, 0), width=10.0, height=rib_width, x0=-5.0, y=center_y)
    draw_rectangle(comp, (3, 0), width=10.0, height=100.0, x0=-5.0, y=-5.0)

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
