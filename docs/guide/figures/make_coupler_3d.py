"""Isometric set-up view of the directional coupler for docs/guide/index.md.

Writes docs/guide/images/guide-coupler-3d.png. The silicon is the layout of
``gf.components.coupler`` extruded over the ``core`` layer of the generic PDK stack; the
domain box comes from the ``metadata.json`` that ``fdtd.Simulation.write()`` writes
locally (physical domain, without the PML). The source and monitor planes sit on the
ports. y and z are stretched for legibility; the script logs the factors. Nothing is
solved and nothing is submitted.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
from _common import LOG, apply_style, save
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from gsim import fdtd

GAP_UM = 0.236
LENGTH_UM = 20.0
BOX_ASPECT = (40.0, 11.0, 7.0)  # drawn aspect of the domain box (x, y, z)


def build_simulation(component):
    """Set-up of the page's snippet: oxide background, silicon index, source on o1."""
    sim = fdtd.Simulation()
    sim.materials(background="SiO2", overrides={"si": 3.4757})
    sim.geometry(component)
    sim.source(
        port="o1", wavelength_um=1.55, wavelength_span_um=0.1, num_wavelengths=101
    )
    sim.domain(padding=1.0)
    sim.solver(cell_size_nm=60)
    return sim


def silicon_faces(component, z0: float, z1: float):
    """Top and side faces of the core-layer polygons, extruded from z0 to z1."""
    dbu = component.kcl.dbu
    tops, sides = [], []
    for polygons in component.get_polygons(by="tuple").values():
        for polygon in polygons:
            points = [(p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()]
            tops.append([(x, y, z1) for x, y in points])
            for (xa, ya), (xb, yb) in zip(points, points[1:] + points[:1], strict=True):
                sides.append([(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)])
    return tops, sides


def main() -> None:
    """Draw the isometric set-up."""
    apply_style()
    gf.gpdk.PDK.activate()
    component = gf.components.coupler(gap=GAP_UM, length=LENGTH_UM)
    sim = build_simulation(component)
    with tempfile.TemporaryDirectory(prefix="guidefig-dc3d-") as scratch:
        sim.write(scratch)  # local files only; nothing is submitted
        meta = json.loads((Path(scratch) / "metadata.json").read_text(encoding="utf8"))
    low, high = (
        meta["grid"]["non_pml_bounds"]["min"],
        meta["grid"]["non_pml_bounds"]["max"],
    )
    LOG.info("physical domain (nm) min %s max %s", low, high)
    (x0, y0, z0), (x1, y1, z1) = ([v / 1000 for v in low], [v / 1000 for v in high])
    stack = gf.gpdk.PDK.layer_stack
    if stack is None:
        raise RuntimeError("the generic PDK has no layer stack")
    core = stack.layers["core"]
    core_z0, core_z1 = core.zmin, core.zmin + core.thickness
    LOG.info(
        "stretch of the drawing: y x%.1f, z x%.1f",
        (BOX_ASPECT[1] / BOX_ASPECT[0]) / ((y1 - y0) / (x1 - x0)),
        (BOX_ASPECT[2] / BOX_ASPECT[0]) / ((z1 - z0) / (x1 - x0)),
    )

    fig = plt.figure(figsize=(8.0, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    tops, sides = silicon_faces(component, core_z0, core_z1)
    ax.add_collection3d(
        Poly3DCollection(
            sides, facecolors="#4a78a8", edgecolors="none", rasterized=True
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            tops, facecolors="#7fb2e5", edgecolors="#23415f", linewidths=0.4
        )
    )

    # Domain box (physical domain; the PML lies outside it).
    corners = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    edges = [
        (a, b)
        for i, a in enumerate(corners)
        for b in corners[i + 1 :]
        if sum(u != v for u, v in zip(a, b, strict=True)) == 1
    ]
    ax.add_collection3d(
        Line3DCollection(edges, colors="0.35", linestyles="--", linewidths=0.8)
    )

    # Source plane on o1, monitor planes on the other ports (y-z planes at the port).
    half = (
        0.5 + 2 * 0.5
    )  # half-width (um) of the drawn plane: port width + 0.5 um each side
    for port in component.ports:
        x, y = port.dcenter
        z_lo, z_hi = core_z0 - 0.5, core_z1 + 0.5
        plane = [
            (x, y - half / 2, z_lo),
            (x, y + half / 2, z_lo),
            (x, y + half / 2, z_hi),
            (x, y - half / 2, z_hi),
        ]
        is_source = port.name == "o1"
        ax.add_collection3d(
            Poly3DCollection(
                [plane],
                facecolors="tab:red" if is_source else "tab:green",
                alpha=0.55,
                edgecolors="none",
            )
        )
        ax.text(
            x + (-1.2 if x < 0 else 1.2),
            y,
            z_hi + 0.3,
            f"{port.name}\n{'source' if is_source else 'monitor'}",
            fontsize=8,
            color="tab:red" if is_source else "darkgreen",
            ha="center",
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect(BOX_ASPECT)
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel("x (µm)", labelpad=2)
    ax.set_ylabel("y (µm)", labelpad=2)
    ax.set_zlabel("z (µm)", labelpad=-2)
    ax.set_yticks([-2, 0, 2])
    ax.set_zticks([-1, 0, 1])  # ty: ignore[call-non-callable]
    ax.set_title(
        f"Directional coupler, gap {GAP_UM * 1000:.0f} nm, L = {LENGTH_UM:.0f} µm: "
        "silicon in SiO$_2$, domain box dashed"
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)
    save(fig, "guide-coupler-3d.png")


if __name__ == "__main__":
    main()
