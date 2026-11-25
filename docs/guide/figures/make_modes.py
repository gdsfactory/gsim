"""Waveguide cross-sections for docs/guide/index.md: strip, rib and slot.

Writes docs/guide/images/guide-mode-sections.png. Each column is one cross-section of the
generic PDK stack in the window ``BoundaryModeSim.set_cross_section`` cuts: the top row
shows the window, the cross-section mesh and the silicon, the bottom row zooms on the TE0
mode computed by ``gsim.femwell.solve_modes`` (|E|^2, normalised to its peak). The mesh
comes from gsim's native-2D mesher and the modes from femwell; everything runs locally and
nothing touches the cloud.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
from _common import LOG, apply_style, save, today
from matplotlib.patches import Rectangle

from gsim.femwell import boundary_field_ratio, epsilon_by_region, solve_modes
from gsim.palace import BoundaryModeSim

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.22  # "core" layer of the generic PDK stack
SLAB_HEIGHT_UM = 0.09  # "slab90" layer
WINDOW_Y = (-2.5, 2.5)  # um, across the guide
WINDOW_Z = (-1.6, 1.9)  # um, vertical

# Silicon rectangles of each cross-section: (y0, y1, z0, z1) in um.
SECTIONS = {
    "Strip 500 x 220 nm": [(-0.25, 0.25, 0.0, CORE_HEIGHT_UM)],
    "Rib 500 x 220 nm, 90 nm slab": [
        (-0.25, 0.25, 0.0, CORE_HEIGHT_UM),
        (-1.5, 1.5, 0.0, SLAB_HEIGHT_UM),
    ],
    "Slot, two 250 nm rails, 100 nm gap": [
        (-0.30, -0.05, 0.0, CORE_HEIGHT_UM),
        (0.05, 0.30, 0.0, CORE_HEIGHT_UM),
    ],
}


def build_component(label: str) -> gf.Component:
    """Layout of one cross-section: long rectangles, cut at x = 0."""
    layer = gf.gpdk.LAYER
    component = gf.Component()
    if label.startswith("Strip"):
        component << gf.c.rectangle((10.0, 0.5), centered=True, layer=layer.WG)
    elif label.startswith("Rib"):
        component << gf.c.rectangle((10.0, 0.5), centered=True, layer=layer.WG)
        component << gf.c.rectangle((10.0, 3.0), centered=True, layer=layer.SLAB90)
    else:
        for sign in (-1, 1):
            rail = component << gf.c.rectangle((10.0, 0.25), centered=True, layer=layer.WG)
            rail.move((0, sign * 0.175))
    return component


def solve(label: str):
    """Mesh the cross-section and return the TE0 mode (largest n_eff, TE-like)."""
    sim = BoundaryModeSim()
    sim.set_output_dir(tempfile.mkdtemp(prefix="guidefig-modes-"))
    sim.set_stack(substrate_thickness=1.0)
    sim.set_geometry(build_component(label))
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=3.0, z_below=1.0, material="sio2")
    sim.set_cross_section("x=0", window=WINDOW_Y, window_z=WINDOW_Z)
    sim.mesh(preset="coarse", refined_mesh_size=0.02, max_mesh_size=0.3, verbose=False)
    if sim.stack is None:
        raise RuntimeError("the simulation has no layer stack")
    eps = epsilon_by_region(sim.mesh_path, sim.stack, wavelength_um=WAVELENGTH_UM)
    modes = solve_modes(
        sim.mesh_path, epsilon=eps, wavelength_um=WAVELENGTH_UM, num_modes=4
    )
    n_clad = float(np.sqrt(eps["sio2"].real))
    guided = [m for m in modes if m.te_fraction > 0.5 and m.n_eff.real > n_clad]
    te0 = max(guided, key=lambda m: m.n_eff.real)
    ratio = boundary_field_ratio(te0)
    LOG.info(
        "%s: n_eff = %.4f, te_fraction = %.3f, boundary field ratio = %.1e, "
        "triangles = %d",
        label,
        te0.n_eff.real,
        te0.te_fraction,
        ratio,
        te0.basis.mesh.t.shape[1],
    )
    if ratio >= 0.01:
        raise RuntimeError(f"{label}: the window clips the mode")
    return te0


def element_intensity(mode):
    """Mean |E|^2 over each triangle, normalised to the peak."""
    (e_x, e_y), e_z = mode.basis.interpolate(mode.E)
    intensity = (np.abs(e_x) ** 2 + np.abs(e_y) ** 2 + np.abs(e_z) ** 2).mean(axis=1)
    return intensity / intensity.max()


def main() -> None:
    """Solve the three cross-sections and draw the figure."""
    apply_style()
    gf.gpdk.PDK.activate()
    fig, axes = plt.subplots(
        2, 3, figsize=(8.0, 4.5), constrained_layout=True, height_ratios=(1.15, 1.0)
    )
    last = None
    for column, (label, rectangles) in enumerate(SECTIONS.items()):
        mode = solve(label)
        mesh = mode.basis.mesh
        top, bottom = axes[0, column], axes[1, column]

        # Top row: window, mesh, silicon, oxide above and below.
        top.triplot(mesh.p[0], mesh.p[1], mesh.t.T, color="0.55", linewidth=0.15, rasterized=True)
        for y0, y1, z0, z1 in rectangles:
            top.add_patch(
                Rectangle((y0, z0), y1 - y0, z1 - z0, facecolor="tab:blue", edgecolor="black", linewidth=0.6)
            )
        top.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        top.text(WINDOW_Y[0] + 0.1, 0.9, "SiO$_2$ cladding", fontsize=8, color="0.25")
        top.text(WINDOW_Y[0] + 0.1, -0.9, "SiO$_2$ box", fontsize=8, color="0.25")
        top.set_xlim(*WINDOW_Y)
        top.set_ylim(*WINDOW_Z)
        top.set_aspect("equal")
        top.grid(False)
        top.set_title(label, fontsize=9)
        top.set_xlabel("y (µm)")

        # Bottom row: TE0 field zoomed on the core.
        last = bottom.tripcolor(
            mesh.p[0],
            mesh.p[1],
            mesh.t.T,
            facecolors=element_intensity(mode),
            cmap="magma",
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        for y0, y1, z0, z1 in rectangles:
            bottom.add_patch(
                Rectangle((y0, z0), y1 - y0, z1 - z0, fill=False, edgecolor="white", linewidth=0.8)
            )
        bottom.set_xlim(-1.2, 1.2)
        bottom.set_ylim(-0.6, 0.8)
        bottom.set_aspect("equal")
        bottom.grid(False)
        bottom.set_xlabel("y (µm)")
        bottom.set_title(f"TE0: $n_{{eff}}$ = {mode.n_eff.real:.3f}", fontsize=9)
    axes[0, 0].set_ylabel("z (µm)")
    axes[1, 0].set_ylabel("z (µm)")
    fig.colorbar(last, ax=axes[1, :], label="|E|² (normalised)", shrink=0.8)
    save(fig, "guide-mode-sections.png")
    LOG.info("run date: %s", today())


if __name__ == "__main__":
    main()
