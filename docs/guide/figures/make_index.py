"""Strip-mode figure for the "Waveguide modes" section of docs/guide/index.md.

TE0 and TM0 of a 500 x 220 nm strip, local femwell solve.

Writes docs/guide/images/index-strip-modes.png. The mesh comes from gsim's native-2D
cross-section mesher (BoundaryModeSim, used only to mesh) and the modes from
gsim.femwell.solve_modes. Generic PDK, no cloud call.
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
WIDTH_UM, HEIGHT_UM = 0.5, 0.22
# um, across the guide; 1.5 um clipped the TM0 tail (boundary field ratio 1.0e-2)
WINDOW_Y = (-2.5, 2.5)
WINDOW_Z = (-1.6, 1.9)  # um, vertical


def solve_strip():
    """Mesh the strip cross-section and solve TE0 and TM0 with femwell."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component << gf.c.rectangle((10.0, WIDTH_UM), centered=True, layer=(1, 0))

    sim = BoundaryModeSim()
    sim.set_output_dir(tempfile.mkdtemp(prefix="guidefig-strip-"))
    sim.set_stack(substrate_thickness=1.0)
    sim.set_geometry(component)
    sim.set_airbox(
        margin_x=3.0, margin_y=3.0, z_above=3.0, z_below=1.0, material="sio2"
    )
    sim.set_cross_section("x=0", window=WINDOW_Y, window_z=WINDOW_Z)
    sim.mesh(preset="coarse", refined_mesh_size=0.02, max_mesh_size=0.3, verbose=False)

    if sim.stack is None:
        raise RuntimeError("the simulation has no layer stack")
    eps = epsilon_by_region(sim.mesh_path, sim.stack, wavelength_um=WAVELENGTH_UM)
    modes = solve_modes(
        sim.mesh_path, epsilon=eps, wavelength_um=WAVELENGTH_UM, num_modes=4
    )
    n_clad = float(np.sqrt(eps["sio2"].real))
    te0 = next(m for m in modes if m.te_fraction > 0.5)
    tm0 = next(m for m in modes if m.te_fraction < 0.5 and m.n_eff.real > n_clad)
    LOG.info("index core / clad: %s %s", float(np.sqrt(eps["core"].real)), n_clad)
    for label, mode in (("TE0", te0), ("TM0", tm0)):
        ratio = boundary_field_ratio(mode)
        LOG.info(
            "%s: n_eff = %.5f, te_fraction = %.3f, boundary field ratio = %.2e",
            label,
            mode.n_eff.real,
            mode.te_fraction,
            ratio,
        )
        if ratio >= 0.01:
            raise RuntimeError("the window clips the mode")
    return te0, tm0


def element_intensity(mode):
    """Mean |E|^2 over each triangle, normalised to the peak."""
    (e_x, e_y), e_z = mode.basis.interpolate(mode.E)
    intensity = (np.abs(e_x) ** 2 + np.abs(e_y) ** 2 + np.abs(e_z) ** 2).mean(axis=1)
    return intensity / intensity.max()


def main() -> None:
    """Solve the strip modes and draw the figure."""
    apply_style()
    te0, tm0 = solve_strip()

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.3), sharey=True, constrained_layout=True
    )
    images = []
    for ax, label, mode in zip(axes, ("TE0", "TM0"), (te0, tm0), strict=True):
        mesh = mode.basis.mesh
        images.append(
            ax.tripcolor(
                mesh.p[0],
                mesh.p[1],
                mesh.t.T,
                facecolors=element_intensity(mode),
                cmap="magma",
                vmin=0,
                vmax=1,
                rasterized=True,
            )
        )
        ax.add_patch(
            Rectangle(
                (-WIDTH_UM / 2, 0),
                WIDTH_UM,
                HEIGHT_UM,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 0.9)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("y (µm)")
        ax.set_title(f"{label}: $n_{{eff}}$ = {mode.n_eff.real:.3f}")
    axes[0].set_ylabel("z (µm)")
    fig.colorbar(images[-1], ax=axes, label="|E|² (normalised)", shrink=0.85)
    save(fig, "index-strip-modes.png")
    LOG.info("run date: %s", today())


if __name__ == "__main__":
    main()
