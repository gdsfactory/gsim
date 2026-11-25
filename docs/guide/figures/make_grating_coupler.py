"""Set-up figure of the grating coupler for docs/guide/index.md.

Writes docs/guide/images/guide-grating-setup.png: the XZ side view of the generic-PDK
elliptical grating coupler as ``gsim.meep.Simulation`` sets it up for a 2D XZ run (the
cells of nbs/meep_2d_xz_gc.ipynb up to the preview), drawn by ``Simulation.plot_2d``.
Nothing is solved and nothing is submitted: ``plot_2d`` only builds the configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
from _common import LOG, apply_style, save

from gsim import meep
from gsim.common.stack import get_stack
from gsim.meep.models.api import Material

FIBER = {
    "x": 25.0,
    "z": 2,
    "angle_deg": -6.0,
    "waist": 5.2,
    "wavelength": 1.55,
    "wavelength_span": 0.06,
    "polarization": "TE",
}


def build_simulation() -> meep.Simulation:
    """The set-up cells of the 2D grating-coupler notebook."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    grating = component.add_ref(gf.components.grating_coupler_elliptical(fiber_angle=0.0))
    straight = component.add_ref(gf.components.straight(length=3))
    straight.connect("o1", grating.ports["o1"])
    component.add_port("o2", port=straight.ports["o2"])

    sim = meep.Simulation()
    sim.geometry(component=component, stack=get_stack())
    sim.materials = {
        "si": Material(refractive_index=3.47),
        "SiO2": Material(refractive_index=1.44),
    }
    sim.solver(resolution=25, mode="2d", y_cut="auto", save_animation=True)
    sim.solver.stop_when_energy_decayed()
    sim.source_fiber(**FIBER)
    sim.monitors = ["o2"]
    # z_bounds=(-1.5, 4.6) is what the notebook's margin_z=(1.5, 0) resolves to (the
    # library's deprecation warning prints this tuple).
    sim.domain(pml=1.0, margin_x=0.5, margin_y=0.5, z_bounds=(-1.5, 4.6), extend_into_pml=True)
    sim.num_freqs = 21
    return sim


def main() -> None:
    """Draw the side view."""
    apply_style()
    sim = build_simulation()
    LOG.info("validate_config: %s", sim.validate_config())
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    sim.plot_2d(slices="y", aspect="auto", ax=ax)
    ax.grid(False)
    LOG.info("axes limits: x %s z %s", ax.get_xlim(), ax.get_ylim())
    LOG.info("title: %s", ax.get_title())
    for text in ax.texts:
        LOG.info("text: %r", text.get_text())
    legend = ax.get_legend()
    if legend is not None:
        LOG.info("legend: %s", [t.get_text() for t in legend.get_texts()])
    save(fig, "guide-grating-setup.png")


if __name__ == "__main__":
    main()
