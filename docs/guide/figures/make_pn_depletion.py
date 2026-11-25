"""Depletion-width figure for docs/guide/index.md.

Writes docs/guide/images/guide-pn-depletion.png: the lateral pn junction of
nbs/tcad_pn_junction.ipynb (two 0.3 um rib halves of 220 nm, two 0.5 um pads with
aluminium electrodes, 1e18 cm^-3 on both sides) with the depletion edges of the abrupt
depletion approximation at 0 V and 2 V reverse bias, and the total width against bias.
The widths come from ``gsim.common.stack.pn_junction`` (closed form, silicon
permittivity 11.9); no carrier solve is run here, so the figure needs neither DEVSIM nor
the cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _common import LOG, apply_style, save
from matplotlib.patches import Rectangle

from gsim.common.stack.pn_junction import (
    built_in_voltage,
    depletion_extents,
    depletion_width,
)

N_A = N_D = 1e18  # cm^-3, as in the notebook
FLANK_UM = 0.3  # width of each junction half
PAD_UM = 0.5  # width of each contact pad
RIB_HEIGHT_UM = 0.22
ELECTRODE_HEIGHT_UM = 0.5
N_COLOR, P_COLOR = "#9ecae1", "#fcae91"


def widths(v_reverse: float) -> tuple[float, float, float]:
    """Total width, p-side and n-side extents of the depleted region, in um."""
    total = depletion_width(N_A, N_D, v_reverse=v_reverse)
    x_p, x_n = depletion_extents(N_A, N_D, w_um=total)
    return total, x_p, x_n


def main() -> None:
    """Draw the section, the zoom on the junction and W(V)."""
    apply_style()
    v_bi = built_in_voltage(N_A, N_D)
    LOG.info("built-in voltage: %.3f V", v_bi)
    for volts in (0.0, 2.0):
        total, x_p, x_n = widths(volts)
        LOG.info(
            "%.0f V: W = %.1f nm (p side %.1f nm, n side %.1f nm)",
            volts,
            total * 1e3,
            x_p * 1e3,
            x_n * 1e3,
        )

    fig = plt.figure(figsize=(8.0, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    section = fig.add_subplot(grid[0, :])
    zoom = fig.add_subplot(grid[1, 0])
    curve = fig.add_subplot(grid[1, 1])

    # Section across the rib, junction at y = 0.
    spans = {
        "n pad": (-FLANK_UM - PAD_UM, -FLANK_UM, N_COLOR),
        "n": (-FLANK_UM, 0.0, N_COLOR),
        "p": (0.0, FLANK_UM, P_COLOR),
        "p pad": (FLANK_UM, FLANK_UM + PAD_UM, P_COLOR),
    }
    for label, (y0, y1, color) in spans.items():
        section.add_patch(
            Rectangle(
                (y0, 0),
                y1 - y0,
                RIB_HEIGHT_UM,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
            )
        )
        section.text((y0 + y1) / 2, -0.05, label, ha="center", va="top", fontsize=8)
    for label, key in (("cathode (Al)", "n pad"), ("anode (Al)", "p pad")):
        y0, y1, _ = spans[key]
        section.add_patch(
            Rectangle(
                (y0, RIB_HEIGHT_UM),
                y1 - y0,
                ELECTRODE_HEIGHT_UM,
                facecolor="#b59a3c",
                edgecolor="black",
                linewidth=0.8,
            )
        )
        section.text(
            (y0 + y1) / 2,
            RIB_HEIGHT_UM + ELECTRODE_HEIGHT_UM + 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    section.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    section.text(0.0, -0.2, "junction", ha="center", va="top", fontsize=8)
    section.add_patch(
        Rectangle(
            (-0.12, 0),
            0.24,
            RIB_HEIGHT_UM,
            fill=False,
            edgecolor="tab:green",
            linewidth=1.4,
            linestyle=":",
        )
    )
    section.annotate(
        "zoom below",
        xy=(0.12, RIB_HEIGHT_UM * 0.5),
        xytext=(0.5, 0.55),
        fontsize=8,
        color="tab:green",
        arrowprops={"arrowstyle": "-", "color": "tab:green", "linewidth": 0.8},
    )
    section.set_xlim(-FLANK_UM - PAD_UM - 0.05, FLANK_UM + PAD_UM + 0.05)
    section.set_ylim(-0.38, 0.95)
    section.set_aspect("equal")
    section.grid(False)
    section.set_xlabel("y (µm), across the guide")
    section.set_ylabel("z (µm)")
    section.set_title(
        "Lateral pn junction: 220 nm doped silicon, 10$^{18}$ cm$^{-3}$ on both sides",
        fontsize=9,
    )

    # Zoom on the junction with the depletion edges.
    zoom.add_patch(
        Rectangle((-0.12, 0), 0.12, RIB_HEIGHT_UM, facecolor=N_COLOR, edgecolor="none")
    )
    zoom.add_patch(
        Rectangle((0.0, 0), 0.12, RIB_HEIGHT_UM, facecolor=P_COLOR, edgecolor="none")
    )
    for volts, color, style in ((0.0, "tab:green", "-"), (2.0, "tab:purple", "--")):
        total, x_p, x_n = widths(volts)
        zoom.add_patch(
            Rectangle(
                (-x_n, 0),
                x_n + x_p,
                RIB_HEIGHT_UM,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                linestyle=style,
            )
        )
        zoom.plot(
            [],
            [],
            color=color,
            linestyle=style,
            label=f"{volts:.0f} V: W = {total * 1e3:.0f} nm",
        )
    zoom.axvline(0.0, color="black", linestyle="--", linewidth=0.6)
    zoom.set_xlim(-0.12, 0.12)
    zoom.set_ylim(-0.02, 0.28)
    zoom.set_aspect("equal")
    zoom.grid(False)
    zoom.set_xlabel("y (µm)")
    zoom.set_ylabel("z (µm)")
    zoom.set_title("Depletion edges (reverse bias)", fontsize=9)
    zoom.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=8,
        framealpha=1,
        ncol=1,
    )

    # W(V).
    volts = np.linspace(0.0, 4.0, 81)
    curve.plot(volts, [widths(v)[0] * 1e3 for v in volts], color="tab:blue")
    for v, color in ((0.0, "tab:green"), (2.0, "tab:purple")):
        curve.plot([v], [widths(v)[0] * 1e3], "o", color=color)
    curve.set_xlabel("Reverse bias (V)")
    curve.set_ylabel("Total depletion width W (nm)")
    curve.set_title("Abrupt-junction depletion approximation", fontsize=9)
    save(fig, "guide-pn-depletion.png")


if __name__ == "__main__":
    main()
