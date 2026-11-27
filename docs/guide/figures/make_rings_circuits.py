"""Rings and circuits figure for docs/guide/index.md.

Writes docs/guide/images/guide-ring-circuit.png. Top: the layouts of the generic-PDK
``ring_single`` and ``mzi`` as gdsfactory builds them. Bottom left: the circuit idea (a
coupler section plus a loop of waveguide). Bottom right: the through-port spectrum of
that circuit, computed locally with SAX from an ideal coupler and a straight-waveguide
model. The coupler and the loop use illustrative numbers that are set in this script;
they are not extracted from the layouts and not from an FDTD run. No cloud call.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import sax
from _common import LOG, apply_style, save
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

COUPLING = 0.02  # power coupling of the ideal coupler (illustrative)
LOOP_LENGTH_UM = 70.0  # waveguide length of the loop (illustrative)
LOOP_LOSS_DB_CM = 5.0  # (illustrative)
NEFF, NG = 2.34, 3.4  # defaults of sax.models.straight


def draw_layout(ax, component, title: str) -> None:
    """Fill the polygons of the component and label its ports."""
    dbu = component.kcl.dbu
    shapes = [
        [(p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()]
        for polygons in component.get_polygons(by="tuple").values()
        for polygon in polygons
    ]
    ax.add_collection(
        PolyCollection(
            shapes, facecolors="#7fb2e5", edgecolors="#23415f", linewidths=0.4
        )
    )
    for port in component.ports:
        x, y = port.dcenter
        ax.plot([x], [y], "o", color="tab:red", markersize=3.5)
        ax.annotate(
            port.name,
            (x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:red",
        )
    x0, y0, x1, y1 = (
        float(v)
        for v in (component.dxmin, component.dymin, component.dxmax, component.dymax)
    )
    ax.set_xlim(x0 - 3, x1 + 3)
    ax.set_ylim(y0 - 3, y1 + 5)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (µm)")
    ax.set_title(f"{title}: {x1 - x0:.0f} x {y1 - y0:.0f} µm", fontsize=9)
    LOG.info("%s bbox: %.3f %.3f %.3f %.3f", title, x0, y0, x1, y1)


def ring_spectrum():
    """Through-port power of an all-pass ring: ideal coupler and straight loop (SAX)."""
    netlist = {
        "instances": {
            "dc": {"component": "coupler", "settings": {"coupling": COUPLING}},
            "loop": {
                "component": "straight",
                "settings": {"length": LOOP_LENGTH_UM, "loss_dB_cm": LOOP_LOSS_DB_CM},
            },
        },
        "connections": {"dc,out1": "loop,in0", "loop,out0": "dc,in1"},
        "ports": {"in": "dc,in0", "thru": "dc,out0"},
    }
    models = {"coupler": sax.models.coupler_ideal, "straight": sax.models.straight}
    circuit, _ = sax.circuit(netlist, models)  # ty: ignore[invalid-argument-type]
    wl = np.linspace(1.50, 1.60, 4001)
    power = np.abs(np.asarray(circuit(wl=wl)[("in", "thru")])) ** 2
    dips = [
        i
        for i in range(1, len(wl) - 1)
        if power[i] < power[i - 1] and power[i] < power[i + 1]
    ]
    fsr_nm = float(np.diff(wl[dips]).mean() * 1000)
    LOG.info(
        "ring spectrum: min %.4f (%.2f dB), max %.4f, %d dips, mean spacing %.2f nm, "
        "lambda^2/(ng L) at 1.55 um = %.2f nm",
        power.min(),
        10 * np.log10(power.min()),
        power.max(),
        len(dips),
        fsr_nm,
        1.55**2 / (NG * LOOP_LENGTH_UM) * 1000,
    )
    return wl, power


def draw_schematic(ax) -> None:
    """The circuit idea: a bus waveguide, a coupler section and a loop."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([0.4, 9.6], [1.0, 1.0], color="#23415f", linewidth=2.2)
    ax.add_patch(Circle((5, 3.55), 1.5, fill=False, edgecolor="#23415f", linewidth=2.2))
    ax.add_patch(
        Rectangle(
            (3.6, 0.55),
            2.8,
            1.35,
            facecolor="#ffe9b3",
            edgecolor="tab:orange",
            linewidth=1.2,
            alpha=0.8,
        )
    )
    ax.text(
        5,
        0.12,
        "coupler section",
        ha="center",
        va="center",
        fontsize=8,
        color="tab:orange",
    )
    ax.text(5, 3.55, "loop\nstraight model", ha="center", va="center", fontsize=8)
    ax.text(0.4, 1.45, "in", ha="left", fontsize=8, color="tab:red")
    ax.text(9.6, 1.45, "thru", ha="right", fontsize=8, color="tab:red")
    ax.add_patch(
        FancyArrowPatch(
            (0.4, 1.0), (1.4, 1.0), arrowstyle="-|>", mutation_scale=10, color="tab:red"
        )
    )
    ax.text(
        9.7,
        4.9,
        "coupler: S-matrix\nfrom FDTD (or analytic)",
        ha="right",
        va="center",
        fontsize=8,
        color="tab:orange",
    )
    ax.annotate(
        "",
        xy=(6.3, 1.9),
        xytext=(7.6, 4.5),
        arrowprops={"arrowstyle": "-", "color": "tab:orange", "linewidth": 0.8},
    )
    ax.set_title("Circuit: coupler section + loop", fontsize=9)


def main() -> None:
    """Draw the figure."""
    apply_style()
    gf.gpdk.PDK.activate()
    ring = gf.get_component("ring_single")
    mzi = gf.get_component("mzi")
    wl, power = ring_spectrum()

    fig, axes = plt.subplots(
        2, 2, figsize=(8.0, 6.2), constrained_layout=True, height_ratios=(1.0, 1.0)
    )
    draw_layout(axes[0, 0], ring, "Ring (gpdk ring_single)")
    draw_layout(axes[0, 1], mzi, "MZI (gpdk mzi)")
    draw_schematic(axes[1, 0])
    spectrum = axes[1, 1]
    spectrum.plot(wl * 1000, 10 * np.log10(power), color="tab:blue")
    spectrum.set_xlabel("Wavelength (nm)")
    spectrum.set_ylabel("Through power (dB)")
    spectrum.set_title(
        f"SAX: \u03ba\u00b2 = {COUPLING}, L = {LOOP_LENGTH_UM:.0f} µm, "
        f"{LOOP_LOSS_DB_CM:g} dB/cm",
        fontsize=9,
    )
    save(fig, "guide-ring-circuit.png")


if __name__ == "__main__":
    main()
