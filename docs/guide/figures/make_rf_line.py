"""Layout figure of the coplanar line for docs/guide/index.md.

Writes docs/guide/images/guide-cpw-layout.png: the ``gsg_line`` of
docs/getting_started_palace.md (ground-signal-ground, generic PDK, metal1) in top view
with its two lumped CPW port elements per end, and its cross-section at mid-length with
the dimensions. The metal and oxide heights come from ``get_stack(substrate_thickness=2.0)``
(the stack that ``DrivenSim.set_stack(substrate_thickness=2.0)`` reads for this PDK); the
cross-section is stretched in z. Needs gdsfactory only; nothing is meshed or run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
from _common import LOG, apply_style, save
from matplotlib.patches import Rectangle

from gsim.common.stack import get_stack

LENGTH, S_WIDTH, G_WIDTH, GAP_WIDTH = 300.0, 10.0, 40.0, 6.0
METAL_COLOR = "#c9a227"


def gsg_line(length=LENGTH, s_width=S_WIDTH, g_width=G_WIDTH, gap_width=GAP_WIDTH):
    """The page's line (docs/getting_started_palace.md)."""
    layer = gf.gpdk.LAYER.M1
    component = gf.Component()
    top = component << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    top.move((0, (g_width + s_width) / 2 + gap_width))
    component << gf.c.rectangle((length, s_width), centered=True, layer=layer)
    bottom = component << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    bottom.move((0, -(g_width + s_width) / 2 - gap_width))
    for name, x, orientation in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
        component.add_port(
            name=name, center=(x, 0), width=s_width, orientation=orientation,
            port_type="electrical", layer=layer,
        )  # fmt: skip
    return component


def dimension(ax, x, y0, y1, text, *, side=1):
    """Vertical dimension arrow at x from y0 to y1 with a text on one side."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops={"arrowstyle": "<->", "linewidth": 0.8})
    ax.text(x + side * 3, (y0 + y1) / 2, text, ha="left" if side > 0 else "right", va="center", fontsize=8)


def main() -> None:
    """Draw top view and cross-section."""
    apply_style()
    gf.gpdk.PDK.activate()
    component = gsg_line()
    stack = get_stack(substrate_thickness=2.0)
    metal = stack.layers["metal1"]
    oxide = stack.dielectrics[0]
    LOG.info(
        "metal1 z %.2f..%.2f um (%.2f um thick); oxide fill z %.1f..%.1f um",
        metal.zmin, metal.zmax, metal.zmax - metal.zmin, oxide["zmin"], oxide["zmax"],
    )  # fmt: skip
    half = (G_WIDTH + S_WIDTH / 2 + GAP_WIDTH)  # outer edge of the ground, 51 um

    fig, (top, section) = plt.subplots(
        2, 1, figsize=(8.0, 5.6), gridspec_kw={"height_ratios": (1.0, 1.1)}, constrained_layout=True
    )

    # Top view.
    dbu = component.kcl.dbu
    for polygons in component.get_polygons(by="tuple").values():
        for polygon in polygons:
            points = [(p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()]
            top.add_patch(plt.Polygon(points, closed=True, facecolor=METAL_COLOR, edgecolor="black", linewidth=0.5))
    for port in component.ports:
        x, y = port.dcenter
        for sign in (-1, 1):
            top.plot([x, x], [y + sign * S_WIDTH / 2, y + sign * (S_WIDTH / 2 + GAP_WIDTH)],
                     color="tab:red", linewidth=2.6, solid_capstyle="butt")  # fmt: skip
        top.text(x + (6 if x < 0 else -6), 0, port.name, color="tab:red", fontsize=8,
                 ha="left" if x < 0 else "right", va="center")  # fmt: skip
    top.text(0, 0, "signal", ha="center", va="center", fontsize=8)
    top.text(0, (G_WIDTH + S_WIDTH) / 2 + GAP_WIDTH, "ground", ha="center", va="center", fontsize=8)
    top.text(0, -(G_WIDTH + S_WIDTH) / 2 - GAP_WIDTH, "ground", ha="center", va="center", fontsize=8)
    top.annotate("", xy=(-LENGTH / 2, -half - 12), xytext=(LENGTH / 2, -half - 12),
                 arrowprops={"arrowstyle": "<->", "linewidth": 0.8})  # fmt: skip
    top.text(0, -half - 14, f"length {LENGTH:.0f} µm", ha="center", va="top", fontsize=8)
    top.set_xlim(-LENGTH / 2 - 25, LENGTH / 2 + 25)
    top.set_ylim(-half - 30, half + 10)
    top.set_aspect("equal")
    top.grid(False)
    top.set_xlabel("x (µm)")
    top.set_ylabel("y (µm)")
    top.set_title("Top view, metal1; the two red bars at each end are one CPW port", fontsize=9)

    # Cross-section at mid-length.
    lim = half + 14
    z_lo, z_hi = oxide["zmin"], oxide["zmax"]
    section.add_patch(Rectangle((-lim, z_lo), 2 * lim, z_hi - z_lo, facecolor="#e8eef5", edgecolor="none"))
    for y0, y1 in ((-half, -half + G_WIDTH), (-S_WIDTH / 2, S_WIDTH / 2), (half - G_WIDTH, half)):
        section.add_patch(
            Rectangle((y0, metal.zmin), y1 - y0, metal.zmax - metal.zmin, facecolor=METAL_COLOR, edgecolor="black", linewidth=0.6)
        )
    section.text(-lim + 3, 0.3, "SiO$_2$ fill", fontsize=8, color="0.3")
    section.text(0, metal.zmax + 0.35, "signal", ha="center", fontsize=8)
    section.text(-half + G_WIDTH / 2, metal.zmax + 0.35, "ground", ha="center", fontsize=8)
    section.text(half - G_WIDTH / 2, metal.zmax + 0.35, "ground", ha="center", fontsize=8)
    for sign in (-1, 1):
        y_a, y_b = sorted((sign * S_WIDTH / 2, sign * (S_WIDTH / 2 + GAP_WIDTH)))
        section.annotate("", xy=(y_a, 0.7), xytext=(y_b, 0.7), arrowprops={"arrowstyle": "<->", "linewidth": 0.8})
        section.text((y_a + y_b) / 2, 0.15, f"gap {GAP_WIDTH:.0f}", ha="center", fontsize=8)
    section.annotate("", xy=(-S_WIDTH / 2, 3.0), xytext=(S_WIDTH / 2, 3.0), arrowprops={"arrowstyle": "<->", "linewidth": 0.8})
    section.text(0, 3.25, f"{S_WIDTH:.0f}", ha="center", fontsize=8)
    section.annotate("", xy=(half - G_WIDTH, 4.2), xytext=(half, 4.2), arrowprops={"arrowstyle": "<->", "linewidth": 0.8})
    section.text(half - G_WIDTH / 2, 4.4, f"{G_WIDTH:.0f}", ha="center", fontsize=8)
    section.set_xlim(-lim, lim)
    section.set_ylim(z_lo - 0.3, z_hi + 0.4)
    section.grid(False)
    section.set_xlabel("y (µm)")
    section.set_ylabel("z (µm), stretched")
    section.set_title(
        f"Cross-section at x = 0: metal1 {metal.zmax - metal.zmin:.1f} µm thick at z = {metal.zmin:.1f} µm", fontsize=9
    )
    save(fig, "guide-cpw-layout.png")


if __name__ == "__main__":
    main()
