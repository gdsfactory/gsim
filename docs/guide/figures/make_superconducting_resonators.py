"""Set-up figure for the "Superconducting resonators" section of docs/guide/index.md.

Writes docs/guide/images/resonator-setup.png: the page's ``coupled_strip`` (read from
the page, not retyped) in top view with its two CPW ports and the airbox of the page's
``set_airbox`` call, and a zoom on one coupling end. gsim has no static plot of Palace
ports and airbox, so the polygons come from the component and the port and airbox
numbers from the page's snippet. Nothing is meshed or run.
"""

from __future__ import annotations

import ast
import re
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
from _common import HERE, LOG, apply_style, save
from matplotlib.patches import Polygon, Rectangle

PAGE = HERE.parent / "index.md"  # the "Superconducting resonators" section
METAL_COLOR = "#c9a227"


def component_from_page():
    """Build ``coupled_strip`` with the source text of the page's first code block."""
    text = PAGE.read_text(encoding="utf-8")
    function = re.search(
        r"(@gf\.cell\ndef coupled_strip.*?\n    return c\n)", text, re.DOTALL
    )
    if function is None:
        raise ValueError("coupled_strip not found on the page")
    # The guide is one page with many snippets: read the numbers from the resonator
    # snippet, which starts at the function.
    snippet = text[function.start() :]
    settings = re.search(r"settings = (\{[^}]*\})", snippet)
    airbox = re.search(
        r"set_airbox\(margin_x=([\d.]+), margin_y=([\d.]+), "
        r"z_above=([\d.]+), z_below=([\d.]+)\)",
        snippet,
    )
    if settings is None or airbox is None:
        raise ValueError("settings or set_airbox call not found on the page")
    gf.gpdk.PDK.activate()
    with tempfile.TemporaryDirectory(prefix="guidefig-resonator-") as scratch:
        script = Path(scratch) / "coupled_strip.py"
        script.write_text(function.group(1), encoding="utf-8")
        namespace = runpy.run_path(
            str(script), init_globals={"gf": gf, "LAYER": gf.gpdk.LAYER.M1}
        )
    cpw = ast.literal_eval(settings.group(1))  # a dict literal of numbers
    margins = dict(
        zip(
            ("margin_x", "margin_y", "z_above", "z_below"),
            map(float, airbox.groups()),
            strict=True,
        )
    )
    return namespace["coupled_strip"](), cpw, margins


def draw_metal(ax, component) -> None:
    """Draw the metal polygons of the component on ``ax``."""
    dbu = component.kcl.dbu
    for polygons in component.get_polygons(by="tuple").values():
        for polygon in polygons:
            points = [(p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()]
            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor=METAL_COLOR,
                    edgecolor="black",
                    linewidth=0.5,
                )
            )


def draw_ports(ax, component, cpw, *, label: bool = True) -> None:
    """Draw the two port elements of each CPW port as red bars on ``ax``."""
    inner = cpw["s_width"] / 2
    half = inner + cpw["gap_width"]
    for port in component.ports:
        x, y = port.dcenter
        # add_cpw_port: the two port elements are the gaps either side of the signal
        # conductor.
        for sign in (-1, 1):
            ax.plot(
                [x, x],
                [y + sign * inner, y + sign * half],
                color="tab:red",
                linewidth=2.6,
                solid_capstyle="butt",
            )
        if label:
            ax.annotate(
                f"port {port.name}",
                (x, y + half),
                xytext=(14, 30),
                textcoords="offset points",
                ha="left",
                fontsize=8,
                color="tab:red",
                arrowprops={"arrowstyle": "-", "color": "tab:red", "linewidth": 0.8},
            )


def main() -> None:
    """Draw the top view and the zoom on one coupling end."""
    apply_style()
    component, cpw, margins = component_from_page()
    x0, y0, x1, y1 = (
        float(v)
        for v in (component.dxmin, component.dymin, component.dxmax, component.dymax)
    )
    LOG.info(
        "component bbox (µm): %s %s %s %s | CPW settings: %s | airbox: %s",
        x0,
        y0,
        x1,
        y1,
        cpw,
        margins,
    )

    fig, (top, zoom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.6),
        gridspec_kw={"height_ratios": (1, 2.6)},
        constrained_layout=True,
    )

    # Top view of the whole structure with the airbox outline.
    draw_metal(top, component)
    draw_ports(top, component, cpw, label=False)
    ax_x0, ax_x1 = x0 - margins["margin_x"], x1 + margins["margin_x"]
    ax_y0, ax_y1 = y0 - margins["margin_y"], y1 + margins["margin_y"]
    top.add_patch(
        Rectangle(
            (ax_x0, ax_y0),
            ax_x1 - ax_x0,
            ax_y1 - ax_y0,
            fill=False,
            edgecolor="tab:blue",
            linestyle="--",
            linewidth=1.0,
        )
    )
    top.text(
        0,
        ax_y1 + 18,
        f"airbox: {margins['margin_y']:.0f} µm margin in y, "
        f"{margins['z_above']:.0f} µm above and "
        f"{margins['z_below']:.0f} µm below the stack; outer wall PMC",
        ha="center",
        va="bottom",
        fontsize=8,
        color="tab:blue",
    )
    top.text(0, 0, "floating strip", ha="center", va="bottom", fontsize=8)
    top.text(x0 + 30, 0, "o1", ha="left", va="bottom", fontsize=8, color="tab:red")
    top.text(x1 - 30, 0, "o2", ha="right", va="bottom", fontsize=8, color="tab:red")
    top.set_xlim(ax_x0 - 60, ax_x1 + 60)
    top.set_ylim(ax_y0 - 25, ax_y1 + 90)
    top.set_aspect("equal")
    top.grid(False)
    top.set_xlabel("x (µm)")
    top.set_ylabel("y (µm)")
    top.set_title(f"Top view, metal1 (strip length {2 * 1000:.0f} µm)")

    # Zoom on the left coupling end.
    draw_metal(zoom, component)
    draw_ports(zoom, component, cpw)
    zoom.set_xlim(-1245, -900)
    zoom.set_ylim(-70, 70)
    zoom.annotate(
        "coupling gap 15 µm",
        xy=(-1007.5, 0),
        xytext=(-1007.5, 62),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    zoom.annotate(
        "feed line",
        xy=(-1115, 0),
        xytext=(-1115, -62),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    zoom.annotate(
        f"CPW gap {cpw['gap_width']:.0f} µm",
        xy=(-1135, 8),
        xytext=(-1135, 62),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    zoom.annotate(
        "floating strip",
        xy=(-950, 0),
        xytext=(-950, -62),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    zoom.text(-1060, 31, "ground plane", ha="center", va="center", fontsize=8)
    zoom.text(-1060, -31, "ground plane", ha="center", va="center", fontsize=8)
    zoom.set_aspect("equal")
    zoom.grid(False)
    zoom.set_xlabel("x (µm)")
    zoom.set_ylabel("y (µm)")
    zoom.set_title(
        "Zoom on the o1 end: the port elements (red) are the two gaps either side "
        "of the signal conductor"
    )
    save(fig, "resonator-setup.png")


if __name__ == "__main__":
    main()
