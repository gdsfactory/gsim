"""Figures for docs/guide/phase-shifters.md.

Writes docs/guide/images/phase-shifter-setup.png (the device of the page's outline,
drawn from the mesh that ``tcad.Simulation.mesh`` makes) and
docs/guide/images/phase-shifter-cv.png (local DEVSIM: carriers across the junction and
C(V) against the depletion approximation).

The device, the simulation and the sweep are the cells of nbs/tcad_pn_junction.ipynb,
executed here, so the figures show the notebook's device and numbers. DEVSIM runs
locally, nothing touches the cloud. Run from any folder; needs the ``tcad`` extra. The
sweep takes about a minute.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import meshio
import numpy as np
from _common import LOG, ROOT, apply_style, save, today
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

NOTEBOOK = ROOT / "nbs" / "tcad_pn_junction.ipynb"
DOPED_COLORS = {"donor": "#9ecae1", "acceptor": "#fcae91"}  # n: blue, p: red
OTHER_COLOR = "#eeeeee"
REGION_COLORS = {
    "air": "white",
    "sio2": "#f0f0f0",
    "slab90": "#bdbdbd",
}  # undoped slab darker than the oxide


def run_notebook_cells(
    cell_ids: tuple[str, ...], scratch: Path, namespace: dict | None = None
) -> dict:
    """Run code cells of the PN junction notebook, in order, as one script.

    The cells are written to a file in ``scratch`` and run with ``runpy``. Pass the
    namespace returned by an earlier call to continue with the same objects.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {
        c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    }
    script = scratch / f"notebook_cells_{'_'.join(cell_ids)}.py"
    script.write_text("\n\n".join(cells[i] for i in cell_ids), encoding="utf-8")
    return runpy.run_path(str(script), init_globals=namespace, run_name="pn_notebook")


def sci(value: float) -> str:
    """Format a doping level as a power of ten in cm^-3 for a Matplotlib label."""
    exponent = int(np.floor(np.log10(value)))
    return f"10$^{{{exponent}}}$ cm$^{{-3}}$"


def setup_figure(ns: dict) -> None:
    """Draw the device cross-section as meshed for the charge solve."""
    mesh = meshio.read(ns["mesh"].mesh_path, file_format="gmsh")
    points = mesh.points[:, :2]
    names = {tag: name for name, (tag, dim) in mesh.field_data.items() if dim == 2}
    triangles = mesh.cells_dict["triangle"]
    tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    LOG.info("2D regions of the charge mesh: %s", sorted(names.values()))

    spans, rib_height, center = ns["spans"], ns["RIB_HEIGHT"], ns["CENTER_Y"]
    dopant = {
        "n_pad": "donor",
        "n_rib": "donor",
        "p_rib": "acceptor",
        "p_pad": "acceptor",
    }

    LOG.info(
        "mesh extent y: %s %s z: %s %s",
        points[:, 0].min(),
        points[:, 0].max(),
        points[:, 1].min(),
        points[:, 1].max(),
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    for tag, name in names.items():
        color = (
            DOPED_COLORS[dopant[name]]
            if name in dopant
            else REGION_COLORS.get(name, OTHER_COLOR)
        )
        ax.add_collection(
            PolyCollection(
                points[triangles[tags == tag]],
                facecolors=color,
                edgecolors="none",
                rasterized=True,
            )
        )
    # Electrodes are holes in the semiconductor mesh: draw them where the layer stack
    # puts them.
    for pad, electrode in ns["electrodes"].items():
        y0, y1 = spans[pad]
        layer = ns["layers"][electrode]
        ax.add_patch(
            Rectangle(
                (y0, layer.zmin),
                y1 - y0,
                layer.thickness,
                facecolor="#b59a3c",
                edgecolor="black",
                linewidth=0.8,
            )
        )
    for y0, y1 in spans.values():
        ax.add_patch(
            Rectangle(
                (y0, 0),
                y1 - y0,
                rib_height,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
            )
        )

    # Labels: doped regions below the silicon, electrodes above them.
    for name, (y0, y1) in spans.items():
        kind = "n" if name.startswith("n") else "p"
        label = f"{kind} pad" if name.endswith("pad") else kind
        ax.text((y0 + y1) / 2, -0.12, label, ha="center", va="top", fontsize=8)
    for pad, electrode in ns["electrodes"].items():
        y0, y1 = spans[pad]
        name = "cathode" if pad.startswith("n") else "anode"
        ax.text(
            (y0 + y1) / 2,
            ns["layers"][electrode].zmax + 0.05,
            f"{name}\n(Al)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axvline(center, color="black", linestyle="--", linewidth=0.9)
    ax.text(center, -0.42, "junction", ha="center", va="top", fontsize=8)

    # Charge-solve window and refinement box, read back from the notebook's settings.
    y_min, y_max, z_min, z_max, size_um = ns["sim"].solver.refinement_boxes[
        0
    ]  # (y0, y1, z0, z1, size) in um
    ax.add_patch(
        Rectangle(
            (y_min, z_min),
            y_max - y_min,
            z_max - z_min,
            fill=False,
            edgecolor="tab:green",
            linewidth=1.6,
            linestyle=":",
        )
    )
    ax.annotate(
        f"refinement box\n({size_um * 1e3:.0f} nm elements)",
        xy=((y_min + y_max) / 2, z_max),
        xytext=(center, 1.02),
        fontsize=8,
        color="tab:green",
        ha="center",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": "tab:green",
            "linewidth": 0.8,
            "shrinkA": 0,
        },
    )
    window = (center - 1.2, center + 1.2)  # sim.domain(window=...) of the notebook
    ax.set_xlim(
        points[:, 0].min() - 0.08, points[:, 0].max() + 0.08
    )  # the window edges are the mesh edges
    ax.set_ylim(-0.75, 1.3)
    for edge in window:
        ax.axvline(edge, color="gray", linestyle="-.", linewidth=0.9)
    ax.text(
        window[0] + 0.03,
        1.2,
        "charge-solve window",
        fontsize=8,
        color="dimgray",
        ha="left",
        va="center",
    )
    # Regions that are neither doped nor declared insulating take no part in the solve
    # (tcad Contact docstring).
    ax.text(
        center + 1.0,
        0.45,
        "SiO$_2$\n(not solved)",
        fontsize=8,
        color="dimgray",
        ha="center",
        va="center",
    )
    ax.text(
        center - 1.0,
        0.3,
        "90 nm slab\n(not solved)",
        fontsize=8,
        color="dimgray",
        ha="center",
        va="bottom",
    )
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("y (µm), across the guide")
    ax.set_ylabel("z (µm)")
    ax.set_title(
        "Device cross-section as meshed for the charge solve; "
        f"every doped region {sci(ns['N_D'])}"
    )
    save(fig, "phase-shifter-setup.png")


def result_figure(ns: dict) -> None:
    """Draw carriers at 0 V and 2 V and C(V) against the depletion approximation."""
    from gsim import tcad
    from gsim.common.stack.pn_junction import PNJunctionConfig

    sweep = ns["sweep"]
    rib_height, center = ns["RIB_HEIGHT"], ns["CENTER_Y"]
    comparison = tcad.compare_capacitance(
        PNJunctionConfig(na_cm3=ns["N_A"], nd_cm3=ns["N_D"]),
        sweep,
        height_um=rib_height,
    )
    LOG.info("biases (V): %s", [float(v) for v in comparison.v_reverse])
    LOG.info(
        "C tcad (pF/cm): %s",
        [round(float(c) * 1e12, 4) for c in comparison.c_tcad_f_per_cm],
    )
    LOG.info(
        "C analytic (pF/cm): %s",
        [round(float(c) * 1e12, 4) for c in comparison.c_analytic_f_per_cm],
    )
    deviation = comparison.c_tcad_f_per_cm / comparison.c_analytic_f_per_cm - 1.0
    LOG.info(
        "deviation per point (%%): %s", [round(100 * float(d), 2) for d in deviation]
    )
    LOG.info("largest relative deviation: %s", float(comparison.max_relative_deviation))
    LOG.info(
        "C monotonically falling: %s",
        bool(np.all(np.diff(sweep.capacitance_f_per_cm) < 0)),
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)
    for point, style in ((sweep.points[0], "-"), (sweep.points[-1], "--")):
        carriers = point.carriers
        cut = np.abs(carriers.y_um - rib_height / 2) < 0.01  # at mid-height of the rib
        order = np.argsort(carriers.x_um[cut])
        y = carriers.x_um[cut][order]
        emptied = (carriers.electrons_cm3[cut] < 1e16) & (
            carriers.holes_cm3[cut] < 1e16
        )
        width_nm = 1e3 * (
            carriers.x_um[cut][emptied].max() - carriers.x_um[cut][emptied].min()
        )
        LOG.info(
            "%.0f V: width of the strip with n and p below 1e16 cm^-3: %.0f nm",
            point.bias_v,
            width_nm,
        )
        left.semilogy(
            y,
            carriers.electrons_cm3[cut][order],
            "C0" + style,
            label=f"electrons, {point.bias_v:.0f} V",
        )
        left.semilogy(
            y,
            carriers.holes_cm3[cut][order],
            "C3" + style,
            label=f"holes, {point.bias_v:.0f} V",
        )
    left.set_xlim(center - 0.15, center + 0.15)
    left.set_ylim(1e8, 3e18)
    left.set_xlabel("y (µm), across the junction")
    left.set_ylabel("Carrier density (cm$^{-3}$)")
    left.set_title("Carriers at mid-height of the rib")
    left.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), framealpha=1)

    right.plot(
        comparison.v_reverse,
        comparison.c_tcad_f_per_cm * 1e12,
        "o-",
        color="tab:blue",
        label="DEVSIM drift-diffusion",
    )
    right.plot(
        comparison.v_reverse,
        comparison.c_analytic_f_per_cm * 1e12,
        "--",
        color="tab:orange",
        label="depletion approximation",
    )
    right.set_xlabel("Reverse bias $V_R$ (V)")
    right.set_ylabel("Capacitance (pF/cm)")
    right.set_title("C(V) of the junction")
    right.legend(loc="upper right", framealpha=1)
    save(fig, "phase-shifter-cv.png")
    LOG.info("run date: %s", today())


def main() -> None:
    """Run the notebook cells and draw both figures of the page."""
    apply_style()
    scratch = Path(tempfile.mkdtemp(prefix="guidefig-pn-"))
    os.chdir(scratch)  # the notebook writes its mesh to ./tcad-sim-pn-junction
    ns = run_notebook_cells(("2", "4"), scratch)  # device, simulation, mesh
    setup_figure(ns)
    # The sweep cell continues with the device and the simulation of the cells above.
    ns = run_notebook_cells(("6",), scratch, ns)
    result_figure(ns)


if __name__ == "__main__":
    main()
