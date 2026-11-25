"""Draw the figures of the "Mesh quality" section of ``docs/api/tcad.md``.

The device is the lateral PN junction of ``nbs/tcad_pn_junction.ipynb``: a
220 nm silicon rib cut into four doped strips along the horizontal axis
(n pad, n half, p half, p pad), with an aluminium electrode on each pad.
Every figure solves it twice, once the way ``gsim.tcad`` did before the
change and once as it does now, and saves a PNG next to this script:

* ``mesh_quality_couples.png``: DEVSIM's own edge couples against the signed
  ones, on a coarse mesh and on the notebook's refined mesh;
* ``mesh_quality_grading.png``: the default mesh against a graded one;
* ``mesh_quality_contact_ends.png``: the potential at the end of a Contact,
  without and with the ends held;
* ``mesh_quality_doping_edges.png``: the nodes a doping box on the top
  surface reaches, without and with the slack on its edges.

There is no public option that restores the earlier behaviour, so the
"before" runs switch off the new code by patching three private names of
``gsim.tcad``: ``device._install_signed_couples``,
``device._write_contact_ends`` and ``doping._EDGE_TOLERANCE_UM``. Those
names can change with the package, and this script has to follow them.

Needs the ``tcad`` extra (DEVSIM) and matplotlib. Run it from the
repository root::

    python docs/assets/tcad/make_mesh_quality.py

It takes about three minutes; the printed numbers are the ones quoted in the
text of the guide.
"""

# ruff: noqa: INP001, SLF001, T201

from __future__ import annotations

import tempfile
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from unittest.mock import patch

import gdsfactory as gf
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import PolyCollection

import gsim.tcad.device as tcad_device
import gsim.tcad.doping as tcad_doping
from gsim import tcad
from gsim.common.cross_section import build_doped_cross_section
from gsim.common.stack.extractor import Layer
from gsim.common.stack.materials import make_doped_materials
from gsim.common.stack.pn_junction import PNJunctionConfig
from gsim.tcad.mesh import UM_TO_CM
from gsim.tcad.runtime import require_devsim
from gsim.tcad.silicon import intrinsic_density_cm3

OUTPUT_DIR = Path(__file__).resolve().parent

CENTER_Y = -20.0  # junction position (um)
FLANK = 0.3  # width of each junction half (um)
PAD = 0.5  # width of each contact pad (um)
RIB_HEIGHT = 0.22  # um
N_A = N_D = 1e18  # cm^-3
BIASES = np.linspace(0.0, 2.0, 5)  # V on the cathode
SOLVED = ("n_pad", "n_rib", "p_rib", "p_pad", "sio2")
SILICON = SOLVED[:4]

#: The depletion edges move with bias: hold the mesh fine over the strip.
BOX = [(CENTER_Y - 0.12, CENTER_Y + 0.12, 0.0, RIB_HEIGHT, 0.01)]
#: The notebook's mesh settings, with and without its refinement box.
NOTEBOOK = {"mesh_preset": "coarse", "refined_mesh_size_um": 0.02}
NOTEBOOK |= {"max_mesh_size_um": 40.0}
COARSE = dict(NOTEBOOK)
REFINED = NOTEBOOK | {"refinement_boxes": BOX}
#: The graded counterpart: halve refined_mesh_size_um to keep the size on
#: the contacts and the junction (see the guide).
GRADED = REFINED | {
    "refined_mesh_size_um": 0.01,
    "mesh_grading": tcad.MeshGrading(growth_per_um=0.25),
}

COLORS = {"before": "#c0392b", "after": "#1f6fb2", "reference": "0.15"}


# ----------------------------------------------------------------------
# The device
# ----------------------------------------------------------------------


def build_geometry() -> tuple[gf.Component, object]:
    """Return the component and stack of the notebook's PN junction."""
    gf.gpdk.PDK.activate()
    spans = {
        "n_pad": (CENTER_Y - FLANK - PAD, CENTER_Y - FLANK),
        "n_rib": (CENTER_Y - FLANK, CENTER_Y),
        "p_rib": (CENTER_Y, CENTER_Y + FLANK),
        "p_pad": (CENTER_Y + FLANK, CENTER_Y + FLANK + PAD),
    }
    electrodes = {"n_pad": "cathode_metal", "p_pad": "anode_metal"}
    component = gf.Component()
    component << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    layers = {}
    for i, (name, (y0, y1)) in enumerate(spans.items()):
        strip = component << gf.c.rectangle((10.0, y1 - y0), layer=(30, i))
        strip.y = (y0 + y1) / 2
        layers[name] = Layer(
            name=name,
            gds_layer=(30, i),
            zmin=0.0,
            zmax=RIB_HEIGHT,
            thickness=RIB_HEIGHT,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    for i, (pad, electrode) in enumerate(electrodes.items()):
        y0, y1 = spans[pad]
        metal = component << gf.c.rectangle((10.0, y1 - y0), layer=(46, i))
        metal.y = (y0 + y1) / 2
        layers[electrode] = Layer(
            name=electrode,
            gds_layer=(46, i),
            zmin=RIB_HEIGHT,
            zmax=RIB_HEIGHT + 0.5,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )
    stack, _ = build_doped_cross_section(
        component,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={
            "layer_specs": layers,
            "materials": make_doped_materials(
                [(name, 1.6e3) for name in spans], permittivity=11.9
            ),
        },
        verbose=False,
    )
    return component, stack


COMPONENT, STACK = build_geometry()


def make_sim(
    solver: dict,
    *,
    oxide: bool = False,
    pad_cm3: float = N_D,
    rib_cm3: float = N_D,
    y_range: tuple[float, float] | None = None,
) -> tcad.Simulation:
    """Configure the diode; ``y_range`` makes each doping a box on the slab."""
    sim = tcad.Simulation()
    sim.geometry(COMPONENT, stack=STACK)
    sim.domain(
        plane="x=0",
        window=(CENTER_Y - 1.2, CENTER_Y + 1.2),
        margin_y=0.5,
        z_above=1.0,
        z_below=1.0,
    )
    for region, level in (("n_pad", pad_cm3), ("n_rib", rib_cm3)):
        sim.doping.add(
            tcad.StepDoping(
                region=region,
                dopant_type="donor",
                concentration_cm3=level,
                y_range=y_range,
            )
        )
    for region, level in (("p_rib", rib_cm3), ("p_pad", pad_cm3)):
        sim.doping.add(
            tcad.StepDoping(
                region=region,
                dopant_type="acceptor",
                concentration_cm3=level,
                y_range=y_range,
            )
        )
    sim.contacts.add(name="cathode", semiconductor="n_pad", electrode="cathode_metal")
    sim.contacts.add(name="anode", semiconductor="p_pad", electrode="anode_metal")
    sim.interfaces.add(name="n_link", region_a="n_pad", region_b="n_rib")
    sim.interfaces.add(name="junction", region_a="n_rib", region_b="p_rib")
    sim.interfaces.add(name="p_link", region_a="p_rib", region_b="p_pad")
    if oxide:
        sim.materials(insulators={"sio2": 3.9})
    sim.solver(**solver)
    return sim


# ----------------------------------------------------------------------
# Switches for the behaviour before each change
# ----------------------------------------------------------------------


def _no_signed_couples(*_args: object, **_kwargs: object) -> None:
    """Leave DEVSIM's own edge couples and node volumes in place."""


def _no_contact_ends(*_args: object, **_kwargs: object) -> dict:
    """Claim no node at the end of a Contact."""
    return {}


def devsim_couples(*, signed: bool) -> AbstractContextManager:
    """Patch context: DEVSIM's own couples when ``signed`` is false."""
    if signed:
        return nullcontext()
    return patch.object(tcad_device, "_install_signed_couples", _no_signed_couples)


def contact_ends(*, held: bool) -> AbstractContextManager:
    """Patch context: no node held at the ends of a Contact when false."""
    if held:
        return nullcontext()
    return patch.object(tcad_device, "_write_contact_ends", _no_contact_ends)


def edge_slack(*, slack: bool) -> AbstractContextManager:
    """Patch context: no slack on the edge of a doping box when false."""
    return patch.object(tcad_doping, "_EDGE_TOLERANCE_UM", 1e-9 if slack else 0.0)


# ----------------------------------------------------------------------
# Meshes and results
# ----------------------------------------------------------------------


def triangles_by_region(mesh_path: Path) -> dict[str, np.ndarray]:
    """Read the mesh in um: triangle corners ``(n, 3, 2)`` of each region."""
    mesh = meshio.read(str(mesh_path))
    names = {int(v[0]): k for k, v in mesh.field_data.items() if v[1] == 2}
    found: dict[str, list[np.ndarray]] = {}
    for block, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != "triangle":
            continue
        for tag in np.unique(tags):
            corners = mesh.points[block.data[tags == tag]][..., :2]
            found.setdefault(names[int(tag)], []).append(corners)
    return {name: np.concatenate(parts) for name, parts in found.items()}


def obtuse_mask(corners: np.ndarray) -> np.ndarray:
    """Mark the triangles with an angle above 90 degrees."""
    a = corners[:, 1] - corners[:, 0]
    b = corners[:, 2] - corners[:, 1]
    c = corners[:, 0] - corners[:, 2]
    dots = np.stack(
        [
            -np.einsum("ij,ij->i", a, c),
            -np.einsum("ij,ij->i", b, a),
            -np.einsum("ij,ij->i", c, b),
        ]
    )
    return (dots < 0).any(axis=0)


def mesh_stats(triangles: dict[str, np.ndarray]) -> dict[str, tuple[int, float]]:
    """Triangle count and obtuse fraction of silicon, oxide and both."""
    groups = {"silicon": SILICON, "oxide": ("sio2",), "solved": SOLVED}
    stats = {}
    for label, regions in groups.items():
        corners = np.concatenate([triangles[r] for r in regions if r in triangles])
        stats[label] = (len(corners), float(obtuse_mask(corners).mean()))
    return stats


def cv(
    sim: tcad.Simulation, biases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sweep the cathode; return V_R, C(V), depletion C and max deviation."""
    sweep = sim.sweep(biases, contact="cathode")
    comparison = tcad.compare_capacitance(
        PNJunctionConfig(na_cm3=N_A, nd_cm3=N_D), sweep, height_um=RIB_HEIGHT
    )
    return (
        np.asarray(comparison.v_reverse),
        np.asarray(comparison.c_tcad_f_per_cm) * 1e12,
        np.asarray(comparison.c_analytic_f_per_cm) * 1e12,
        float(comparison.max_relative_deviation),
    )


def live_device(sim: tcad.Simulation) -> tuple[tcad_device.Device, str]:
    """Return the DEVSIM device of a simulation that has solved, and its name."""
    device = sim._device
    if device is None:
        msg = "the simulation has no DEVSIM device: solve or sweep it first"
        raise RuntimeError(msg)
    if device.name is None:
        msg = "the DEVSIM device of the simulation was released"
        raise RuntimeError(msg)
    return device, device.name


def node_values(sim: tcad.Simulation, region: str, *names: str) -> dict:
    """Read node models of a region of the live DEVSIM device (x, y in um)."""
    devsim = require_devsim()
    _, device_name = live_device(sim)
    values = {
        name: np.asarray(
            devsim.get_node_model_values(device=device_name, region=region, name=name)
        )
        for name in names
    }
    for axis in ("x", "y"):
        if axis in values:
            values[axis] = values[axis] / UM_TO_CM
    return values


def report(label: str, **numbers: object) -> None:
    """Print one line of numbers for the text of the guide."""
    print(f"{label}: " + ", ".join(f"{k}={v}" for k, v in numbers.items()), flush=True)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def draw_triangles(
    ax: plt.Axes, corners: np.ndarray, window: tuple[float, float, float, float]
) -> None:
    """Outline the triangles in a window, filling the obtuse ones."""
    h0, h1, z0, z1 = window
    centre = corners.mean(axis=1)
    keep = (
        (centre[:, 0] > h0)
        & (centre[:, 0] < h1)
        & (centre[:, 1] > z0)
        & (centre[:, 1] < z1)
    )
    corners = corners[keep]
    obtuse = obtuse_mask(corners)
    ax.add_collection(
        PolyCollection(
            corners[~obtuse].tolist(),
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.25,
        )
    )
    ax.add_collection(
        PolyCollection(
            corners[obtuse].tolist(),
            facecolors=COLORS["before"],
            alpha=0.45,
            edgecolors="0.35",
            linewidths=0.25,
        )
    )
    ax.set_xlim(h0, h1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal")


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure next to this script."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}", flush=True)


# ----------------------------------------------------------------------
# The figures
# ----------------------------------------------------------------------


def figure_couples(root: Path) -> None:
    """DEVSIM's own couples against the signed ones, on two meshes."""
    curves = {}
    coarse_triangles = None
    for mesh_name, solver in (("coarse", COARSE), ("refined", REFINED)):
        for signed in (False, True):
            with devsim_couples(signed=signed):
                sim = make_sim(solver)
                mesh = sim.mesh(output_dir=root / f"couples_{mesh_name}_{signed}")
                if mesh_name == "coarse" and signed:
                    coarse_triangles = triangles_by_region(mesh.mesh_path)
                curves[mesh_name, signed] = cv(sim, BIASES)
            v_r, c, ana, deviation = curves[mesh_name, signed]
            report(
                f"couples {mesh_name} {'signed' if signed else 'devsim'}",
                C_pF_per_cm=np.round(c, 3).tolist(),
                max_deviation=round(deviation, 3),
            )
    report("couples depletion", C_pF_per_cm=np.round(ana, 3).tolist())
    if coarse_triangles is None:
        msg = "the coarse mesh with signed couples was not read"
        raise RuntimeError(msg)
    report(
        "couples coarse mesh",
        **{k: (n, round(f, 3)) for k, (n, f) in mesh_stats(coarse_triangles).items()},
    )

    fig, (ax_mesh, ax_cv) = plt.subplots(
        1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    window = (CENTER_Y - 0.55, CENTER_Y + 0.55, -0.02, RIB_HEIGHT + 0.02)
    for region in SILICON:
        draw_triangles(ax_mesh, coarse_triangles[region], window)
    ax_mesh.set_xlabel("h (µm)")
    ax_mesh.set_ylabel("z (µm)")
    ax_mesh.set_title("Coarse mesh, obtuse triangles shaded", fontsize=10)
    ax_cv.plot(
        v_r, ana, "--", color=COLORS["reference"], label="Depletion approximation"
    )
    styles = {
        ("coarse", False): ("s:", COLORS["before"], "coarse mesh, DEVSIM couples"),
        ("coarse", True): ("s-", COLORS["after"], "coarse mesh, signed couples"),
        ("refined", False): ("o:", "#e08a7e", "refined mesh, DEVSIM couples"),
        ("refined", True): ("o-", "#6fa8dc", "refined mesh, signed couples"),
    }
    for key, (style, color, label) in styles.items():
        ax_cv.plot(
            curves[key][0], curves[key][1], style, color=color, ms=4, label=label
        )
    ax_cv.set_xlabel("Reverse bias $V_R$ (V)")
    ax_cv.set_ylabel("Capacitance (pF/cm)")
    ax_cv.legend(fontsize=8)
    ax_cv.set_title("C(V) of the PN junction", fontsize=10)
    save(fig, "mesh_quality_couples.png")


def figure_grading(root: Path) -> None:
    """The default mesh against a graded one, with the oxide solved."""
    # Without the refinement box, meshes only: the same junction size.
    for name, solver in (
        ("default", COARSE),
        ("graded", GRADED | {"refinement_boxes": None}),
    ):
        sim = make_sim(solver, oxide=True)
        mesh = sim.mesh(output_dir=root / f"grading_nobox_{name}")
        stats = mesh_stats(triangles_by_region(mesh.mesh_path))
        report(
            f"grading without box, {name}",
            **{k: (n, round(f, 3)) for k, (n, f) in stats.items()},
        )

    results = {}
    for name, solver in (("default", REFINED), ("graded", GRADED)):
        sim = make_sim(solver, oxide=True)
        mesh = sim.mesh(output_dir=root / f"grading_{name}")
        triangles = triangles_by_region(mesh.mesh_path)
        stats = mesh_stats(triangles)
        t0 = time.perf_counter()
        curve = cv(sim, BIASES[::2])
        results[name] = (triangles, stats, curve)
        report(
            f"grading {name}",
            **{k: (n, round(f, 3)) for k, (n, f) in stats.items()},
            C_pF_per_cm=np.round(curve[1], 3).tolist(),
            max_deviation=round(curve[3], 3),
            solve_s=round(time.perf_counter() - t0, 1),
        )
    report("grading depletion", C_pF_per_cm=np.round(curve[2], 3).tolist())

    fig = plt.figure(figsize=(12.5, 3.8))
    # wspace leaves room for the y label of the right panel between the panels.
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.5)
    window = (CENTER_Y - 0.9, CENTER_Y + 0.9, -0.7, 0.95)
    for i, (name, title) in enumerate(
        (("default", "Default growth"), ("graded", "Graded, 0.25 per µm"))
    ):
        triangles, stats, _ = results[name]
        ax = fig.add_subplot(grid[0, i])
        for region in SOLVED:
            draw_triangles(ax, triangles[region], window)
        n_solved, _ = stats["solved"]
        _, oxide_share = stats["oxide"]
        ax.set_title(
            f"{title}: {n_solved} triangles,\n{oxide_share:.0%} obtuse in the oxide",
            fontsize=9,
        )
        ax.set_xlabel("h (µm)")
        if i == 0:
            ax.set_ylabel("z (µm)")
    ax = fig.add_subplot(grid[0, 2])
    v_r, _, ana, _ = results["default"][2]
    ax.plot(v_r, ana, "--", color=COLORS["reference"], label="Depletion approximation")
    ax.plot(
        v_r,
        results["default"][2][1],
        "o-",
        color=COLORS["before"],
        ms=4,
        label="default",
    )
    ax.plot(
        v_r, results["graded"][2][1], "s-", color=COLORS["after"], ms=4, label="graded"
    )
    ax.set_xlabel("Reverse bias $V_R$ (V)")
    ax.set_ylabel("Capacitance (pF/cm)")
    ax.set_title("C(V), oxide in the solve", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "mesh_quality_grading.png")


def figure_contact_ends(root: Path) -> None:
    """The potential at the end of a Contact, without and with the end held."""
    pad_cm3, rib_cm3, bias = 1e20, 1e17, 1.0
    runs = {}
    for held in (False, True):
        with contact_ends(held=held):
            sim = make_sim(REFINED, oxide=True, pad_cm3=pad_cm3, rib_cm3=rib_cm3)
            sim.mesh(output_dir=root / f"ends_{held}")
            point = sim.solve(bias, contact="cathode")
            nodes = {
                region: node_values(sim, region, "x", "y", "Potential")
                for region in ("n_pad", "n_rib", "sio2")
            }
            device, device_name = live_device(sim)
            ends = [
                (end.name, end.region, *(v / UM_TO_CM for v in end.point))
                for contact_ends_ in device.contact_ends.values()
                for end in contact_ends_
            ]
            v_t = float(
                require_devsim().get_parameter(
                    device=device_name, region="n_pad", name="V_t"
                )
            )
        runs[held] = (nodes, ends, point)
        report(
            f"contact ends held={held}",
            end_nodes=[(name, round(x, 3), round(y, 3)) for name, _, x, y in ends],
            C_pF_per_cm=round(point.capacitance_f_per_cm * 1e12, 4),
            cathode_A_per_cm=point.currents_a_per_cm["cathode"],
        )

    n_i = intrinsic_density_cm3(300.0)
    expected = {
        region: bias + v_t * np.arcsinh(level / (2.0 * n_i))
        for region, level in (("n_pad", pad_cm3), ("n_rib", rib_cm3))
    }
    for held, (nodes, _, _) in runs.items():
        data = nodes["n_rib"]
        at_end = (np.abs(data["y"] - RIB_HEIGHT) < 1e-6) & (
            np.abs(data["x"] - (CENTER_Y - FLANK)) < 1e-6
        )
        report(
            f"contact ends held={held}",
            rib_end_node_V=np.round(data["Potential"][at_end], 4).tolist(),
            expected_V={k: round(float(v), 4) for k, v in expected.items()},
        )
    h_end = CENTER_Y - FLANK
    fig, (ax_mesh, ax_v) = plt.subplots(
        1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1.1]}
    )
    nodes, ends, _ = runs[True]
    window = (h_end - 0.3, h_end + 0.3, 0.0, 0.45)
    for region, color, size in (
        ("sio2", "0.7", 14),
        ("n_rib", "C1", 22),
        ("n_pad", "C0", 22),
    ):
        data = nodes[region]
        inside = (
            (data["x"] > window[0])
            & (data["x"] < window[1])
            & (data["y"] > window[2])
            & (data["y"] < window[3])
        )
        label = {"sio2": "oxide", "n_rib": "n rib", "n_pad": "n pad"}[region]
        ax_mesh.scatter(
            data["x"][inside], data["y"][inside], s=size, color=color, label=label
        )
    ax_mesh.plot(
        [h_end - 0.3, h_end],
        [RIB_HEIGHT, RIB_HEIGHT],
        "k-",
        lw=3,
        alpha=0.6,
        label="cathode Contact",
    )
    ax_mesh.scatter(
        [e[2] for e in ends],
        [e[3] for e in ends],
        s=110,
        facecolors="none",
        edgecolors=COLORS["after"],
        linewidths=1.5,
        label="end nodes the Contact holds",
    )
    ax_mesh.set_xlim(window[0], window[1])
    ax_mesh.set_ylim(window[2], window[3])
    ax_mesh.set_aspect("equal")
    ax_mesh.set_xlabel("h (µm)")
    ax_mesh.set_ylabel("z (µm)")
    ax_mesh.set_title("Mesh nodes at the end of the cathode Contact", fontsize=10)
    ax_mesh.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    for held, color, label in (
        (False, COLORS["before"], "end not held"),
        (True, COLORS["after"], "end held"),
    ):
        for region, style in (("n_pad", "o-"), ("n_rib", "s--")):
            data = runs[held][0][region]
            top = np.abs(data["y"] - RIB_HEIGHT) < 1e-6
            near = np.abs(data["x"] - h_end) < 0.3
            keep = top & near
            order = np.argsort(data["x"][keep])
            ax_v.plot(
                data["x"][keep][order],
                data["Potential"][keep][order],
                style,
                color=color,
                ms=3.5,
                label=f"{label}, {region.replace('_', ' ')}",
            )
    for i, level in enumerate(expected.values()):
        ax_v.axhline(
            level,
            color=COLORS["reference"],
            lw=0.7,
            ls=":",
            label="metal bias + equilibrium potential" if i == 0 else None,
        )
    ax_v.axvline(h_end, color="0.6", lw=0.6)
    ax_v.set_xlabel("h along the top surface (µm)")
    ax_v.set_ylabel("Potential (V)")
    ax_v.set_title(f"Potential at {bias:.0f} V on the cathode", fontsize=10)
    ax_v.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "mesh_quality_contact_ends.png")


def figure_doping_edges(root: Path) -> None:
    """The nodes a doping box reaches, without and with the edge slack."""
    runs = {}
    for slack in (False, True):
        with edge_slack(slack=slack):
            sim = make_sim(REFINED, y_range=(0.0, RIB_HEIGHT))
            sim.mesh(output_dir=root / f"doping_{slack}")
            point = sim.solve(1.0, contact="cathode")
            nodes = {
                region: node_values(sim, region, "x", "y", "NetDoping")
                for region in SILICON
            }
        runs[slack] = nodes
        total = sum(len(n["x"]) for n in nodes.values())
        undoped = sum(int((n["NetDoping"] == 0).sum()) for n in nodes.values())
        top = sum(
            int((np.abs(n["y"] - RIB_HEIGHT) < 1e-6).sum()) for n in nodes.values()
        )
        report(
            f"doping edges slack={slack}",
            silicon_nodes=total,
            top_surface_nodes=top,
            undoped_nodes=undoped,
            C_pF_per_cm=round(point.capacitance_f_per_cm * 1e12, 4),
            cathode_A_per_cm=point.currents_a_per_cm["cathode"],
        )

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    for ax, slack, title in (
        (
            axes[0],
            False,
            "Strict edge: the top-surface nodes read 0.22000000000000003 µm",
        ),
        (axes[1], True, "With the 1e-9 µm slack"),
    ):
        for region in SILICON:
            data = runs[slack][region]
            doped = data["NetDoping"] != 0
            ax.scatter(data["x"][doped], data["y"][doped], s=5, color=COLORS["after"])
            ax.scatter(
                data["x"][~doped],
                data["y"][~doped],
                s=14,
                marker="x",
                color=COLORS["before"],
            )
        ax.set_xlim(CENTER_Y - 0.6, CENTER_Y + 0.6)
        ax.set_ylim(-0.03, RIB_HEIGHT + 0.03)
        ax.set_ylabel("z (µm)")
        ax.set_title(title, fontsize=9)
    axes[1].set_xlabel("h (µm)")
    axes[0].scatter([], [], s=5, color=COLORS["after"], label="doped node")
    axes[0].scatter(
        [], [], s=14, marker="x", color=COLORS["before"], label="node with no doping"
    )
    axes[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "mesh_quality_doping_edges.png")


def main() -> None:
    """Draw the four figures."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        figure_couples(root)
        figure_grading(root)
        figure_contact_ends(root)
        figure_doping_edges(root)
    print(f"total {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
