# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### QPDK Coupled Resonator — Eigenmode Analysis
#
# This notebook builds a compact coupled resonator from QPDK cells, converts etch
# layers to conductor geometry, and runs a Palace eigenmode simulation to find the eigenfrequency and mode close to 7.78 GHz.
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install qpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %%
import gdsfactory as gf
from qpdk import PDK, cells
from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.tech import LAYER, route_bundle_sbend_cpw

PDK.activate()


@gf.cell
def resonator_compact(coupling_gap: float = 20.0) -> gf.Component:
    """Compact coupled resonator with S-bend CPW routes."""
    c = gf.Component()

    res = c << cells.resonator_coupled(
        coupling_straight_length=300, coupling_gap=coupling_gap
    )
    res.movex(-res.size_info.width / 4)

    left = c << cells.straight()
    right = c << cells.straight()

    w = res.size_info.width + 100
    left.move((-w, 0))
    right.move((w, 0))

    route_bundle_sbend_cpw(
        c,
        [left["o2"], right["o1"]],
        [res["coupling_o1"], res["coupling_o2"]],
        cross_section=cpw_with_airbridges(
            airbridge_spacing=250.0, airbridge_padding=20.0
        ),
    )

    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 100))

    c.add_port(name="o1", port=left["o1"])
    c.add_port(name="o2", port=right["o2"])
    return c


component = resonator_compact(coupling_gap=15.0)
_c = component.copy()
_c.draw_ports()
_c

# %% [markdown]
# ### Convert QPDK etch layers to conductor geometry

# %%
import warnings

import klayout.db as kdb
from qpdk.tech import LAYER as QPDK_LAYER

from gsim.common.polygon_utils import decimate

sim_area_layer = (QPDK_LAYER.SIM_AREA[0], QPDK_LAYER.SIM_AREA[1])
etch_layer = (QPDK_LAYER.M1_ETCH[0], QPDK_LAYER.M1_ETCH[1])

CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

layout = component.kdb_cell.layout()
sim_region = kdb.Region(
    component.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
)
etch_region = kdb.Region(component.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer)))

etch_polys = decimate(list(etch_region.each()))
etch_region = kdb.Region()
for poly in etch_polys:
    etch_region.insert(poly)

if sim_region.is_empty():
    warnings.warn("No polygons found on SIM_AREA", stacklevel=2)
if etch_region.is_empty():
    warnings.warn("No polygons found on M1_ETCH", stacklevel=2)

conductor_region = sim_region - etch_region

etched = gf.Component("etched_component")
el = etched.kdb_cell.layout()
for name, region in [
    ("SUPERCONDUCTOR", conductor_region),
    ("SUBSTRATE", sim_region),
    ("VACUUM", sim_region),
]:
    idx = el.layer(*CPW_LAYERS[name])
    etched.kdb_cell.shapes(idx).insert(region)

for port in component.ports:
    etched.add_port(name=port.name, port=port)

etched

# %% [markdown]
# ### Configure simulation

# %%
from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace import EigenmodeSim

# Build a CPW stack matching the etched component layers
substrate_thickness = 500
vacuum_thickness = 500

stack = LayerStack(pdk_name="qpdk")
stack.layers["SUBSTRATE"] = Layer(
    name="SUBSTRATE",
    gds_layer=(1, 0),
    zmin=0.0,
    zmax=substrate_thickness,
    thickness=substrate_thickness,
    material="sapphire",
    layer_type="dielectric",
)
stack.layers["SUPERCONDUCTOR"] = Layer(
    name="SUPERCONDUCTOR",
    gds_layer=(2, 0),
    zmin=substrate_thickness,
    zmax=substrate_thickness,
    thickness=0,
    material="aluminum",
    layer_type="conductor",
)
stack.layers["VACUUM"] = Layer(
    name="VACUUM",
    gds_layer=(3, 0),
    zmin=substrate_thickness,
    zmax=substrate_thickness + vacuum_thickness,
    thickness=vacuum_thickness,
    material="vacuum",
    layer_type="dielectric",
)
stack.dielectrics = [
    {
        "name": "substrate",
        "zmin": 0.0,
        "zmax": substrate_thickness,
        "material": "sapphire",
    },
    {
        "name": "vacuum",
        "zmin": substrate_thickness,
        "zmax": substrate_thickness + vacuum_thickness,
        "material": "vacuum",
    },
]
stack.materials = {
    "sapphire": MATERIALS_DB["sapphire"].to_dict(),
    "aluminum": MATERIALS_DB["aluminum"].to_dict(),
    "vacuum": MATERIALS_DB["vacuum"].to_dict(),
}

# %%
sim = EigenmodeSim()
sim.set_geometry(etched)
sim.set_stack(stack)
sim.add_cpw_port("o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.add_cpw_port("o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.set_eigenmode(num_modes=1, target=7.78e9, save=1)

# %%
sim.set_output_dir("./sim_qpdk_resonator")
sim.mesh(preset="default", margin=0)
sim.plot_mesh(show_groups=["superconductor", "P", "sapphire", "vacuum"])

# %%
sim.write_config()
results = sim.run()

# %% [markdown]
# ### Field visualization — top view at conductor layer

# %%
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

pv.OFF_SCREEN = True

freq_ghz = results["eig.csv"]

results_dir = Path(results["eig.csv"]).parent

# Load volume field data
vol_dir = results_dir / "paraview/eigenmode/Cycle000001"
print(vol_dir)
vol_path = sorted(vol_dir.rglob("*.pvtu"))[-1]
vol = pv.read(str(vol_path))
print(f"Volume: {vol.n_points:,} points, {vol.n_cells:,} cells")

# %%
import pandas as pd

data = pd.read_csv(results["eig.csv"])
data.columns = data.columns.str.strip()
freq_ghz = data["Re{f} (GHz)"][0] + 1j * data["Im{f} (GHz)"][0]
data

# %%
import matplotlib.pyplot as plt

# Slice volume at conductor plane (z = substrate_thickness)
vol_slice = vol.slice(normal="z", origin=(0, 0, substrate_thickness))

# Build interpolation grid from data bounds
vpts = vol_slice.points
x_pad, y_pad = 5, 5
xi = np.linspace(vpts[:, 0].min() - x_pad, vpts[:, 0].max() + x_pad, 800)
yi = np.linspace(vpts[:, 1].min() - y_pad, vpts[:, 1].max() + y_pad, 400)
Xi, Yi = np.meshgrid(xi, yi)


def plot_topview(pts, data, title, cmap="turbo"):
    gi = griddata(pts[:, :2], data, (Xi, Yi), method="linear")
    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = np.nanpercentile(gi, 98)
    im = ax.pcolormesh(Xi, Yi, gi, cmap=cmap, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    valid = ~np.isnan(gi)
    if valid.any():
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        ax.set_xlim(xi[cols][0], xi[cols][-1])
        ax.set_ylim(yi[rows][0], yi[rows][-1])
    fig.tight_layout(pad=0.5)
    plt.show()


# %%
plot_topview(
    vpts,
    np.linalg.norm(vol_slice.point_data["E_real"], axis=1),
    f"Electric field |E| at {freq_ghz:.4f} GHz — (V/m)",
)
