# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### QPDK Coupled Resonator — Driven Simulation
#
# This notebook builds a compact coupled resonator from QPDK cells, converts etch
# layers to conductor geometry, and runs a Palace driven simulation to extract
# S-parameters around the resonance near 7.5 GHz.
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator
# supporting eigenmode, driven (S-parameter), and electrostatic simulations.
# This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a
# microstrip transmission line with via ports.
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
from gsim.palace import DrivenSim

# Build a CPW stack matching the etched component layers
substrate_thickness = 100
vacuum_thickness = 100

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

sim = DrivenSim()
sim.set_geometry(etched)
sim.set_stack(stack)
sim.add_cpw_port(
    "o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0, offset=2.5
)
sim.add_cpw_port(
    "o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0, offset=2.5
)
sim.set_driven(fmin=7.75e9, fmax=7.8e9, num_points=100)

# %%
sim.set_output_dir("./sim_qpdk_resonator")
sim.mesh(preset="graded", margin=0)
# sim.mesh(preset="coarse", margin=0)
# sim.plot_mesh(
#     show_groups=["superconductor", "P", "sapphire", "vacuum"], interactive=False
# )

# %%
sim.write_config()
# results = sim.run()

import time

t0 = -time.time()
results = sim.run_local(use_apptainer=False, num_threads=4, num_processes=1)

t0 += time.time()
print(f"Simulation took {t0:.2f} seconds")

# %%
results

# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(results["port-S.csv"])
df.columns = df.columns.str.strip()

freq = df["f (GHz)"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

# Magnitude plot
ax1.plot(freq, df["|S[1][1]| (dB)"], marker=".", label="S11")
ax1.plot(freq, df["|S[2][1]| (dB)"], marker=".", label="S21")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("S-Parameters — QPDK Coupled Resonator")
ax1.legend()
ax1.grid(True)

# Phase plot
ax2.plot(freq, df["arg(S[1][1]) (deg.)"], marker=".", label="S11")
ax2.plot(freq, df["arg(S[2][1]) (deg.)"], marker=".", label="S21")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Phase (deg)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# %%

# %%

# %%
