# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Electrostatic Capacitance Extraction with Palace
#
# This notebook demonstrates electrostatic simulation using `gsim.palace.ElectrostaticSim` to extract the capacitance matrix between conductor terminals.
#
# We use the IHP `cmim` (MIM capacitor) cell. The bottom plate is on Metal5 and the top plate on TopMetal1, connected by a 10x10 array of Vmim vias through a thin MIM dielectric (0.19 um SiO2).
#
# **Limitation:** The current terminal system assigns one terminal per layer. Structures with multiple electrodes on the same layer (e.g., interdigitated capacitors) are not yet supported.
#
# **Requirements:**
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown]
# ### Load IHP MIM capacitor

# %%
from ihp import PDK, cells

PDK.activate()

cap_width = 10.0  # um
cap_length = 10.0  # um

# IHP cmim: Metal5 (bottom plate, MINUS) -> MIM dielectric (0.19 um SiO2)
#   -> 10x10 Vmim vias (0.42 um, pitch 0.94 um) -> TopMetal1 (top plate, PLUS)
c = cells.cmim(width=cap_width, length=cap_length).copy()
print("Ports:", [(p.name, tuple(p.center)) for p in c.ports])

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown]
# ### Configure ElectrostaticSim

# %%
from gsim.palace import ElectrostaticSim

sim = ElectrostaticSim()

sim.set_output_dir("./palace-sim-electrostatic")
sim.set_geometry(c)
sim.set_stack(substrate_thickness=2.0)
sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=5)

# Metal5 = bottom plate (MINUS), TopMetal1 = top plate (PLUS)
sim.add_terminal("T1", layer="metal5")
sim.add_terminal("T2", layer="topmetal1")

sim.set_electrostatic()

print(sim.validate_config())

# %% [markdown]
# ### Mesh and generate config

# %%
# Vmim vias are 0.42 um with 0.52 um gaps; MIM dielectric is 0.19 um thick
sim.mesh(preset="fine", refined_mesh_size=0.1, merge_via_distance=0)

# %%
# sim.plot_mesh(show_groups=["metal", "topmetal", "via", "dielectric", "SiO2__vmim"])

# sim.plot_mesh(show_groups=["metal5", "topmetal1", "vmim", "SiO2__vmim"])

sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "sio2__None", "air__sio2", "air__passive"],
    interactive=True,
)

# %% [markdown]
# ### Analytical estimate
#
# For the MIM capacitor: C = epsilon_0 * epsilon_r * A / d
#
# In this estimate, epsilon_r is read from the active PDK-derived stack material table for SiO2, the MIM drawing dimensions are read from the geometry `MIMdrawing` polygon bounding box, and we report two spacings:
# - d_topmetal1 = topmetal1_bottom - metal5_top
# - d_vmim = vmim_bottom - metal5_top

# %%
import scipy.constants as const

# Pull SiO2 epsilon_r and plate spacings from the active stack derived from the current PDK.
stack = sim._resolve_stack()
sio2_props = stack.materials.get("sio2") or stack.materials.get("SiO2")
if sio2_props is None or sio2_props.get("permittivity") is None:
    raise ValueError("Could not find SiO2 permittivity in active stack materials")
eps_r = float(sio2_props["permittivity"])

metal5 = stack.layers.get("metal5")
topmetal1 = stack.layers.get("topmetal1")
vmim = stack.layers.get("vmim")
if metal5 is None or topmetal1 is None or vmim is None:
    raise ValueError("Could not find metal5/topmetal1/vmim in active stack layers")

metal5_top = metal5.zmin + metal5.thickness
topmetal1_bottom = topmetal1.zmin
vmim_bottom = vmim.zmin

d_topmetal1_um = topmetal1_bottom - metal5_top
d_vmim_um = vmim_bottom - metal5_top
if d_topmetal1_um <= 0:
    raise ValueError(f"Non-physical topmetal1 spacing: {d_topmetal1_um} um")
if d_vmim_um <= 0:
    raise ValueError(f"Non-physical vmim spacing: {d_vmim_um} um")

d_topmetal1 = d_topmetal1_um * 1e-6  # m
d_vmim = d_vmim_um * 1e-6  # m

# Read MIM drawing dimensions from geometry.
geom_component = sim.geometry.component
mim_polys = geom_component.get_polygons(by="name").get("MIMdrawing", [])
if not mim_polys:
    raise ValueError("Could not find MIMdrawing polygons in geometry")

dbu = geom_component.kcl.dbu
left = min(poly.bbox().left for poly in mim_polys)
right = max(poly.bbox().right for poly in mim_polys)
bottom = min(poly.bbox().bottom for poly in mim_polys)
top = max(poly.bbox().top for poly in mim_polys)

mim_width_um = (right - left) * dbu
mim_length_um = (top - bottom) * dbu
A_mim = (mim_width_um * 1e-6) * (mim_length_um * 1e-6)  # m^2

C_analytical_topmetal1 = const.epsilon_0 * eps_r * A_mim / d_topmetal1
C_analytical_vmim = const.epsilon_0 * eps_r * A_mim / d_vmim
C_analytical = C_analytical_topmetal1

print(f"eps_r(SiO2) = {eps_r}")
print(f"MIM drawing from geometry: {mim_width_um:.3f} x {mim_length_um:.3f} um")
print(
    f"Analytical (d=topmetal1_bottom-metal5_top={d_topmetal1_um:.3f} um): {C_analytical_topmetal1 * 1e15:.1f} fF"
)
print(
    f"Analytical (d=vmim_bottom-metal5_top={d_vmim_um:.3f} um):      {C_analytical_vmim * 1e15:.1f} fF"
)

# %% [markdown]
# ### Run on cloud
#
# Uncomment to submit to GDSFactory+ cloud. The result should be a capacitance matrix CSV.

# %%
results = sim.run()

# %% [markdown]
# ### Load and analyze results

# %%
import csv
from pathlib import Path

import numpy as np

results_dir = Path(results["terminal-C.csv"]).parent


def read_palace_csv(path):
    """Read a Palace output CSV, returning header and data as numpy array."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader])
    return [h.strip() for h in header], data


# Capacitance matrix
header, C_matrix = read_palace_csv(results_dir / "terminal-C.csv")
print("Capacitance matrix (F):")
print(f"  C[1,1] = {C_matrix[0, 1]:+.4e} F  ({C_matrix[0, 1] * 1e15:+.3f} fF)")
print(f"  C[1,2] = {C_matrix[0, 2]:+.4e} F  ({C_matrix[0, 2] * 1e15:+.3f} fF)")
print(f"  C[2,1] = {C_matrix[1, 1]:+.4e} F  ({C_matrix[1, 1] * 1e15:+.3f} fF)")
print(f"  C[2,2] = {C_matrix[1, 2]:+.4e} F  ({C_matrix[1, 2] * 1e15:+.3f} fF)")

# Mutual capacitance
_, Cm = read_palace_csv(results_dir / "terminal-Cm.csv")
print(f"\nMutual capacitance C_m[1,2] = {Cm[0, 2] * 1e15:.3f} fF")

# Domain energy
_, E = read_palace_csv(results_dir / "domain-E.csv")
print(
    f"\nStored electric energy: {E[0, 1]:.4e} J (excitation 1), {E[1, 1]:.4e} J (excitation 2)"
)

# %% [markdown]
# ### Compare with analytical estimate

# %%
import scipy.constants as const

C_palace = abs(C_matrix[0, 1])

stack = sim._resolve_stack()
sio2_props = stack.materials.get("sio2") or stack.materials.get("SiO2")
if sio2_props is None or sio2_props.get("permittivity") is None:
    raise ValueError("Could not find SiO2 permittivity in active stack materials")
eps_r = float(sio2_props["permittivity"])

metal5 = stack.layers.get("metal5")
topmetal1 = stack.layers.get("topmetal1")
vmim = stack.layers.get("vmim")
if metal5 is None or topmetal1 is None or vmim is None:
    raise ValueError("Could not find metal5/topmetal1/vmim in active stack layers")

metal5_top = metal5.zmin + metal5.thickness
topmetal1_bottom = topmetal1.zmin
vmim_bottom = vmim.zmin

d_topmetal1_um = topmetal1_bottom - metal5_top
d_vmim_um = vmim_bottom - metal5_top
if d_topmetal1_um <= 0:
    raise ValueError(f"Non-physical topmetal1 spacing: {d_topmetal1_um} um")
if d_vmim_um <= 0:
    raise ValueError(f"Non-physical vmim spacing: {d_vmim_um} um")

d_topmetal1 = d_topmetal1_um * 1e-6
d_vmim = d_vmim_um * 1e-6

geom_component = sim.geometry.component
mim_polys = geom_component.get_polygons(by="name").get("MIMdrawing", [])
if not mim_polys:
    raise ValueError("Could not find MIMdrawing polygons in geometry")

dbu = geom_component.kcl.dbu
left = min(poly.bbox().left for poly in mim_polys)
right = max(poly.bbox().right for poly in mim_polys)
bottom = min(poly.bbox().bottom for poly in mim_polys)
top = max(poly.bbox().top for poly in mim_polys)

mim_width_um = (right - left) * dbu
mim_length_um = (top - bottom) * dbu
A_mim = (mim_width_um * 1e-6) * (mim_length_um * 1e-6)

C_analytical_topmetal1 = const.epsilon_0 * eps_r * A_mim / d_topmetal1
C_analytical_vmim = const.epsilon_0 * eps_r * A_mim / d_vmim
C_analytical = C_analytical_topmetal1

print(f"Palace result:                                {C_palace * 1e15:.3f} fF")
print(
    f"Analytical (d=topmetal1_bottom-metal5_top):   {C_analytical_topmetal1 * 1e15:.1f} fF"
)
print(
    f"Analytical (d=vmim_bottom-metal5_top):        {C_analytical_vmim * 1e15:.1f} fF"
)
print(
    f"Ratio Palace / topmetal1-based analytical:    {C_palace / C_analytical_topmetal1:.3f}"
)
print(
    f"Ratio Palace / vmim-based analytical:         {C_palace / C_analytical_vmim:.3f}"
)
print(f"Using A_mim = {mim_width_um:.3f} x {mim_length_um:.3f} um^2 from MIMdrawing")
print(f"d_topmetal1 = {d_topmetal1_um:.3f} um, d_vmim = {d_vmim_um:.3f} um")

# %%
