# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: gsim
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
c.plot()

# %% [markdown]
# ### Configure ElectrostaticSim

# %%
from gsim.palace import ElectrostaticSim

sim = ElectrostaticSim()

sim.set_output_dir("./palace-sim-electrostatic")
sim.set_geometry(c)
sim.set_stack(air_above=100.0, substrate_thickness=2.0)

# Metal5 = bottom plate (MINUS), TopMetal1 = top plate (PLUS)
sim.add_terminal("T1", layer="metal5")
sim.add_terminal("T2", layer="topmetal1")

sim.set_electrostatic()

print(sim.validate_config())

# %% [markdown]
# ### Mesh and generate config

# %%
# Vmim vias are 0.42 um with 0.52 um gaps; MIM dielectric is 0.19 um thick
sim.mesh(preset="fine", margin=20, refined_mesh_size=0.1, merge_via_distance=0)

# %%
# sim.plot_mesh(show_groups=["metal", "topmetal", "via", "dielectric", "SiO2__vmim"])

sim.plot_mesh(show_groups=["metal5", "topmetal1", "vmim", "SiO2__vmim"])

# %% [markdown]
# ### Analytical estimate
#
# For the MIM capacitor: C = epsilon_0 * epsilon_r * A / d
#
# The MIM dielectric (SiO2, eps_r ~ 4.1) is 0.19 um thick between Metal5 top (z=5.38) and the Vmim bottom (z=5.58). The MIM drawing is 10.72 x 10.72 um. The Vmim vias (tungsten, conductive) extend the TopMetal1 terminal down to the MIM dielectric surface.

# %%
import scipy.constants as const

# MIM dielectric area (MIM drawing: 10.72 x 10.72 um)
eps_r = 4.1  # SiO2 permittivity from IHP stack
A_mim = (10.72e-6) * (10.72e-6)  # m^2
d_mim = (5.58 - 5.38) * 1e-6  # m (Metal5 top to Vmim bottom = MIM dielectric)

C_analytical = const.epsilon_0 * eps_r * A_mim / d_mim
print(f"Analytical capacitance: {C_analytical * 1e15:.1f} fF")
print(f"  A_mim = 10.72 x 10.72 um^2, d = {d_mim * 1e6:.2f} um, eps_r = {eps_r}")

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

eps_r = 4.1
A_mim = (10.72e-6) * (10.72e-6)
d_mim = (5.58 - 5.38) * 1e-6

C_analytical = const.epsilon_0 * eps_r * A_mim / d_mim

print(f"Palace result:       {C_palace * 1e15:.3f} fF")
print(f"Analytical (C=eA/d): {C_analytical * 1e15:.1f} fF")
print(f"Ratio:               {C_palace / C_analytical:.3f}")

# %%
