# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: gsim
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running Palace Simulations: Microstrip
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a microstrip transmission line with via ports.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown]
# ### Load a pcell from IHP PDK

# %%
import gdsfactory as gf
from ihp import LAYER, PDK, cells

PDK.activate()

c = gf.Component()
r1 = c << cells.straight_metal(length=1000, width=14)

r = c.get_region(layer=LAYER.TopMetal2drawing)
r_sized = r.sized(+20000)
c.add_polygon(r_sized, layer=LAYER.Metal1drawing)


c.add_ports(r1.ports)

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown]
# ### Configure and run simulation with DrivenSim

# %%
from gsim.palace import DrivenSim

# Create simulation object
sim = DrivenSim()

# Set output directory
sim.set_output_dir("./palace-sim-microstrip")

# Set the component geometry
sim.set_geometry(c)

# Configure layer stack from active PDK
sim.set_stack(substrate_thickness=2.0, air_above=300.0)

# Configure via ports (Metal1 ground plane to TopMetal2 signal)
for port in c.ports:
    sim.add_port(port.name, from_layer="metal1", to_layer="topmetal2", geometry="via")

# Configure driven simulation (frequency sweep for S-parameters)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=80)

# Validate configuration
print(sim.validate_config())

# %%
# Generate mesh (presets: "coarse", "default", "fine")
sim.mesh(preset="default")

# %%
# Static PNG
sim.plot_mesh(show_groups=["metal", "P"], interactive=False)

# Interactive
# sim.plot_mesh(show_groups=["metal", "P"], interactive=True)

# %% [markdown]
# ### Run simulation on GDSFactory+ Cloud

# %%
# Run simulation on GDSFactory+ cloud
results = sim.run()

# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(results["port-S.csv"])
df.columns = df.columns.str.strip()

freq = df["f (GHz)"]

fig, axes = plt.subplots(2, 2, figsize=(8, 5))

# S11 Magnitude
axes[0, 0].plot(freq, df["|S[1][1]| (dB)"], marker=".")
axes[0, 0].set_xlabel("Frequency (GHz)")
axes[0, 0].set_ylabel("Magnitude (dB)")
axes[0, 0].set_title("S11 Magnitude")
axes[0, 0].grid(True)

# S21 Magnitude
axes[0, 1].plot(freq, df["|S[2][1]| (dB)"], marker=".", color="tab:orange")
axes[0, 1].set_xlabel("Frequency (GHz)")
axes[0, 1].set_ylabel("Magnitude (dB)")
axes[0, 1].set_title("S21 Magnitude")
axes[0, 1].grid(True)

# S11 Phase
axes[1, 0].plot(freq, df["arg(S[1][1]) (deg.)"], marker=".")
axes[1, 0].set_xlabel("Frequency (GHz)")
axes[1, 0].set_ylabel("Phase (deg)")
axes[1, 0].set_title("S11 Phase")
axes[1, 0].grid(True)

# S21 Phase
axes[1, 1].plot(freq, df["arg(S[2][1]) (deg.)"], marker=".", color="tab:orange")
axes[1, 1].set_xlabel("Frequency (GHz)")
axes[1, 1].set_ylabel("Phase (deg)")
axes[1, 1].set_title("S21 Phase")
axes[1, 1].grid(True)

plt.tight_layout()
