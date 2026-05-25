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
# # Palace Straight Dielectric Waveguide - Wave Ports
#
# This notebook sets up a straight dielectric waveguide and runs a Palace driven simulation using waveport boundary conditions.

# %% [markdown]
# ### Build straight waveguide geometry

# %%
from ubcpdk import PDK, cells

PDK.activate()

c = cells.straight(length=10.0)

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown]
# ### Configure Palace driven simulation

# %%
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("./palace-sim-straight-waveport")
sim.set_geometry(c)

sim.set_stack(
    substrate_thickness=2.0,
    add_oxide_dielectric=False,
    add_passivation_dielectric=False,
)

sim.set_numerical(
    solver_type="MUMPS",
    max_iterations=1,
    tolerance=1e-8,
)

for i, port in enumerate(c.ports):
    sim.add_wave_port(
        str(port.name),
        layer="core",
        z_margin=1.5,
        lateral_margin=1.5,
        max_size=False,
        mode=1,
        excited=(i == 0),
    )

sim.set_driven(fmin=190e12, fmax=200e12, num_points=21)

print(sim.validate_config())

# %% [markdown]
# ### Generate mesh

# %%
sim.set_airbox(
    margin_x=0.0,
    margin_y=2.0,
    z_above=0.5,
    z_below=0.5,
)

sim.mesh(
    preset="fine",
    refined_mesh_size=0.04,
    max_mesh_size=0.2,
    fmax=200e12,
)

# %%
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None"],
    interactive=True,
)

# %% [markdown]
# ### Run simulation

# %%
import json
from pathlib import Path

config_path = sim.write_config(photonic=True)
cfg = json.loads(Path(config_path).read_text())
print("Solver.Linear:", cfg["Solver"]["Linear"])

results = sim.run_local(num_processes=4, verbose=True)

# %% [markdown]
# ### Plot S-parameters

# %%
results.plot_interactive()

# %%
results.plot_interactive(phase=True)
