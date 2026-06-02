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
# # Running MEEP Simulations
#
# [MEEP](https://meep.readthedocs.io/) is an open-source FDTD electromagnetic simulator. This notebook demonstrates using the `gsim.meep` API to run an S-parameter simulation on a photonic Y-branch.
#
# **Requirements:**
#
# - UBC PDK: `uv pip install ubcpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown]
# ### Load a pcell from UBC PDK

# %%
from ubcpdk import PDK, cells

PDK.activate()

c = cells.straight(length=20.0)

c

# %% [markdown]
# ### Configure and run simulation

# %%
from gsim import meep
from gsim.meep.models.api import Material

sim = meep.Simulation()

sim.geometry(component=c, z_crop="auto")
sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.num_freqs = 11
sim.monitors = ["o1", "o2"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=20, save_animation=True, verbose_interval=5.0)
sim.solver.stop_after_sources(time=100)

print(sim.validate_config())

# %%
sim.plot_2d(slices="xyz")

# %% [markdown]
# ### Run simulation on cloud

# %%
# Run on GDSFactory+ cloud
result = sim.run()

# %%
result.plot_interactive()

# %%
result.plot_interactive(phase=True)

# %%
result.show_animation()

# %%
