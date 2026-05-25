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

# %% [markdown] papermill={"duration": 0.004114, "end_time": "2026-04-04T19:34:48.414266", "exception": false, "start_time": "2026-04-04T19:34:48.410152", "status": "completed"}
# # Running MEEP Simulations
#
# [MEEP](https://meep.readthedocs.io/) is an open-source FDTD electromagnetic simulator. This notebook demonstrates using the `gsim.meep` API to run an S-parameter simulation on a photonic Y-branch.
#
# **Requirements:**
#
# - UBC PDK: `uv pip install ubcpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001984, "end_time": "2026-04-04T19:34:48.419440", "exception": false, "start_time": "2026-04-04T19:34:48.417456", "status": "completed"}
# ### Load a pcell from UBC PDK

# %% papermill={"duration": 2.555359, "end_time": "2026-04-04T19:34:50.976607", "exception": false, "start_time": "2026-04-04T19:34:48.421248", "status": "completed"}
from ubcpdk import PDK, cells

PDK.activate()

c = cells.coupler_ring(length_x=10)
c

# %% [markdown] papermill={"duration": 0.000651, "end_time": "2026-04-04T19:34:50.978192", "exception": false, "start_time": "2026-04-04T19:34:50.977541", "status": "completed"}
# ### Configure and run simulation

# %% papermill={"duration": 0.054199, "end_time": "2026-04-04T19:34:51.033141", "exception": false, "start_time": "2026-04-04T19:34:50.978942", "status": "completed"}
from gsim import meep
from gsim.common.stack import get_stack

stack = get_stack()  # auto-detects active PDK

sim = meep.Simulation()

sim.geometry(component=c, stack=stack, z_crop="auto")
sim.materials = {"si": 3.47, "SiO2": 1.44}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.monitors = ["o1", "o2", "o3", "o4"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=20, save_animation=True, verbose_interval=5.0)
sim.solver.stop_when_energy_decayed()

print(sim.validate_config())

# %% papermill={"duration": 0.589482, "end_time": "2026-04-04T19:34:51.624922", "exception": false, "start_time": "2026-04-04T19:34:51.035440", "status": "completed"}
sim.plot_2d(slices="xyz")

# %% [markdown] papermill={"duration": 0.000875, "end_time": "2026-04-04T19:34:51.626998", "exception": false, "start_time": "2026-04-04T19:34:51.626123", "status": "completed"}
# ### Run simulation on cloud

# %% papermill={"duration": 702.426764, "end_time": "2026-04-04T19:46:34.055246", "exception": false, "start_time": "2026-04-04T19:34:51.628482", "status": "completed"}
# Run on GDSFactory+ cloud
result = sim.run()

# %% papermill={"duration": 0.161886, "end_time": "2026-04-04T19:46:34.218349", "exception": false, "start_time": "2026-04-04T19:46:34.056463", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.014716, "end_time": "2026-04-04T19:46:34.234877", "exception": false, "start_time": "2026-04-04T19:46:34.220161", "status": "completed"}
result.plot_interactive(phase=True)

# %% papermill={"duration": 0.013619, "end_time": "2026-04-04T19:46:34.249771", "exception": false, "start_time": "2026-04-04T19:46:34.236152", "status": "completed"}
result.show_animation()
