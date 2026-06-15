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

# %% [markdown] papermill={"duration": 0.003534, "end_time": "2026-04-04T19:47:36.591070", "exception": false, "start_time": "2026-04-04T19:47:36.587536", "status": "completed"}
# # Running MEEP Simulations
#
# [MEEP](https://meep.readthedocs.io/) is an open-source FDTD electromagnetic simulator. This notebook demonstrates using the `gsim.meep` API to run an S-parameter simulation on a photonic Y-branch.
#
# **Requirements:**
#
# - UBC PDK: `uv pip install ubcpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001996, "end_time": "2026-04-04T19:47:36.595345", "exception": false, "start_time": "2026-04-04T19:47:36.593349", "status": "completed"}
# ### Load a pcell from UBC PDK

# %% papermill={"duration": 3.17493, "end_time": "2026-04-04T19:47:39.772105", "exception": false, "start_time": "2026-04-04T19:47:36.597175", "status": "completed"}
from ubcpdk import PDK, cells

PDK.activate()

c = cells.ebeam_y_1550()
c

# %% [markdown] papermill={"duration": 0.000956, "end_time": "2026-04-04T19:47:39.774192", "exception": false, "start_time": "2026-04-04T19:47:39.773236", "status": "completed"}
# ### Configure and run simulation

# %% papermill={"duration": 0.073708, "end_time": "2026-04-04T19:47:39.849379", "exception": false, "start_time": "2026-04-04T19:47:39.775671", "status": "completed"}
from gsim import meep
from gsim.common.stack import get_stack
from gsim.meep.models.api import Material

stack = get_stack()  # auto-detects active PDK

sim = meep.Simulation()

sim.geometry(component=c, stack=stack, z_crop="auto")
sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.monitors = ["o1", "o2", "o3"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=20, simplify_tol=0.01, save_animation=True, verbose_interval=5.0)
sim.solver.stop_when_energy_decayed()

print(sim.validate_config())

# %% papermill={"duration": 0.92567, "end_time": "2026-04-04T19:47:40.775948", "exception": false, "start_time": "2026-04-04T19:47:39.850278", "status": "completed"}
sim.plot_2d(slices="xyz")

# %% [markdown] papermill={"duration": 0.000928, "end_time": "2026-04-04T19:47:40.778972", "exception": false, "start_time": "2026-04-04T19:47:40.778044", "status": "completed"}
# ### Run simulation on cloud

# %% papermill={"duration": 131.672047, "end_time": "2026-04-04T19:49:52.452547", "exception": false, "start_time": "2026-04-04T19:47:40.780500", "status": "completed"}
# Run on GDSFactory+ cloud
result = sim.run()

# %% papermill={"duration": 0.139914, "end_time": "2026-04-04T19:49:52.593404", "exception": false, "start_time": "2026-04-04T19:49:52.453490", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.048194, "end_time": "2026-04-04T19:49:52.643091", "exception": false, "start_time": "2026-04-04T19:49:52.594897", "status": "completed"}
result.plot_interactive(phase=True)

# %% papermill={"duration": 0.008188, "end_time": "2026-04-04T19:49:52.652646", "exception": false, "start_time": "2026-04-04T19:49:52.644458", "status": "completed"}
result.show_animation()
