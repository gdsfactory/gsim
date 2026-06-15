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

# %% [markdown] papermill={"duration": 0.001383, "end_time": "2026-04-04T19:25:07.580291", "exception": false, "start_time": "2026-04-04T19:25:07.578908", "status": "completed"}
# # Running MEEP Simulations
#
# [MEEP](https://meep.readthedocs.io/) is an open-source FDTD electromagnetic simulator. This notebook demonstrates using the `gsim.meep` API to run an S-parameter simulation on a photonic Y-branch.
#
# **Requirements:**
#
# - UBC PDK: `uv pip install ubcpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.000615, "end_time": "2026-04-04T19:25:07.581837", "exception": false, "start_time": "2026-04-04T19:25:07.581222", "status": "completed"}
# ### Load a pcell from UBC PDK

# %% papermill={"duration": 3.633209, "end_time": "2026-04-04T19:25:11.215600", "exception": false, "start_time": "2026-04-04T19:25:07.582391", "status": "completed"}
from ubcpdk import PDK, cells

PDK.activate()

c = cells.coupler()

c

# %% [markdown] papermill={"duration": 0.000654, "end_time": "2026-04-04T19:25:11.217244", "exception": false, "start_time": "2026-04-04T19:25:11.216590", "status": "completed"}
# ### Configure and run simulation

# %% papermill={"duration": 0.04458, "end_time": "2026-04-04T19:25:11.262469", "exception": false, "start_time": "2026-04-04T19:25:11.217889", "status": "completed"}
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
sim.source(port="o1", wavelength=1.55, wavelength_span=0.04)
sim.monitors = ["o1", "o2", "o3", "o4"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=20, save_animation=True, verbose_interval=5.0)
sim.num_freqs = 21
sim.solver.stop_when_energy_decayed()

print(sim.validate_config())

# %% papermill={"duration": 0.655465, "end_time": "2026-04-04T19:25:11.918961", "exception": false, "start_time": "2026-04-04T19:25:11.263496", "status": "completed"}
sim.plot_2d(slices="xyz")

# %% [markdown] papermill={"duration": 0.001614, "end_time": "2026-04-04T19:25:11.921774", "exception": false, "start_time": "2026-04-04T19:25:11.920160", "status": "completed"}
# ### Run simulation on cloud

# %% papermill={"duration": 411.550742, "end_time": "2026-04-04T19:32:03.473783", "exception": false, "start_time": "2026-04-04T19:25:11.923041", "status": "completed"}
# Run on GDSFactory+ cloud
result = sim.run()

# %% papermill={"duration": 0.145999, "end_time": "2026-04-04T19:32:03.620766", "exception": false, "start_time": "2026-04-04T19:32:03.474767", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.014307, "end_time": "2026-04-04T19:32:03.636329", "exception": false, "start_time": "2026-04-04T19:32:03.622022", "status": "completed"}
result.plot_interactive(phase=True)

# %% papermill={"duration": 0.006012, "end_time": "2026-04-04T19:32:03.644294", "exception": false, "start_time": "2026-04-04T19:32:03.638282", "status": "completed"}
result.show_animation()
