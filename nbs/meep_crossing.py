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

# %% [markdown] papermill={"duration": 0.003076, "end_time": "2026-04-22T11:27:09.006814", "exception": false, "start_time": "2026-04-22T11:27:09.003738", "status": "completed"}
# # Running MEEP Simulations
#
# [MEEP](https://meep.readthedocs.io/) is an open-source FDTD electromagnetic simulator. This notebook demonstrates using the `gsim.meep` API to run an S-parameter simulation on a photonic Y-branch.
#
# **Requirements:**
#
# - UBC PDK: `uv pip install ubcpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.002639, "end_time": "2026-04-22T11:27:09.011654", "exception": false, "start_time": "2026-04-22T11:27:09.009015", "status": "completed"}
# ### Load a pcell from UBC PDK

# %% papermill={"duration": 3.211213, "end_time": "2026-04-22T11:27:12.224777", "exception": false, "start_time": "2026-04-22T11:27:09.013564", "status": "completed"}
from ubcpdk import PDK, cells

PDK.activate()

c = cells.ebeam_crossing4()

c

# %% [markdown] papermill={"duration": 0.000659, "end_time": "2026-04-22T11:27:12.226368", "exception": false, "start_time": "2026-04-22T11:27:12.225709", "status": "completed"}
# ### Configure and run simulation

# %% papermill={"duration": 0.0558, "end_time": "2026-04-22T11:27:12.282705", "exception": false, "start_time": "2026-04-22T11:27:12.226905", "status": "completed"}
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
sim.num_freqs = 51
sim.solver.stop_when_energy_decayed()

print(sim.validate_config())

xsxs

# %% papermill={"duration": 0.837901, "end_time": "2026-04-22T11:27:13.121534", "exception": false, "start_time": "2026-04-22T11:27:12.283633", "status": "completed"}
sim.plot_2d(slices="xyz")

# %% [markdown] papermill={"duration": 0.000831, "end_time": "2026-04-22T11:27:13.123545", "exception": false, "start_time": "2026-04-22T11:27:13.122714", "status": "completed"}
# ### Run simulation on cloud

# %% papermill={"duration": 202.030979, "end_time": "2026-04-22T11:30:35.155272", "exception": false, "start_time": "2026-04-22T11:27:13.124293", "status": "completed"}
# Run on GDSFactory+ cloud
result = sim.run()

# %% papermill={"duration": 0.12756, "end_time": "2026-04-22T11:30:35.284407", "exception": false, "start_time": "2026-04-22T11:30:35.156847", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.013, "end_time": "2026-04-22T11:30:35.298964", "exception": false, "start_time": "2026-04-22T11:30:35.285964", "status": "completed"}
result.plot_interactive(phase=True)

# %% papermill={"duration": 0.00833, "end_time": "2026-04-22T11:30:35.308764", "exception": false, "start_time": "2026-04-22T11:30:35.300434", "status": "completed"}
result.show_animation()
