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

# %% [markdown] papermill={"duration": 0.0015, "end_time": "2026-06-12T07:45:50.781381", "exception": false, "start_time": "2026-06-12T07:45:50.779881", "status": "completed"}
# # 2D FDTD
#
# This notebook demonstrates **2D effective-index FDTD** simulations using `gsim.meep`.
#
# 2D simulations collapse the z-dimension, making them 10–100× faster than full 3D. They use an effective-index approximation and enforce TE polarization.
#
# **When to use 2D:**
# - Quick design-space exploration and parameter sweeps
# - Verifying port connectivity and mode coupling before committing to 3D
# - Components where vertical confinement is well-described by an effective index
#
# **Requirements:**
#
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.000836, "end_time": "2026-06-12T07:45:50.783250", "exception": false, "start_time": "2026-06-12T07:45:50.782414", "status": "completed"}
# ### Load a component from GDSFactory

# %% papermill={"duration": 1.129147, "end_time": "2026-06-12T07:45:51.913152", "exception": false, "start_time": "2026-06-12T07:45:50.784005", "status": "completed"}
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.components.coupler(gap=0.5)
c

# %% [markdown] papermill={"duration": 0.000681, "end_time": "2026-06-12T07:45:51.914774", "exception": false, "start_time": "2026-06-12T07:45:51.914093", "status": "completed"}
# ### Configure 2D simulation
#
# The only difference from a 3D simulation is `sim.solver.is_3d = False`.
# This collapses the z-dimension, ignores sidewall angles, and enforces TE polarization.

# %% papermill={"duration": 0.057654, "end_time": "2026-06-12T07:45:51.973158", "exception": false, "start_time": "2026-06-12T07:45:51.915504", "status": "completed"}
from gsim import meep
from gsim.common.stack import get_stack
from gsim.meep.models.api import Material

stack = get_stack()  # auto-detects active PDK

sim = meep.Simulation()

sim.geometry(component=c, stack=stack)
sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.monitors = ["o1", "o2", "o3"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=25, is_3d=False)
sim.num_freqs = 21
sim.solver.stop_when_energy_decayed()

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.000629, "end_time": "2026-06-12T07:45:51.974554", "exception": false, "start_time": "2026-06-12T07:45:51.973925", "status": "completed"}
# ### Preview geometry

# %% papermill={"duration": 0.523111, "end_time": "2026-06-12T07:45:52.498247", "exception": false, "start_time": "2026-06-12T07:45:51.975136", "status": "completed"}
sim.plot_2d(slices="z")

# %% [markdown] papermill={"duration": 0.00076, "end_time": "2026-06-12T07:45:52.500544", "exception": false, "start_time": "2026-06-12T07:45:52.499784", "status": "completed"}
# For an interactive preview, use `plot_2d_interactive()`. It returns a Plotly
# figure where you can zoom, pan, and toggle individual layers, materials, PML
# regions, and ports on/off via the legend.

# %% papermill={"duration": 0.093389, "end_time": "2026-06-12T07:45:52.594757", "exception": false, "start_time": "2026-06-12T07:45:52.501368", "status": "completed"}
sim.plot_2d_interactive()

# %% [markdown] papermill={"duration": 0.000948, "end_time": "2026-06-12T07:45:52.596741", "exception": false, "start_time": "2026-06-12T07:45:52.595793", "status": "completed"}
# ### Run 2D simulation on cloud

# %% papermill={"duration": 130.177283, "end_time": "2026-06-12T07:48:02.774893", "exception": false, "start_time": "2026-06-12T07:45:52.597610", "status": "completed"}
result = sim.run()

# %% papermill={"duration": 0.030581, "end_time": "2026-06-12T07:48:02.807369", "exception": false, "start_time": "2026-06-12T07:48:02.776788", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.013849, "end_time": "2026-06-12T07:48:02.822704", "exception": false, "start_time": "2026-06-12T07:48:02.808855", "status": "completed"}
result.plot_interactive(phase=True)

# %% papermill={"duration": 0.001271, "end_time": "2026-06-12T07:48:02.825457", "exception": false, "start_time": "2026-06-12T07:48:02.824186", "status": "completed"}
