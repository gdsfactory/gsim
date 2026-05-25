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

# %% [markdown] papermill={"duration": 0.00153, "end_time": "2026-04-22T12:34:54.306606", "exception": false, "start_time": "2026-04-22T12:34:54.305076", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000813, "end_time": "2026-04-22T12:34:54.308552", "exception": false, "start_time": "2026-04-22T12:34:54.307739", "status": "completed"}
# ### Load a pcell from UBC PDK

# %% papermill={"duration": 1.274009, "end_time": "2026-04-22T12:34:55.583235", "exception": false, "start_time": "2026-04-22T12:34:54.309226", "status": "completed"}
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.components.coupler(gap=0.5)
c

# %% [markdown] papermill={"duration": 0.000704, "end_time": "2026-04-22T12:34:55.584888", "exception": false, "start_time": "2026-04-22T12:34:55.584184", "status": "completed"}
# ### Configure 2D simulation
#
# The only difference from a 3D simulation is `sim.solver.is_3d = False`.
# This collapses the z-dimension, ignores sidewall angles, and enforces TE polarization.

# %% papermill={"duration": 0.06344, "end_time": "2026-04-22T12:34:55.648999", "exception": false, "start_time": "2026-04-22T12:34:55.585559", "status": "completed"}
from gsim import meep
from gsim.common.stack import get_stack

stack = get_stack()  # auto-detects active PDK

sim = meep.Simulation()

sim.geometry(component=c, stack=stack)
sim.materials = {"si": 3.47, "SiO2": 1.44}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.monitors = ["o1", "o2", "o3"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=25, is_3d=False)
sim.num_freqs = 21
sim.solver.stop_when_energy_decayed()

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.000603, "end_time": "2026-04-22T12:34:55.650410", "exception": false, "start_time": "2026-04-22T12:34:55.649807", "status": "completed"}
# ### Preview geometry

# %% papermill={"duration": 0.768434, "end_time": "2026-04-22T12:34:56.419446", "exception": false, "start_time": "2026-04-22T12:34:55.651012", "status": "completed"}
sim.plot_2d(slices="z")

# %% [markdown] papermill={"duration": 0.000828, "end_time": "2026-04-22T12:34:56.421323", "exception": false, "start_time": "2026-04-22T12:34:56.420495", "status": "completed"}
# ### Run 2D simulation on cloud

# %% papermill={"duration": 34.623659, "end_time": "2026-04-22T12:35:31.045720", "exception": false, "start_time": "2026-04-22T12:34:56.422061", "status": "completed"}
result = sim.run()

# %% papermill={"duration": 0.132918, "end_time": "2026-04-22T12:35:31.180318", "exception": false, "start_time": "2026-04-22T12:35:31.047400", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.011145, "end_time": "2026-04-22T12:35:31.192567", "exception": false, "start_time": "2026-04-22T12:35:31.181422", "status": "completed"}
result.plot_interactive(phase=True)

# %% papermill={"duration": 0.000965, "end_time": "2026-04-22T12:35:31.194975", "exception": false, "start_time": "2026-04-22T12:35:31.194010", "status": "completed"}
