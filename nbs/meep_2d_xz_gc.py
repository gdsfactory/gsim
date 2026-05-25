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

# %% [markdown] papermill={"duration": 0.002197, "end_time": "2026-04-22T12:26:09.180844", "exception": false, "start_time": "2026-04-22T12:26:09.178647", "status": "completed"}
# # 2D Grating Coupler
#
# This notebook demonstrates **2D FDTD in the XZ cross-section plane** using `gsim.meep`. Unlike the top-down XY effective-index sim, this one models the vertical stack (substrate / BOX / core / cladding) and a Gaussian-beam fiber source above the chip — the standard grating-coupler workflow.
#
#
# **Requirements:** GDSFactory+ account for cloud simulation.

# %% [markdown] papermill={"duration": 0.000763, "end_time": "2026-04-22T12:26:09.182596", "exception": false, "start_time": "2026-04-22T12:26:09.181833", "status": "completed"}
# ### Load a grating coupler + feed straight

# %% papermill={"duration": 1.246279, "end_time": "2026-04-22T12:26:10.429520", "exception": false, "start_time": "2026-04-22T12:26:09.183241", "status": "completed"}
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component()
gc = gf.components.grating_coupler_elliptical(fiber_angle=0.0)

gc_r = c.add_ref(gc)
s_r = c.add_ref(gf.components.straight(length=3))
s_r.connect("o1", gc_r.ports["o1"])
c.add_port("o2", port=s_r.ports["o2"])
c

# %% [markdown] papermill={"duration": 0.000767, "end_time": "2026-04-22T12:26:10.431296", "exception": false, "start_time": "2026-04-22T12:26:10.430529", "status": "completed"}
# ### Configure the XZ 2D simulation
#
# Key differences from the XY notebook:
# - `sim.solver(plane="xz", is_3d=False)` picks the vertical cross-section sim.
# - `sim.source_fiber(...)` replaces the port-based mode source.
# - `sim.monitors = ["o2"]` monitors the waveguide end (feed straight).

# %% papermill={"duration": 0.052953, "end_time": "2026-04-22T12:26:10.484942", "exception": false, "start_time": "2026-04-22T12:26:10.431989", "status": "completed"}
from gsim import meep
from gsim.common.stack import get_stack

stack = get_stack()  # auto-detects active PDK

sim = meep.Simulation()

sim.geometry(component=c, stack=stack)
sim.materials = {"si": 3.47, "SiO2": 1.44}

sim.solver(resolution=25, is_3d=False, plane="xz", save_animation=True)
sim.solver.stop_when_energy_decayed()

sim.source_fiber(
    x=25.0,
    z=2,
    angle_deg=-6.0,
    waist=5.2,
    wavelength=1.55,
    wavelength_span=0.06,
    polarization="TE",
)

sim.monitors = ["o2"]
sim.domain(pml=1.0, margin=0.5)
sim.num_freqs = 21

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.00072, "end_time": "2026-04-22T12:26:10.486504", "exception": false, "start_time": "2026-04-22T12:26:10.485784", "status": "completed"}
# ### Preview the XZ cross-section

# %% papermill={"duration": 0.855926, "end_time": "2026-04-22T12:26:11.343222", "exception": false, "start_time": "2026-04-22T12:26:10.487296", "status": "completed"}
sim.plot_2d(slices="y")

# %% [markdown] papermill={"duration": 0.005152, "end_time": "2026-04-22T12:26:11.351144", "exception": false, "start_time": "2026-04-22T12:26:11.345992", "status": "completed"}
# ### Run the simulation

# %% papermill={"duration": 132.689187, "end_time": "2026-04-22T12:28:24.042091", "exception": false, "start_time": "2026-04-22T12:26:11.352904", "status": "completed"}
result = sim.run()

# %% papermill={"duration": 0.156844, "end_time": "2026-04-22T12:28:24.199924", "exception": false, "start_time": "2026-04-22T12:28:24.043080", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.006374, "end_time": "2026-04-22T12:28:24.210107", "exception": false, "start_time": "2026-04-22T12:28:24.203733", "status": "completed"}
result.show_animation()

# %% papermill={"duration": 0.008708, "end_time": "2026-04-22T12:28:24.221683", "exception": false, "start_time": "2026-04-22T12:28:24.212975", "status": "completed"}
