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

# %% [markdown] papermill={"duration": 0.001879, "end_time": "2026-07-06T15:20:20.052281", "exception": false, "start_time": "2026-07-06T15:20:20.050402", "status": "completed"}
# # 2D Grating Coupler
#
# This notebook demonstrates **2D FDTD in the XZ cross-section plane** using `gsim.meep`. Unlike the top-down XY effective-index sim, this one models the vertical stack (substrate / BOX / core / cladding) and a Gaussian-beam fiber source above the chip — the standard grating-coupler workflow.
#
#
# **Requirements:** GDSFactory+ account for cloud simulation.

# %% [markdown] papermill={"duration": 0.001059, "end_time": "2026-07-06T15:20:20.054799", "exception": false, "start_time": "2026-07-06T15:20:20.053740", "status": "completed"}
# ### Load a grating coupler + feed straight

# %% papermill={"duration": 1.175557, "end_time": "2026-07-06T15:20:21.231280", "exception": false, "start_time": "2026-07-06T15:20:20.055723", "status": "completed"}
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component()
gc = gf.components.grating_coupler_elliptical(fiber_angle=0.0)

gc_r = c.add_ref(gc)
s_r = c.add_ref(gf.components.straight(length=3))
s_r.connect("o1", gc_r.ports["o1"])
c.add_port("o2", port=s_r.ports["o2"])
c

# %% [markdown] papermill={"duration": 0.000798, "end_time": "2026-07-06T15:20:21.233232", "exception": false, "start_time": "2026-07-06T15:20:21.232434", "status": "completed"}
# ### Configure the XZ 2D simulation
#
# Key differences from the XY notebook:
# - `sim.solver(mode="2d", y_cut="auto")` picks the vertical XZ cross-section sim.
# - `sim.source_fiber(...)` replaces the port-based mode source.
# - `sim.monitors = ["o2"]` monitors the waveguide end (feed straight).

# %% papermill={"duration": 0.055659, "end_time": "2026-07-06T15:20:21.289674", "exception": false, "start_time": "2026-07-06T15:20:21.234015", "status": "completed"}
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

sim.solver(resolution=25, mode="2d", y_cut="auto", save_animation=True)
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
sim.domain(pml=1.0, margin_x=0.5, margin_y=0.5, margin_z=(1.5, 0))
sim.num_freqs = 21

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.000729, "end_time": "2026-07-06T15:20:21.291360", "exception": false, "start_time": "2026-07-06T15:20:21.290631", "status": "completed"}
# ### Preview the XZ cross-section

# %% papermill={"duration": 0.513783, "end_time": "2026-07-06T15:20:21.805866", "exception": false, "start_time": "2026-07-06T15:20:21.292083", "status": "completed"}
sim.plot_2d(slices="y")

# %% [markdown] papermill={"duration": 0.000872, "end_time": "2026-07-06T15:20:21.808274", "exception": false, "start_time": "2026-07-06T15:20:21.807402", "status": "completed"}
# ### Run the simulation

# %% papermill={"duration": 264.33169, "end_time": "2026-07-06T15:24:46.140801", "exception": false, "start_time": "2026-07-06T15:20:21.809111", "status": "completed"}
result = sim.run()

# %% papermill={"duration": 0.126819, "end_time": "2026-07-06T15:24:46.269650", "exception": false, "start_time": "2026-07-06T15:24:46.142831", "status": "completed"}
result.plot_interactive()

# %% papermill={"duration": 0.006356, "end_time": "2026-07-06T15:24:46.277574", "exception": false, "start_time": "2026-07-06T15:24:46.271218", "status": "completed"}
result.show_animation()

# %% papermill={"duration": 0.001424, "end_time": "2026-07-06T15:24:46.280804", "exception": false, "start_time": "2026-07-06T15:24:46.279380", "status": "completed"}
