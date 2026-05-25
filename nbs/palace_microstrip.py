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

# %% [markdown] papermill={"duration": 0.001283, "end_time": "2026-04-18T15:42:27.031755", "exception": false, "start_time": "2026-04-18T15:42:27.030472", "status": "completed"}
# # Running Palace Simulations: Microstrip
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a microstrip transmission line with via ports.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.000633, "end_time": "2026-04-18T15:42:27.034454", "exception": false, "start_time": "2026-04-18T15:42:27.033821", "status": "completed"}
# ### Load a pcell from IHP PDK

# %% papermill={"duration": 1.252975, "end_time": "2026-04-18T15:42:28.287987", "exception": false, "start_time": "2026-04-18T15:42:27.035012", "status": "completed"}
import gdsfactory as gf
from ihp import LAYER, PDK, cells

PDK.activate()

c = gf.Component()
r1 = c << cells.straight_metal(length=1000, width=14)

r = c.get_region(layer=LAYER.TopMetal2drawing)
r_sized = r.sized(+20000)
c.add_polygon(r_sized, layer=LAYER.Metal1drawing)


c.add_ports(r1.ports)

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown] papermill={"duration": 0.000649, "end_time": "2026-04-18T15:42:28.289520", "exception": false, "start_time": "2026-04-18T15:42:28.288871", "status": "completed"}
# ### Configure and run simulation with DrivenSim

# %% papermill={"duration": 0.473096, "end_time": "2026-04-18T15:42:28.763192", "exception": false, "start_time": "2026-04-18T15:42:28.290096", "status": "completed"}
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

# Create simulation object
sim = DrivenSim()

# Set output directory
sim.set_output_dir("./palace-sim-microstrip")

# Set the component geometry
sim.set_geometry(c)

# Configure layer stack from active PDK
stack = get_stack(air_above=300.0)  # auto-detects active PDK
sim.set_stack(stack)

# Configure via ports (Metal1 ground plane to TopMetal2 signal)
for port in c.ports:
    sim.add_port(port.name, from_layer="metal1", to_layer="topmetal2", geometry="via")

# Configure driven simulation (frequency sweep for S-parameters)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)

# Validate configuration
print(sim.validate_config())

# %% papermill={"duration": 1.070671, "end_time": "2026-04-18T15:42:29.834871", "exception": false, "start_time": "2026-04-18T15:42:28.764200", "status": "completed"}
# Generate mesh (presets: "coarse", "default", "fine")
sim.mesh(preset="default")

# %% papermill={"duration": 0.717334, "end_time": "2026-04-18T15:42:30.553126", "exception": false, "start_time": "2026-04-18T15:42:29.835792", "status": "completed"}
# Static PNG
sim.plot_mesh(show_groups=["metal", "P"])

# %% [markdown] papermill={"duration": 0.001594, "end_time": "2026-04-18T15:42:30.556259", "exception": false, "start_time": "2026-04-18T15:42:30.554665", "status": "completed"}
# ### Run simulation on GDSFactory+ Cloud

# %% papermill={"duration": 235.489245, "end_time": "2026-04-18T15:46:26.048567", "exception": false, "start_time": "2026-04-18T15:42:30.559322", "status": "completed"}
# Run simulation on GDSFactory+ cloud
results = sim.run()

# %% papermill={"duration": 0.262001, "end_time": "2026-04-18T15:46:26.320448", "exception": false, "start_time": "2026-04-18T15:46:26.058447", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.017669, "end_time": "2026-04-18T15:46:26.339969", "exception": false, "start_time": "2026-04-18T15:46:26.322300", "status": "completed"}
results.plot_interactive(phase=True)
