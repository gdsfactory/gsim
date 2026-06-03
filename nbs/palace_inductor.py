# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.003748, "end_time": "2026-05-18T13:26:54.014979", "exception": false, "start_time": "2026-05-18T13:26:54.011231", "status": "completed"}
# # Running Palace Simulations
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a spiral inductor with Metal1 guard ring.
#
# **Requirements:**
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - gsim with Palace backend
#

# %% [markdown] papermill={"duration": 0.001625, "end_time": "2026-05-18T13:26:54.019193", "exception": false, "start_time": "2026-05-18T13:26:54.017568", "status": "completed"}
# ### Build inductor + guard ring
#
# **Known PDK limitation:** `gf.components.inductor` accepts a `turns` parameter but does not use it in geometry construction. The spiral is always single-turn regardless of the value passed."""
#

# %% papermill={"duration": 2.353525, "end_time": "2026-05-18T13:26:56.374264", "exception": false, "start_time": "2026-05-18T13:26:54.020739", "status": "completed"}
import gdsfactory as gf
from ihp import PDK

PDK.activate()

c = gf.components.inductor(
    width=2,
    space=2.1,
    diameter=50,
    turns=1,
    layer_metal="TopMetal2drawing",
    layer_inductor="INDdrawing",
    layer_metal_pin="TopMetal2drawing",
    layers_no_fill=("NoMetFillerdrawing",),
).copy()

# Define guard ring dimensions based on the inductor's bounding box
bbox = c.bbox()
xmin, ymin = bbox.left, bbox.bottom
xmax, ymax = bbox.right, bbox.top

margin_outer = 0.0
margin_inner = -15.0

xlo, xro = xmin - margin_outer, xmax + margin_outer
ybo, yto = ymin - margin_outer, ymax + margin_outer
xli, xri = xmin - margin_inner, xmax + margin_inner
ybi, yti = ymin - margin_inner, ymax + margin_inner

w_v = xli - xlo  # Width vertical walls
h_h = yto - yti  # Height horizontal walls
over = 0.5  # Overlap for Gmsh to fuse the pieces

# Left wall
c.add_ref(
    gf.components.rectangle(
        size=(w_v + over, yto - ybo), layer="Metal1drawing", centered=True
    )
).move((xlo + w_v / 2 + over / 2, (yto + ybo) / 2))
# Right wall
c.add_ref(
    gf.components.rectangle(
        size=(w_v + over, yto - ybo), layer="Metal1drawing", centered=True
    )
).move((xro - w_v / 2 - over / 2, (yto + ybo) / 2))
# Top wall
c.add_ref(
    gf.components.rectangle(
        size=(xro - xlo, h_h + over), layer="Metal1drawing", centered=True
    )
).move(((xro + xlo) / 2, yto - h_h / 2 - over / 2))
# Bottom wall
c.add_ref(
    gf.components.rectangle(
        size=(xro - xlo, h_h + over), layer="Metal1drawing", centered=True
    )
).move(((xro + xlo) / 2, ybo + h_h / 2 + over / 2))

cc = c.copy()

c.draw_ports()
c.plot()

# %% [markdown] papermill={"duration": 0.000971, "end_time": "2026-05-18T13:26:56.376142", "exception": false, "start_time": "2026-05-18T13:26:56.375171", "status": "completed"}
# ### Configure and run simulation with DrivenSim

# %% papermill={"duration": 0.576828, "end_time": "2026-05-18T13:26:56.953899", "exception": false, "start_time": "2026-05-18T13:26:56.377071", "status": "completed"}
from gsim.palace import DrivenSim

# Create simulation object
sim = DrivenSim()

# Set output directory
sim.set_output_dir("./palace-sim-inductor-guardring")

# Set the component geometry
sim.set_geometry(cc)

# Configure layer stack from active PDK
sim.set_stack(substrate_thickness=180.0, include_substrate=True)

# Configure ports
sim.add_port(
    "P1", from_layer="metal1", to_layer="topmetal2", geometry="via", excited=True
)
sim.add_port(
    "P2", from_layer="metal1", to_layer="topmetal2", geometry="via", excited=True
)

# Configure driven simulation (frequency sweep for S-parameters)
sim.set_driven(fmin=10e9, fmax=200e9, num_points=50)

# Validate configuration
print(sim.validate_config())

# %% papermill={"duration": 1.930523, "end_time": "2026-05-18T13:26:58.885247", "exception": false, "start_time": "2026-05-18T13:26:56.954724", "status": "completed"}
# Generate mesh (presets: "coarse", "default", "fine")
sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=5)
sim.mesh(preset="default", refined_mesh_size=1.5)

# %% papermill={"duration": 0.706534, "end_time": "2026-05-18T13:26:59.592690", "exception": false, "start_time": "2026-05-18T13:26:58.886156", "status": "completed"}
# sim.plot_mesh(show_groups=["metal", "P"])
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "sio2__None", "air__sio2"],
    interactive=True,
)

# %% [markdown] papermill={"duration": 0.001592, "end_time": "2026-05-18T13:26:59.596145", "exception": false, "start_time": "2026-05-18T13:26:59.594553", "status": "completed"}
# ### Run simulation on cloud

# %% papermill={"duration": 134.955712, "end_time": "2026-05-18T13:29:14.553533", "exception": false, "start_time": "2026-05-18T13:26:59.597821", "status": "completed"}
# Run simulation on GDSFactory+ cloud
results = sim.run()

# %% papermill={"duration": 0.104371, "end_time": "2026-05-18T13:29:14.660424", "exception": false, "start_time": "2026-05-18T13:29:14.556053", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.012891, "end_time": "2026-05-18T13:29:14.675333", "exception": false, "start_time": "2026-05-18T13:29:14.662442", "status": "completed"}
results.plot_interactive(phase=True)
