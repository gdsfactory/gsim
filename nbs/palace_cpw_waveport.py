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

# %% [markdown] papermill={"duration": 0.005746, "end_time": "2026-06-12T07:04:20.151072", "exception": false, "start_time": "2026-06-12T07:04:20.145326", "status": "completed"}
# # Palace CPW Simulation — Wave Ports
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a CPW (coplanar waveguide) structure with **wave ports**.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.003686, "end_time": "2026-06-12T07:04:20.157977", "exception": false, "start_time": "2026-06-12T07:04:20.154291", "status": "completed"}
# ### Define GSG electrode

# %% papermill={"duration": 2.54829, "end_time": "2026-06-12T07:04:22.708430", "exception": false, "start_time": "2026-06-12T07:04:20.160140", "status": "completed"}
import gdsfactory as gf
from ihp import LAYER, PDK

PDK.activate()


@gf.cell
def gsg_electrode(
    length: float = 800,
    s_width: float = 20,
    g_width: float = 40,
    gap_width: float = 15,
    layer=LAYER.TopMetal2drawing,
) -> gf.Component:
    """
    Create a GSG (Ground-Signal-Ground) electrode.

    Args:
        length: horizontal length of the electrodes
        s_width: width of the signal (center) electrode
        g_width: width of the ground electrodes
        gap_width: gap between signal and ground electrodes
        layer: layer for the metal
    """
    c = gf.Component()

    r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    r1.move((0, (g_width + s_width) / 2 + gap_width))

    _r2 = c << gf.c.rectangle((length, s_width), centered=True, layer=layer)

    r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    r3.move((0, -(g_width + s_width) / 2 - gap_width))

    c.add_port(
        name="o1",
        center=(-length / 2, 0),
        width=s_width,
        orientation=180,
        port_type="electrical",
        layer=layer,
    )

    c.add_port(
        name="o2",
        center=(length / 2, 0),
        width=s_width,
        orientation=0,
        port_type="electrical",
        layer=layer,
    )

    return c


c = gsg_electrode()
cc = c.copy()
cc.draw_ports()
cc

# %% [markdown] papermill={"duration": 0.000784, "end_time": "2026-06-12T07:04:22.710470", "exception": false, "start_time": "2026-06-12T07:04:22.709686", "status": "completed"}
# ### Configure simulation

# %% papermill={"duration": 0.639064, "end_time": "2026-06-12T07:04:23.350467", "exception": false, "start_time": "2026-06-12T07:04:22.711403", "status": "completed"}
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("./palace-sim-cpw-waveport")
sim.set_geometry(c)

stack = get_stack()  # auto-detects active PDK
sim.set_stack(stack)
sim.set_airbox(margin_x=0.0, margin_y=50, z_above=100.0, z_below=100.0)

# Wave ports — max_size fills the full domain boundary
sim.add_wave_port("o1", layer="topmetal2", max_size=True, mode=1, excited=True)
sim.add_wave_port("o2", layer="topmetal2", max_size=True, mode=1, excited=False)

sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.000814, "end_time": "2026-06-12T07:04:23.352349", "exception": false, "start_time": "2026-06-12T07:04:23.351535", "status": "completed"}
# ### Generate mesh

# %% papermill={"duration": 5.823922, "end_time": "2026-06-12T07:04:29.177017", "exception": false, "start_time": "2026-06-12T07:04:23.353095", "status": "completed"}
sim.mesh(preset="default", refined_mesh_size=2.0, max_mesh_size=40.0, fmax=150e9)

# %% papermill={"duration": 1.826372, "end_time": "2026-06-12T07:04:31.004882", "exception": false, "start_time": "2026-06-12T07:04:29.178510", "status": "completed"}
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "SiO2__None", "SiO2__passive", "air__passive"],
    interactive=True,
)

# %% [markdown] papermill={"duration": 0.003212, "end_time": "2026-06-12T07:04:31.011514", "exception": false, "start_time": "2026-06-12T07:04:31.008302", "status": "completed"}
# ### Run simulation

# %% papermill={"duration": 572.564296, "end_time": "2026-06-12T07:14:03.578757", "exception": false, "start_time": "2026-06-12T07:04:31.014461", "status": "completed"}
results = sim.run()

# %% [markdown] papermill={"duration": 0.003248, "end_time": "2026-06-12T07:14:03.585116", "exception": false, "start_time": "2026-06-12T07:14:03.581868", "status": "completed"}
# ### Plot S-parameters

# %% papermill={"duration": 0.161449, "end_time": "2026-06-12T07:14:03.749486", "exception": false, "start_time": "2026-06-12T07:14:03.588037", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.014154, "end_time": "2026-06-12T07:14:03.767046", "exception": false, "start_time": "2026-06-12T07:14:03.752892", "status": "completed"}
results.plot_interactive(phase=True)
