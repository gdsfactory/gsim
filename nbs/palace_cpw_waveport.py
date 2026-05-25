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

# %% [markdown] papermill={"duration": 0.005198, "end_time": "2026-04-17T08:35:45.350343", "exception": false, "start_time": "2026-04-17T08:35:45.345145", "status": "completed"}
# # Palace CPW Simulation — Wave Ports
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a CPW (coplanar waveguide) structure with **wave ports**.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.002142, "end_time": "2026-04-17T08:35:45.355931", "exception": false, "start_time": "2026-04-17T08:35:45.353789", "status": "completed"}
# ### Define GSG electrode

# %% papermill={"duration": 1.318149, "end_time": "2026-04-17T08:35:46.675848", "exception": false, "start_time": "2026-04-17T08:35:45.357699", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000751, "end_time": "2026-04-17T08:35:46.677892", "exception": false, "start_time": "2026-04-17T08:35:46.677141", "status": "completed"}
# ### Configure simulation

# %% papermill={"duration": 0.479072, "end_time": "2026-04-17T08:35:47.158023", "exception": false, "start_time": "2026-04-17T08:35:46.678951", "status": "completed"}
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("./palace-sim-cpw-waveport")
sim.set_geometry(c)

stack = get_stack(air_above=100.0, air_below=100.0)  # auto-detects active PDK
sim.set_stack(stack)

# Wave ports — max_size fills the full domain boundary
sim.add_wave_port("o1", layer="topmetal2", max_size=True, mode=1, excited=True)
sim.add_wave_port("o2", layer="topmetal2", max_size=True, mode=1, excited=False)

sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.000725, "end_time": "2026-04-17T08:35:47.159654", "exception": false, "start_time": "2026-04-17T08:35:47.158929", "status": "completed"}
# ### Generate mesh

# %% papermill={"duration": 4.746906, "end_time": "2026-04-17T08:35:51.907178", "exception": false, "start_time": "2026-04-17T08:35:47.160272", "status": "completed"}
sim.mesh(
    preset="default",
    refined_mesh_size=2.0,
    max_mesh_size=40.0,
    fmax=150e9,
    margin_x=0,
    margin_y=50.0,
)

# %% papermill={"duration": 0.907891, "end_time": "2026-04-17T08:35:52.816150", "exception": false, "start_time": "2026-04-17T08:35:51.908259", "status": "completed"}
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "SiO2__None", "SiO2__passive", "air__passive"],
    interactive=True,
)

# %% [markdown] papermill={"duration": 0.003419, "end_time": "2026-04-17T08:35:52.823583", "exception": false, "start_time": "2026-04-17T08:35:52.820164", "status": "completed"}
# ### Run simulation

# %% papermill={"duration": 671.248603, "end_time": "2026-04-17T08:47:04.075888", "exception": false, "start_time": "2026-04-17T08:35:52.827285", "status": "completed"}
results = sim.run()

# %% [markdown] papermill={"duration": 0.002779, "end_time": "2026-04-17T08:47:04.084457", "exception": false, "start_time": "2026-04-17T08:47:04.081678", "status": "completed"}
# ### Plot S-parameters

# %% papermill={"duration": 0.24935, "end_time": "2026-04-17T08:47:04.336748", "exception": false, "start_time": "2026-04-17T08:47:04.087398", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.017403, "end_time": "2026-04-17T08:47:04.357808", "exception": false, "start_time": "2026-04-17T08:47:04.340405", "status": "completed"}
results.plot_interactive(phase=True)
