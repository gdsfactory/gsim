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

# %% [markdown] papermill={"duration": 0.003093, "end_time": "2026-05-18T13:21:13.924709", "exception": false, "start_time": "2026-05-18T13:21:13.921616", "status": "completed"}
# # Palace CPW Simulation — Lumped Ports
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a CPW (coplanar waveguide) structure with **lumped ports**.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001148, "end_time": "2026-05-18T13:21:13.928081", "exception": false, "start_time": "2026-05-18T13:21:13.926933", "status": "completed"}
# ### Define GSG electrode

# %% papermill={"duration": 2.554843, "end_time": "2026-05-18T13:21:16.483929", "exception": false, "start_time": "2026-05-18T13:21:13.929086", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000789, "end_time": "2026-05-18T13:21:16.485840", "exception": false, "start_time": "2026-05-18T13:21:16.485051", "status": "completed"}
# ### Configure simulation

# %% papermill={"duration": 0.619004, "end_time": "2026-05-18T13:21:17.105521", "exception": false, "start_time": "2026-05-18T13:21:16.486517", "status": "completed"}
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("./palace-sim-cpw-lumped")
sim.set_geometry(c)

stack = get_stack(air_above=100.0, air_below=100.0)  # auto-detects active PDK
sim.set_stack(stack)

# CPW lumped ports — offset defaults to length/2 (flush with conductor edge)
sim.add_cpw_port("o1", layer="topmetal2", s_width=20, gap_width=15, excited=True)
sim.add_cpw_port("o2", layer="topmetal2", s_width=20, gap_width=15, excited=False)

sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.000747, "end_time": "2026-05-18T13:21:17.107199", "exception": false, "start_time": "2026-05-18T13:21:17.106452", "status": "completed"}
# ### Generate mesh

# %% papermill={"duration": 0.844215, "end_time": "2026-05-18T13:21:17.952069", "exception": false, "start_time": "2026-05-18T13:21:17.107854", "status": "completed"}
sim.set_airbox(margin_x=50, margin_y=0, z_above=100, z_below=100)

sim.mesh(
    preset="coarse",
    # refined_mesh_size=2.0,
    # max_mesh_size=40.0,
    # fmax=150e9,
    margin_x=50.0,
    margin_y=0,
)

# %% papermill={"duration": 0.555541, "end_time": "2026-05-18T13:21:18.509127", "exception": false, "start_time": "2026-05-18T13:21:17.953586", "status": "completed"}
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "SiO2__None", "SiO2__passive", "air__passive"],
    interactive=True,
)

# %% [markdown] papermill={"duration": 0.001812, "end_time": "2026-05-18T13:21:18.513433", "exception": false, "start_time": "2026-05-18T13:21:18.511621", "status": "completed"}
# ### Run simulation

# %% papermill={"duration": 97.647775, "end_time": "2026-05-18T13:22:56.162915", "exception": false, "start_time": "2026-05-18T13:21:18.515140", "status": "completed"}
results = sim.run()

# %% [markdown] papermill={"duration": 0.002577, "end_time": "2026-05-18T13:22:56.168483", "exception": false, "start_time": "2026-05-18T13:22:56.165906", "status": "completed"}
# ### Plot S-parameters

# %% papermill={"duration": 0.189017, "end_time": "2026-05-18T13:22:56.359908", "exception": false, "start_time": "2026-05-18T13:22:56.170891", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.012315, "end_time": "2026-05-18T13:22:56.374751", "exception": false, "start_time": "2026-05-18T13:22:56.362436", "status": "completed"}
results.plot_interactive(phase=True)

# %% papermill={"duration": 0.002349, "end_time": "2026-05-18T13:22:56.379283", "exception": false, "start_time": "2026-05-18T13:22:56.376934", "status": "completed"}
