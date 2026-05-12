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

# %% [markdown] papermill={"duration": 0.001655, "end_time": "2026-04-18T15:49:11.926549", "exception": false, "start_time": "2026-04-18T15:49:11.924894", "status": "completed"}
# # Palace CPW Simulation — Lumped Ports
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a CPW (coplanar waveguide) structure with **lumped ports**.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.000942, "end_time": "2026-04-18T15:49:11.928627", "exception": false, "start_time": "2026-04-18T15:49:11.927685", "status": "completed"}
# ### Define GSG electrode

# %% papermill={"duration": 1.289763, "end_time": "2026-04-18T15:49:13.219276", "exception": false, "start_time": "2026-04-18T15:49:11.929513", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000817, "end_time": "2026-04-18T15:49:13.221395", "exception": false, "start_time": "2026-04-18T15:49:13.220578", "status": "completed"}
# ### Configure simulation

# %% papermill={"duration": 0.592297, "end_time": "2026-04-18T15:49:13.814782", "exception": false, "start_time": "2026-04-18T15:49:13.222485", "status": "completed"}
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("./palace-sim-cpw-lumped")
sim.set_geometry(c)
sim.set_stack(substrate_thickness=2.0, air_above=100.0, air_below=100.0)

# CPW lumped ports — offset defaults to length/2 (flush with conductor edge)
sim.add_cpw_port("o1", layer="topmetal2", s_width=20, gap_width=15, excited=True)
sim.add_cpw_port("o2", layer="topmetal2", s_width=20, gap_width=15, excited=False)

sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)

print(sim.validate_config())

# %% [markdown] papermill={"duration": 0.001197, "end_time": "2026-04-18T15:49:13.817164", "exception": false, "start_time": "2026-04-18T15:49:13.815967", "status": "completed"}
# ### Generate mesh

# %% papermill={"duration": 1.612178, "end_time": "2026-04-18T15:49:15.430189", "exception": false, "start_time": "2026-04-18T15:49:13.818011", "status": "completed"}
sim.mesh(
    preset="default",
    # refined_mesh_size=2.0,
    # max_mesh_size=40.0,
    # fmax=150e9,
    margin_x=50.0,
    margin_y=0,
)

# %% papermill={"duration": 0.577498, "end_time": "2026-04-18T15:49:16.008713", "exception": false, "start_time": "2026-04-18T15:49:15.431215", "status": "completed"}
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "SiO2__None", "SiO2__passive", "air__passive"],
    interactive=True,
)

# %% [markdown] papermill={"duration": 0.002318, "end_time": "2026-04-18T15:49:16.013622", "exception": false, "start_time": "2026-04-18T15:49:16.011304", "status": "completed"}
# ### Run simulation

# %% papermill={"duration": 123.129284, "end_time": "2026-04-18T15:51:19.145033", "exception": false, "start_time": "2026-04-18T15:49:16.015749", "status": "completed"}
sim.write_config()
results = sim.run()

# %% [markdown] papermill={"duration": 0.002571, "end_time": "2026-04-18T15:51:19.151041", "exception": false, "start_time": "2026-04-18T15:51:19.148470", "status": "completed"}
# ### Plot S-parameters

# %% papermill={"duration": 0.244907, "end_time": "2026-04-18T15:51:19.399008", "exception": false, "start_time": "2026-04-18T15:51:19.154101", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.01808, "end_time": "2026-04-18T15:51:19.420771", "exception": false, "start_time": "2026-04-18T15:51:19.402691", "status": "completed"}
results.plot_interactive(phase=True)

# %% papermill={"duration": 0.003691, "end_time": "2026-04-18T15:51:19.427545", "exception": false, "start_time": "2026-04-18T15:51:19.423854", "status": "completed"}
