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

# %% [markdown] papermill={"duration": 0.00345, "end_time": "2026-04-04T05:08:46.583945", "exception": false, "start_time": "2026-04-04T05:08:46.580495", "status": "completed"}
# # Running Palace Simulations
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a CPW (coplanar waveguide) structure.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001823, "end_time": "2026-04-04T05:08:46.589529", "exception": false, "start_time": "2026-04-04T05:08:46.587706", "status": "completed"}
# ### Load a pcell from IHP PDK

# %% papermill={"duration": 1.626697, "end_time": "2026-04-04T05:08:48.218023", "exception": false, "start_time": "2026-04-04T05:08:46.591326", "status": "completed"}
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

    # Top ground electrode
    r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    r1.move((0, (g_width + s_width) / 2 + gap_width))

    # Center signal electrode
    _r2 = c << gf.c.rectangle((length, s_width), centered=True, layer=layer)

    # Bottom ground electrode
    r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    r3.move((0, -(g_width + s_width) / 2 - gap_width))

    # Add ports at the signal center (one per side)
    # The CPW port API computes the gap element surfaces from s_width and gap_width
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

# %% [markdown] papermill={"duration": 0.000656, "end_time": "2026-04-04T05:08:48.219611", "exception": false, "start_time": "2026-04-04T05:08:48.218955", "status": "completed"}
# ### Configure and run simulation with DrivenSim

# %% papermill={"duration": 0.467892, "end_time": "2026-04-04T05:08:48.688108", "exception": false, "start_time": "2026-04-04T05:08:48.220216", "status": "completed"}
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

# Create simulation object
sim_lumped = DrivenSim()

# Set output directory
sim_lumped.set_output_dir("./palace-sim-cpw")

# Set the component geometry
sim_lumped.set_geometry(c)

# Configure layer stack from active PDK
stack = get_stack(air_above=100.0, air_below=100.0)  # auto-detects active PDK
sim_lumped.set_stack(stack)

# Configure left CPW port (single port at signal center)
sim_lumped.add_cpw_port(
    "o1",
    layer="topmetal2",
    s_width=20,
    gap_width=15,
    length=1.0,
    # offset=2.5,
    excited=True,
)

# Configure right CPW port (single port at signal center)
sim_lumped.add_cpw_port(
    "o2",
    layer="topmetal2",
    s_width=20,
    gap_width=15,
    length=1.0,
    # offset=2.5,
    excited=False,
)

# Configure driven simulation (frequency sweep for S-parameters)
sim_lumped.set_driven(fmin=1e9, fmax=100e9, num_points=300)

# Validate configuration
print(sim_lumped.validate_config())

# %% [markdown]
# ### Configure simulation with DrivenSim for WavePorts

# %%
# Create simulation object
sim_waveport = DrivenSim()

# Set output directory
sim_waveport.set_output_dir("./palace-sim-cpw-waveport")

# Set the component geometry
sim_waveport.set_geometry(c)

# Reuse the same stack from active PDK
sim_waveport.set_stack(stack)

# Configure left CPW port (single port at signal center)
sim_waveport.add_wave_port("o1", layer="topmetal2", max_size=True, mode=1, excited=True)

# Configure right CPW port (single port at signal center)
sim_waveport.add_wave_port(
    "o2", layer="topmetal2", max_size=True, mode=1, excited=False
)

# Configure driven simulation (frequency sweep for S-parameters)
sim_waveport.set_driven(fmin=1e9, fmax=100e9, num_points=300)

# Validate configuration
print(sim_waveport.validate_config())

# %% papermill={"duration": 1.602879, "end_time": "2026-04-04T05:08:50.291850", "exception": false, "start_time": "2026-04-04T05:08:48.688971", "status": "completed"}
# Generate mesh with planar conductors (presets: "coarse", "default", "fine")
sim_lumped.mesh(
    preset="default",
    refined_mesh_size=2.0,
    max_mesh_size=40.0,
    fmax=150e9,
    margin_x=50.0,
    margin_y=0,
)

# Use default refinement with much finer custom sizing for waveports
sim_waveport.mesh(
    preset="default",
    refined_mesh_size=2.0,
    max_mesh_size=40.0,
    fmax=150e9,
    margin_x=0,
    margin_y=50.0,
)

# %%
# Solid view — coloured surfaces per physical group, boundary transparent
sim_lumped.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "SiO2__None", "SiO2__passive", "air__passive"],
    interactive=True,
)

# %% papermill={"duration": 0.900347, "end_time": "2026-04-04T05:08:51.193094", "exception": false, "start_time": "2026-04-04T05:08:50.292747", "status": "completed"}
# Solid view — coloured surfaces per physical group, boundary transparent
sim_waveport.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "SiO2__passive", "air__passive"],
    interactive=True,
)

# %% [markdown] papermill={"duration": 0.00199, "end_time": "2026-04-04T05:08:51.198503", "exception": false, "start_time": "2026-04-04T05:08:51.196513", "status": "completed"}
# ### Run simulation on cloud

# %% papermill={"duration": 502.375746, "end_time": "2026-04-04T05:17:13.575895", "exception": false, "start_time": "2026-04-04T05:08:51.200149", "status": "completed"}
sim_lumped.write_config()
results_lumped = sim_lumped.run()

# %% papermill={"duration": 0.158415, "end_time": "2026-04-04T05:17:13.736142", "exception": false, "start_time": "2026-04-04T05:17:13.577727", "status": "completed"}
sim_waveport.write_config()
results_waveport = sim_waveport.run()

# %%
import matplotlib.pyplot as plt

from gsim.palace import load_sparams

sp_lumped = load_sparams(results_lumped.files)
sp_waveport = load_sparams(results_waveport.files)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

ax1.plot(sp_lumped.freq, sp_lumped.s21.db, label="S21 (lumped)")
ax1.plot(sp_waveport.freq, sp_waveport.s21.db, "--", label="S21 (waveport)")

ax2.plot(sp_lumped.freq, sp_lumped.s21.deg, label="S21 (lumped)")
ax2.plot(sp_waveport.freq, sp_waveport.s21.deg, "--", label="S21 (waveport)")

ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("S21 — Lumped vs Waveport")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Phase (deg)")
ax2.legend()
ax2.grid(True)

fig.tight_layout()
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

ax1.plot(sp_lumped.freq, sp_lumped.s11.db, label="S11 (lumped)")
ax1.plot(sp_waveport.freq, sp_waveport.s11.db, "--", label="S11 (waveport)")

ax2.plot(sp_lumped.freq, sp_lumped.s11.deg, label="S11 (lumped)")
ax2.plot(sp_waveport.freq, sp_waveport.s11.deg, "--", label="S11 (waveport)")

ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("S11 — Lumped vs Waveport")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Phase (deg)")
ax2.legend()
ax2.grid(True)

fig.tight_layout()
plt.show()

# %%
