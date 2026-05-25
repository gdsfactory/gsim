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

# %% [markdown]
# # Palace Directional Coupler Simulation - Wave Ports
#
# This notebook mirrors the geometry in [meep_dc.ipynb](meep_dc.ipynb), but configures a Palace driven simulation with waveport boundary conditions.
#
# MEEP-specific features (for example FDTD PML/animation details) are omitted.

# %% [markdown]
# ### Load the same directional coupler geometry

# %%
from pathlib import Path

import gdsfactory.component as gfc
from ubcpdk import PDK, cells

# Use a workspace-local temp path for layer preview artifacts.
gds_tmp = Path.cwd() / ".gdsfactory_tmp"
gds_tmp.mkdir(parents=True, exist_ok=True)
gfc.GDSDIR_TEMP = gds_tmp

PDK.activate()

c = cells.coupler()

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown]
# ### Configure Palace driven simulation with waveports

# %%
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("./palace-sim-dc-waveport")
sim.set_geometry(c)

sim.set_stack(
    substrate_thickness=2.0,
    add_oxide_dielectric=False,
    add_passivation_dielectric=False,
)

# Use direct sparse solve backend.
sim.set_numerical(
    solver_type="MUMPS",
    max_iterations=1,
    tolerance=1e-8,
)

for i, port in enumerate(c.ports):
    sim.add_wave_port(
        str(port.name),
        layer="core",
        z_margin=1.5,
        lateral_margin=1.5,
        max_size=False,
        mode=1,
        excited=(i == 0),
    )

# Around 1.55 um (~193.5 THz).
sim.set_driven(fmin=190e12, fmax=200e12, num_points=21)

print(sim.validate_config())

# %%
import gdsfactory as gf

active = gf.get_active_pdk()
print("Active PDK:", active.name)

stack = sim._resolve_stack()
print("Stack layers:")
for name, layer in stack.layers.items():
    print(
        f"  - {name}: type={layer.layer_type}, material={layer.material}, "
        f"gds={tuple(layer.gds_layer)}, z=({layer.zmin}, {layer.zmax})"
    )

print("\nMaterial properties (selected):")
for key in ("si", "sio2", "SiO2", "air"):
    if key in stack.materials:
        print(f"  - {key}: {stack.materials[key]}")

print("\nDielectric regions:")
for d in stack.dielectrics:
    print(f"  - {d}")

# %% [markdown]
# ### Generate mesh

# %%
sim.set_airbox(
    margin_x=0.0,
    margin_y=1.5,
    z_above=0.5,
    z_below=0.5,
)

# fmax=220 THz -> lambda0~1.36 um. In Si (n~3.45), lambda~0.395 um,
# so for >=5 points per wavelength use mesh size <=0.079 um.
sim.mesh(
    preset="fine",
    refined_mesh_size=0.05,
    max_mesh_size=0.25,
    fmax=220e12,
    curve_fit_mode="bspline",
    curve_fit_layers=["core", "core2"],
    curve_fit_tolerance_um=0.01,
    curve_fit_min_points=8,
    high_order_elements=True,
    high_order_order=2,
    high_order_optimize=True,
)

# %%
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None"],
    interactive=True,
)

# %% [markdown]
# ### Run simulation

# %%
for p in sim._last_mesh_result.port_info:
    print(
        f"{p['type']} P{p['portnumber']}: "
        f"z=({p['zmin']:.3f}, {p['zmax']:.3f}), "
        f"x=({p['xmin']:.3f}, {p['xmax']:.3f}), "
        f"y=({p['ymin']:.3f}, {p['ymax']:.3f})"
    )

# %%
import json
from pathlib import Path

# For photonic sims, bypass conductor-oriented mesh validation.
config_path = sim.write_config(photonic=True)
cfg = json.loads(Path(config_path).read_text())
print("Solver.Linear:", cfg["Solver"]["Linear"])

results = sim.run_local(num_processes=16, verbose=True)

# %% [markdown]
# ### Plot S-parameters

# %%
results.plot_interactive()

# %%
results.plot_interactive(phase=True)
