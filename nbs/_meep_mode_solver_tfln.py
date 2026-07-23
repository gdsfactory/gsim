# ---
# jupyter:
#   jupytext:
#     jupytext_version: 1.19.2
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MEEP Mode Solver — TFLN Ridge Waveguide
#
# Fundamental TE mode of thin-film lithium niobate ridge waveguide at λ=1.55 µm.
# Reference: Ying Li et al., ACS Omega 2023, 8(10), 9644–9651.
#
# **Design:** SiO2 cladding / LiNbO3 slab 220 nm / LiNbO3 ridge 180 nm (total 400 nm),
# ridge width 1.1 µm, sidewall angle 17°.
#
# **Expected:** n_eff ~ 1.85, n_group ~ 2.20.
# ``background_material="sio2"`` fills unpatterned space with SiO2.

# %% [markdown]
# ### Imports

# %%
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

import gsim.meep as gm
from gsim.common.stack.extractor import Layer, LayerStack

plt.close()

gf.gpdk.PDK.activate()

# %% [markdown]
# ### LiNbO₃ material (Zelmon 1997)
#
# LiNbO3 is birefringent and is now registered as a uniaxial material
# in ``gsim.common.stack.materials.MATERIALS_DB`` (notebook no longer
# manually registers it).  Both ordinary and extraordinary Sellmeier
# models from Zelmon et al. (JOSA B 14(12), 3319--3322, 1997) are included:
#
# Ordinary:
#   n_o^2 = 1 + 2.6734 lam^2/(lam^2 - 0.01764) + 1.2290 lam^2/(lam^2 - 0.05914)
#             + 12.614 lam^2/(lam^2 - 474.60)
#
# Extraordinary:
#   n_e^2 = 1 + 2.9804 lam^2/(lam^2 - 0.02047) + 0.5981 lam^2/(lam^2 - 0.0666)
#             + 8.9543 lam^2/(lam^2 - 416.08)
#
# For x-cut TFLN the TE mode's dominant E-field aligns with the
# extraordinary axis (zz).


# %% [markdown]
# ### Build the GDS component

# %%
SLAB_WIDTH = 5.0  # um --- wide slab
CORE_WIDTH = 1.1  # um --- w0 from reference design
LENGTH = 10.0  # um --- waveguide length (arbitrary for mode solving)

c = gf.Component()

# LiNbO3 ridge (layer 2) --- narrow core on top
c.add_polygon(
    [
        (-LENGTH / 2, -CORE_WIDTH / 2),
        (LENGTH / 2, -CORE_WIDTH / 2),
        (LENGTH / 2, CORE_WIDTH / 2),
        (-LENGTH / 2, CORE_WIDTH / 2),
    ],
    layer=(2, 0),
)

# Ports at both ends
c.add_port(
    name="o1",
    center=(-LENGTH / 2, 0),
    width=CORE_WIDTH,
    orientation=180,
    layer=(1, 0),
)
c.add_port(
    name="o2",
    center=(LENGTH / 2, 0),
    width=CORE_WIDTH,
    orientation=0,
    layer=(1, 0),
)

print(f"Component: {c.name}")
print(f"  Ports:  {[p.name for p in c.ports]}")
print(f"  Layers: {list(c.layers)}")

# %% [markdown]
# ### Layer stack
#
# SiO2 fills background via ``background_material="sio2"``.

# %%
SLAB_THICKNESS = 0.22  # um  (h3)
CORE_THICKNESS = 0.40  # um --- total LiNbO3 thickness
RIDGE_THICKNESS = CORE_THICKNESS - SLAB_THICKNESS  # 0.18 um

layers = {
    "box": Layer(
        name="box",
        gds_layer=(0, 0),
        zmin=-1.0,
        zmax=0.0,
        thickness=1.0,
        material="sio2",
        layer_type="dielectric",
    ),
    "slab": Layer(
        name="slab",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=SLAB_THICKNESS,
        thickness=SLAB_THICKNESS,
        material="linbo3",
        layer_type="dielectric",
    ),
    "ridge": Layer(
        name="ridge",
        gds_layer=(2, 0),
        zmin=SLAB_THICKNESS,
        zmax=CORE_THICKNESS,
        thickness=RIDGE_THICKNESS,
        material="linbo3",
        layer_type="dielectric",
        sidewall_angle=17.0,
    ),
}
stack = LayerStack(layers=layers)

print("Layer stack (+ SiO2 background):")
for name, l in stack.layers.items():
    print(
        f"  {name:6s}  z=[{l.zmin:+.3f}, {l.zmax:+.3f}]  "
        f"t={l.thickness:.3f}  material={l.material}  gds={l.gds_layer}"
    )

# %% [markdown]
# ### Solve

# %%
WAVELENGTH = 1.55  # um
RESOLUTION = 64  # grid points per um
PML_THICKNESS = WAVELENGTH  # um

sim = gm.Simulation(
    geometry=gm.Geometry(component=c, stack=stack),
    domain=gm.Domain(
        pml=PML_THICKNESS,
        margin_z=(0.0, 0.5),
    ),
)
sim.mode_solver.wavelengths = [WAVELENGTH]
sim.mode_solver.fundamental().at_port("o1")
sim.mode_solver.y_span = SLAB_WIDTH
sim.mode_solver.n_field_y = 1000
sim.mode_solver.n_field_z = 1000
sim.mode_solver.background_material = "sio2"

sweep = sim.solve_modes()
mode = sweep.at(WAVELENGTH).band(1)

print(f"n_eff     = {mode.n_eff}")
print(f"n_group   = {mode.n_group}")
print(f"kdom      = {[f'{k:.6f}' for k in mode.kdom]}")
print(f"band      = {mode.band_num}, parity = {mode.parity}")
print(f"fields    = {list(mode.fields.keys())}")
for comp, arr in mode.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

# %% [markdown]
# ### Index profile

# %%
mode.plot_index(show=True)

# %% [markdown]
# ### Mode profile

# %%
mode.plot_mode(
    components="all",
    norm="abs",
    geometry=True,
    suptitle=(
        f"TFLN ridge waveguide fundamental TE mode\n"
        f"(λ={WAVELENGTH:.2f} µm, w0={CORE_WIDTH:.1f} µm, "
        f"h_slab={SLAB_THICKNESS * 1000:.0f} nm, h_ridge={RIDGE_THICKNESS * 1000:.0f} nm, "
        f"bg=SiO2)"
    ),
    show=True,
)
