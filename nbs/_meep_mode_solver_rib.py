# ---
# jupyter:
#   jupytext:
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
# # MEEP Mode Solver - Rib Waveguide
#
# Compute the fundamental TE mode of a **silicon rib waveguide** at a
# single wavelength, returning `n_eff` and the full 2D (Y, Z) mode
# profile.
#
# **Rib cross-section:**
#  - SiO2 box (2 um)
#  - Si slab (70 nm, partially etched)
#  - Si rib (150 nm on top of slab -> total 220 nm)
#
# The rib provides lateral confinement via effective-index contrast
# between the ridge and the thinner side slabs.

# %% [markdown]
# ### Imports & MEEP check

# %%
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

import gsim.meep as gm
from gsim.common.stack.extractor import Layer, LayerStack

try:
    import meep as mp

    print(f"MEEP {mp.__version__} ready")
except ImportError as err:
    raise SystemExit(
        "MEEP not found. Install it via conda-forge:\n"
        "    conda install -c conda-forge pymeep"
    ) from err

plt.close()

gf.gpdk.PDK.activate()

# %% [markdown]
# ### Build the GDS component
#
# We manually create a component with **two Si layers** - a wide slab and
# a narrow rib on top.  The mode solver later slices through both layers
# at the port Y-position to reconstruct the 2D (X,Z) geometry.

# %%
SLAB_WIDTH = 3.0  # um
RIB_WIDTH = 0.5  # um
LENGTH = 10.0  # um

c = gf.Component()

# Si rib layer (layer 2) - narrow ridge on top
c.add_polygon(
    [
        (-LENGTH / 2, -RIB_WIDTH / 2),
        (LENGTH / 2, -RIB_WIDTH / 2),
        (LENGTH / 2, RIB_WIDTH / 2),
        (-LENGTH / 2, RIB_WIDTH / 2),
    ],
    layer=(2, 0),
)

# Ports at both ends
c.add_port(
    name="o1", center=(-LENGTH / 2, 0), width=RIB_WIDTH, orientation=180, layer=(1, 0)
)
c.add_port(
    name="o2", center=(LENGTH / 2, 0), width=RIB_WIDTH, orientation=0, layer=(1, 0)
)

print(f"Component: {c.name}")
print(f"  Ports:  {[p.name for p in c.ports]}")
print(f"  Layers: {list(c.layers)}")

# %% [markdown]
# ### Layer stack
#
# Define the vertical material profile.  Each non-air layer maps a GDS
# layer to a Z-range.

# %%
layers = {
    "ox": Layer(
        name="box",
        gds_layer=(0, 0),
        zmin=-1,
        zmax=0.0,
        thickness=1.0,
        material="sio2",
        layer_type="dielectric",
    ),
    "slab": Layer(
        name="slab",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.07,
        thickness=0.07,
        material="si",
        layer_type="dielectric",
    ),
    "rib": Layer(
        name="rib",
        gds_layer=(2, 0),
        zmin=0.07,
        zmax=0.22,
        thickness=0.15,
        material="si",
        layer_type="dielectric",
    ),
}
stack = LayerStack(layers=layers)

print("Layer stack:")
for name, l in stack.layers.items():
    print(
        f"  {name:6s}  z=[{l.zmin:+.3f}, {l.zmax:+.3f}]  t={l.thickness:.3f}  material={l.material}"
    )

# %% [markdown]
# ### Solve the fundamental mode
#
# Use the declarative :class:`Simulation` + :class:`ModeSolver` API.

# %%
WAVELENGTH = 1.55  # um
RESOLUTION = 32
PML_THICKNESS = 1 * WAVELENGTH


sim = gm.Simulation(
    geometry=gm.Geometry(component=c, stack=stack),
    domain=gm.Domain(
        pml=PML_THICKNESS,
        margin_z_above=0.5,
    ),
)
sim.mode_solver.wavelengths = [WAVELENGTH]
sim.mode_solver.fundamental().at_port("o1")
sim.mode_solver.y_span = SLAB_WIDTH
sim.mode_solver.n_field_y = 100
sim.mode_solver.n_field_z = 100

sweep = sim.solve_modes()
mode = sweep.at(WAVELENGTH).band(1)

print(f"n_eff    = {mode.n_eff:.6f}")
print(f"n_group  = {mode.n_group}")
print(f"kdom     = {[f'{k:.6f}' for k in mode.kdom]}")
print(f"band     = {mode.band_num}, parity = {mode.parity}")
print(f"fields   = {list(mode.fields.keys())}")
for comp, arr in mode.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

# %% [markdown]
# ### Refractive index profile
#
# Use ``ModeResult.plot_index()`` which auto-computes ``n`` from the
# stored :class:`LayerStack` and grid arrays.

# %%
mode.plot_index(show=True)


# %% [markdown]
# ### 2D mode profile — all field components
#
# Use ``ModeResult.plot_mode()`` with ``components="all"`` and
# ``geometry=True`` to overlay structural geometry boundaries.

# %%
mode.plot_mode(
    components="all",
    norm="abs",
    geometry=True,
    suptitle=(
        f"Rib waveguide fundamental TE mode\n"
        f"(λ={WAVELENGTH:.2f} µm, slab={SLAB_WIDTH:.1f} µm, rib={RIB_WIDTH:.1f} µm)"
    ),
    show=True,
)
