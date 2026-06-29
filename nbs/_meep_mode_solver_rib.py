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

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.meep import (
    mode_y_grid,
    mode_z_grid,
    refractive_index_profile,
    solve_cross_section_mode,
)

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
SLAB_WIDTH = 2.0  # um
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

# %%
WAVELENGTH = 1.55  # um
RESOLUTION = 64
pml_thickness = 0 * WAVELENGTH

y_span = SLAB_WIDTH
z_margin = (0, 0.5)  # asymmetric: 0 below, 0.5 above

z_min = min(l.zmin for l in stack.layers.values())
z_max = max(l.zmax for l in stack.layers.values())
actual_y_span = y_span + 2 * pml_thickness
actual_z_span = (z_max - z_min) + z_margin[0] + z_margin[1] + 2 * pml_thickness

y_grid = mode_y_grid(
    n_points=max(round(actual_y_span * RESOLUTION), 1),
    y_span=y_span,
    pml_thickness=pml_thickness,
)
z_grid = mode_z_grid(
    stack,
    n_points=max(round(actual_z_span * RESOLUTION), 1),
    z_margin=z_margin,
    pml_thickness=pml_thickness,
)
result = solve_cross_section_mode(
    component=c,
    stack=stack,
    port="o1",
    wavelength=WAVELENGTH,
    band_num=1,
    parity="NO_PARITY",
    resolution=RESOLUTION,
    field_y_grid=y_grid,
    field_z_grid=z_grid,
    z_margin=z_margin,
    pml_thickness=pml_thickness,
)

print(f"n_eff    = {result.n_eff:.6f}")
print(f"n_group  = {result.n_group}")
print(f"kdom     = {[f'{k:.6f}' for k in result.kdom]}")
print(f"band     = {result.band_num}, parity = {result.parity}")
print(f"fields   = {list(result.fields.keys())}")
for comp, arr in result.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

# %% [markdown]
# ### 2D mode profile - dominant component
#
# The fundamental TE-like rib mode has its primary electric field in Ey.

# %%
y_um = y_grid
z_um = z_grid
# Right: refractive index distribution

fig, ax2 = plt.subplots(1, 1, figsize=(7, 5))
n_yz = refractive_index_profile(
    stack,
    WAVELENGTH,
    z_grid=z_um,
    y_grid=y_um,
    component=c,
    port="o1",
)


im2 = ax2.pcolormesh(y_um, z_um, n_yz, shading="auto", cmap="rainbow", alpha=0.85)
plt.colorbar(im2, ax=ax2, label="n")
ax2.set_xlabel("y (um)")
ax2.set_ylabel("z (um)")
ax2.set_title("Refractive index")
ax2.set_aspect("equal")


for dom_comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    # dom_comp = max(result.fields, key=lambda k: np.abs(result.fields[k]).max())
    field_2d = np.abs(result.fields[dom_comp])
    nz, ny = field_2d.shape

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    # Field amplitude
    im = ax1.pcolormesh(y_um, z_um, field_2d, shading="auto", cmap="inferno")
    im2 = ax1.pcolormesh(y_um, z_um, n_yz, shading="auto", cmap="Greys", alpha=0.1)
    plt.colorbar(im, ax=ax1, label=f"|{dom_comp}| (arb. units)")
    ax1.set_xlabel("y (um)")
    ax1.set_ylabel("z (um)")
    ax1.set_title(f"|{dom_comp}|  n_eff={result.n_eff:.4f}")
    ax1.set_aspect("equal")

    fig.suptitle(
        f"Rib waveguide fundamental TE mode  \n"
        f"(lambda={WAVELENGTH:.2f} um, slab={SLAB_WIDTH:.1f}um, rib={RIB_WIDTH:.1f}um)",
        fontweight="bold",
    )
    fig.tight_layout()
plt.show()
