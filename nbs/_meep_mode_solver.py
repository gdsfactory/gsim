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

# %% [markdown]
# # Standalone MEEP Eigenmode Solver
#
# The `gsim.meep` module now exposes a standalone eigenmode solver for
# **1D slab modes** and **2D waveguide cross-section modes**. This runs
# MEEP locally — no cloud job required — and returns the effective index,
# field profiles, and wavevectors.
#
# **Requirements:**
#
# - Local MEEP installation: `conda install -c conda-forge pymeep`
# - gdsfactory for cross-section mode and geometry handling
#
# **API:**
#
# | Function | What it does |
# |---|---|
# | `solve_slab_mode(stack, wavelength)` | 1D slab mode from a layer stack |
# | `solve_cross_section_mode(component, stack, port=..., wavelength)` | 2D waveguide cross-section at a port |
# | `sim.solve_mode(port=..., wavelength)` | Simulation wrapper — delegates to one of the above |
#
# All return a `ModeResult` with `.n_eff`, `.fields`, `.kdom`, `.n_group`, etc.

# %% [markdown]
# ### Setup

# %%
import gdsfactory as gf
from gsim.common.stack import get_stack

gf.gpdk.PDK.activate()

try:
    import meep as mp

    print(f"MEEP {mp.__version__} ready")
    HAS_MEEP = True
except ImportError:
    print(
        "MEEP not found. Install it via conda-forge:\n"
        "    conda install -c conda-forge pymeep"
    )
    HAS_MEEP = False

stack = get_stack()

# Show the layer stack
for name, layer in stack.layers.items():
    print(
        f"  {name:20s}  z=[{layer.zmin:6.3f}, {layer.zmax:6.3f}]  "
        f"t={layer.thickness:.3f}  material={layer.material}"
    )

# %% [markdown]
# ### 1. Slab mode — uniform layer stack
#
# A slab mode solver treats the stack as infinite in *x* and *y*. The mode
# propagates along *x* with field variation only along *z*. This is the 1D
# effective-index building block used for variational 2D approximations.

# %%
from gsim.meep import solve_slab_mode

result = solve_slab_mode(
    stack=stack,
    wavelength=1.55,
    band_num=1,          # fundamental mode
    parity="NO_PARITY",  # no symmetry constraint
    resolution=32,
)

print(f"n_eff   = {result.n_eff:.6f}")
print(f"n_group = {result.n_group}")
print(f"kdom    = {[f'{k:.6f}' for k in result.kdom]}")
print(f"band    = {result.band_num}, parity = {result.parity}")

# %% [markdown]
# ### 2. Accessing field profiles
#
# `ModeResult.fields` is a dict of complex-valued numpy arrays keyed by
# field component name (`"Ex"`, `"Ey"`, `"Ez"`, `"Hx"`, `"Hy"`, `"Hz"`).

# %%
if globals().get("result") and result.fields:
    import numpy as np

    for comp, arr in result.fields.items():
        print(f"{comp}: shape={arr.shape}, |max|={np.abs(arr).max():.6f}")
else:
    print("No mode result available — run the slab mode cell first.")

# %% [markdown]
# ### 3. Cross-section mode — straight waveguide at a port
#
# For a 2D waveguide cross-section, the solver extracts the component
# geometry at the port's *y*-coordinate, builds a 2D XZ MEEP cell with the
# layer stack, and computes the guided mode propagating along *x*.

# %%
from gsim.meep import solve_cross_section_mode

c = gf.components.straight(length=10, width=0.5)

result = solve_cross_section_mode(
    component=c,
    stack=stack,
    port="o1",            # auto-extracts y-cut + x-span from port
    wavelength=1.55,
    band_num=1,
    resolution=32,
)

print(f"n_eff   = {result.n_eff:.6f}")
print(f"n_group = {result.n_group}")
print(f"fields  = {list(result.fields.keys())}")

# %% [markdown]
# ### 4. Cross-section at an arbitrary position
#
# When you don't have a port, specify `position=(x, y)` and an explicit
# `x_span` for the simulation cell width.

# %%
result = solve_cross_section_mode(
    component=c,
    stack=stack,
    position=(5.0, 0.0),   # (x, y) — y is used for the cut plane
    x_span=2.0,            # total x-width of the cell in µm
    wavelength=1.55,
    band_num=1,
    resolution=32,
)

print(f"n_eff = {result.n_eff:.6f}")

# %% [markdown]
# ### 5. Using the Simulation wrapper
#
# `Simulation.solve_mode()` resolves the stack and materials internally,
# then delegates to the appropriate standalone function. If a component
# with `port` or `position` is set, it runs cross-section mode; otherwise
# it falls back to slab mode.

# %%
from gsim import meep as meep_mod

sim = meep_mod.Simulation()
sim.geometry.component = c
sim.geometry.stack = stack

result = sim.solve_mode(port="o1", wavelength=1.55)

print(f"n_eff = {result.n_eff:.6f}")

# %% [markdown]
# ### 6. Wavelength sweep (broadband n_eff)
#
# Loop over wavelengths to compute the dispersion curve.

# %%
wavelengths = [1.50, 1.52, 1.54, 1.55, 1.56, 1.58, 1.60]
n_effs = []

for wl in wavelengths:
    result_mode = sim.solve_mode(port="o1", wavelength=wl, band_num=1)
    n_effs.append(result_mode.n_eff)
    print(f"  λ = {wl:.2f} µm → n_eff = {result_mode.n_eff:.6f}")

# %%
import matplotlib.pyplot as plt

if n_effs:
    fig, ax = plt.subplots()
    ax.plot(wavelengths, n_effs, "o-")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("n_eff")
    ax.set_title("Dispersion — fundamental TE mode")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
else:
    print("No dispersion data — run the wavelength sweep cell first.")
