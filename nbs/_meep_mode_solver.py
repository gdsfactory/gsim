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
# - gdsfactory for cross-section mode and Geometry handling
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
# ### 1. Slab mode — uniform layer stack
#
# A slab mode solver treats the stack as infinite in *x* and *y*. The mode
# propagates along *x* with field variation only along *z*. This is the 1D
# effective-index building block used for variational 2D approximations.

# %%
from gsim.common.stack import get_stack
from gsim.meep import solve_slab_mode

stack = get_stack()  # defaults to active PDK stack

result = solve_slab_mode(
    stack=stack,
    wavelength=1.55,
    band_num=1,          # fundamental mode
    parity="NO_PARITY",  # no symmetry constraint
    resolution=32,
)

print(f"n_eff = {result.n_eff:.4f}")
print(f"n_group = {result.n_group}")
print(f"kdom = {result.kdom}")
print(f"band = {result.band_num}, parity = {result.parity}")

# %% [markdown]
# ### 2. Accessing field profiles
#
# `ModeResult.fields` is a dict of complex-valued numpy arrays keyed by
# field component name (`"Ex"`, `"Ey"`, `"Ez"`, `"Hx"`, `"Hy"`, `"Hz"`).

# %%
import numpy as np

for comp, arr in result.fields.items():
    print(f"{comp}: shape={arr.shape}, |max|={np.abs(arr).max():.4f}")

# %% [markdown]
# ### 3. Cross-section mode — straight waveguide at a port
#
# For a 2D waveguide cross-section, the solver extracts the component
# geometry at the port's *y*-coordinate, builds a 2D XZ MEEP cell with the
# layer stack, and computes the guided mode propagating along *x*.

# %%
import gdsfactory as gf
from gsim.meep import solve_cross_section_mode

gf.gpdk.PDK.activate()

c = gf.components.straight(length=10, width=0.5)

result = solve_cross_section_mode(
    component=c,
    stack=stack,
    port="o1",            # auto-extracts y-cut + x-span from port
    wavelength=1.55,
    band_num=1,
    resolution=32,
)

print(f"n_eff = {result.n_eff:.4f}")
print(f"n_group = {result.n_group}")
print(f"fields: {list(result.fields.keys())}")

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

print(f"n_eff = {result.n_eff:.4f}")

# %% [markdown]
# ### 5. Using the Simulation wrapper
#
# `Simulation.solve_mode()` resolves the stack and materials internally,
# then delegates to the appropriate standalone function. If a component
# with `port` or `position` is set, it runs cross-section mode; otherwise
# it falls back to slab mode.

# %%
from gsim import meep

sim = meep.Simulation()
sim.geometry.component = c
sim.geometry.stack = stack

result = sim.solve_mode(port="o1", wavelength=1.55)

print(f"n_eff = {result.n_eff:.4f}")

# %% [markdown]
# ### 6. Wavelength sweep (broadband n_eff)
#
# Loop over wavelengths to compute dispersion curves.

# %%
wavelengths = [1.50, 1.52, 1.54, 1.55, 1.56, 1.58, 1.60]
n_effs = []

for wl in wavelengths:
    result = sim.solve_mode(port="o1", wavelength=wl, band_num=1)
    n_effs.append(result.n_eff)
    print(f"  λ = {wl:.2f} µm → n_eff = {result.n_eff:.4f}")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(wavelengths, n_effs, "o-")
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("n_eff")
ax.set_title("Dispersion — fundamental TE mode")
ax.grid(True, alpha=0.3)
fig.tight_layout()
