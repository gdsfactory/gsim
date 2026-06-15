# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MEEP API Styles
#
# The `gsim.meep` API supports three equivalent styles for configuring simulations.
# All three produce identical `Simulation` objects — pick whichever reads best for your use case.

# %%
from ubcpdk import PDK, cells

PDK.activate()
c = cells.ebeam_y_1550()

# %% [markdown]
# ## 1. Callable style (recommended)
#
# Updates fields in place — only the fields you pass change, others keep their current values. No extra imports needed.

# %%
from gsim import meep
from gsim.meep.models.api import Material

sim = meep.Simulation()

sim.geometry(component=c, z_crop="auto")
sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.num_freqs = 11
sim.monitors = ["o1", "o2", "o3"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=20, simplify_tol=0.01, save_animation=True, verbose_interval=5.0)

# %% [markdown]
# ## 2. Attribute style
#
# One field per line. Most explicit — good when you need to set fields conditionally.

# %%
from gsim import meep
from gsim.meep.models.api import Material

sim = meep.Simulation()

sim.geometry.component = c
sim.geometry.z_crop = "auto"

sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}

sim.source.port = "o1"
sim.source.wavelength = 1.55
sim.source.wavelength_span = 0.01
sim.num_freqs = 11

sim.monitors = ["o1", "o2", "o3"]

sim.domain.pml = 1.0
sim.domain.margin = 0.5

sim.solver.resolution = 20
sim.solver.simplify_tol = 0.01
sim.solver.save_animation = True
sim.solver.verbose_interval = 5.0

# %% [markdown]
# ## 3. Constructor style
#
# Replaces the entire sub-object — fields you don't pass reset to defaults. Requires importing model classes, but useful when building configs programmatically or from saved presets.

# %%
from gsim import meep
from gsim.meep import FDTD, Domain, Geometry, Material, ModeSource

sim = meep.Simulation()

sim.geometry = Geometry(component=c, z_crop="auto")
sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}
sim.source = ModeSource(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.num_freqs = 11
sim.monitors = ["o1", "o2", "o3"]
sim.domain = Domain(pml=1.0, margin=0.5)
sim.solver = FDTD(
    resolution=20, simplify_tol=0.01, save_animation=True, verbose_interval=5.0
)

# %% [markdown]
# ## Combining styles
#
# All three styles can be freely combined. For example, use callable for bulk setup, then attribute for conditional tweaks:

# %%
from gsim import meep
from gsim.meep.models.api import Material

sim = meep.Simulation()

sim.geometry(component=c, z_crop="auto")
sim.materials = {
    "si": Material(refractive_index=3.47),
    "SiO2": Material(refractive_index=1.44),
}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.num_freqs = 11
sim.monitors = ["o1", "o2", "o3"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=20, simplify_tol=0.01)

# Conditional tweaks with attribute style
debug = True
if debug:
    sim.solver.save_animation = True
    sim.solver.verbose_interval = 5.0
