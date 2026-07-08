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
# # MEEP Mode Solver - TFLN Ridge Waveguide
#
# Compute the fundamental TE mode of a **thin-film lithium niobate (TFLN)**
# ridge waveguide at lambda = 1.55 um.
#
# **Reference design** (*Ying Li et al.*, ACS Omega 2023, 8(10), 9644--9651):
#  - SiO2 cladding everywhere (no Si substrate)
#  - LiNbO3 slab: 220 nm thick, extends laterally
#  - LiNbO3 ridge: 180 nm on top of slab -> total 400 nm
#  - Ridge width w0 = 1.1 um
#  - Sidewall angle 17deg -> **supported via ``mp.Prism``**
#    with native ``sidewall_angle`` parameter (in radians).
#    The n_eff difference vs. vertical sidewalls is ~0.0045
#    (1.8533 vs 1.8578).
#
# **Key concept** --- ``background_material="sio2"`` sets the MEEP
# ``default_material`` to SiO2.  Any
# space not covered by an explicit layer is filled with SiO2 instead of air.
#
# **Expected results**:
#  - n_eff ~ 1.85 (fundamental TE-like mode, mode_index=0)
#  - n_group ~ 2.20

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
# ### LiNbO3 material (Zelmon 1997)
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
#
# Two LiNbO3 layers:
#  - **Slab** (layer 1,0): wide rectangle (thin-film slab)
#  - **Ridge** (layer 2,0): narrow rectangle (core on top of slab)
#
# The SiO2 BOX is provided by ``background_material="sio2"`` --- no
# separate GDS layer needed.  Layers without GDS polygons are treated
# as full-width backgrounds (``mode_solver.py:434--435``).
#
# **Sidewall angle note**: set ``sidewall_angle=17.0`` on the ridge
# :class:`Layer` to create a trapezoidal ridge cross-section via
# ``mp.Prism`` with native ``sidewall_angle``.

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
# Vertical material profile.  SiO2 fills all background space via
# ``background_material="sio2"``.
#
# | Region  | Material | Z-range (um)   | GDS layer |
# |---------|----------|----------------|-----------|
# | Ridge   | LiNbO3   | 0.22 ... 0.40    | (2,0) |
# | Slab    | LiNbO3   | 0.00 ... 0.22    | (1,0) |
# | BOX     | SiO2     | -2.0 ... 0.0     | (0,0) --- no polygons |

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
# ### Solve the fundamental TE-like mode
#
# Use the declarative :class:`Simulation` + :class:`ModeSolver` API.
# Grids are auto-constructed from ``y_span``, ``n_field_y``, and
# ``n_field_z`` settings on the mode solver.

# %%
WAVELENGTH = 1.55  # um
RESOLUTION = 64  # grid points per um
PML_THICKNESS = WAVELENGTH  # um

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
# ``index=True`` to overlay the refractive index as a greyscale underlay.

# %%
mode.plot_mode(
    components="all",
    norm="abs",
    index=True,
    suptitle=(
        f"TFLN ridge waveguide fundamental TE mode\n"
        f"(λ={WAVELENGTH:.2f} µm, w0={CORE_WIDTH:.1f} µm, "
        f"h_slab={SLAB_THICKNESS * 1000:.0f} nm, h_ridge={RIDGE_THICKNESS * 1000:.0f} nm, "
        f"bg=SiO2)"
    ),
    show=True,
)
