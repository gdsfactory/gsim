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
# ridge waveguide at lambda = 1.55 um --- reproducing the *Optical Waveguide*
# section of the Tidy3D TFLN EOM example.
# https://www.flexcompute.com/tidy3d/examples/notebooks/TFLNTidy3d/
#
# **Reference design** (*Ying Li et al.*, ACS Omega 2023, 8(10), 9644--9651):
#  - SiO2 cladding everywhere (Tidy3D sets ``medium=SiO2`` --- **no Si substrate**)
#  - LiNbO3 slab: 220 nm thick, extends laterally
#  - LiNbO3 ridge: 180 nm on top of slab -> total 400 nm
#  - Ridge width w0 = 1.1 um
#  - Sidewall angle 17deg (PolySlab with taper) -> **approximated as vertical**
#    here because the GDS-based mode solver uses rectangular polygons.
#    The n_eff difference is < 0.01 for ridge waveguides at this aspect ratio.
#
# **Key concept** --- ``background_material="sio2"`` sets the MEEP
# ``default_material`` to SiO2, matching Tidy3D's ``medium=SiO2``.  Any
# space not covered by an explicit layer is filled with SiO2 instead of air.
#
# **Expected results** (Tidy3D reference):
#  - n_eff ~ 1.85 (fundamental TE-like mode, mode_index=0)
#  - n_group ~ 2.20

# %% [markdown]
# ### Imports & MEEP check

# %%
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import (
    MATERIALS_DB,
    DispersionModel,
    MaterialProperties,
    SellmeierTerm,
    ValidityRange,
)
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
# ### Register LiNbO3 material (Zelmon 1997, extraordinary axis)
#
# LiNbO3 is birefringent.  The Tidy3D example uses
# ``LiNbO3.Zelmon1997(1)`` --- the extraordinary axis.  For x-cut TFLN
# the TE mode's dominant E-field aligns with this axis.
#
# Sellmeier (Zelmon et al., JOSA B 14(12), 3319--3322, 1997):
#   n_e^2 = 1 + 2.9804 lambda^2/(lambda^2 - 0.02047) + 0.5981 lambda^2/(lambda^2 - 0.0666)
#             + 8.9543 lambda^2/(lambda^2 - 416.08)

# %%
MATERIALS_DB["linbo3"] = MaterialProperties(
    permittivity=4.9064,
    dispersion_models=[
        DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=2.9804, C=0.02047),
                SellmeierTerm(B=0.5981, C=0.0666),
                SellmeierTerm(B=8.9543, C=416.08),
            ],
            epsilon_inf=1.0,
            validity=ValidityRange(valid_wavelength=(0.4, 5.0)),
            source="Zelmon et al. 1997 (LiNbO3 extraordinary, e-polarized)",
        ),
    ],
)

from gsim.common.stack.materials import resolve_material_at_wavelength

resolved = resolve_material_at_wavelength("linbo3", 1.55)
n_linbo3 = (
    np.sqrt(resolved.permittivity)
    if resolved and resolved.permittivity
    else np.sqrt(4.9064)
)
print(f"n(LiNbO3 e-axis) at 1.55 um = {n_linbo3:.4f}")
resolved_sio2 = resolve_material_at_wavelength("sio2", 1.55)
n_sio2 = (
    np.sqrt(resolved_sio2.permittivity)
    if resolved_sio2 and resolved_sio2.permittivity
    else 1.444
)
print(f"n(SiO2) at 1.55 um = {n_sio2:.4f}")

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
# **Sidewall angle note**: the Tidy3D example uses ``PolySlab(sidewall_angle=17deg)``
# which produces a trapezoidal ridge cross-section.  GDS polygons are
# rectangular, so we approximate with vertical sidewalls.

# %%
SLAB_WIDTH = 12.0  # um --- wide slab (matches Tidy3D plane_size)
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
# ``background_material="sio2"``, matching Tidy3D's ``medium=SiO2``.
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

# %%
WAVELENGTH = 1.55  # um
RESOLUTION = 32  # grid points per um
PML_THICKNESS = WAVELENGTH  # um --- absorbing BCs prevent lateral standing waves

y_span = SLAB_WIDTH
z_margin = (0.0, 1)  # asymmetric: no margin below BOX, 0.5 um above ridge

z_min = min(l.zmin for l in stack.layers.values())
z_max = max(l.zmax for l in stack.layers.values())
actual_y_span = y_span + 2 * PML_THICKNESS
actual_z_span = (z_max - z_min) + z_margin[0] + z_margin[1] + 2 * PML_THICKNESS

y_grid = mode_y_grid(
    n_points=max(round(actual_y_span * RESOLUTION), 1),
    y_span=y_span,
    pml_thickness=PML_THICKNESS,
)
z_grid = mode_z_grid(
    stack,
    n_points=max(round(actual_z_span * RESOLUTION), 1),
    z_margin=z_margin,
    pml_thickness=PML_THICKNESS,
)

result = solve_cross_section_mode(
    component=c,
    stack=stack,
    port="o1",
    y_span=y_span,
    wavelength=WAVELENGTH,
    band_num=1,
    parity="NO_PARITY",
    resolution=RESOLUTION,
    field_y_grid=y_grid,
    field_z_grid=z_grid,
    z_margin=z_margin,
    pml_thickness=PML_THICKNESS,
    background_material="sio2",
)

print(f"n_eff     = {result.n_eff}  (Tidy3D ref: ~1.85)")
print(f"n_group   = {result.n_group}     (Tidy3D ref: ~2.20)")
print(f"kdom      = {[f'{k:.6f}' for k in result.kdom]}")
print(f"band      = {result.band_num}, parity = {result.parity}")
print(f"fields    = {list(result.fields.keys())}")
for comp, arr in result.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

# %% [markdown]
# ### 2D mode profiles --- all six field components
#
# The fundamental TE-like mode of the TFLN ridge waveguide has its
# primary electric field component in E_y (TE-like polarization).

# %%
y_um = y_grid
z_um = z_grid

n_yz = refractive_index_profile(
    stack,
    WAVELENGTH,
    z_grid=z_um,
    y_grid=y_um,
    component=c,
    port="o1",
    background_material="sio2",
)

# Refractive index map
fig, ax_rn = plt.subplots(1, 1, figsize=(7, 5))
im_rn = ax_rn.pcolormesh(y_um, z_um, n_yz, shading="auto", cmap="rainbow", alpha=0.85)
plt.colorbar(im_rn, ax=ax_rn, label="n")
ax_rn.set_xlabel("y (um)")
ax_rn.set_ylabel("z (um)")
ax_rn.set_title("Refractive index --- TFLN ridge waveguide")
ax_rn.set_aspect("equal")
fig.tight_layout()

# Field amplitude for each component
for dom_comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    field_2d = np.abs(result.fields[dom_comp])

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    im = ax1.pcolormesh(y_um, z_um, field_2d, shading="auto", cmap="inferno")
    ax1.pcolormesh(y_um, z_um, n_yz, shading="auto", cmap="Greys", alpha=0.1)
    plt.colorbar(im, ax=ax1, label=f"|{dom_comp}| (arb. units)")
    ax1.set_xlabel("y (um)")
    ax1.set_ylabel("z (um)")
    ax1.set_title(f"|{dom_comp}|  n_eff={result.n_eff:.4f}")
    ax1.set_aspect("equal")

    fig.suptitle(
        f"TFLN ridge waveguide fundamental TE mode\n"
        f"(lambda={WAVELENGTH:.2f} um, w0={CORE_WIDTH:.1f} um, "
        f"h_slab={SLAB_THICKNESS * 1000:.0f} nm, h_ridge={RIDGE_THICKNESS * 1000:.0f} nm, "
        f"bg=SiO2)",
        fontweight="bold",
    )
    fig.tight_layout()

plt.show()
