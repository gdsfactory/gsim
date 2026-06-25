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
# # Standalone MEEP Eigenmode Solver
#
# The `gsim.meep` module exposes a standalone eigenmode solver for
# **1D slab modes** and **2D waveguide cross-section modes**. This runs
# MEEP locally - no cloud job required - and returns the effective index,
# field profiles, and wavevectors.
#
# **API overview:**
#
# | Function | What it does |
# |---|---|
# | `solve_slab_mode(stack, wavelength)` | 1D slab mode from a layer stack |
# | `solve_cross_section_mode(component, stack, port=..., wavelength)` | 2D waveguide cross-section at a port |
# | `mode_z_grid(stack, n_points)` | Z-axis coordinates for field arrays |
# | `refractive_index_profile(stack, z_grid, wavelength)` | n(z) from the layer stack |
# | `sim.solve_mode(port=..., wavelength)` | Simulation wrapper - delegates to one of the above |
#
# All solvers return a `ModeResult` with `.n_eff`, `.fields`, `.kdom`, `.n_group`, etc.

# %% [markdown]
# ## Part 1 - Quick Start

# %% [markdown]
# ### Setup and imports

# %%
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

from gsim.common.stack import get_stack
from gsim.meep import (
    mode_x_grid,
    mode_z_grid,
    refractive_index_profile,
    solve_cross_section_mode,
    solve_slab_mode,
)

HAS_MEEP = False
try:
    import meep as mp

    HAS_MEEP = True
    print(f"MEEP {mp.__version__} ready")
except ImportError as err:
    raise SystemExit(
        "MEEP not found. Install it via conda-forge:\n"
        "    conda install -c conda-forge pymeep"
    ) from err

gf.gpdk.PDK.activate()
stack = get_stack()

# %% [markdown]
# ### Slab mode in 3 lines

# %%
result = solve_slab_mode(
    stack=stack,
    wavelength=1.55,
    band_num=1,
    parity="NO_PARITY",
    resolution=32,
)

print(f"n_eff   = {result.n_eff:.6f}")
print(f"n_group = {result.n_group}")
print(f"kdom    = {[f'{k:.6f}' for k in result.kdom]}")
print(f"band    = {result.band_num}, parity = {result.parity}")

# %% [markdown]
# ### Cross-section mode in 3 lines

# %%
c = gf.components.straight(length=10, width=0.5)

result = solve_cross_section_mode(
    component=c,
    stack=stack,
    port="o1",
    wavelength=1.55,
    band_num=1,
    resolution=32,
)

print(f"n_eff   = {result.n_eff:.6f}")
print(f"n_group = {result.n_group}")
print(f"fields  = {list(result.fields.keys())}")


# %% [markdown]
# ## Part 2 - Mode Results

# %% [markdown]
# ### Understanding ModeResult
#
# `ModeResult` is a Pydantic model containing everything the solver
# extracted from MEEP:
#
# | Attribute | Type | Description |
# |---|---|---|
# | `n_eff` | `float` | Effective index |
# | `wavelength` | `float` | Free-space wavelength (um) |
# | `frequency` | `float` | Frequency in MEEP units (1/um) |
# | `fields` | `dict[str, np.ndarray]` | Complex field arrays keyed by component |
# | `kdom` | `list[float]` | Dominant wavevector [kx, ky, kz] |
# | `n_group` | `float \| None` | Group index (from `mode.group_velocity`) |
# | `band_num` | `int` | Mode band index (1 = fundamental) |
# | `parity` | `str` | Parity constraint used |

# %%
for comp, arr in result.fields.items():
    print(f"{comp}: shape={arr.shape}, |max|={np.abs(arr).max():.6f}")

# %% [markdown]
# ### Z-grid and refractive index profile
#
# Use the library functions `mode_z_grid()` and
# `refractive_index_profile()` to reconstruct the coordinate system and
# material profile. These replace the notebook's previous hardcoded
# `eps_map`.

# %%
nz = max(round(2.22 * 32), 1)  # span x resolution for default SOI stack
z_um = mode_z_grid(stack, n_points=nz)
n_profile = refractive_index_profile(stack, z_um, wavelength=1.55)

print(f"Z grid: {z_um[0]:.4f} ... {z_um[-1]:.4f} um  ({len(z_um)} points)")
print(f"Index range: {n_profile.min():.4f} - {n_profile.max():.4f}")

# %% [markdown]
# ### Mode profile with index overlay
#
# Plot the dominant electric field component magnitude alongside the
# refractive-index profile. The slab fundamental TE mode has its primary
# field in `Ey`.

# %%
result_slab = solve_slab_mode(stack=stack, wavelength=1.55, resolution=32)
nz = len(next(iter(result_slab.fields.values())))
z_slab = mode_z_grid(stack, n_points=nz)
n_prof = refractive_index_profile(stack, z_slab, wavelength=1.55)

dom_comp = max(result_slab.fields, key=lambda k: np.abs(result_slab.fields[k]).max())

fig, ax = plt.subplots(figsize=(7, 4))
color_primary = "C0"

field = result_slab.fields[dom_comp]
ax.plot(
    z_slab, np.abs(field), color=color_primary, linewidth=1.5, label=f"|{dom_comp}|"
)
ax.plot(
    z_slab,
    field.real,
    color=color_primary,
    alpha=0.35,
    linestyle="--",
    linewidth=1,
    label=f"Re({dom_comp})",
)
ax.plot(
    z_slab,
    field.imag,
    color=color_primary,
    alpha=0.35,
    linestyle=":",
    linewidth=1,
    label=f"Im({dom_comp})",
)

ax.set_xlabel("z (um)")
ax.set_ylabel("Field amplitude (arb. units)")

ax_idx = ax.twinx()
ax_idx.plot(z_slab, n_prof, color="C3", linewidth=1.5, label="n(z)")
ax_idx.set_ylabel("Refractive index")
ax_idx.set_ylim(bottom=0.8)

ax.set_title(
    f"Mode profile - {dom_comp}  (lambda={result_slab.wavelength:.2f} um, "
    f"n_eff={result_slab.n_eff:.4f})"
)
ax.grid(True, alpha=0.2)
fig.tight_layout()


# %% [markdown]
# ### Field component grid
#
# All six E- and H-field components in a subplot grid.

# %%
comps = sorted(
    result_slab.fields.keys(),
    key=lambda c: ("ExEyEzHxHyHz".index(c) if c in "ExEyEzHxHyHz" else 99),
)
n_cols = min(3, len(comps))
n_rows = (len(comps) + n_cols - 1) // n_cols

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
)
for idx, comp in enumerate(comps):
    ax = axes[idx // n_cols][idx % n_cols]
    arr = result_slab.fields[comp]
    ax.plot(z_slab, np.abs(arr), linewidth=1.2, label=f"|{comp}|")
    ax.set_title(comp)
    ax.set_xlabel("z (um)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)

for idx in range(len(comps), n_rows * n_cols):
    axes[idx // n_cols][idx % n_cols].set_visible(False)

fig.suptitle(
    f"Field components - slab mode  lambda={result_slab.wavelength:.2f} um",
    fontweight="bold",
)
fig.tight_layout()


# %% [markdown]
# ### Cross-section mode — 2D (X, Z) field maps
#
# :func:`solve_cross_section_mode` returns **full 2D arrays** in
# ``ModeResult.fields`` with shape ``(nz, nx)`` — the Z-axis (rows) and
# X-axis (columns) span the cell cross-section at the Y-cut plane.
#
# Use :func:`mode_z_grid` for the Z-axis and :func:`mode_x_grid` for the
# X-axis to reconstruct the coordinate system.

# %%
c = gf.components.straight(length=10, width=0.5)

result_xz = solve_cross_section_mode(
    component=c,
    stack=stack,
    port="o1",
    wavelength=1.55,
    band_num=1,
    resolution=32,
)

print(f"n_eff = {result_xz.n_eff:.6f}")
for comp, arr in result_xz.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

nz, nx = next(iter(result_xz.fields.values())).shape
x_um = mode_x_grid(n_points=nx, x_span=2.5)
z_um = mode_z_grid(stack, n_points=nz)

print(f"\nX grid: {x_um[0]:.3f} … {x_um[-1]:.3f} µm  ({nx} points)")
print(f"Z grid: {z_um[0]:.3f} … {z_um[-1]:.3f} µm  ({nz} points)")

# %% [markdown]
# ### Dominant field component (|E|)
#
# The fundamental TE-like mode has its primary electric field in Ey.
# The 2D colour map shows the mode confined to the waveguide core region.

# %%
dom_comp = max(result_xz.fields, key=lambda k: np.abs(result_xz.fields[k]).max())
field_2d = np.abs(result_xz.fields[dom_comp])

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.pcolormesh(x_um, z_um, field_2d, shading="auto", cmap="inferno")
cbar = plt.colorbar(im, ax=ax, label=f"|{dom_comp}| (arb. units)")
ax.set_xlabel("x (µm)")
ax.set_ylabel("z (µm)")
ax.set_title(
    f"Cross-section mode profile |{dom_comp}|  "
    f"lambda={result_xz.wavelength:.2f} um, n_eff={result_xz.n_eff:.4f}"
)
ax.set_aspect("equal")
fig.tight_layout()

# %% [markdown]
# ### Refractive index overlay
#
# Overlay the layer-stack boundaries on top of the field map to see
# how the mode is shaped by the material distribution.  The X-extent is
# determined by the GDS geometry at the cut plane — inside the waveguide
# core the Si layer is present; outside it the SiO2 cladding fills the
# cell.

# %%
n_prof_2d = refractive_index_profile(stack, z_um, wavelength=1.55)
n_xz = np.tile(n_prof_2d[:, np.newaxis], (1, nx))

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.pcolormesh(x_um, z_um, field_2d, shading="auto", cmap="inferno", alpha=0.85)
cbar = plt.colorbar(im, ax=ax, label=f"|{dom_comp}| (arb. units)")
ct = ax.contour(x_um, z_um, n_xz, levels=[1.4, 2.0, 3.0], colors="cyan", linewidths=0.8)
ax.clabel(ct, fmt="n=%.1f", fontsize=7)
ax.set_xlabel("x (µm)")
ax.set_ylabel("z (µm)")
ax.set_title(
    f"|{dom_comp}| with refractive-index contours  (n_eff={result_xz.n_eff:.4f})"
)
ax.set_aspect("equal")
fig.tight_layout()

# %% [markdown]
# ### All field components (2D grid)
#
# Each E- and H-component as a 2D colour map.  For a straight waveguide
# propagating along X, the transverse components (Ey, Ez, Hy, Hz)
# dominate; the longitudinal components (Ex, Hx) are typically small.

# %%
comps = sorted(
    result_xz.fields.keys(),
    key=lambda c: ("ExEyEzHxHyHz".index(c) if c in "ExEyEzHxHyHz" else 99),
)
n_cols = 3
n_rows = (len(comps) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), squeeze=False)
for idx, comp in enumerate(comps):
    ax = axes[idx // n_cols][idx % n_cols]
    a2 = np.abs(result_xz.fields[comp])
    im = ax.pcolormesh(x_um, z_um, a2, shading="auto", cmap="inferno")
    plt.colorbar(im, ax=ax, label=f"|{comp}|")
    ax.set_title(comp)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_aspect("equal")

for idx in range(len(comps), n_rows * n_cols):
    axes[idx // n_cols][idx % n_cols].set_visible(False)

fig.suptitle(
    f"All field components — cross-section mode  "
    f"lambda={result_xz.wavelength:.2f} um, n_eff={result_xz.n_eff:.4f}",
    fontweight="bold",
)
fig.tight_layout()


# %% [markdown]
# ## Part 3 - Advanced

# %% [markdown]
# ### Simulation wrapper
#
# `Simulation.solve_mode()` resolves the stack and materials internally,
# then delegates to the standalone functions.

# %%
from gsim import meep as meep_mod

sim = meep_mod.Simulation()
sim.geometry.component = c
sim.geometry.stack = stack

result = sim.solve_mode(port="o1", wavelength=1.55)
print(f"n_eff = {result.n_eff:.6f}")

# %% [markdown]
# ### Wavelength sweep & dispersion curve

# %%
wavelengths = [1.50, 1.52, 1.54, 1.55, 1.56, 1.58, 1.60]
n_effs: list[float] = []
n_groups: list[float | None] = []

for wl in wavelengths:
    r = sim.solve_mode(port="o1", wavelength=wl, band_num=1)
    n_effs.append(r.n_eff)
    n_groups.append(r.n_group)
    print(f"  lambda = {wl:.2f} um -> n_eff = {r.n_eff:.6f}, n_group = {r.n_group}")

fig, ax1 = plt.subplots()
(line1,) = ax1.plot(wavelengths, n_effs, "o-", label="n_eff")
ax1.set_xlabel("Wavelength (um)")
ax1.set_ylabel("n_eff")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
group_vals = [g for g in n_groups if g is not None]
if group_vals:
    (line2,) = ax2.plot(wavelengths, n_groups, "s--", color="C1", label="n_group")
    ax2.set_ylabel("n_group")
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
else:
    ax1.legend(loc="upper right")

ax1.set_title("Dispersion - fundamental TE mode")
fig.tight_layout()


# %% [markdown]
# ### Group velocity validation
#
# MEEP's ``get_eigenmode`` returns ``mode.group_velocity`` directly,
# giving the group index as ``n_group = 1/v_g``.  A fine wavelength
# sweep shows the dispersion and group index together.

# %%
wl_fine = [1.45 + i * 0.005 for i in range(61)]  # 1.45-1.75 um
n_eff_fine: list[float] = []
n_group_meep: list[float | None] = []

for wl in wl_fine:
    r = sim.solve_mode(
        port="o1",
        wavelength=wl,
        band_num=1,
    )
    n_eff_fine.append(r.n_eff)
    n_group_meep.append(r.n_group)

wl_arr = np.array(wl_fine)
n_arr = np.array(n_eff_fine)

fig, ax1 = plt.subplots()
(line1,) = ax1.plot(wl_arr, n_arr, ".-", label="n_eff", linewidth=1)
ax1.set_xlabel("Wavelength (um)")
ax1.set_ylabel("n_eff")
ax1.grid(True, alpha=0.3)

meep_valid = [(wl, ng) for wl, ng in zip(wl_fine, n_group_meep) if ng is not None]
if meep_valid:
    wl_v, ng_v = zip(*meep_valid)
    ax2 = ax1.twinx()
    (line2,) = ax2.plot(
        wl_v, ng_v, "s-", color="C1", label="n_group (MEEP vg)", markersize=3
    )
    ax2.set_ylabel("n_group")
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
else:
    ax1.legend(loc="upper right")

ax1.set_title("Dispersion - fundamental TE mode (fine sweep)")
fig.tight_layout()

# %% [markdown]
# ### Multi-band modes
#
# The symmetric SOI slab (Si 220 nm core, SiO2 cladding) at lambda=1.55 um
# has V-parameter approx 1.41 - it supports a single TE mode (TE0).
# Bands beyond 1 are **leaky / radiation modes**: MEEP's MPB omega-solve
# can converge to them because the finite PML-bounded cell discretises
# the radiation continuum.  The library logs a warning for modes with
# ``n_eff`` below the minimum cladding index.

# %%
from gsim.common.stack.extractor import Layer, LayerStack


def _soi_slab(t_si: float) -> LayerStack:
    return LayerStack(
        layers={
            "box": Layer(
                name="box",
                gds_layer=(0, 0),
                zmin=-2.0,
                zmax=0.0,
                thickness=2.0,
                material="sio2",
                layer_type="dielectric",
            ),
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=t_si,
                thickness=t_si,
                material="si",
                layer_type="dielectric",
            ),
            "clad": Layer(
                name="clad",
                gds_layer=(2, 0),
                zmin=t_si,
                zmax=t_si + 2.0,
                thickness=2.0,
                material="sio2",
                layer_type="dielectric",
            ),
        }
    )


soi = _soi_slab(0.22)
n_si = 3.4777
n_sio2 = 1.4440
V_param = (2 * np.pi / 1.55) * 0.22 / 2 * np.sqrt(n_si**2 - n_sio2**2)
print(
    f"V-parameter = {V_param:.3f}  ->  {max(1, int(2 * V_param / np.pi) + 1)} guided TE mode(s)"
)

for band in range(1, 5):
    try:
        r = solve_slab_mode(
            stack=soi,
            wavelength=1.55,
            band_num=band,
            parity="NO_PARITY",
            resolution=64,
        )
        tag = "GUIDED" if r.n_eff > n_sio2 else "LEAKY"
        print(
            f"  band {band}: n_eff={r.n_eff:.6f}  ({tag})"
            f"  n_group={r.n_group}"
            f"  kdom={[f'{k:.4f}' for k in r.kdom[:2]]}..."
        )
    except RuntimeError as exc:
        print(f"  band {band}:  not found ({exc})")

# %% [markdown]
# ### Multi-band field profiles
#
# Plotting |Ey| for each available band on the SOI slab. Only TE0
# (band 1) is physically guided; higher bands may be MPB artefacts.

# %%
for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for band in range(1, 5):
        try:
            r = solve_slab_mode(
                stack=soi,
                wavelength=1.55,
                band_num=band,
                parity="NO_PARITY",
                resolution=64,
            )
            nz_r = len(next(iter(r.fields.values())))
            zz = mode_z_grid(soi, n_points=nz_r)
            if comp in r.fields:
                tag = "guided" if r.n_eff > n_sio2 else "leaky"
                ax.plot(
                    zz,
                    np.abs(r.fields[comp]),
                    linewidth=1.3,
                    label=f"band {band} ({tag}) n={r.n_eff:.4f}",
                )
        except RuntimeError:
            pass

    ax.set_xlabel("z (um)")
    ax.set_ylabel(f"|{comp}| (arb. units)")
    ax.set_title("Multi-band slab modes - SOI 220 nm")
    ax.legend(fontsize="small", loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

# %% [markdown]
# ### Parity modes
#
# MEEP's eigenmode solver supports parity constraints along Y and Z.
# For the symmetric SOI slab stack, ``EVEN_Y`` selects the fundamental
# TE0 mode.  ``ODD_Y`` selects the first odd-symmetry mode --
# this may be a leaky/radiation mode if the waveguide is single-mode
# (V < pi/2 approx 1.57).

# %%
parities = ["NO_PARITY", "EVEN_Y", "ODD_Y"]
for parity in parities:
    try:
        r = solve_slab_mode(
            stack=soi,
            wavelength=1.55,
            band_num=1,
            parity=parity,
            resolution=64,
        )
        tag = "guided" if r.n_eff > n_sio2 else "leaky"
        print(f"  {parity:10s}: n_eff={r.n_eff:.6f}  ({tag})")
    except RuntimeError as exc:
        print(f"  {parity:10s}: not found ({exc})")

# %% [markdown]
# ## Part 4 - Validation

# %% [markdown]
# ### Analytical SOI slab benchmark
#
# Validate the MEEP solver against the analytical transcendental equation
# for a symmetric dielectric slab waveguide.
#
# **Structure:** Si core (n=3.4777 @ 1.55 um) with thickness *t* = 220 nm,
# SiO2 cladding (n=1.4440 @ 1.55 um).
#
# **Transcendental equation for TE modes:**
#
# Even modes:  $$\tan(\kappa t/2) = \frac{\gamma}{\kappa}$$
# Odd modes:   $$\tan(\kappa t/2) = -\frac{\kappa}{\gamma}$$
#
# where $$\kappa = \sqrt{n_\mathrm{core}^2 k_0^2 - \beta^2}, \quad
# \gamma = \sqrt{\beta^2 - n_\mathrm{clad}^2 k_0^2}, \quad k_0 = 2\pi/\lambda$$

# %% [markdown]
# ### Analytical solver (inline)


# %%
def solve_slab_analytical(
    wavelength: float,
    n_core: float,
    n_clad: float,
    t_core: float,
    polarization: str = "TE",
) -> dict[int, float]:
    """Solve the symmetric slab transcendental equation.

    Args:
        wavelength: Free-space wavelength (um).
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        t_core: Core thickness (um).
        polarization: ``"TE"`` or ``"TM"``.

    Returns:
        Dict mapping mode number (0=fundamental) to n_eff.
    """
    k0 = 2.0 * np.pi / wavelength

    def _te_even(beta: float) -> float:
        kappa = np.sqrt(max(n_core**2 * k0**2 - beta**2, 0))
        gamma = np.sqrt(max(beta**2 - n_clad**2 * k0**2, 0))
        if kappa == 0:
            return -1e6
        return np.tan(kappa * t_core / 2) - gamma / kappa

    def _te_odd(beta: float) -> float:
        kappa = np.sqrt(max(n_core**2 * k0**2 - beta**2, 0))
        gamma = np.sqrt(max(beta**2 - n_clad**2 * k0**2, 0))
        if gamma == 0:
            return -1e6
        return np.tan(kappa * t_core / 2) + kappa / gamma

    if polarization == "TE":
        fns = [_te_even, _te_odd]
    else:
        # TM: scale by (n_core/n_clad)^2 at the interface
        def _tm_even(beta: float) -> float:
            kappa = np.sqrt(max(n_core**2 * k0**2 - beta**2, 0))
            gamma = np.sqrt(max(beta**2 - n_clad**2 * k0**2, 0))
            if kappa == 0:
                return -1e6
            return np.tan(kappa * t_core / 2) - gamma / kappa * (n_core / n_clad) ** 2

        def _tm_odd(beta: float) -> float:
            kappa = np.sqrt(max(n_core**2 * k0**2 - beta**2, 0))
            gamma = np.sqrt(max(beta**2 - n_clad**2 * k0**2, 0))
            if gamma == 0:
                return -1e6
            return np.tan(kappa * t_core / 2) + kappa / gamma * (n_clad / n_core) ** 2

        fns = [_tm_even, _tm_odd]

    beta_min = n_clad * k0
    beta_max = n_core * k0

    results: dict[int, float] = {}
    mode_idx = 0
    for fn in fns:
        # Scan beta range for sign changes
        betas = np.linspace(beta_min + 1e-4, beta_max - 1e-4, 10000)
        vals = np.array([fn(b) for b in betas])
        sign_changes = np.where(np.diff(np.signbit(vals)))[0]
        for si in sign_changes:
            b_lo = betas[si]
            b_hi = betas[min(si + 1, len(betas) - 1)]
            try:
                from scipy.optimize import bisect

                beta_root = bisect(fn, b_lo, b_hi, xtol=1e-12)
                n_eff = beta_root / k0

                # Reject spurious "roots" caused by tan(kappa*t/2) -> +/-inf
                # (pole crossing, not a genuine zero crossing)
                kappa = np.sqrt(max(n_core**2 * k0**2 - beta_root**2, 0))
                gamma = np.sqrt(max(beta_root**2 - n_clad**2 * k0**2, 0))
                if gamma <= 0 or kappa <= 0:
                    continue
                # Reject spurious "roots" where tan(kappa*t/2) -> +/-inf (pole crossing).
                # A genuine root gives fn(root) ~= 0; a pole crossing gives
                # |fn(root)| >> 0 because the sign change occurs through +/-inf,
                # not through zero.
                kappa = np.sqrt(max(n_core**2 * k0**2 - beta_root**2, 0))
                gamma = np.sqrt(max(beta_root**2 - n_clad**2 * k0**2, 0))
                if gamma <= 0 or kappa <= 0:
                    continue
                kt2 = kappa * t_core / 2.0
                # tan poles at odd multiples of pi/2: |tan| -> inf
                if abs(np.tan(kt2)) > 1e4:
                    continue

                if n_eff not in results.values():
                    results[mode_idx] = n_eff
                    mode_idx += 1
            except Exception:
                pass

    return results


# %%
# Analytical benchmark parameters
lambda_bench = 1.55
n_si = 3.4777
n_sio2 = 1.4440
t_si = 0.22

analytical = solve_slab_analytical(
    wavelength=lambda_bench,
    n_core=n_si,
    n_clad=n_sio2,
    t_core=t_si,
    polarization="TE",
)
print("Analytical TE slab modes (Si 220nm / SiO2):")
for mode, n_eff in sorted(analytical.items()):
    print(f"  TE{mode}: n_eff = {n_eff:.6f}")

# %% [markdown]
# ### Compare MEEP with analytical

# %%
from gsim.common.stack.extractor import LayerStack


def _make_soi_stack(t_si: float) -> LayerStack:
    """Build a symmetric SOI slab stack: SiO2 / Si / SiO2."""
    layers = {
        "box": Layer(
            name="box",
            gds_layer=(0, 0),
            zmin=-2.0,
            zmax=0.0,
            thickness=2.0,
            material="sio2",
            layer_type="dielectric",
        ),
        "core": Layer(
            name="core",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=t_si,
            thickness=t_si,
            material="si",
            layer_type="dielectric",
        ),
        "clad": Layer(
            name="clad",
            gds_layer=(2, 0),
            zmin=t_si,
            zmax=t_si + 2.0,
            thickness=2.0,
            material="sio2",
            layer_type="dielectric",
        ),
    }
    return LayerStack(layers=layers)


soi_stack = _make_soi_stack(t_si)

# MEEP fundamental TE0
r_meep = solve_slab_mode(
    stack=soi_stack,
    wavelength=lambda_bench,
    band_num=1,
    parity="EVEN_Y",
    resolution=64,
)

if 0 in analytical:
    n_analytical = analytical[0]
    rel_error = abs(r_meep.n_eff - n_analytical) / n_analytical
    print(f"MEEP  TE0: n_eff = {r_meep.n_eff:.6f}")
    print(f"Anal. TE0: n_eff = {n_analytical:.6f}")
    print(f"Relative error: {rel_error:.2e}")
else:
    print("No analytical TE0 mode found.")

# Compare higher-order modes if available
for band in range(2, 5):
    try:
        r_meep_b = solve_slab_mode(
            stack=soi_stack,
            wavelength=lambda_bench,
            band_num=band,
            parity="NO_PARITY",
            resolution=64,
        )
        print(f"\nMEEP  band {band}: n_eff = {r_meep_b.n_eff:.6f}")
    except RuntimeError:
        pass


# %% [markdown]
# ## Part 5 - Parameter Studies

# %% [markdown]
# ### Core thickness sweep
#
# Vary the silicon core thickness and re-solve the slab mode. Thicker
# cores -> higher confinement -> larger n_eff.

# %%
thicknesses = [0.15, 0.18, 0.22, 0.25, 0.30, 0.35, 0.40]
n_eff_thick: list[float] = []

for t_si in thicknesses:
    wg_stack = _make_soi_stack(t_si)
    r = solve_slab_mode(
        stack=wg_stack,
        wavelength=1.55,
        band_num=1,
        parity="NO_PARITY",
        resolution=32,
    )
    n_eff_thick.append(r.n_eff)
    print(f"  t_Si = {t_si:.2f} um -> n_eff = {r.n_eff:.6f}")

fig, ax = plt.subplots()
ax.plot(thicknesses, n_eff_thick, "s-", linewidth=1.5)
ax.set_xlabel("Si core thickness (um)")
ax.set_ylabel("n_eff")
ax.set_title("n_eff vs core thickness  (lambda = 1.55 um, slab mode)")
ax.grid(True, alpha=0.3)
fig.tight_layout()


# %% [markdown]
# ### Summary
#
# | Feature | API |
# |---|---|
# | Slab modes (1D) | `solve_slab_mode(stack, wavelength=...)` |
# | Cross-section modes (2D) | `solve_cross_section_mode(component, stack, port=..., wavelength=...)` |
# | Simulation wrapper | `sim.solve_mode(port=..., wavelength=...)` |
# | Z-grid utility | `mode_z_grid(stack, n_points)` |
# | X-grid utility | `mode_x_grid(n_points, x_span)` |
# | Index profile | `refractive_index_profile(stack, z_grid, wavelength)` |
# | Field profiles (slab) | `result.fields["Ey"]` — 1D complex array along *z* |
# | Field profiles (cross-section) | `result.fields["Ey"]` — 2D complex array ``(nz, nx)`` |
# | Dispersion sweep | Loop over wavelengths -> n_eff(lambda) |
# | Group index | `result.n_group` — direct from `mode.group_velocity` |
# | Multi-band | `band_num=1, 2, 3, ...` |
# | Parity modes | `parity="EVEN_Y"` etc. |
# | Core thickness sweep | Vary `layer.thickness` on a copy of the stack -> n_eff(t) |
# | Analytical validation | Transcendental equation benchmark for SOI slab |
#
# **Next steps:**
#
# - Use the slab effective index as input to the variational 2D
#   approximation (issue #156)
# - Compare mode-solver n_eff with full FDTD frequency-domain results
# - Run the same sweeps with `palace` (finite-element) for cross-validation
