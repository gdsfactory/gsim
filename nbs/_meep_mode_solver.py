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
# MEEP locally — no cloud job required — and returns the effective index,
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
# | `sim.solve_mode(port=..., wavelength)` | Simulation wrapper — delegates to one of the above |
#
# All solvers return a `ModeResult` with `.n_eff`, `.fields`, `.kdom`, `.n_group`, etc.

# %% [markdown]
# ## Part 1 — Quick Start

# %% [markdown]
# ### Setup and imports

# %%
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

from gsim.common.stack import get_stack
from gsim.meep import (
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
# ## Part 2 — Mode Results

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
nz = max(round(2.22 * 32), 1)  # span × resolution for default SOI stack
z_um = mode_z_grid(stack, n_points=nz)
n_profile = refractive_index_profile(stack, z_um, wavelength=1.55)

print(f"Z grid: {z_um[0]:.4f} … {z_um[-1]:.4f} um  ({len(z_um)} points)")
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
    f"Mode profile — {dom_comp}  (lambda={result_slab.wavelength:.2f} um, "
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
    ax.plot(
        z_slab, arr.real, alpha=0.35, linestyle="--", linewidth=0.8, label=f"Re({comp})"
    )
    ax.set_title(comp)
    ax.set_xlabel("z (um)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)

for idx in range(len(comps), n_rows * n_cols):
    axes[idx // n_cols][idx % n_cols].set_visible(False)

fig.suptitle(
    f"Field components — slab mode  lambda={result_slab.wavelength:.2f} um",
    fontweight="bold",
)
fig.tight_layout()


# %% [markdown]
# ## Part 3 — Advanced

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

ax1.set_title("Dispersion — fundamental TE mode")
fig.tight_layout()


# %% [markdown]
# ### Group velocity validation
#
# Compare MEEP's two-pass `n_group` (via `compute_group_index=True`)
# with the numerical derivative `n_g = n_eff - lambda · dn_eff/dlambda`.

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

# Numerical group index via central finite difference
ng_numeric = n_arr[1:-1] - wl_arr[1:-1] * np.gradient(n_arr, wl_arr)[1:-1]

fig, ax = plt.subplots()
ax.plot(wl_arr, n_arr, ".-", label="n_eff", linewidth=1)
ax.plot(wl_arr[1:-1], ng_numeric, ".-", label="n_group (numeric diff)", linewidth=1)
ax.set_xlabel("Wavelength (um)")
ax.set_ylabel("Index")
ax.set_title("Dispersion — fundamental TE mode (fine sweep)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()


# %% [markdown]
# ### Group velocity from MEEP two-pass solver
#
# Using `compute_group_index=True`, MEEP rebuilds the simulation with
# Bloch-periodic boundary conditions at the mode's propagation
# wavevector, enabling `mode.group_velocity`.

# %%
wl_meep = np.linspace(1.50, 1.60, 6)
n_group_from_meep: list[float | None] = []

for wl in wl_meep:
    r = solve_slab_mode(
        stack=stack,
        wavelength=wl,
        band_num=1,
        compute_group_index=True,
    )
    n_group_from_meep.append(r.n_group)

fig, ax = plt.subplots()
ax.plot(wl_arr[1:-1], ng_numeric, "o-", label="n_group (numeric diff)", markersize=4)
ax.plot(wl_meep, n_group_from_meep, "s--", label="n_group (MEEP vg)", markersize=6)
ax.set_xlabel("Wavelength (um)")
ax.set_ylabel("n_group")
ax.set_title("Group index validation — MEEP vg vs numeric differentiation")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()


# %% [markdown]
# ### Multi-band modes

# %%
max_bands = 4
for band in range(1, max_bands + 1):
    try:
        r = solve_slab_mode(
            stack=stack,
            wavelength=1.55,
            band_num=band,
            parity="NO_PARITY",
            resolution=32,
        )
        print(
            f"  band {band}: n_eff={r.n_eff:.6f}"
            f"  n_group={r.n_group}"
            f"  kdom={[f'{k:.4f}' for k in r.kdom[:2]]}…"
        )
    except RuntimeError as exc:
        print(f"  band {band}:  not found ({exc})")

# %% [markdown]
# ### Multi-band field profiles

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for band in range(1, 5):
    try:
        r = solve_slab_mode(
            stack=stack,
            wavelength=1.55,
            band_num=band,
            parity="NO_PARITY",
            resolution=32,
        )
        nz_r = len(next(iter(r.fields.values())))
        zz = mode_z_grid(stack, n_points=nz_r)
        if "Ey" in r.fields:
            ax.plot(
                zz,
                np.abs(r.fields["Ey"]),
                linewidth=1.3,
                label=f"TE{band - 1}  n={r.n_eff:.4f}",
            )
    except RuntimeError:
        pass

ax.set_xlabel("z (um)")
ax.set_ylabel("|Ey| (arb. units)")
ax.set_title("Multi-band TE slab modes")
ax.legend(fontsize="small", loc="upper right")
ax.grid(True, alpha=0.2)
fig.tight_layout()


# %% [markdown]
# ### Parity modes
#
# MEEP's eigenmode solver supports parity constraints along Y and Z.
# For a symmetric slab stack, `EVEN_Y` selects TE modes and `ODD_Y`
# selects TM modes.

# %%
parities = ["NO_PARITY", "EVEN_Y", "ODD_Y", "EVEN_Z", "ODD_Z"]
for parity in parities:
    try:
        r = solve_slab_mode(
            stack=stack,
            wavelength=1.55,
            band_num=1,
            parity=parity,
            resolution=32,
        )
        print(f"  {parity:10s}: n_eff={r.n_eff:.6f}")
    except RuntimeError as exc:
        print(f"  {parity:10s}: not found ({exc})")

# %% [markdown]
# ## Part 4 — Validation

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
from gsim.common.stack.extractor import Layer, LayerStack


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
# ## Part 5 — Parameter Studies

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
# | Index profile | `refractive_index_profile(stack, z_grid, wavelength)` |
# | Field profiles | `result.fields["Ey"]` — 1D complex array along *z* |
# | Dispersion sweep | Loop over wavelengths -> n_eff(lambda) |
# | Group index | `compute_group_index=True` for MEEP vg, or numeric diff |
# | Multi-band | `band_num=1, 2, 3, …` |
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
