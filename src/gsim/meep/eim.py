"""Variational effective-index method (varEIM) for 2D FDTD.

Computes a spatially-varying effective permittivity from a 3D layer stack,
weighted by the vertical slab-mode intensity, so that a 2D (z-collapsed)
simulation reproduces the vertical confinement that bulk-index substitution
ignores.

The effective permittivity at a lateral point ``(x, y)`` relative to a fixed
reference point ``r`` is (Hammer & Ivanova 2009):

    eps_eff(x, y) = n_eff^2(r)
                    + integral[ (eps(x,y,z) - eps(r,z)) * |Phi_r(z)|^2 dz ]
                      / integral[ |Phi_r(z)|^2 dz ]

where ``n_eff(r)`` and ``Phi_r(z)`` are the effective index and field profile of
the fundamental TE vertical slab mode at the reference point. In the core region
the perturbation vanishes (``eps_eff = n_eff^2``); in the cladding it is
``n_eff^2`` reduced by a mode-weighted average, giving the physically correct
(reduced) lateral index contrast.

This implementation is TE-only and uses a single reference mode, so it is valid
only where the vertical mode profile is consistent across the device (strips,
rings, MMIs). It breaks down for mode-converting transitions (e.g.
strip-to-slot).

Reference:
    H. J. W. M. Hammer and O. V. Ivanova, "Effective index approximations of
    photonic crystal slabs: a 2-to-1-D assessment," Opt. Quant. Electron. 41,
    267 (2009).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from gsim.common.cross_section import _layer_shapely_polys
from gsim.common.stack.materials import resolve_material_at_wavelength

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def build_eps_z(
    component: gf.Component,
    layer_stack: LayerStack,
    x: float,
    y: float,
    wavelength_um: float,
    *,
    nz: int = 400,
    z_range: tuple[float, float] | None = None,
    background_material: str = "air",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 1D vertical permittivity profile eps(z) at lateral point (x, y).

    Samples the layer stack along z at a fixed ``(x, y)``: dielectrics provide
    the background (cladding/box/substrate), patterned layers override the
    background wherever their polygons cover the point. Where two patterned
    layers overlap in z (e.g. core and slab), the higher-permittivity material
    wins, matching gsim's "highest-index = core" convention.

    Args:
        component: gdsfactory Component (may contain references).
        layer_stack: LayerStack describing layers and dielectrics.
        x: Lateral X coordinate of the vertical cut (um).
        y: Lateral Y coordinate of the vertical cut (um).
        wavelength_um: Wavelength for material index lookup (um).
        nz: Number of z samples (cell centers).
        z_range: (zmin, zmax) for the cut. Defaults to ``layer_stack.get_z_range()``.
        background_material: Material name for z samples not covered by any
            dielectric or layer (typically the top air region).

    Returns:
        (z_grid, eps_z): the z sample coordinates (um) and permittivity at each.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    # get_polygons(merge=True) mutates, which is disabled on locked cached
    # cells; work on an unlocked copy.
    if getattr(component, "locked", False):
        component = component.copy()

    z_lo, z_hi = z_range if z_range is not None else layer_stack.get_z_range()
    dz = (z_hi - z_lo) / nz
    z_grid = z_lo + (np.arange(nz) + 0.5) * dz

    eps_cache: dict[str, float] = {}

    def eps_of(material: str) -> float:
        if material not in eps_cache:
            resolved = resolve_material_at_wavelength(material, wavelength_um)
            eps = None if resolved is None else resolved.permittivity_scalar
            eps_cache[material] = float(eps) if eps is not None else 1.0
        return eps_cache[material]

    # Background pass: fill from dielectrics, fall back to background_material.
    eps_z = np.full(nz, eps_of(background_material), dtype=float)
    for d in sorted(layer_stack.dielectrics, key=lambda d: d["zmin"]):
        band = (z_grid >= d["zmin"]) & (z_grid < d["zmax"])
        eps_z[band] = eps_of(d["material"])

    # Coverage pass: patterned layers override the background where they cover
    # (x, y); on overlapping z bands the higher-permittivity layer wins.
    pt = Point(x, y)
    dbu = getattr(getattr(component, "kcl", None), "dbu", 0.001)
    for layer in layer_stack.layers.values():
        gds_layer = getattr(layer, "gds_layer", None)
        if gds_layer is None:
            continue
        gds_layer_tuple = (int(gds_layer[0]), int(gds_layer[1]))
        polys = _layer_shapely_polys(component, gds_layer_tuple, dbu)
        if not polys:
            continue
        if not unary_union(polys).covers(pt):
            continue
        eps_layer = eps_of(layer.material)
        band = (z_grid >= layer.zmin) & (z_grid < layer.zmax)
        eps_z[band] = np.maximum(eps_z[band], eps_layer)

    return z_grid, eps_z


def solve_slab_mode(
    z_grid: np.ndarray,
    eps_z: np.ndarray,
    wavelength_um: float,
) -> tuple[float, np.ndarray]:
    """Solve the fundamental TE vertical slab mode of a 1D permittivity profile.

    Solves the 1D Helmholtz eigenproblem on a uniform z grid::

        d^2 Phi / dz^2 + k0^2 eps(z) Phi = k0^2 n_eff^2 Phi

    with Dirichlet boundaries (Phi -> 0 at the domain edges). The largest
    eigenvalue is ``n_eff^2`` and its eigenvector is the field profile ``Phi``.

    Args:
        z_grid: Uniform z sample coordinates (um).
        eps_z: Permittivity at each z sample.
        wavelength_um: Wavelength (um).

    Returns:
        (n_eff, Phi): effective index and the (L2-normalized) field profile,
        sampled on ``z_grid``.
    """
    from scipy.linalg import eigh_tridiagonal

    dz = float(z_grid[1] - z_grid[0])
    k0 = 2.0 * np.pi / wavelength_um

    # A = (1/k0^2) D2 + diag(eps), with D2 the standard 2nd-derivative stencil.
    off = (1.0 / k0**2) * (1.0 / dz**2) * np.ones(len(z_grid) - 1)
    diag = (1.0 / k0**2) * (-2.0 / dz**2) + eps_z

    # Fundamental mode = largest eigenvalue (= n_eff^2).
    n = len(z_grid)
    eigvals, eigvecs = eigh_tridiagonal(
        diag, off, select="i", select_range=(n - 1, n - 1)
    )
    n_eff = float(np.sqrt(eigvals[-1]))

    phi = eigvecs[:, -1]
    phi = phi / np.sqrt(np.trapezoid(phi**2, z_grid))
    return n_eff, phi


def _cladding_eps(layer_stack, z_value: float, wavelength_um: float) -> float:
    """Permittivity of the background dielectric covering ``z_value``.

    This is the physical cladding the guiding layer is embedded in (e.g. the
    oxide around an SOI strip), used as the floor for the variational result.
    """
    for d in layer_stack.dielectrics:
        if d["zmin"] <= z_value < d["zmax"]:
            resolved = resolve_material_at_wavelength(d["material"], wavelength_um)
            if resolved is not None and resolved.permittivity_scalar is not None:
                return float(resolved.permittivity_scalar)
    return 1.0


def variational_effective_permittivity(
    xy: tuple[float, float],
    reference_xy: tuple[float, float],
    sim,
    wavelength: float = 1.55,
    *,
    nz: int = 400,
    eps_floor: float | None = None,
) -> float:
    """Variational effective permittivity at ``xy`` relative to ``reference_xy``.

    Implements the Hammer & Ivanova formula by solving the vertical slab mode at
    ``reference_xy`` and weighting the local permittivity difference by the mode
    intensity. Geometry (component + layer stack) is taken from ``sim``.

    Args:
        xy: (x, y) location where the effective permittivity is evaluated.
        reference_xy: (x, y) reference point for the slab mode. Must lie in a
            single-mode guiding region; the same reference is used for every
            ``xy`` to keep the perturbation consistent.
        sim: A ``gsim.meep.Simulation`` carrying ``sim.geometry.component`` and
            ``sim.geometry.stack``.
        wavelength: Wavelength in um.
        nz: Number of z samples for the profiles and slab solve.
        eps_floor: Lower floor on the returned permittivity. For points whose
            vertical column cannot support the reference mode (e.g. pure
            cladding under a strongly confined strip mode) the raw variational
            value can drop near zero or negative. ``None`` (default) clamps to
            the physical cladding permittivity at the guiding plane (e.g. oxide)
            so evanescent behaviour in gaps stays physical; pass a float to
            override.

    Returns:
        Effective permittivity at ``xy`` (square it back for index:
        ``n = sqrt(eps)``).
    """
    component = sim.geometry.component
    layer_stack = sim.geometry.stack

    z_grid, eps_reference = build_eps_z(
        component, layer_stack, reference_xy[0], reference_xy[1], wavelength, nz=nz
    )
    n_eff, phi = solve_slab_mode(z_grid, eps_reference, wavelength)

    z_grid, eps_local = build_eps_z(
        component, layer_stack, xy[0], xy[1], wavelength, nz=nz
    )

    intensity = np.abs(phi) ** 2
    weighted_shift = np.trapezoid((eps_local - eps_reference) * intensity, z_grid)
    mode_power = np.trapezoid(intensity, z_grid)
    eps = n_eff**2 + weighted_shift / mode_power

    if eps_floor is None:
        z_guiding = float(z_grid[np.argmax(intensity)])
        eps_floor = _cladding_eps(layer_stack, z_guiding, wavelength)
    return max(float(eps), eps_floor)


def fit_medium(
    eps_eff_values: float | list[float],
    wavelengths: float | list[float],
) -> float:
    """Convert effective permittivity to a constant effective index (stub).

    Single-wavelength, zero-dispersion placeholder: returns ``sqrt(eps_eff)`` so
    the value can be dropped straight into ``sim.materials``. The dispersive fit
    (eps_eff(lambda) -> Sellmeier -> Lorentzian poles via
    ``gsim.meep.materials.sellmeier_to_lorentzian_poles``) replaces this later.

    Args:
        eps_eff_values: One effective permittivity, or a list (only length 1 is
            supported until dispersive fitting lands).
        wavelengths: The matching wavelength(s) in um.

    Returns:
        Constant effective refractive index ``n = sqrt(eps_eff)``.
    """
    # TODO(#156): fit eps_eff(lambda) to a dispersive MEEP-compatible model.
    eps_list = (
        [eps_eff_values]
        if isinstance(eps_eff_values, (int, float))
        else list(eps_eff_values)
    )
    wl_list = (
        [wavelengths] if isinstance(wavelengths, (int, float)) else list(wavelengths)
    )
    if len(eps_list) != 1 or len(wl_list) != 1:
        raise NotImplementedError(
            "Dispersive fitting is not implemented yet; pass a single "
            "(eps_eff, wavelength) pair (zero-dispersion stub)."
        )
    return float(np.sqrt(eps_list[0]))
