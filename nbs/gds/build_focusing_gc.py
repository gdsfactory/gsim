"""GDSFactory PCell for a focusing apodized grating coupler.

Adapted from FDTD-Bench problem 07 and the Tidy3D FocusedApodGC example:
https://github.com/doplaydo/FDTD-Bench/tree/main/problems/07_focusing_grating_coupler
https://www.flexcompute.com/tidy3d/examples/notebooks/FocusedApodGC/
"""

from __future__ import annotations

import math

import gdsfactory as gf
import numpy as np
from gdsfactory import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec


def _grating_parameters(
    *,
    periods: int,
    minimum_feature_um: float,
    fiber_tilt_deg: float,
    cladding_index: float,
    wavelength_um: float,
    apodization_per_um: float,
    full_thickness_effective_index: float,
    partial_thickness_effective_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the period and fill fraction of every grating tooth."""
    grating_angle_rad = math.asin(
        math.sin(math.radians(fiber_tilt_deg)) / cladding_index
    )
    first_period_um = wavelength_um / (
        full_thickness_effective_index - cladding_index * math.sin(grating_angle_rad)
    )
    first_fill_fraction = (first_period_um - minimum_feature_um) / first_period_um

    periods_um = []
    fill_fractions = []
    distance_um = 0.0
    for _ in range(periods):
        fill_fraction = first_fill_fraction - apodization_per_um * distance_um
        period_um = wavelength_um / (
            partial_thickness_effective_index
            - cladding_index * math.sin(grating_angle_rad)
            + fill_fraction
            * (full_thickness_effective_index - partial_thickness_effective_index)
        )
        periods_um.append(period_um)
        fill_fractions.append(fill_fraction)
        distance_um += period_um
    return np.asarray(periods_um), np.asarray(fill_fractions)


def _annular_sector_points(
    *,
    center_x_um: float,
    outer_radius_um: float,
    opening_angle_deg: float,
    inner_radius_um: float = 0.0,
    arc_tolerance_um: float = 0.0005,
) -> np.ndarray:
    """Return polygon vertices for one circular or annular sector."""

    def point_count(radius_um: float) -> int:
        half_step_rad = math.acos(1 - arc_tolerance_um / radius_um)
        ideal_segments = abs(angle_span_rad) / (2 * half_step_rad)
        return max(4, 1 + math.floor(ideal_segments + 0.5))

    half_angle_rad = math.radians(opening_angle_deg) / 2
    start_angle_rad = -half_angle_rad
    angle_span_rad = 2 * half_angle_rad
    points = []
    if inner_radius_um <= 0:
        points.append((center_x_um, 0.0))

    outer_angles = np.linspace(
        start_angle_rad, half_angle_rad, point_count(outer_radius_um)
    )
    points.extend(
        (
            center_x_um + outer_radius_um * math.cos(angle_rad),
            outer_radius_um * math.sin(angle_rad),
        )
        for angle_rad in outer_angles
    )

    if inner_radius_um > 0:
        inner_angles = np.linspace(
            half_angle_rad, start_angle_rad, point_count(inner_radius_um)
        )
        points.extend(
            (
                center_x_um + inner_radius_um * math.cos(angle_rad),
                inner_radius_um * math.sin(angle_rad),
            )
            for angle_rad in inner_angles
        )
    return np.asarray(points)


@gf.cell
def focusing_grating_coupler(
    *,
    opening_angle_deg: float = 40.0,
    taper_length_um: float = 16.0,
    extension_length_um: float = 1.0,
    periods: int = 25,
    minimum_feature_um: float = 0.100,
    waveguide_length_um: float = 2.0,
    fiber_tilt_deg: float = 14.5,
    cladding_index: float = 1.44,
    wavelength_um: float = 1.55,
    apodization_per_um: float = 0.031,
    full_thickness_effective_index: float = 2.8537467,
    partial_thickness_effective_index: float = 2.4211880,
    arc_tolerance_um: float = 0.0005,
    etch_layer: LayerSpec = "DEEP_ETCH",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Return a focusing apodized grating-coupler PCell for the generic PDK.

    The full silicon footprint is drawn on the cross-section's ``WG`` layer.
    Grating gaps are drawn on ``DEEP_ETCH``. The generic-PDK layer stack then
    resolves unetched regions to 220 nm silicon and etched gaps to a 130 nm
    silicon slab.

    Args:
        opening_angle_deg: Grating and taper opening angle.
        taper_length_um: Radius of the focusing taper.
        extension_length_um: Unetched extension after the final tooth.
        periods: Number of apodized grating periods.
        minimum_feature_um: Initial minimum gap width.
        waveguide_length_um: Length of the straight waveguide before the taper.
        fiber_tilt_deg: Fiber angle from the surface normal.
        cladding_index: Cladding refractive index.
        wavelength_um: Design wavelength.
        apodization_per_um: Linear fill-fraction reduction.
        full_thickness_effective_index: Frozen 220 nm slab effective index.
        partial_thickness_effective_index: Frozen 130 nm slab effective index.
        arc_tolerance_um: Maximum circular-arc tessellation error.
        etch_layer: Generic-PDK partial-etch mask layer.
        cross_section: Waveguide cross-section; defaults to gpdk ``strip``.
    """
    cross_section_object = gf.get_cross_section(cross_section)
    waveguide_layer = cross_section_object.layer
    waveguide_width_um = cross_section_object.width
    if waveguide_layer is None:
        raise ValueError("cross_section must define a waveguide layer")

    periods_um, fill_fractions = _grating_parameters(
        periods=periods,
        minimum_feature_um=minimum_feature_um,
        fiber_tilt_deg=fiber_tilt_deg,
        cladding_index=cladding_index,
        wavelength_um=wavelength_um,
        apodization_per_um=apodization_per_um,
        full_thickness_effective_index=full_thickness_effective_index,
        partial_thickness_effective_index=partial_thickness_effective_index,
    )
    opening_angle_rad = math.radians(opening_angle_deg)
    focus_x_um = waveguide_length_um - (
        waveguide_width_um
        / 2
        / math.sin(opening_angle_rad / 2)
        * math.cos(opening_angle_rad / 2)
    )
    final_radius_um = taper_length_um + float(np.sum(periods_um))
    final_radius_um += extension_length_um

    component = Component()
    component.add_polygon(
        _annular_sector_points(
            center_x_um=focus_x_um,
            outer_radius_um=final_radius_um,
            opening_angle_deg=opening_angle_deg,
            arc_tolerance_um=arc_tolerance_um,
        ),
        layer=waveguide_layer,
    )
    component.add_polygon(
        [
            (waveguide_length_um, -waveguide_width_um / 2),
            (0.0, -waveguide_width_um / 2),
            (0.0, waveguide_width_um / 2),
            (waveguide_length_um, waveguide_width_um / 2),
        ],
        layer=waveguide_layer,
    )

    previous_outer_radius_um = taper_length_um
    for period_um, fill_fraction in zip(periods_um, fill_fractions, strict=True):
        tooth_outer_radius_um = previous_outer_radius_um + period_um
        gap_outer_radius_um = tooth_outer_radius_um - fill_fraction * period_um
        component.add_polygon(
            _annular_sector_points(
                center_x_um=focus_x_um,
                outer_radius_um=gap_outer_radius_um,
                inner_radius_um=previous_outer_radius_um,
                opening_angle_deg=opening_angle_deg,
                arc_tolerance_um=arc_tolerance_um,
            ),
            layer=etch_layer,
        )
        previous_outer_radius_um = tooth_outer_radius_um

    component.add_port(
        name="o1",
        center=(0.0, 0.0),
        width=waveguide_width_um,
        orientation=180,
        layer=waveguide_layer,
        port_type="optical",
    )
    component.info["wavelength_um"] = wavelength_um
    component.info["fiber_tilt_deg"] = fiber_tilt_deg
    return component


__all__ = ["focusing_grating_coupler"]
