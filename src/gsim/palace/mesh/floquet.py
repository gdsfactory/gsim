"""Floquet boundary configuration from measured periodic mesh translations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from gsim.palace.models import EigenmodeConfig


def periodic_boundary_config(
    groups: dict,
    eigenmode_config: EigenmodeConfig,
    periodic_axis: str | None,
    periodic_translation: tuple[float, float, float] | None,
) -> dict:
    """Return Palace's periodic boundary block using lengths in mesh units.

    Imported meshes may supply x, y, or z translations; the GDS mesher supports
    x and y. Missing geometry metadata is an error, even if the user supplied
    an expected period, because that expectation must be checked against it.
    """
    axis = (periodic_axis or "").lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(
            "Floquet eigenmode requires a periodic axis set in mesh(). "
            "Use mesh(periodic_axis='x') or mesh(periodic_axis='y')."
        )

    boundary_surfaces = groups.get("boundary_surfaces", {})
    donor_info = boundary_surfaces.get("periodic_donor")
    receiver_info = boundary_surfaces.get("periodic_receiver")
    if donor_info is None or receiver_info is None:
        raise ValueError(
            "Floquet enabled but periodic donor/receiver boundaries were not "
            "found in the generated mesh."
        )
    attributes = []
    for boundary_info in (donor_info, receiver_info):
        physical_group = boundary_info.get("phys_group")
        values = (
            physical_group if isinstance(physical_group, list) else [physical_group]
        )
        if not values or any(
            not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError("Floquet periodic boundary attributes must be positive.")
        attributes.append(sorted(values))

    if periodic_translation is None:
        raise ValueError(
            "Floquet requires the mesh periodic_translation in mesh units. "
            "Regenerate the periodic mesh to record its actual period."
        )
    axis_index = "xyz".index(axis)
    if (
        len(periodic_translation) != 3
        or not all(math.isfinite(value) for value in periodic_translation)
        or periodic_translation[axis_index] <= 0
        or any(
            value != 0
            for index, value in enumerate(periodic_translation)
            if index != axis_index
        )
    ):
        raise ValueError(
            "periodic_translation must be a finite, positive translation "
            f"along the periodic {axis} axis."
        )
    wave_vector = eigenmode_config.compute_floquet_wave_vector(
        periodic_axis=cast(Literal["x", "y", "z"], axis),
        periodic_length=periodic_translation[axis_index],
    )
    return {
        "FloquetWaveVector": wave_vector,
        "BoundaryPairs": [
            {
                "DonorAttributes": attributes[0],
                "ReceiverAttributes": attributes[1],
                "Translation": list(periodic_translation),
            }
        ],
    }
