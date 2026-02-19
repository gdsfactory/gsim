"""Pre-built RF layer stacks for common transmission-line topologies.

These helpers return a ``(LayerStack, LAYER_MAP)`` pair ready to use with
the Palace meshing and simulation pipeline.
"""

from __future__ import annotations

from gdsfactory.technology import LayerLevel, LayerStack


def cpw_layer_stack(
    conductor_thickness: float = 0,
    substrate_thickness: float = 1500,
    vacuum_thickness: float = 1500,
) -> tuple[LayerStack, dict[str, tuple[int, int]]]:
    """Return a basic CPW (coplanar waveguide) layer stack.

    Layers
    ------
    * **SUBSTRATE** (1, 0) — dielectric below the conductor plane.
    * **CONDUCTOR** (2, 0) — metal plane (zero-thickness by default for 2-D
      boundary-element treatment).
    * **VACUUM** (3, 0) — air / vacuum above the conductor.

    Args:
        conductor_thickness: Conductor metal thickness (µm), usually 0 for
            sheet-impedance models.
        substrate_thickness: Dielectric thickness (µm).
        vacuum_thickness: Air gap above conductor (µm).

    Returns:
        ``(layer_stack, LAYER)`` where *LAYER* maps names to ``(layer, datatype)``
        tuples.
    """
    LAYER = {
        "SUBSTRATE": (1, 0),
        "CONDUCTOR": (2, 0),
        "VACUUM": (3, 0),
    }

    ls = LayerStack()

    ls.layers["SUBSTRATE"] = LayerLevel(
        layer=LAYER["SUBSTRATE"],
        zmin=0.0,
        thickness=substrate_thickness,
        material="FR4",
        info={"description": "Dielectric substrate"},
    )

    ls.layers["CONDUCTOR"] = LayerLevel(
        layer=LAYER["CONDUCTOR"],
        zmin=substrate_thickness,
        thickness=conductor_thickness,
        material="conductor",
        info={"description": "Conductor plane"},
    )

    ls.layers["VACUUM"] = LayerLevel(
        layer=LAYER["VACUUM"],
        zmin=substrate_thickness + conductor_thickness,
        thickness=vacuum_thickness,
        material="Vacuum",
        info={"description": "Vacuum above conductor"},
    )

    return ls, LAYER
