"""Fingerprint of the inputs that shape a Palace mesh.

A mesh fixes the polygons, the stack regions, the physical groups and the port
surfaces of the simulation that made it. A simulation that reuses the mesh
must keep those inputs. :func:`mesh_inputs_fingerprint` records them, as one
canonical string per input, so two records can be compared and
:func:`describe_changes` can name what differs.

Inputs that only size the elements (mesh settings, ``mesh_resolution``,
permittivity values) and Port settings (excitation, impedance, R/L/C, wave
modes) are left out: they can change without a new mesh.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from gsim.palace.models.ports import (
    CPWPortConfig,
    PortConfig,
    TwoTerminalPortConfig,
    WavePortConfig,
)

if TYPE_CHECKING:
    from gsim.common import LayerStack

# A port declaration of any kind: lumped, CPW, wave or two-terminal
PortDeclaration = PortConfig | CPWPortConfig | WavePortConfig | TwoTerminalPortConfig

# Sections of a fingerprint in the order describe_changes() lists them, with
# the label of each. A trailing ":" marks a section keyed per name.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("geometry", "geometry (layout polygons)"),
    ("component_ports", "component ports (names, positions, orientations, widths)"),
    ("layer:", "stack layer '{}'"),
    ("dielectrics", "stack dielectric regions"),
    ("material:", "material '{}' (air-like or not)"),
    ("airbox", "airbox (set_airbox)"),
    ("pec_blocks", "PEC blocks (add_pec)"),
    ("symmetry_plane", "symmetry plane"),
    ("port:", "port '{}' geometry"),
)


def _canonical(value: Any) -> str:
    """One JSON string per value, independent of dict key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def port_geometry(cfg: PortDeclaration) -> tuple[str, tuple]:
    """The name of a port declaration and the fields that shape its surface.

    Args:
        cfg: A port declaration of a simulation.

    Returns:
        ``(key, geometry)``: the declared name (``plus_port`` for a two-terminal
        port) and a tuple of the fields the mesher reads to place the port
        surface, led by the port kind. Port settings are not in it.

    Raises:
        TypeError: For an object that is not a port declaration.
    """
    if isinstance(cfg, PortConfig):
        return cfg.name, (
            "lumped",
            cfg.geometry,
            cfg.layer,
            cfg.from_layer,
            cfg.to_layer,
            cfg.length,
            cfg.offset,
        )
    if isinstance(cfg, CPWPortConfig):
        return cfg.name, (
            "cpw",
            cfg.layer,
            cfg.s_width,
            cfg.gap_width,
            cfg.length,
            cfg.offset,
        )
    if isinstance(cfg, WavePortConfig):
        return cfg.name, (
            "wave",
            cfg.layer,
            float(cfg.z_margin),
            cfg.lateral_margin,
            cfg.max_size,
        )
    if isinstance(cfg, TwoTerminalPortConfig):
        return cfg.plus_port, ("two_terminal", cfg.minus_port, cfg.layer)
    raise TypeError(f"Not a port declaration: {type(cfg).__name__}")


def _geometry_digest(component: Any) -> str:
    """sha256 of the layout polygons, grouped by GDS ``(layer, datatype)``.

    The polygons come from the call ``extract_geometry`` makes. They are
    grouped by GDS tuple, not by layer index, because named aliases share a
    tuple. ``Polygon.to_s()`` gives exact integer database-unit coordinates,
    holes included, and sorting makes the order of the shapes irrelevant.
    Tuples with no polygons are left out, so a layer that another layout
    registers in the shared layout does not count as a change.
    """
    if component is None:
        return "none"
    layout = component.kcl.layout
    polygons_by_index = component.get_polygons(layers=list(layout.layer_indexes()))
    by_tuple: dict[tuple[int, int], list[str]] = {}
    for index, polygons in polygons_by_index.items():
        if not polygons or not layout.is_valid_layer(index):
            continue
        info = layout.get_info(index)
        by_tuple.setdefault((info.layer, info.datatype), []).extend(
            poly.to_s() for poly in polygons
        )

    digest = hashlib.sha256()
    digest.update(f"dbu={layout.dbu!r}\n".encode())
    for (layer, datatype), shapes in sorted(by_tuple.items()):
        digest.update(f"layer={layer}/{datatype}\n".encode())
        for shape in sorted(shapes):
            digest.update(f"{shape}\n".encode())
    return digest.hexdigest()


def _component_ports(component: Any) -> list[tuple]:
    """The component port fields the port code reads, in port order."""
    if component is None:
        return []
    return [
        (
            p.name,
            float(p.center[0]),
            float(p.center[1]),
            None if p.orientation is None else float(p.orientation),
            float(p.width),
        )
        for p in component.ports
    ]


def _via_conductive(stack: LayerStack, material: str) -> bool:
    """Whether the mesher meshes a via of this material as a volume.

    Read as ``mesh/geometry.py`` reads it: a conductivity above 0. For an
    anisotropic (list) value, any element above 0.
    """
    conductivity = stack.materials.get(material, {}).get("conductivity", 0.0)
    if isinstance(conductivity, list | tuple):
        return any(float(c) > 0 for c in conductivity)
    return conductivity is not None and float(conductivity) > 0


def mesh_inputs_fingerprint(
    *,
    component: Any,
    stack: LayerStack,
    airbox_config: Mapping[str, Any],
    pec_blocks: Iterable[Any],
    symmetry_planes: Iterable[Any],
    port_configs: Iterable[PortDeclaration],
) -> dict[str, str]:
    """Record the inputs that shape a mesh, one canonical string per input.

    Args:
        component: The gdsfactory component, or None.
        stack: The resolved layer stack, as the mesher receives it.
        airbox_config: The ``set_airbox`` settings (``{}`` when unset).
        pec_blocks: The ``add_pec`` blocks.
        symmetry_planes: The symmetry planes.
        port_configs: Every port declaration: lumped, CPW, wave and
            two-terminal ports, in that order.

    Returns:
        ``{key: value}``. ``geometry`` is a sha256 hex digest (``"none"``
        without a component); every other value is canonical JSON. Ports are
        keyed by name, so declaring a port again in another list position gives
        the same fingerprint.
    """
    from gsim.palace.mesh.geometry import is_air_like_material

    fingerprint = {
        "geometry": _geometry_digest(component),
        "component_ports": _canonical(_component_ports(component)),
    }

    referenced: list[str] = []
    for name, layer in stack.layers.items():
        entry = layer.model_dump(mode="json", exclude={"mesh_resolution"})
        if layer.layer_type == "via":
            entry["via_conductive"] = _via_conductive(stack, layer.material)
        fingerprint[f"layer:{name}"] = _canonical(entry)
        referenced.append(layer.material)

    fingerprint["dielectrics"] = _canonical(stack.dielectrics)
    referenced.extend(
        d["material"] for d in stack.dielectrics if d.get("material") is not None
    )
    for material in dict.fromkeys(referenced):
        fingerprint[f"material:{material}"] = _canonical(
            is_air_like_material(stack, material)
        )

    fingerprint["airbox"] = _canonical(dict(airbox_config))
    fingerprint["pec_blocks"] = _canonical(
        [b.model_dump(mode="json") for b in pec_blocks]
    )
    fingerprint["symmetry_plane"] = _canonical(
        [
            p.model_dump(mode="json", exclude={"verify_symmetry"})
            for p in symmetry_planes
        ]
    )

    ports: dict[str, list[tuple]] = {}
    for cfg in port_configs:
        key, geometry = port_geometry(cfg)
        ports.setdefault(key, []).append(geometry)
    for key, geometries in ports.items():
        fingerprint[f"port:{key}"] = _canonical(geometries)

    return fingerprint


def describe_changes(
    recorded: Mapping[str, str], current: Mapping[str, str]
) -> list[str]:
    """Label every input that differs between two fingerprints.

    Args:
        recorded: The fingerprint the mesh was made from.
        current: The fingerprint of the current inputs.

    Returns:
        One label per changed, added or removed input, in a fixed section
        order (geometry first, ports last); within a section, in the order
        the inputs appear, recorded first. Empty when nothing differs.
    """
    changed = [
        key
        for key in dict.fromkeys([*recorded, *current])
        if recorded.get(key) != current.get(key)
    ]
    labels: list[str] = []
    for section, label in _SECTIONS:
        for key in changed:
            prefix, _, name = key.partition(":")
            if (f"{prefix}:" if name else prefix) != section:
                continue
            if section == "port:" and key not in recorded:
                labels.append(f"port '{name}' (added)")
            elif section == "port:" and key not in current:
                labels.append(f"port '{name}' (removed)")
            else:
                labels.append(label.format(name))
    return labels
