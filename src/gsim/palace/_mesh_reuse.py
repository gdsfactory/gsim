"""Port declarations and Port settings of a simulation that reuses a mesh.

A reused mesh fixes the kind and geometry of every port: the surfaces are in
the mesh. What each simulation may still choose are the Port settings
(excitation, impedance, R/L/C, wave-port mode solve).
:func:`merge_port_declarations` combines the source's port declarations with
the ones the receiver made itself (or, on a receiver that already reuses a
mesh, gives its ports the source's Port geometry back), and
:func:`sync_port_settings` writes the receiver's Port settings into its copy of
the extracted ports.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from gsim.palace._mesh_fingerprint import port_geometry
from gsim.palace.models.ports import (
    CPWPortConfig,
    PortConfig,
    TwoTerminalPortConfig,
    WavePortConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from gsim.palace._mesh_fingerprint import PortDeclaration
    from gsim.palace.ports.config import PalacePort

# The port lists of a simulation, in the order the mesh inputs list them
PORT_LISTS = ("ports", "cpw_ports", "wave_ports", "two_terminal_ports")

# The fields of each port kind that port_geometry() reads, and the name; every
# other field is a Port setting
_GEOMETRY_FIELDS: dict[type, frozenset[str]] = {
    PortConfig: frozenset(
        ("name", "geometry", "layer", "from_layer", "to_layer", "length", "offset")
    ),
    CPWPortConfig: frozenset(
        ("name", "geometry", "layer", "s_width", "gap_width", "length", "offset")
    ),
    WavePortConfig: frozenset(
        ("name", "layer", "z_margin", "lateral_margin", "max_size")
    ),
    TwoTerminalPortConfig: frozenset(("plus_port", "minus_port", "layer")),
}


def _port_settings(cfg: PortDeclaration) -> set[str]:
    """The names of the Port settings of a port declaration."""
    return set(type(cfg).model_fields) - _GEOMETRY_FIELDS[type(cfg)]


def _with_settings_of(cfg: PortDeclaration, own: PortDeclaration) -> PortDeclaration:
    """A copy of ``cfg`` that takes the Port settings ``own`` shares with it.

    The kind and geometry are the ones of ``cfg``. A setting the two kinds do
    not share keeps the value of ``cfg``.
    """
    shared = _port_settings(cfg) & _port_settings(own)
    return cfg.model_copy(
        deep=True,
        update={field: copy.deepcopy(getattr(own, field)) for field in shared},
    )


def merge_port_declarations(
    source: Mapping[str, Sequence[PortDeclaration]],
    receiver: Mapping[str, Sequence[PortDeclaration]],
    *,
    source_name: str,
    receiver_name: str,
    take_geometry_back: bool = False,
) -> dict[str, list[PortDeclaration]]:
    """Merge the receiver's port declarations into the source's, by name.

    Args:
        source: The source's port lists, keyed as :data:`PORT_LISTS`.
        receiver: The receiver's port lists, keyed the same way.
        source_name: Class name of the source, for the messages.
        receiver_name: Class name of the receiver, for the messages.
        take_geometry_back: True when the receiver already holds a reused
            mesh, so its ports came from a source. A receiver port with
            another kind or geometry then becomes a copy of the source's
            declaration with the receiver's Port settings, and a receiver
            port the source lacks is dropped; nothing is refused.

    Returns:
        The receiver's new port lists, in the source's order. A port the
        receiver declared with the source's kind and geometry is its own
        object, with its own settings; every other port is a deep copy of
        the source's declaration.

    Raises:
        ValueError: Unless ``take_geometry_back``: if a receiver port has the
            name of a source port but another kind or geometry, or if it has
            no source port at all.
    """
    declared: dict[str, list[PortDeclaration]] = {}
    for kind in PORT_LISTS:
        for cfg in receiver.get(kind, ()):
            declared.setdefault(port_geometry(cfg)[0], []).append(cfg)

    kept: set[int] = set()
    merged: dict[str, list[PortDeclaration]] = {kind: [] for kind in PORT_LISTS}
    for kind in PORT_LISTS:
        for cfg in source.get(kind, ()):
            key, geometry = port_geometry(cfg)
            match = None
            for own in declared.get(key, ()):
                own_geometry = port_geometry(own)[1]
                if own_geometry == geometry:
                    match = own
                elif take_geometry_back:
                    match = _with_settings_of(cfg, own)
                else:
                    raise ValueError(
                        f"Port '{key}' of this {receiver_name} does not match "
                        f"the reused mesh: it is declared as {own_geometry}, "
                        f"the mesh of the source {source_name} has {geometry}. "
                        "A reused mesh fixes the kind and geometry of every "
                        "port. Declare the port with the source's geometry, or "
                        "call mesh() on this simulation instead."
                    )
            if match is None:
                merged[kind].append(cfg.model_copy(deep=True))
            else:
                merged[kind].append(match)
                kept.add(id(match))

    extra = [
        key for key, cfgs in declared.items() for cfg in cfgs if id(cfg) not in kept
    ]
    if extra and not take_geometry_back:
        names = ", ".join(f"'{key}'" for key in extra)
        port, has, it = (
            ("Ports", "have", "them") if len(extra) > 1 else ("Port", "has", "it")
        )
        raise ValueError(
            f"{port} {names} {has} no surface in the reused mesh: the source "
            f"{source_name} does not declare {it}. Remove {it} from this "
            f"{receiver_name}, or declare {it} on the source and call mesh() "
            "on the source again."
        )
    return merged


def sync_port_settings(
    palace_ports: Sequence[PalacePort],
    port_info: Iterable[dict[str, Any]],
    port_configs: Iterable[PortDeclaration],
    *,
    simulation_type: str,
) -> None:
    """Write the Port settings of the declarations into the extracted ports.

    Only the Port settings change, in place: ``excited``, ``impedance``,
    R/L/C and the wave-port mode solve (``mode``, ``modes``,
    ``excite_modes``, ``pilot_paths``, the de-embedding ``offset``,
    ``eigensolver_*``), and ``"Z0"`` in the mesh's port info. Order, names,
    kinds and positions stay as the mesh has them. The values are the ones
    port configuration and extraction give a simulation that meshes itself,
    so equal declarations change nothing.

    Args:
        palace_ports: The extracted ports of the simulation's mesh.
        port_info: The port info of that mesh (``MeshResult.port_info``).
        port_configs: The simulation's port declarations, of every kind.
        simulation_type: The simulation's type. Only a driven simulation
            excites its lumped ports, as in ``_configure_ports_on_component``.
    """
    declared = {port_geometry(cfg)[0]: cfg for cfg in port_configs}
    for port in palace_ports:
        cfg = declared.get(port.name)
        if isinstance(cfg, PortConfig):
            # A lumped offset moved the surface; it is not a de-embedding
            # distance, so PalacePort.offset stays.
            port.excited = cfg.excited and simulation_type == "driven"
            port.impedance = cfg.impedance
            port.resistance = cfg.resistance
            port.inductance = cfg.inductance
            port.capacitance = cfg.capacitance
        elif isinstance(cfg, CPWPortConfig | TwoTerminalPortConfig):
            port.excited = cfg.excited
            port.impedance = cfg.impedance
        elif isinstance(cfg, WavePortConfig):
            port.excited = cfg.excited
            port.mode = cfg.mode
            port.offset = cfg.offset
            port.eigensolver_type = cfg.eigensolver_type
            port.eigensolver_tol = cfg.eigensolver_tol
            port.eigensolver_ksp_tol = cfg.eigensolver_ksp_tol
            port.eigensolver_max_size = cfg.eigensolver_max_size
            port.eigensolver_verbose = cfg.eigensolver_verbose
            port.modes = tuple(int(m) for m in cfg.modes)
            port.excite_modes = cfg.excite_modes
            port.pilot_paths = {
                int(m): [[float(c) for c in point] for point in path]
                for m, path in (cfg.pilot_paths or {}).items()
            }

    impedance = {port.name: port.impedance for port in palace_ports}
    for entry in port_info:
        if "Z0" in entry and entry.get("name") in impedance:
            entry["Z0"] = impedance[entry["name"]]
