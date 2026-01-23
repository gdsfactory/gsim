from __future__ import annotations

from collections import deque

import gdsfactory as gf
import vlsir
import vlsir.circuit_pb2 as vckt

# Mapping from string names to proto enum values
_SPICE_TYPE_MAP: dict[str, int] = {
    "SUBCKT": vckt.SpiceType.SUBCKT,
    "RESISTOR": vckt.SpiceType.RESISTOR,
    "CAPACITOR": vckt.SpiceType.CAPACITOR,
    "INDUCTOR": vckt.SpiceType.INDUCTOR,
    "MOS": vckt.SpiceType.MOS,
    "DIODE": vckt.SpiceType.DIODE,
    "BIPOLAR": vckt.SpiceType.BIPOLAR,
    "VSOURCE": vckt.SpiceType.VSOURCE,
    "ISOURCE": vckt.SpiceType.ISOURCE,
    "VCVS": vckt.SpiceType.VCVS,
    "VCCS": vckt.SpiceType.VCCS,
    "CCCS": vckt.SpiceType.CCCS,
    "CCVS": vckt.SpiceType.CCVS,
    "TLINE": vckt.SpiceType.TLINE,
}


def _make_param(key: str, val: float | str) -> vlsir.Param:
    """Create a VLSIR Param from a key-value pair."""
    param = vlsir.Param(name=key)
    if isinstance(val, float):
        param.value.double_value = val
    elif isinstance(val, int):
        param.value.int64_value = val
    else:
        param.value.literal = str(val)
    return param


def _spice_type_to_proto(spice_type: str | int) -> int:
    """Convert string or int spice_type to proto enum value."""
    if isinstance(spice_type, int):
        return spice_type
    return _SPICE_TYPE_MAP.get(spice_type.upper(), vckt.SpiceType.SUBCKT)


def _process_schematic(
    schematic: dict,
    full_netlist: dict,
    package: vckt.Package,
    lib_set: set[str],
    ext_modules: dict[str, vckt.ExternalModule],
    processed_subckt: set[str],
    domain: str = "gsim",
) -> vckt.Module:
    """Process a single schematic level and return its Module.

    This function handles both leaf devices (with vlsir metadata) and
    recursive sub-components (treated as SUBCKTs).

    Args:
        schematic: The schematic dict for this component level
        full_netlist: The complete recursive netlist from get_netlist()
        package: The Package being built (modified in place)
        lib_set: Set of library names (modified in place)
        ext_modules: Dict of external modules by model name (modified in place)
        processed_subckt: Set of already processed subcircuit names (modified in place)
        domain: Domain name for VLSIR qualified names

    Returns:
        The Module representing this schematic level
    """
    # Classify instances into devices (vlsir metadata), routing, and subckts
    device_instances = []
    routing_instances = []
    subckt_instances = []  # instances that reference other components in the netlist

    for inst_name, inst_data in schematic.get("instances", {}).items():
        info = inst_data.get("info", {})
        component_name = inst_data.get("component", "")

        if "vlsir" in info:
            # Leaf device with SPICE model
            device_instances.append(inst_name)
            lib_set.add(info["vlsir"]["spice_lib"])
        elif component_name in full_netlist:
            # This instance references another component in the netlist → SUBCKT
            subckt_instances.append(inst_name)
        else:
            # No vlsir metadata and not a subckt → routing
            routing_instances.append(inst_name)

    # Build connectivity graph including routing and direct device connections
    node_id = 0

    # First pass: assign nodes via routing connectivity (BFS on routing graph)
    routing_graph: dict[str, set[str]] = {r: set() for r in routing_instances}

    for net in schematic.get("nets", []):
        p1_comp, _ = net["p1"].split(",")
        p2_comp, _ = net["p2"].split(",")

        if p1_comp in routing_graph and p2_comp in routing_graph:
            routing_graph[p1_comp].add(p2_comp)
            routing_graph[p2_comp].add(p1_comp)

    # BFS to find connected routing components → electrical nodes
    visited: set[str] = set()
    routing_to_node: dict[str, str] = {}

    for start in routing_instances:
        if start in visited:
            continue
        queue = deque([start])
        node_name = f"net_{node_id}"
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            routing_to_node[curr] = node_name
            queue.extend(routing_graph[curr] - visited)
        node_id += 1

    # Second pass: map device/subckt ports to nodes
    # Also handle direct device-to-device and device-to-subckt connections
    all_device_like = set(device_instances) | set(subckt_instances)
    device_port_nodes: dict[str, dict[str, str]] = {d: {} for d in all_device_like}

    # Track direct connections between device-like instances for node merging
    # Format: list of (inst1, port1, inst2, port2) tuples
    direct_connections: list[tuple[str, str, str, str]] = []

    for net in schematic.get("nets", []):
        p1_comp, p1_port = net["p1"].split(",")
        p2_comp, p2_port = net["p2"].split(",")

        p1_is_device = p1_comp in all_device_like
        p2_is_device = p2_comp in all_device_like
        p1_is_routing = p1_comp in routing_to_node
        p2_is_routing = p2_comp in routing_to_node

        if p1_is_device and p2_is_routing:
            device_port_nodes[p1_comp][p1_port] = routing_to_node[p2_comp]
        elif p2_is_device and p1_is_routing:
            device_port_nodes[p2_comp][p2_port] = routing_to_node[p1_comp]
        elif p1_is_device and p2_is_device:
            # Direct device-to-device connection
            direct_connections.append((p1_comp, p1_port, p2_comp, p2_port))

    # Process direct device-to-device connections
    # Create new nodes or merge existing ones
    for inst1, port1, inst2, port2 in direct_connections:
        node1 = device_port_nodes[inst1].get(port1)
        node2 = device_port_nodes[inst2].get(port2)

        if node1 is not None and node2 is not None:
            # Both already assigned - they should be the same node
            # (In a more complex implementation, we'd merge nodes here)
            pass
        elif node1 is not None:
            # inst1.port1 has a node, assign it to inst2.port2
            device_port_nodes[inst2][port2] = node1
        elif node2 is not None:
            # inst2.port2 has a node, assign it to inst1.port1
            device_port_nodes[inst1][port1] = node2
        else:
            # Neither has a node, create a new one
            new_node = f"net_{node_id}"
            node_id += 1
            device_port_nodes[inst1][port1] = new_node
            device_port_nodes[inst2][port2] = new_node

    # Recursively process subcircuits first
    for inst_name in subckt_instances:
        inst_data = schematic["instances"][inst_name]
        component_name = inst_data["component"]

        if component_name not in processed_subckt:
            processed_subckt.add(component_name)
            sub_schematic = full_netlist[component_name]
            sub_module = _process_schematic(
                sub_schematic,
                full_netlist,
                package,
                lib_set,
                ext_modules,
                processed_subckt,
                domain,
            )
            package.modules.append(sub_module)

    # Collect unique ExternalModules for leaf devices
    for inst_name in device_instances:
        info = schematic["instances"][inst_name]["info"]["vlsir"]
        model = info["model"]

        if model not in ext_modules:
            qname = vlsir.utils.QualifiedName(name=model, domain=domain)
            spice_type = _spice_type_to_proto(info.get("spice_type", "SUBCKT"))
            ext_mod = vckt.ExternalModule(name=qname, spicetype=spice_type)

            for port_name in info["port_order"]:
                ext_mod.signals.append(vckt.Signal(name=port_name, width=1))
                ext_mod.ports.append(
                    vckt.Port(signal=port_name, direction=vckt.Port.Direction.INOUT)
                )

            for key, val in info.get("params", {}).items():
                ext_mod.parameters.append(_make_param(key, val))

            ext_modules[model] = ext_mod

    # Build the module for this level
    module_name = schematic.get("name", "unnamed")
    # Try to get name from the schematic's settings or instances
    for inst_data in schematic.get("instances", {}).values():
        if "component" in inst_data:
            # The parent component name might be derivable
            break

    module = vckt.Module(name=module_name)

    # Collect all nodes used by devices and subckts
    node_set: set[str] = set()
    for ports in device_port_nodes.values():
        node_set.update(ports.values())

    for node_name in sorted(node_set):
        module.signals.append(vckt.Signal(name=node_name, width=1))

    # Create instances for leaf devices
    for inst_name in device_instances:
        info = schematic["instances"][inst_name]["info"]["vlsir"]
        port_map = info.get("port_map", {})

        inst = vckt.Instance(name=inst_name)
        inst.module.external.CopyFrom(
            vlsir.utils.QualifiedName(name=info["model"], domain=domain)
        )

        for key, val in info.get("params", {}).items():
            inst.parameters.append(_make_param(key, val))

        for gds_port, node in device_port_nodes[inst_name].items():
            vlsir_port = port_map.get(gds_port, gds_port).lower()
            inst.connections.append(
                vckt.Connection(
                    portname=vlsir_port, target=vckt.ConnectionTarget(sig=node)
                )
            )

        module.instances.append(inst)

    # Create instances for subckts
    for inst_name in subckt_instances:
        inst_data = schematic["instances"][inst_name]
        component_name = inst_data["component"]

        inst = vckt.Instance(name=inst_name)
        inst.module.local = component_name

        # Connect subckt ports to nodes
        for gds_port, node in device_port_nodes[inst_name].items():
            inst.connections.append(
                vckt.Connection(
                    portname=gds_port.lower(), target=vckt.ConnectionTarget(sig=node)
                )
            )

        module.instances.append(inst)

    return module


def to_vlsir_circuit(
    top: gf.Component, domain: str = "gsim"
) -> tuple[vckt.Package, list[str]]:
    """Convert a gdsfactory Component to a VLSIR circuit package.

    Extracts the recursive netlist from the component, identifies device instances
    (those with 'vlsir' metadata), routing instances, and sub-components (treated
    as SUBCKTs). Builds a connectivity graph to determine electrical nodes,
    including direct device-to-device connections.

    Args:
        top: The top-level gdsfactory Component to convert. Device instances must
            have 'vlsir' info containing 'model', 'spice_lib', 'port_order', and
            optionally 'port_map', 'params', and 'spice_type'.
        domain: Domain name for VLSIR qualified names (default: "gsim")

    Returns:
        A tuple of (package, lib_list) where:
            - package: A vckt.Package with the circuit representation
            - lib_list: List of unique SPICE library names required by the devices
    """
    # Get full recursive netlist
    full_netlist = top.get_netlist(recursive=True)
    top_schematic = full_netlist[top.name]

    # Initialize package and tracking structures
    package = vckt.Package(domain=domain)
    lib_set: set[str] = set()
    ext_modules: dict[str, vckt.ExternalModule] = {}
    processed_subckt: set[str] = {top.name}  # Mark top as processed

    # Process the top-level schematic (and recursively any subckts)
    top_module = _process_schematic(
        top_schematic,
        full_netlist,
        package,
        lib_set,
        ext_modules,
        processed_subckt,
        domain,
    )
    top_module.name = top.name  # Ensure top module has correct name

    # Add external modules to package
    package.ext_modules.extend(ext_modules.values())

    # Add top module last (convention: top module is last in modules list)
    package.modules.append(top_module)

    return package, list(lib_set)
