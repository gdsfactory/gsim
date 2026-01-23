# VLSIR-GDSFactory Connection Logic

---

**Status** : Implemented in `netlist.py`

## Overview

GDSFactory connects components in one of three ways:

1. **`.connect()` method** - Physically arranges components side-by-side by aligning ports
2. **`gdsfactory.routing` utilities** - Draws routing between components using PDK rules
3. **Implicit logical links** - Inform routing but are not persistent state

## Implementation

The `to_vlsir_circuit()` function in `netlist.py` extracts connectivity from GDSFactory's recursive netlist using the following rules:

### Instance Classification

1. If an instance has [VLSIR Metadata](./METADATA_SPEC.md) → **device instance** (leaf SPICE element)
2. If an instance references another component in the netlist → **subcircuit instance** (SUBCKT)
3. Otherwise → **routing instance** (used for connectivity)

### Electrical Node Discovery

The algorithm uses BFS (breadth-first search) to discover electrical nodes:

1. Build a graph of routing-to-routing connections from the `nets` array
2. Find connected components in this graph via BFS traversal
3. Each connected component of routing instances becomes a single electrical node (e.g., `net_0`, `net_1`, ...)
4. Device ports are mapped to nodes based on their connections to routing instances

### Connection Flow

```
GDSFactory Component
        ↓
get_netlist(recursive=True)
        ↓
Parse instances → devices (vlsir) | routing | subckts
        ↓
Build routing connectivity graph from nets
        ↓
BFS to find connected routing components → electrical nodes
        ↓
Process direct device-to-device connections → new nodes
        ↓
Recursively process sub-components → nested Modules
        ↓
Generate VLSIR Package with ExternalModules + Modules
```

## Usage Example

```python
import gdsfactory as gf
from gsim.vlsir import to_vlsir_circuit

@gf.cell
def my_circuit():
    c = gf.Component()

    # Add device with vlsir metadata
    r1 = c << resistor(resistance=1000.0)
    r2 = c << resistor(resistance=2000.0)

    # Add routing (no vlsir metadata)
    w = c << wire(length=10)

    # Connect via ports
    w.connect("o1", r1.ports["n"])
    r2.connect("p", w.ports["o2"])

    return c

top = my_circuit()
package, libs = to_vlsir_circuit(top)
```

## Key Points

- Routing instances form the "glue" that defines electrical connectivity
- **Direct device-to-device connections** are now supported - when two devices connect directly without routing, a shared electrical node is automatically created
- The `port_map` in VLSIR metadata translates GDSFactory port names to SPICE port names
- **Recursive SUBCKTs** are supported - sub-components containing devices are processed recursively and represented as local Module references in VLSIR

## Supported Connection Types

| Connection Type                     | Support | Description                        |
| ----------------------------------- | ------- | ---------------------------------- |
| Device → Routing → Device           | ✅      | Standard routed connection         |
| Device → Device (direct)            | ✅      | Direct port-to-port connection     |
| Device → Routing → Routing → Device | ✅      | Chained routing (merged nodes)     |
| Nested SUBCKTs                      | ✅      | Recursive sub-component processing |
