# Multi-Element Lumped Ports for CPW

**Status**: Implemented

## Problem

Coplanar waveguide (CPW) structures require multi-element lumped ports to properly excite the CPW mode. Single-element
ports treat each gap as independent and result in incorrect S-parameters (no transmission).

## Background

In a CPW (Ground-Signal-Ground), the E-field directions are opposite in the two gaps:

```text
Ground 2  ═══════════════════
            ↓ E-field (-Y)     Gap2
Signal    ═══════════════════
            ↑ E-field (+Y)     Gap1
Ground 1  ═══════════════════
```

Palace supports this via multi-element ports:

```json
{
  "Index": 1,
  "R": 56.02,
  "Elements": [
    {"Attributes": [gap1_surface], "Direction": "+Y"},
    {"Attributes": [gap2_surface], "Direction": "-Y"}
  ]
}
```

## Reference

- [Palace CPW Example](https://awslabs.github.io/palace/stable/examples/cpw/)
- [Palace CPW config](https://github.com/awslabs/palace/blob/main/examples/cpw/cpw_lumped_uniform.json)

## Implementation

### API Usage

```python
from gplugins.palace_api import configure_cpw_port, extract_cpw_ports

# Configure two ports as a CPW pair
configure_cpw_port(
    port_upper=c.ports['gap_upper'],  # gap between signal and upper ground
    port_lower=c.ports['gap_lower'],  # gap between signal and lower ground
    layer='topmetal2',
    length=5.0,
    impedance=50.0,
)

# Extract CPW ports
cpw_ports = extract_cpw_ports(c, stack)

# Generate mesh with CPW ports
result = generate_mesh(
    component=c,
    stack=stack,
    ports=[],           # regular ports
    cpw_ports=cpw_ports, # CPW ports
    output_dir="./sim",
)
```

### How It Works

1. `configure_cpw_port()` links two gdsfactory ports as CPW elements

   - Stores `palace_type='cpw_element'` in port.info
   - Assigns `cpw_group` ID to link the pair
   - Auto-detects +/- directions based on Y positions

1. `extract_cpw_ports()` groups CPW elements into `CPWPort` objects

   - Each CPWPort has `upper_center` and `lower_center`
   - `get_element_directions()` returns `("+Y", "-Y")` or similar

1. Mesh generator creates separate surfaces for each element

   - Physical groups: `P1_E0`, `P1_E1` for port 1 elements

1. Config generator outputs multi-element format:

```json
{
  "Index": 1,
  "R": 50.0,
  "Excitation": 1,
  "Elements": [
    {"Attributes": [phys_group_E0], "Direction": "-Y"},
    {"Attributes": [phys_group_E1], "Direction": "+Y"}
  ]
}
```

## Files

- `ports/config.py` - `CPWPort` class, `configure_cpw_port()`, `extract_cpw_ports()`
- `mesh/generator.py` - `_add_ports()` handles CPW surfaces, `_generate_palace_config()` outputs Elements array
