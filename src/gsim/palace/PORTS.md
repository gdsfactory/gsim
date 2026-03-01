# Palace Port Types Explained

This document explains how ports work in gds2palace/Palace and how to define them for different transmission line
configurations.

## Port Types Overview

### In-Plane Port

- **Orientation**: Horizontal surface in the XY plane
- **Location**: Sits on a **single** metal layer
- **Definition**: `target_layername='TopMetal2'`
- **GDS shape**: Rectangle with finite width and length

```text
      ════════════════  ← metal layer
         ▓▓▓▓▓▓▓▓       ← port surface (horizontal)
```

### Via Port

- **Orientation**: Vertical surface in the XZ or YZ plane
- **Location**: Spans **between two** metal layers
- **Definition**: `from_layername='Metal3', to_layername='TopMetal2'`
- **GDS shape**: Line (degenerate rectangle) - one dimension is minimal

```text
      ════════════════  ← TopMetal2
            ║
            ║  ← port surface (vertical, like a wall)
            ║
      ════════════════  ← Metal3
```

Note: "Via port" does NOT mean a port on via structures. It's named this way because it spans the vertical gap between
layers, similar to how a via connects layers.

## How Palace Lumped Ports Work

A lumped port is a **2D surface** with:

- `R` = impedance (typically 50Ω)
- `Direction` = positive current flow direction (X, Y, Z, -X, -Y, -Z)

Palace integrates the E-field across this surface to compute voltage. The port acts as a lumped resistor connected
between whatever conductors touch the port surface.

## GDS Port Geometry

### In-Plane Ports: Rectangle Required

In-plane ports need a **rectangle** in the GDS (not just a line):

```python
# Port dimensions from GDS bounding box
xmin, xmax, ymin, ymax = polygon.bbox

# For Y-direction port (typical CPW facet):
width  = xmax - xmin   # CPW cross-section width
length = ymax - ymin   # extent perpendicular to port face

# For X-direction port:
width  = ymax - ymin
length = xmax - xmin
```

### Via Ports: Line Expected

Via ports expect essentially a line in the GDS:

- The smaller XY dimension determines orientation
- The port surface is created vertically between the two layers

## Transmission Line Configurations

### Case 1: Microstrip (Ground on Different Layer)

Use a **via port** spanning from signal layer to ground layer:

```text
TopMetal2 (signal) ════════════
                        ║ ← via port surface (vertical)
Metal1 (ground)    ════════════
```

Configuration:

```python
configure_port(
    port,
    type="via",
    from_layer="metal1",      # ground layer
    to_layer="topmetal2",     # signal layer
    impedance=50.0,
)
```

### Case 2: CPW (Signal/Ground on Same Layer)

Use an **in-plane port** spanning the gap between signal and ground:

```text
Ground ═══╡    ║    ╞═══ Signal ═══╡    ║    ╞═══ Ground
          └────┘                   └────┘
          port 1                   port 2
          (in gap)                 (in gap)
```

The port rectangle spans **from signal edge to ground edge** (across the gap).

Configuration:

```python
configure_port(
    port,
    type="lumped",
    layer="topmetal2",  # layer where CPW lives
    length=5.0,         # port extent along direction
    impedance=50.0,
)
```

## Converting gdsfactory Ports to Palace Format

gdsfactory ports have:

- `center`: (x, y) position
- `width`: port width (e.g., CPW signal + gaps)
- `orientation`: angle in degrees (0=east, 90=north, 180=west, 270=south)

Using palace_api:

```python
from gplugins.palace_api import configure_port, extract_ports

# Configure ports
configure_port(c.ports['o1'], type='lumped', layer='topmetal2', length=5.0)
configure_port(c.ports['o2'], type='lumped', layer='topmetal2', length=5.0)

# Extract for simulation
ports = extract_ports(c, stack)
```

## Why Ports Must Touch Both Signal and Ground

A lumped port is essentially a **virtual VNA probe**. To compute S-parameters, Palace needs:

1. **Voltage** = potential difference between two conductors
1. **Current** = flow between those conductors

Palace computes these by:

- **Voltage**: Integrating E-field across the port surface (from one conductor to the other)
- **Current**: Integrating H-field around the port boundary

If the port only touches one conductor, voltage is undefined - there's no second reference point.

```text
                Port touching both (CORRECT):

Ground ═══════╡▓▓▓▓▓▓╞═══════ Signal
              ↑     ↑
              └──┬──┘
           E-field integrated → Voltage


                Port touching only signal (WRONG):

Ground ═══════      ▓▓▓▓▓▓═══════ Signal
                    ↑
                    No reference → Voltage undefined
```

### Real-World Analogy

When probing a CPW with a VNA:

- Signal pin touches the center conductor
- Ground pins touch the ground planes
- Measurement happens **across the gap**

The port rectangle represents exactly this: the cross-section where your virtual probe connects signal to ground.

## Multi-Element CPW Ports

For proper CPW mode excitation, use `configure_cpw_port()` to link two gap ports:

```python
from gplugins.palace_api import configure_cpw_port

# CPW has two gaps with opposite E-field directions
configure_cpw_port(
    port_upper=c.ports['gap_upper'],  # signal-to-ground2 gap
    port_lower=c.ports['gap_lower'],  # ground1-to-signal gap
    layer='topmetal2',
    length=5.0,
    impedance=50.0,
)
```

This generates a multi-element lumped port in Palace:

```json
{
  "Index": 1,
  "R": 50.0,
  "Elements": [
    {"Attributes": [gap1_surface], "Direction": "+Y"},
    {"Attributes": [gap2_surface], "Direction": "-Y"}
  ]
}
```

## Key Points

1. **One port = one rectangle** defining the port surface
1. **Port must touch both signal and ground** - it's the bridge between them
1. **Direction** tells Palace the positive current flow direction
1. **In-plane ports** need actual area (rectangle), not just a line
1. **Via ports** are for vertical connections between layers
1. **CPW ports** need two elements with opposite directions
