# gsim

Palace EM Simulation API for GDSFactory.

## Overview

Gsim bridges the gap between circuit layout design (using [GDSFactory](https://gdsfactory.github.io/gdsfactory/)) and electromagnetic simulation (using [Palace](https://awslabs.github.io/palace/)). It automates the conversion of IC component layouts into simulation-ready mesh files and configuration.

## Features

- **Layer Stack Extraction**: Extract layer stacks from PDK definitions with a comprehensive material properties database
- **Port Configuration**: Convert GDSFactory ports into Palace-compatible port definitions (inplane, via, and CPW ports)
- **Mesh Generation**: Generate GMSH-compatible finite element meshes with configurable quality presets

## Installation

```bash
pip install gsim
```

## Quick Start

```python
from gsim.palace import (
    get_stack,
    configure_inplane_port,
    extract_ports,
    generate_mesh,
    MeshConfig,
)

# Get layer stack from active PDK
stack = get_stack()

# Configure ports on your component
configure_inplane_port(c.ports["o1"], layer="topmetal2", length=5.0)
configure_inplane_port(c.ports["o2"], layer="topmetal2", length=5.0)

# Extract configured ports
ports = extract_ports(c, stack)

# Generate mesh
result = generate_mesh(
    component=c,
    stack=stack,
    ports=ports,
    output_dir="./simulation",
    config=MeshConfig.default(),
)
```

## Port Types

| Type    | Description                                                       |
| ------- | ----------------------------------------------------------------- |
| Inplane | Horizontal ports on single metal layer for CPW gaps               |
| Via     | Vertical ports between two metal layers for microstrip structures |
| CPW     | Multi-element ports for proper Coplanar Waveguide excitation      |

## Mesh Presets

| Preset  | Refined Size | Max Size | Use Case                                   |
| ------- | ------------ | -------- | ------------------------------------------ |
| Coarse  | 10.0 um      | 600.0 um | Fast iteration (~2.5 elements/wavelength)  |
| Default | 5.0 um       | 300.0 um | Balanced accuracy (~5 elements/wavelength) |
| Fine    | 2.0 um       | 70.0 um  | High accuracy (~10 elements/wavelength)    |
