# Gsim 0.0.0

> a GDSFactory Simulation Plugin

![gsim-logo](./docs/assets/img/gsim-small.png)

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

For development:

```bash
git clone https://github.com/doplaydo/gsim
cd gsim
pip install -e .[dev]
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

## Mesh Presets

| Preset  | Refined Size | Max Size | Use Case                          |
| ------- | ------------ | -------- | --------------------------------- |
| Coarse  | 10.0 µm      | 600.0 µm | Fast iteration (~2.5 elements/λ)  |
| Default | 5.0 µm       | 300.0 µm | Balanced accuracy (~5 elements/λ) |
| Fine    | 2.0 µm       | 70.0 µm  | High accuracy (~10 elements/λ)    |

## Port Types

- **Inplane ports**: Horizontal ports on single metal layer for CPW gaps
- **Via ports**: Vertical ports between two metal layers for microstrip structures
- **CPW ports**: Multi-element ports for proper Coplanar Waveguide excitation

## Documentation

See the [documentation](https://doplaydo.github.io/gsim/) for detailed API reference and examples.

## License

Copyright 2026 GDSFactory
