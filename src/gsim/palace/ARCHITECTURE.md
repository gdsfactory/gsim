# Palace-API Architecture

Clean gdsfactory integration for Palace EM simulation mesh generation.

## Structure

```
gplugins/palace_api/
├── __init__.py         # Main exports
│
├── stack/              # PDK → LayerStack
│   ├── __init__.py     # get_stack(), load_stack_yaml()
│   ├── extractor.py    # LayerStack, Layer, extract_from_pdk()
│   ├── materials.py    # MaterialProperties, MATERIALS_DB
│   └── visualization.py # print_stack(), plot_stack()
│
├── ports/              # gdsfactory port → PalacePort
│   ├── __init__.py     # configure_port(), extract_ports()
│   └── config.py       # PalacePort, CPWPort, PortType
│
├── mesh/               # Mesh generation
│   ├── __init__.py     # generate_mesh(), MeshConfig
│   ├── pipeline.py     # High-level API with presets
│   ├── generator.py    # Core mesh generation
│   └── gmsh_utils.py   # Gmsh utility functions
│
└── tests/
    ├── test_stack.py
    └── test_mesh.py
```

## Data Flow

```
gdsfactory Component + PDK
         │
         ▼
┌─────────────────────────────────────┐
│  Stack Extraction                   │
│  • get_stack() → LayerStack         │
│  • Extracts layers, materials, z    │
│  • Generates dielectric regions     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Port Configuration                 │
│  • configure_port() → port.info     │
│  • extract_ports() → PalacePort[]   │
│  • Supports lumped, via, CPW        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Mesh Generation                    │
│  • generate_mesh()                  │
│  • Creates geometry in gmsh         │
│  • Assigns physical groups          │
│  • Generates palace.msh + config    │
└──────────────┬──────────────────────┘
               │
               ▼
         palace.msh + config.json
```

## Usage

```python
from gplugins.palace_api import (
    get_stack,
    configure_port,
    extract_ports,
    generate_mesh,
    MeshConfig,
)

# Get component
c = gf.get_component("inductor")

# Configure
stack = get_stack()
for port in c.ports:
    configure_port(port, type='via', from_layer='metal1', to_layer='topmetal2')
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

| Type     | Parameters                           | Use Case                                        |
| -------- | ------------------------------------ | ----------------------------------------------- |
| `via`    | `from_layer`, `to_layer`             | Vertical port between metal layers (microstrip) |
| `lumped` | `layer`, `length`                    | In-plane port on single layer (CPW gap)         |
| `cpw`    | Two ports via `configure_cpw_port()` | Multi-element CPW excitation                    |

## Mesh Presets

| Preset                 | refined_mesh_size | max_mesh_size | Use Case               |
| ---------------------- | ----------------- | ------------- | ---------------------- |
| `MeshConfig.coarse()`  | 10.0 µm           | 600.0 µm      | Fast iteration         |
| `MeshConfig.default()` | 5.0 µm            | 300.0 µm      | Balanced (COMSOL-like) |
| `MeshConfig.fine()`    | 2.0 µm            | 70.0 µm       | High accuracy          |

## Physical Groups in Mesh

The generated mesh contains these named physical groups:

**Volumes (3D):**

- Material names: `SiO2`, `air`, `passive`, `silicon`
- `airbox` - surrounding air box

**Surfaces (2D):**

- Conductors: `{layer}_xy`, `{layer}_z` (e.g., `metal1_xy`, `topmetal2_z`)
- Ports: `P1`, `P2`, ... or `P1_E0`, `P1_E1` for CPW elements
- Boundary: `Absorbing_boundary`

## Dependencies

- gdsfactory (kfactory, klayout)
- gmsh
- numpy
- PyYAML
- plotly (optional, for stack visualization)
