# gsim

Electromagnetic simulation for photonics and electronics, powered by [GDSFactory+](https://gdsfactory.com).

## Overview

gsim connects GDSFactory layout designs to multiple EM solvers for photonic and electronic simulation. It handles
geometry extraction, mesh generation, port configuration, and cloud execution so you can go from GDS to S-parameters
with minimal boilerplate.

## Solvers

| Module        | Solver                                      | Method | Use Case                                               |
| ------------- | ------------------------------------------- | ------ | ------------------------------------------------------ |
| `gsim.palace` | [Palace](https://awslabs.github.io/palace/) | FEM    | RF/microwave, impedance extraction, driven simulations |
| `gsim.meep`   | [Meep](https://meep.readthedocs.io/)        | FDTD   | Photonic components, S-parameters, mode propagation    |

## Features

- **Layer stack extraction** — build 3D geometry from PDK layer stacks
- **Port configuration** — convert GDSFactory ports into solver-compatible definitions
- **Mesh generation** — GMSH finite-element meshes with configurable quality presets (Palace)
- **Cloud execution** — upload, run, and download results via `gsim.gcloud`
- **Visualization** — solver-agnostic 3D/2D component preview (PyVista, Matplotlib)

## Installation

```bash
pip install gsim
```

## Quick Start — Palace (RF/Microwave)

```python
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_geometry(component)
sim.set_stack(air_above=300.0)
sim.add_cpw_port("o1", layer="topmetal2", s_width=10, gap_width=6, length=5)
sim.set_driven(fmin=1e9, fmax=100e9)
sim.set_output_dir("./sim")
sim.mesh(preset="fine")
results = sim.run()
```

## Quick Start — Meep (Photonics)

```python
from gsim import meep

sim = meep.Simulation()
sim.geometry(component=ybranch, z_crop="auto")
sim.materials = {"si": 3.47, "SiO2": 1.44}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01, num_freqs=11)
sim.monitors = ["o1", "o2"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=32, simplify_tol=0.01)
result = sim.run()
```

## API Reference

See the [API docs](api.md) for full details.
