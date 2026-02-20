# gsim

Palace EM Simulation API for GDSFactory.

## Overview

Gsim bridges the gap between circuit layout design (using [GDSFactory](https://gdsfactory.github.io/gdsfactory/)) and
electromagnetic simulation (using [Palace](https://awslabs.github.io/palace/)).

## Installation

```bash
pip install gsim
```

## Quick Start

```python
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_geometry(component)
sim.set_stack(air_above=300.0)
sim.add_cpw_port("P2", "P1", layer="topmetal2", length=5.0)
sim.set_driven(fmin=1e9, fmax=100e9)
sim.mesh("./sim", preset="fine")
results = sim.simulate()
```

See [Palace Example](nbs/palace.md) for a complete walkthrough.
