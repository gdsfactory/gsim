#!/usr/bin/env python3
"""Generate MEEP simulation config for ebeam_y_1550.

Equivalent to the test-meep.ipynb notebook but as a simple script.
Outputs layout.gds, sim_config.json, and run_meep.py to ./meep-sim-test/.
"""

import os
from pathlib import Path

import gdsfactory as gf
from ubcpdk import PDK, cells

from gsim import meep

PDK.activate()

# 1. Create component
component = cells.ebeam_y_1550()

# 2. Configure simulation (declarative API)
sim = meep.Simulation()

# Geometry
sim.geometry.component = component
sim.geometry.z_crop = "auto"  # stack resolved from active PDK

# Materials
sim.materials = {"si": 3.47, "SiO2": 1.44}

# Source — spectral window + auto fwidth
sim.source.port = "o1"
sim.source.wavelength = 1.55
sim.source.bandwidth = 0.01
sim.source.num_freqs = 11

# Monitors — just port names (wavelength info lives on source)
sim.monitors = ["o1", "o2", "o3"]

# Domain
sim.domain.pml = 1.0
sim.domain.margin = 0.5  # auto-extend ports into PML (margin + pml = 1.5um)

# Solver
sim.solver.resolution = 20
sim.solver.stopping = meep.DFTDecay(max_time=200, threshold=1e-3, min_time=100)
sim.solver.simplify_tol = 0.01
sim.solver.save_geometry = True
sim.solver.save_fields = True
sim.solver.save_animation = True
sim.solver.verbose_interval = 5.0

# 3. Validate
result = sim.validate_config()
if not result.valid:
    raise SystemExit(f"Invalid config: {result.errors}")

# 4. Write config (auto-extends waveguide ports into PML and stores original bbox)
output_dir = sim.write_config(Path(__file__).parent / "meep-sim-test")
print(f"Config written to: {output_dir}")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(Path(output_dir) / f)
    print(f"  {f} ({size:,} bytes)")

# 5. Verify component_bbox in generated config
import json

config_data = json.loads((Path(output_dir) / "sim_config.json").read_text())
bbox = config_data.get("component_bbox")
print(f"\ncomponent_bbox (original, before port extension): {bbox}")
