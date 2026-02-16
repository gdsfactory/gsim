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
sim.geometry.component = component
sim.geometry.stack = None  # resolved from active PDK
sim.geometry.z_crop = "auto"

sim.materials = {"si": 3.47, "SiO2": 1.44}

sim.source = meep.ModeSource()  # auto: fwidth ~3x monitor bw, auto port

sim.monitors = [
    meep.ModeMonitor(port="o1", wavelength=1.55, bandwidth=0.01, num_freqs=11),
    meep.ModeMonitor(port="o2", wavelength=1.55, bandwidth=0.01, num_freqs=11),
    meep.ModeMonitor(port="o3", wavelength=1.55, bandwidth=0.01, num_freqs=11),
]

# Domain: 0.5um margins, 1um PML, auto-extend ports into PML (margin + pml = 1.5um)
sim.domain = meep.Domain(pml=1.0, margin=0.5)

sim.solver = meep.FDTD(
    resolution=20,
    stopping=meep.DFTDecay(max_time=200, threshold=1e-3, min_time=100),
    simplify_tol=0.01,
)

sim.diagnostics = meep.Diagnostics(
    save_geometry=True, save_fields=True, save_animation=True, verbose_interval=5.0
)

sim.output_dir = Path(__file__).parent / "meep-sim-test"

# 3. Validate
result = sim.validate_config()
if not result.valid:
    raise SystemExit(f"Invalid config: {result.errors}")

# 4. Write config (auto-extends waveguide ports into PML and stores original bbox)
output_dir = sim.write_config()
print(f"Config written to: {output_dir}")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(Path(output_dir) / f)
    print(f"  {f} ({size:,} bytes)")

# 5. Verify component_bbox in generated config
import json

config_data = json.loads((Path(output_dir) / "sim_config.json").read_text())
bbox = config_data.get("component_bbox")
print(f"\ncomponent_bbox (original, before port extension): {bbox}")
