#!/usr/bin/env python3
"""Generate MEEP simulation config for ebeam_y_1550.

Equivalent to the test-meep.ipynb notebook but as a simple script.
Outputs layout.gds, sim_config.json, and run_meep.py to ./meep-sim-test/.
"""

import os
from pathlib import Path

import gdsfactory as gf
from ubcpdk import PDK, cells

from gsim.meep import MeepSim

PDK.activate()

# 1. Create component
component = cells.ebeam_y_1550()

# 2. Configure simulation
sim = MeepSim()
sim.set_geometry(component)
sim.set_stack()
sim.set_domain(0.5)
sim.set_z_crop()
sim.set_material("si", refractive_index=3.47)
sim.set_material("SiO2", refractive_index=1.44)
sim.set_wavelength(
    wavelength=1.55,
    bandwidth=0.1,
    num_freqs=11,
    run_after_sources=100,
    stop_when_decayed=True,
    decay_threshold=1e-3,
)
sim.set_resolution(pixels_per_um=20)
sim.set_symmetry(y=-1)
sim.set_output_dir("./meep-sim-test")

# 3. Validate
result = sim.validate_config()
if not result.valid:
    raise SystemExit(f"Invalid config: {result.errors}")

# 4. Write config
output_dir = sim.write_config()
print(f"Config written to: {output_dir}")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(Path(output_dir) / f)
    print(f"  {f} ({size:,} bytes)")
