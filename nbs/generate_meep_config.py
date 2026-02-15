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
# Domain: 0.5um margins, 1um PML, auto-extend ports into PML (margin_xy + dpml = 1.5um)
sim.set_domain(0.5)
sim.set_z_crop()
sim.set_material("si", refractive_index=3.47)
sim.set_material("SiO2", refractive_index=1.44)
sim.set_wavelength(wavelength=1.55, bandwidth=0.01, num_freqs=11)
sim.set_source()  # auto: fwidth ~3x monitor bw
sim.set_stopping(mode="dft_decay", max_time=200, threshold=1e-3, dft_min_run_time=100)
sim.set_resolution(pixels_per_um=20)
sim.set_accuracy(
    simplify_tol=0.01,
    eps_averaging=True,
    verbose_interval=5.0,
)
sim.set_diagnostics(save_geometry=True, save_fields=True, save_animation=True)
sim.set_output_dir(Path(__file__).parent / "meep-sim-test")

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
