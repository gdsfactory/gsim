"""Generate MEEP simulation config for ebeam_y_1550.

Equivalent to the test-meep.ipynb notebook but as a simple script.
Outputs layout.gds, sim_config.json, and run_meep.py to ./meep-sim-test/.
"""

from pathlib import Path

from ubcpdk import PDK, cells

from gsim import meep

PDK.activate()

c = cells.ebeam_y_1550()

sim = meep.Simulation()

sim.geometry.component = c
sim.geometry.z_crop = "auto"

sim.materials = {"si": 3.47, "SiO2": 1.44}

sim.source.port = "o1"
sim.source.wavelength = 1.55
sim.source.bandwidth = 0.01
sim.source.num_freqs = 11

sim.monitors = ["o1", "o2", "o3"]

sim.solver.resolution = 20
sim.solver.stopping = meep.DFTDecay(max_time=200, threshold=1e-3, min_time=100)
sim.solver.simplify_tol = 0.01
sim.solver.save_geometry = True
sim.solver.save_fields = True
sim.solver.save_animation = True
sim.solver.verbose_interval = 5.0

result = sim.validate_config()
if not result.valid:
    raise SystemExit(f"Invalid config: {result.errors}")

output_dir = sim.write_config(Path(__file__).parent / "meep-sim-test")
print(f"Config written to: {output_dir}")
for p in sorted(Path(output_dir).iterdir()):
    print(f"  {p.name} ({p.stat().st_size:,} bytes)")
