"""Interactive 2D FDTD preview of a taper_sc_nc component."""

import gdsfactory as gf

from gsim import meep
from gsim.common.stack import get_stack

gf.gpdk.PDK.activate()

c = gf.components.taper_sc_nc()
stack = get_stack()

sim = meep.Simulation()
sim.geometry(component=c, stack=stack)
sim.materials = {"si": 3.47, "SiO2": 1.44}
sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
sim.monitors = ["o1", "o2"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=25, is_3d=False)
sim.solver.stop_when_energy_decayed()

sim.validate_config()

fig = sim.plot_2d_interactive()
fig.show()
