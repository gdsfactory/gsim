# gsim

[![codecov](https://codecov.io/gh/gdsfactory/gsim/branch/main/graph/badge.svg)](https://codecov.io/gh/gdsfactory/gsim)

Photonic FDTD and RF/microwave simulation for GDSFactory, powered by [GDSFactory+](https://gdsfactory.com).

| Engine                            | Status and use                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------- |
| **GDSFactory FDTD** (`gsim.fdtd`) | **Primary.** New photonic FDTD simulations supported by its current capabilities. |
| **Palace** (`gsim.palace`)        | **Established.** FEM simulations for RF and microwave devices.                    |
| **Meep** (`gsim.meep`)            | **Supported.** Photonic workflows that are not yet available in GDSFactory FDTD.  |

## Photonic FDTD

**GDSFactory FDTD (`gsim.fdtd`) is the primary engine for new photonic simulations.** Meep (`gsim.meep`) remains
supported for capabilities not yet available in GDSFactory FDTD.

### Quick start

```python
import gdsfactory as gf

from gsim import fdtd

gf.gpdk.PDK.activate()
component = gf.get_component("mmi1x2")

simulation = fdtd.Simulation()
simulation.materials(background="SiO2", overrides={"si": 3.4757})
simulation.geometry(component)
simulation.source(port="o1", wavelength_span_um=0.1)

result = simulation.run(check_cache=True)
result.plot_plotly()
```

[Start with the MMI tutorial](nbs/fdtd_mmi_gpdk.md) or read the [GDSFactory FDTD guide](api/fdtd.md).

## GDSFactory FDTD roadmap

Over the next six months, we plan to add:

- Material dispersion
- Anisotropic materials
- Periodic boundary conditions
- Gradients for inverse design
- Feature parity and Meep migration

Meep will be deprecated only after GDSFactory FDTD reaches the required feature coverage and a documented migration path
is available.

*Last updated: August 2026.*

## Palace for RF and microwave

`gsim.palace` is the FEM engine for driven RF and microwave simulations, eigenmodes, electrostatics, and impedance
extraction.

### Palace quick start

A coplanar line on the generic PDK, driven through two lumped CPW ports. The mesh is built locally and the solve runs on
GDSFactory+ cloud.

```python
import gdsfactory as gf

from gsim.palace import DrivenSim

gf.gpdk.PDK.activate()


@gf.cell
def gsg_line(length=300, s_width=10, g_width=40, gap_width=6, layer=gf.gpdk.LAYER.M1):
    """Ground-signal-ground line with one port at each end of the signal."""
    c = gf.Component()
    ground_top = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    ground_top.move((0, (g_width + s_width) / 2 + gap_width))
    c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
    ground_bottom = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    ground_bottom.move((0, -(g_width + s_width) / 2 - gap_width))
    for name, x, orientation in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
        c.add_port(name=name, center=(x, 0), width=s_width, orientation=orientation, port_type="electrical", layer=layer)
    return c


sim = DrivenSim()
sim.set_output_dir("./palace-quickstart")
sim.set_geometry(gsg_line())
sim.set_stack(substrate_thickness=2.0)
sim.set_airbox(margin_x=0, margin_y=50.0, z_above=100.0, z_below=100.0)
for port in ("o1", "o2"):
    sim.add_cpw_port(port, layer="metal1", s_width=10, gap_width=6, length=5.0)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)

print(sim.validate_config())
sim.mesh(preset="coarse")

results = sim.run(check_cache=True)  # cloud job; reuses a finished job with identical inputs
results.plot_interactive()
```

`run()` writes `config.json` itself, so `mesh()` is the only step you have to call before it. `preset="coarse"` keeps the quick
start small; use `"default"` or `"fine"` once the geometry is right.

[Start with the Palace CPW tutorial](nbs/palace_cpw_lumped.md) or read the [Palace guide](api/palace.md).

## Installation

```bash
pip install gsim
```

Two optional extras bring in the solvers behind `gsim.tcad` and `gsim.femwell`:

```bash
pip install "gsim[tcad]"     # DEVSIM, for the charge-transport solver
pip install "gsim[femwell]"  # femwell and scikit-fem, for the mode solver
```

## API guides

[GDSFactory FDTD](api/fdtd.md) · [Meep](api/meep.md) · [Palace](api/palace.md) · [TCAD](api/tcad.md) · [Common](api/common.md) ·
[Cloud](api/cloud.md)

## API reference

The API reference is generated from the Python source and docstrings. Browse [`gsim.fdtd`][gsim.fdtd],
[`gsim.meep`][gsim.meep], [`gsim.palace`][gsim.palace], [`gsim.common`][gsim.common], or
[`gsim.gcloud`][gsim.gcloud]. Start with [`fdtd.Simulation`][gsim.fdtd.Simulation],
[`meep.Simulation`][gsim.meep.Simulation], or [`palace.DrivenSim`][gsim.palace.DrivenSim]. API entries link back to
guides that reference them.
