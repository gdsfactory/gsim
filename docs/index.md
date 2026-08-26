# gsim

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
simulation.materials(background="SiO2")
simulation.geometry(component)
simulation.source(port="o1", wavelength_span_um=0.1)

result = simulation.run(check_cache=True)
result.plot_plotly()
```

[Start with the MMI tutorial](nbs/fdtd_mmi_gpdk.md) or browse the [GDSFactory FDTD API](api/fdtd.md).

## GDSFactory FDTD roadmap

Over the next six months, we plan to add:

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

[Start with the Palace CPW tutorial](nbs/palace_cpw_lumped.md) or browse the [Palace API](api/palace.md).

## Installation

```bash
pip install gsim
```

## API reference

[GDSFactory FDTD](api/fdtd.md) · [Meep](api/meep.md) · [Palace](api/palace.md) · [Common](api/common.md) ·
[Cloud](api/cloud.md)
