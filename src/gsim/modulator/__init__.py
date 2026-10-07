"""gsim.modulator — the electro-optic modulator workflow, end to end.

One device description drives every Stage of a traveling-wave modulator
Study: charge transport through the Phase shifter, the carrier coupling
those Bias points imply, the optical and RF Modes, and the whole-device
figures of merit the line Stage combines them into. Each Stage is
configured through its own callable section, derives its own
Cross-section Window where it meshes at all, and caches its result until
something upstream of it changes.

Backends are optional installs: importing this package never requires
them. The full workflow installs with ``pip install 'gsim[modulator]'``.

One call configures every Stage of the usual lateral PN Phase shifter
over a component and stack you already have::

    from gsim.modulator import pn_phase_shifter

    study = pn_phase_shifter(component=comp, stack=stack, device=device)
    report = study.report()

The preset only writes defaults, so every section stays yours to change.
It never draws anything: the component and the stack are always yours.
``demo_phase_shifter`` draws a rib device when an example or a test needs
one to point at, and is scaffolding for exactly that — not a way to
describe a real device.

Configuring the Stages one by one is the same workflow spelled out::

    from gsim.modulator import Device, Study

    study = Study(
        component=comp,
        stack=stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
        ),
    )
    study.charge(biases=[0.0, -1.0, -2.0])
    sweep = study.charge.run()
    response = study.carriers.run()
    study.optical(wavelength_um=1.55)
    modes = study.optical.run()
    n_group = study.optical.group_index()
    study.rf(frequencies_hz=[10e9, 40e9])
    line_params = study.rf.run()
    study.line(length_um=3000.0)
    report = study.report()
"""

from gsim.modulator.carriers import (
    CarrierResponse,
    CarrierResponseSweep,
    CarriersStage,
    MaterialResponse,
)
from gsim.modulator.charge import ChargeStage
from gsim.modulator.demo import (
    DemoPhaseShifter,
    RibPhaseShifter,
    demo_phase_shifter,
    rib_phase_shifter,
)
from gsim.modulator.device import Device
from gsim.modulator.layout import (
    Contact,
    DeviceLayout,
    Interface,
    Span,
    derive_layout,
)
from gsim.modulator.line import ExportRoundTrip, LineStage
from gsim.modulator.optical import (
    GroupIndex,
    OpticalMode,
    OpticalStage,
    OpticalSweep,
)
from gsim.modulator.palace_route import PalaceMode
from gsim.modulator.preset import pn_phase_shifter
from gsim.modulator.rf import RFStage
from gsim.modulator.route import EMRoute
from gsim.modulator.stage import StageNotRunError
from gsim.modulator.study import Study

__all__ = [
    "CarrierResponse",
    "CarrierResponseSweep",
    "CarriersStage",
    "ChargeStage",
    "Contact",
    "DemoPhaseShifter",
    "Device",
    "DeviceLayout",
    "EMRoute",
    "ExportRoundTrip",
    "GroupIndex",
    "Interface",
    "LineStage",
    "MaterialResponse",
    "OpticalMode",
    "OpticalStage",
    "OpticalSweep",
    "PalaceMode",
    "RFStage",
    "RibPhaseShifter",
    "Span",
    "StageNotRunError",
    "Study",
    "demo_phase_shifter",
    "derive_layout",
    "pn_phase_shifter",
    "rib_phase_shifter",
]
