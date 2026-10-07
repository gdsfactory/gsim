# Modulator API

`gsim.modulator` is the electro-optic modulator workflow: one device
description drives every Stage of a traveling-wave Study, from charge
transport through the Phase shifter to the whole-device figures of merit.
Backends are optional installs — importing this package never requires one.

The package exports the Study, its Stages, the result types they return and
the presets that configure them. Everything else is imported by its
submodule path, as `from gsim.modulator.staircase import StaircaseCrossSection`.

## The Study

A Study owns the device description and the Cross-section plane, and hands
each Stage its own section. Re-assigning `component`, `stack`, `device` or
`plane` drops the derived layout and every Stage's result.

::: gsim.modulator.Study
    options:
      show_source: false
      inherited_members: false

## Stage sections

Every Stage is configured by calling its section, run by `run()`, and read
back afterwards; a Stage caches its result until something upstream of it
changes (ADR 0001). Configuration and results are namespaced per Stage:

```python
study.charge(biases=[0.0, -1.0, -2.0])
study.carriers(wavelength_um=1.55)
study.optical(wavelength_um=1.55)
study.rf(frequencies_hz=[10e9, 40e9], route="palace")
study.line(length_um=3000.0)
```

A Route is selected by name — `route="femwell"` or `route="palace"` — rather
than by passing a Route object: the Route classes are not exported, so the
name string is the whole selection surface.

::: gsim.modulator.ChargeStage
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.CarriersStage
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.OpticalStage
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.RFStage
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.LineStage
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.StageNotRunError
    options:
      show_source: false

::: gsim.modulator.EMRoute
    options:
      show_source: false

## Device description

::: gsim.modulator.Device
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.Contact
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.Interface
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.Span
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.DeviceLayout
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.derive_layout
    options:
      show_source: false

## Results

::: gsim.modulator.CarrierResponse
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.CarrierResponseSweep
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.MaterialResponse
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.OpticalMode
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.OpticalSweep
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.GroupIndex
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.PalaceMode
    options:
      show_source: false
      inherited_members: false

## Presets

A preset writes defaults into every Stage of the usual lateral PN Phase
shifter and nothing else; each section stays yours to change afterwards. A
preset never draws anything — the component and the stack are always yours.

::: gsim.modulator.pn_phase_shifter
    options:
      show_source: false

::: gsim.modulator.rib_phase_shifter
    options:
      show_source: false

::: gsim.modulator.RibPhaseShifter
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.demo_phase_shifter
    options:
      show_source: false

::: gsim.modulator.DemoPhaseShifter
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.ExportRoundTrip
    options:
      show_source: false
      inherited_members: false

## MZM physics — `gsim.modulator.twmzm`

The traveling-wave MZM transfer function and its bandwidths, on plain arrays
rather than on a Study: the electro-optic and walk-off responses, the drive
range and the chirp. Imported by module path.

::: gsim.modulator.twmzm.mzm_transfer
    options:
      show_source: false

::: gsim.modulator.twmzm.mzm_transfer_figures
    options:
      show_source: false

::: gsim.modulator.twmzm.MZMTransferFigures
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.twmzm.eo_response
    options:
      show_source: false

::: gsim.modulator.twmzm.eo_bandwidth
    options:
      show_source: false

::: gsim.modulator.twmzm.segmented_eo_response
    options:
      show_source: false

::: gsim.modulator.twmzm.walkoff_bandwidth
    options:
      show_source: false

::: gsim.modulator.twmzm.walkoff_bandwidth_dispersive
    options:
      show_source: false

::: gsim.modulator.twmzm.mzm_drive_range
    options:
      show_source: false

::: gsim.modulator.twmzm.mzm_chirp
    options:
      show_source: false

::: gsim.modulator.twmzm.vpi_length_vcm
    options:
      show_source: false

::: gsim.modulator.twmzm.DriveConfiguration
    options:
      show_source: false
      inherited_members: false

## The report — `gsim.modulator.report`

What `Study.report()` returns, and the cross-Route comparison it carries.
Imported by module path.

::: gsim.modulator.report.TWMZMReport
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.report.LoadedLineComparison
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.report.OpticalPhaseSweep
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.report.twmzm_figures_of_merit
    options:
      show_source: false

## The Staircase — `gsim.modulator.staircase`

The Carrier map, averaged into Strips and redrawn as a Cross-section both EM
Stages mesh. Imported by module path and not exported from the package: a
Staircase is what a Stage builds, not something a user assembles.

::: gsim.modulator.staircase.StaircaseCrossSection
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.staircase.CarrierCoupling
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.staircase.ElectrodeSpec
    options:
      show_source: false
      inherited_members: false

::: gsim.modulator.staircase.ConductorModel
    options:
      show_source: false
