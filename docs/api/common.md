# Common API

`gsim.common` itself is a narrow façade: it re-exports the geometry and
stack names of the two sections below and nothing else. Everything else in
the package — `modes`, `circuit`, `carriers`, `transmission_line`,
`interpolate`, `mesh_regions` — is imported by its submodule path, as
`from gsim.common.transmission_line import RFLineParams`, so importing
`gsim.common` never pulls a backend's dependencies in with it.

## Geometry

::: gsim.common.Geometry
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.GeometryModel
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.Prism
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.extract_geometry_model
    options:
      show_source: false

## Stack

::: gsim.common.LayerStack
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.Layer
    options:
      show_source: false
      inherited_members: false
      members: false

## PN Junction

Depletion model after Sze & Ng, *Physics of Semiconductor Devices*, ch. 2:
the quick analytic path beside the charge-transport solve, and the
reference its numeric `C(V)` is validated against.

::: gsim.common.stack.PNJunctionConfig
    options:
      show_source: false

::: gsim.common.stack.built_in_voltage
    options:
      show_source: false

::: gsim.common.stack.depletion_width
    options:
      show_source: false

::: gsim.common.stack.depletion_extents
    options:
      show_source: false

::: gsim.common.stack.junction_capacitance_per_area
    options:
      show_source: false

## Visualization

::: gsim.common.viz.plot_prisms_3d
    options:
      show_source: false

::: gsim.common.viz.plot_prism_slices
    options:
      show_source: false

::: gsim.common.viz.create_web_export
    options:
      show_source: false

::: gsim.common.viz.export_3d_mesh
    options:
      show_source: false

## Transmission line

The Traveling-wave electrode as a transmission line, described with no
modulator in sight: what a mode solve says about it, its per-unit-length
circuit, how the junction's series-RC branch loads it, and what a
periodically loaded electrode behaves as. The physics that turns these
into device figures of merit lives in `gsim.modulator`.

::: gsim.common.transmission_line.RFLineParams
    options:
      show_source: false

::: gsim.common.transmission_line.line_params_from_neff
    options:
      show_source: false

::: gsim.common.transmission_line.line_params_from_gamma
    options:
      show_source: false

::: gsim.common.transmission_line.rlgc_from_line_params
    options:
      show_source: false

::: gsim.common.transmission_line.JunctionBranch
    options:
      show_source: false

::: gsim.common.transmission_line.series_rc_from_admittance
    options:
      show_source: false

::: gsim.common.transmission_line.loaded_line_params
    options:
      show_source: false

::: gsim.common.transmission_line.section_abcd
    options:
      show_source: false

::: gsim.common.transmission_line.segmented_period_abcd
    options:
      show_source: false

::: gsim.common.transmission_line.segmented_line_params
    options:
      show_source: false

::: gsim.common.transmission_line.segmented_line
    options:
      show_source: false

::: gsim.common.transmission_line.bragg_fraction
    options:
      show_source: false

## Circuit export

The compact-model handoff to circuit simulators: the Traveling-wave
electrode as a two-port, the junction as a tabulated series-RC model
file, and the plain-numpy readers and driven-line responses that prove
the round trip. The junction model file format is specified in
`write_junction_model`.

::: gsim.common.circuit.line_smatrix
    options:
      show_source: false

::: gsim.common.circuit.write_touchstone
    options:
      show_source: false

::: gsim.common.circuit.read_touchstone
    options:
      show_source: false

::: gsim.common.circuit.sax_line_model
    options:
      show_source: false

::: gsim.common.circuit.write_junction_model
    options:
      show_source: false

::: gsim.common.circuit.read_junction_model
    options:
      show_source: false

::: gsim.common.circuit.terminated_response
    options:
      show_source: false

::: gsim.common.circuit.line_driven_response
    options:
      show_source: false
