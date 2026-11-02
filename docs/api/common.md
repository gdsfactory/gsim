# Shared components guide

Use shared geometry and layer-stack objects to describe components and their fabrication stack across simulation engines.
The links below open the API reference for each object or helper.

`gsim.common` itself is a narrow façade: it re-exports the geometry and
stack names of the two sections below and nothing else. Everything else in
the package — `modes`, `circuit`, `carriers`, `junction`,
`transmission_line`, `twmzm`, `staircase`, `interpolate`, `mesh_regions` —
is imported by its submodule path, as
`from gsim.common.transmission_line import RFLineParams`, so importing
`gsim.common` never pulls a backend's dependencies in with it.

## Geometry

- [`Geometry`][gsim.common.Geometry]
- [`GeometryModel`][gsim.common.GeometryModel]
- [`Prism`][gsim.common.Prism]
- [`extract_geometry_model`][gsim.common.extract_geometry_model]

## Stack

- [`LayerStack`][gsim.common.LayerStack]
- [`Layer`][gsim.common.Layer]

## PN Junction

Depletion model after Sze & Ng, *Physics of Semiconductor Devices*, ch. 2:
the quick analytic path beside the charge-transport solve, and the
reference its numeric `C(V)` is validated against.

- [`PNJunctionConfig`][gsim.common.stack.PNJunctionConfig]
- [`built_in_voltage`][gsim.common.stack.built_in_voltage]
- [`depletion_width`][gsim.common.stack.depletion_width]
- [`depletion_extents`][gsim.common.stack.depletion_extents]
- [`junction_capacitance_per_area`][gsim.common.stack.junction_capacitance_per_area]

## Visualization

- [`plot_prisms_3d`][gsim.common.viz.plot_prisms_3d]
- [`plot_prism_slices`][gsim.common.viz.plot_prism_slices]
- [`create_web_export`][gsim.common.viz.create_web_export]
- [`export_3d_mesh`][gsim.common.viz.export_3d_mesh]

## Transmission line

The Traveling-wave electrode as a transmission line: what a mode solve says
about it, its per-unit-length circuit, how the junction's series-RC branch
loads it, and what a periodically loaded electrode behaves as. The
[TW-MZM notebook](../nbs/tcad_twmzm.md) turns these into device figures of
merit.

- [`RFLineParams`][gsim.common.transmission_line.RFLineParams]
- [`line_params_from_neff`][gsim.common.transmission_line.line_params_from_neff]
- [`line_params_from_gamma`][gsim.common.transmission_line.line_params_from_gamma]
- [`rlgc_from_line_params`][gsim.common.transmission_line.rlgc_from_line_params]
- [`JunctionBranch`][gsim.common.junction.JunctionBranch]
- [`series_rc_from_admittance`][gsim.common.junction.series_rc_from_admittance]
- [`loaded_line_params`][gsim.common.transmission_line.loaded_line_params]
- [`section_abcd`][gsim.common.transmission_line.section_abcd]
- [`segmented_period_abcd`][gsim.common.transmission_line.segmented_period_abcd]
- [`segmented_line_params`][gsim.common.transmission_line.segmented_line_params]
- [`segmented_line`][gsim.common.transmission_line.segmented_line]
- [`bragg_fraction`][gsim.common.transmission_line.bragg_fraction]

## Circuit export

The compact-model handoff to circuit simulators: the Traveling-wave
electrode as a two-port, the junction as a tabulated series-RC model
file, and the plain-numpy readers and driven-line responses that prove
the round trip. The junction model file format is specified in
`write_junction_model`.

- [`line_smatrix`][gsim.common.circuit.line_smatrix]
- [`write_touchstone`][gsim.common.circuit.write_touchstone]
- [`read_touchstone`][gsim.common.circuit.read_touchstone]
- [`sax_line_model`][gsim.common.circuit.sax_line_model]
- [`write_junction_model`][gsim.common.circuit.write_junction_model]
- [`read_junction_model`][gsim.common.circuit.read_junction_model]
- [`terminated_response`][gsim.common.circuit.terminated_response]
- [`line_driven_response`][gsim.common.circuit.line_driven_response]

## Traveling-wave MZM figures of merit

The traveling-wave Mach-Zehnder modulator on plain arrays: the electro-optic and walk-off responses, $V_\pi L$, the
static transfer, drive range and chirp (`gsim.common.twmzm`), and the whole-device report wired from RF line parameters
and an optical bias sweep (`gsim.common.twmzm_report`). The [TW-MZM notebook](../nbs/tcad_twmzm.md) uses both.

- [`eo_response`][gsim.common.twmzm.eo_response]
- [`eo_bandwidth`][gsim.common.twmzm.eo_bandwidth]
- [`segmented_eo_response`][gsim.common.twmzm.segmented_eo_response]
- [`walkoff_bandwidth`][gsim.common.twmzm.walkoff_bandwidth]
- [`walkoff_bandwidth_dispersive`][gsim.common.twmzm.walkoff_bandwidth_dispersive]
- [`vpi_length_vcm`][gsim.common.twmzm.vpi_length_vcm]
- [`mzm_transfer`][gsim.common.twmzm.mzm_transfer]
- [`mzm_transfer_figures`][gsim.common.twmzm.mzm_transfer_figures]
- [`MZMTransferFigures`][gsim.common.twmzm.MZMTransferFigures]
- [`mzm_drive_range`][gsim.common.twmzm.mzm_drive_range]
- [`mzm_chirp`][gsim.common.twmzm.mzm_chirp]
- [`twmzm_figures_of_merit`][gsim.common.twmzm_report.twmzm_figures_of_merit]
- [`TWMZMReport`][gsim.common.twmzm_report.TWMZMReport]
- [`OpticalPhaseSweep`][gsim.common.twmzm_report.OpticalPhaseSweep]
- [`LoadedLineComparison`][gsim.common.twmzm_report.LoadedLineComparison]

## Staircase cross-section

A carrier map averaged into piecewise-constant strips and redrawn as a meshable cross-section, the representation a
mode solver with one material per region can take. Each strip carries the carrier coupling's optical permittivity or RF
conductivity.

- [`build_staircase_cross_section`][gsim.common.staircase.build_staircase_cross_section]
- [`StaircaseCrossSection`][gsim.common.staircase.StaircaseCrossSection]
- [`CarrierCoupling`][gsim.common.staircase.CarrierCoupling]
- [`OpticalStripMaterial`][gsim.common.staircase.OpticalStripMaterial]
- [`RFStripMaterial`][gsim.common.staircase.RFStripMaterial]
- [`ElectrodeSpec`][gsim.common.staircase.ElectrodeSpec]
- [`surroundings_from_section`][gsim.common.staircase.surroundings_from_section]
- [`staircase_profile`][gsim.common.staircase.staircase_profile]
