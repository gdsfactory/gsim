# Shared components guide

Use shared geometry and layer-stack objects to describe components and their fabrication stack across simulation engines.
The links below open the API reference for each object or helper.

## Geometry

- [`Geometry`][gsim.common.Geometry]
- [`GeometryModel`][gsim.common.GeometryModel]
- [`Prism`][gsim.common.Prism]
- [`extract_geometry_model`][gsim.common.extract_geometry_model]

## Stack

- [`LayerStack`][gsim.common.LayerStack]
- [`Layer`][gsim.common.Layer]

## PN Junction

Depletion model after Sze & Ng, *Physics of Semiconductor Devices*, ch. 2,
plus the 1D free-carrier plasma-dispersion model for the complex optical
permittivity and the doping-profile geometry builders.

- [`PNJunctionConfig`][gsim.common.stack.PNJunctionConfig]
- [`make_pn_junction_profile`][gsim.common.stack.make_pn_junction_profile]
- [`make_segmented_junction_profile`][gsim.common.stack.make_segmented_junction_profile]
- [`make_doping_profile`][gsim.common.stack.make_doping_profile]
- [`junction_epsilon_profile`][gsim.common.stack.junction_epsilon_profile]
- [`carrier_profile_1d`][gsim.common.stack.carrier_profile_1d]
- [`epsilon_eff_relative`][gsim.common.stack.epsilon_eff_relative]
- [`optical_params`][gsim.common.stack.optical_params]
- [`refractive_index`][gsim.common.stack.refractive_index]
- [`drude_relaxation_times`][gsim.common.stack.drude_relaxation_times]
- [`built_in_voltage`][gsim.common.stack.built_in_voltage]
- [`depletion_width`][gsim.common.stack.depletion_width]
- [`depletion_extents`][gsim.common.stack.depletion_extents]
- [`junction_capacitance_per_area`][gsim.common.stack.junction_capacitance_per_area]
- [`select_junction_mode`][gsim.common.stack.select_junction_mode]

## Visualization

- [`plot_prisms_3d`][gsim.common.viz.plot_prisms_3d]
- [`plot_prism_slices`][gsim.common.viz.plot_prism_slices]
- [`create_web_export`][gsim.common.viz.create_web_export]
- [`export_3d_mesh`][gsim.common.viz.export_3d_mesh]
