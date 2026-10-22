# GDSFactory FDTD guide

`gsim.fdtd` separates simulation setup into geometry, materials, source,
monitors, domain, and solver concerns. It generates the transfer mesh and
validated `config.json`, submits one source to one cloud job, and returns typed
results.

Use [`Simulation`][gsim.fdtd.Simulation] to configure a run and
[`FDTDResult`][gsim.fdtd.FDTDResult] to inspect its output.

## Configure and run

```python
from gsim import fdtd

sim = fdtd.Simulation(pdk=gpdk)
sim.materials(background="SiO2")
sim.geometry(component=mmi, geometry_tolerance_nm=10)
sim.source(
    port="o1",
    wavelength_um=1.55,
    wavelength_span_um=0.1,
    num_wavelengths=101,
)
sim.domain(padding=0.75, pml_cells=16)
sim.solver(
    cell_size_nm=40,
    energy_decay_fraction=1e-6,
    max_wall_seconds=3600,
)

result = sim.run(check_cache=True)
result.plot()
result.plot_plotly()
```

The transfer mesh preserves the resolved CAD boundaries exactly and uses coarse
tetrahedra in volume interiors. Its default maximum target size is 1 µm;
`mesh_size_nm` remains available as an advanced override. Gmsh conforms to every
explicit component, hole, gap, tooth, and port edge even when it is smaller than
that target. `geometry_tolerance_nm` is a geometry-error bound, not a mesh edge
size.

Transfer meshing is independent of `cell_size_nm`, which requests the actual Yee
grid used by the FDTD solver and defaults to 60 nm. Source sweeps default to 101
wavelengths. Guided-port monitors are implicit; one port source returns one
S-matrix column.

If a PDK uses a virtual routing layer for optical ports, map that GDS layer to
the physical LayerStack entry that owns the guided core:

```python
sim.geometry(component=waveguide, port_layer_map={(100, 0): "core"})
```

The marker layer is not added to the mesh. The mapped physical entry supplies
the port's material and vertical position, and must contain the port location.
Unmapped ports still need to match resolved LayerStack geometry directly.

Linear PDK sidewall angles are represented as continuous ruled solids between
their exact bottom and top contours. If polygon offsetting changes component,
hole, or edge topology, meshing safely falls back to vertical slices bounded by
`geometry_tolerance_nm` instead of constructing an invalid loft.

Use `write(path)` to generate `mesh.msh` and `config.json` without submitting.
The original flat constructor keywords remain accepted for compatibility.

## Material dispersion

By default, each material reaches the solver as its index at the source
wavelength. A broadband run therefore uses that one index at every wavelength.
`write()` and `run()` emit one warning that names every material whose card
varies over the source band, and by how much.

`sim.materials(dispersion=True)` sends those cards as dispersive models instead.
The deployed GDSFactory FDTD solver does not accept them yet, so a run with
`dispersion=True` currently fails on the cloud.

| Material card | `config.json` material entry with `dispersion=True` |
| --- | --- |
| Constant, lossless | `refractive_index` |
| Sellmeier, lossless | `dispersion.sellmeier`, with `wavelength_range_nm` |
| Anything else (tables, equations, loss) | `dispersion.table` of `n` and `k` |

Tables are sampled at 21 evenly spaced wavelengths, plus the knots of a
tabulated card inside the band, so that narrow features are kept. The sampled
range is the source band plus a 10 % margin on each side; a side keeps its
margin only if the card is still valid there. A card that does not cover the
band itself raises `FDTDConfigError`.

A continuous-wave or zero-span source needs only the index at its wavelength,
so every material is written as a constant there. The solver encodes loss only
in a dispersion table, so a lossy card raises unless the source is pulsed and
`dispersion=True`. `sim.materials(overrides={"si": 3.47})` always gives a
constant material.

Domain padding is specified in micrometers, using `padding` as a shared fallback
and `padding_x`, `padding_y`, or `padding_z` to override individual axes. Each
accepts a nonnegative scalar (equal padding on both sides) or a
`(negative_side, positive_side)` pair. Zero padding is allowed.

```python
sim.domain(
    padding=1,
    padding_x=(0, 2),  # -x, +x
    padding_y=0.5,    # both y sides
    padding_z=(1, 2), # below, above
)
```

Explicit axis padding is measured from the resolved geometry bounds. Without
`padding_z`, automatic Z sizing uses the PDK background stack when present,
otherwise the shared padding. The shared fallback retains automatic guided-port
alignment. Explicit axis padding must be zero on a side containing a guided
port: waveguide extensions into padding are not currently supported, and a
nonzero value on that side raises an error. For a bend with an input on the
negative X face and an output on the positive Y face, use
`padding_x=(0, 1)` and `padding_y=(1, 0)` to pad its other sides.

The old `padding_um` domain argument and `background_padding_um` constructor
argument have been removed; configure `sim.domain(padding=...)` instead.

Optional `x_bounds`, `y_bounds`, and `z_bounds` tuples set the non-PML physical
domain in micrometers. Omitted axes remain automatically sized from the
component, PDK stack, and padding. Explicit bounds must contain the geometry,
sources, and monitors; guided ports must remain on a domain face. The backend
adds PML outside these bounds. The resulting box is written to `config.json` as
`grid.non_pml_bounds`, in nanometers.
Do not combine explicit bounds and explicit per-axis padding on the same axis;
the shared padding fallback can still be used for other axes.

```python
sim.domain(
    x_bounds=(0.0, 35.0),
    y_bounds=(-12.0, 12.0),
    z_bounds=(-1.0, 1.55),
)
```

## Setup visualization

FDTD setup views use the ported GDSFactory FDTD Three.js viewer and do not depend on
PyVista. All physical groups are shown by default. Smaller material groups are
listed first, followed by the largest volumetric group and then the ports.

```python
sim.plot_3d()                                   # solid interactive geometry
sim.plot_3d(show_mesh=True)                     # add Gmsh surface edges
sim.plot_3d(zoom_to_cursor=False)               # zoom toward the view center
sim.plot_2d(axis="z", position_um=0.11)         # filled cross-section
sim.plot_2d(axis="x", position_um=0, show_mesh=True)  # add cell edges
```

Omit `position_um` to start at the geometry midpoint. The 2D viewer includes a
slider that moves the plane through the mesh. Both methods return a standalone
`MeshViewer`. Set `show_mesh=True` when element edges are useful, or use
`viewer.save("mesh.html")` to share the visualization outside a notebook.
Zooming moves toward the pointer by default; set `zoom_to_cursor=False` to zoom
toward the view center instead. The same choice is available in the viewer's
**Zoom toward** controls.

The viewer also summarizes the physical domain, estimated Yee-grid dimensions
and total cell count (including PML). These values are estimates; the solver
backend is authoritative when it constructs the final grid.

## Sources

`PortSource` requires an explicit `sim.source(port="...")` selection before
artifacts can be written or a job can run. A guided PDK port maps to an
eigenmode source. Selecting a `vertical_te` or `vertical_tm` PDK port instead
derives a Gaussian beam and fiber monitor from the port metadata.

Replace `sim.source` to use another physical source:

```python
sim.source = fdtd.DipoleSource(
    position_um=(0, 0, 0.11),
    current_axis="z",
    wavelength_um=1.55,
)

sim.source = fdtd.LineCurrentSource(
    position_um=(0, -0.5, 0.11),
    line_axis="y",
    current_axis="z",
    length_um=1,
)
```

`GaussianBeamSource` exposes the aperture center and size, propagation and
polarization vectors, focal point, waist, and optional background index. Only
`PortSource` on a guided port produces normalized S-parameters. Other source
types return outgoing port amplitudes and powers.

## Plane monitors

For notebook diagnostics, add center-wavelength field intensity images with one
call after configuring geometry, source and solver:

```python
sim.add_monitors()               # Optical port images and an XY overview.
sim.add_monitors(xy=False)       # Alternative: optical port images only.
sim.add_monitors(z_um=0.11)      # Alternative: set the XY height in micrometers.
sim.add_monitors(margin_um=1.0)  # Alternative: more cladding around each port.
```

These are alternative calls, not a sequence: monitor names must be unique.
`xy=True` is the default; `z_um=None` chooses the resolved geometry's vertical
midpoint and is ignored when `xy=False`. Each optical port's plane follows its
own width and its owning PDK layer's vertical bounds, including negative layer
thickness. `margin_um=0.75` adds that much cladding on each side of the cross
section; set it to zero for only the active region or increase it for wider field
tails. This is a geometric window, not an automatically mode-converged aperture.
The plane is three grid cells inward along its normal; the XY overview spans the component's
bounding box. All these planes use the source center wavelength, record intensity
without flux, and leave implicit guided-port responses unchanged. They show the
driven field, not independent source eigenmode profiles. Existing custom monitors
are preserved, and name collisions fail before adding any of these planes.
Changing geometry/source/solver later requires recreating the diagnostic planes.
This helper only configures monitors; it does not mesh or submit a job. Explicit
domain bounds still need to contain the requested planes. Dimensional suffixes
follow the existing `center_um`/`position_um` conventions; the domain's `padding`
rename is specific to its directional-padding API.

Additional plane monitors can independently record flux, scalar heatmaps, and
Gaussian fiber overlap:

```python
sim.monitors.add_plane(
    name="top",
    center_um=(0, 0, 1),
    size_um=(10, 6, 0),
    normal="+z",
    flux=True,
    heatmap=fdtd.Heatmap(
        quantity="intensity",
        wavelengths_um=[1.55],
    ),
)
```

For a vertical `PortSource`, request a field image from its implicit port plane:

```python
sim.source(
    port="o2",
    vertical_monitor_heatmap=fdtd.Heatmap(
        quantity="abs_e",
        wavelengths_um=[1.55],
    ),
)
```

Monitor names are unique. Use `add()`, `remove()`, and `clear()` to manage the
collection. A plane's size must be zero along its normal.

## Results

```python
result.s_parameters.plot()
result.s_parameters.plot_plotly()
s_parameter_table = result.s_parameters.to_dataframe()

result.port_outputs.plot(quantity="modal_power")
port_table = result.port_outputs.to_dataframe()

# Net monitor flux can be useful as a diagnostic normalization.
result.plot_plotly(normalize_to="top")

# For a Gaussian source, use its analytic incident power and mask weak tails.
coupling = fdtd.gaussian_coupling_efficiency(
    result, sim.source, port="o1"
)
coupling.plot_plotly()

# An eigenmode-normalized fiber overlap is an amplitude; plot its power.
fiber = fdtd.fiber_coupling_efficiency(result.monitors["fiber"])
fiber.plot_plotly()

top = result.monitors["top"]
top.plot_flux()
top.plot_heatmap(wavelength_um=1.55)
```

Heatmap arrays are loaded lazily from their `.npy` sidecars. S-parameter plots
gap samples flagged below the source noise floor by default. Convergence, grid,
timing, and resolved runtime settings remain available on the result object.

Simulation setup views remain separate from these result plots: `sim.plot_*()`
inspects the submitted geometry and mesh, while `result.plot*()` displays solver
output.

## API reference

- [`Simulation`][gsim.fdtd.Simulation]
- [`PortSource`][gsim.fdtd.PortSource]
- [`DipoleSource`][gsim.fdtd.DipoleSource]
- [`LineCurrentSource`][gsim.fdtd.LineCurrentSource]
- [`GaussianBeamSource`][gsim.fdtd.GaussianBeamSource]
- [`PlaneMonitor`][gsim.fdtd.PlaneMonitor]
- [`FDTDResult`][gsim.fdtd.FDTDResult]
