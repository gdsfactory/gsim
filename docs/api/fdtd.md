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

### Virtual port layers

When a guided port uses a virtual routing marker, gsim can find its physical
core through the PDK cross-section. Pass a cross-section to `add_port()` to
retain its complete section description on the port:

```python
import gdsfactory as gf

routing = gf.cross_section.cross_section(
    width=0.5,
    layer=(100, 0),  # Virtual port marker.
    sections=(gf.cross_section.Section(width=0.5, layer=(1, 0)),),
)
waveguide.add_port(
    name="o1", center=(0, 0), orientation=180, cross_section=routing
)
sim.geometry(component=waveguide)
```

Here the physical core is drawn on `(1, 0)` and must already have a
material-bearing entry in the PDK LayerStack. The marker is not extruded.
gsim reads gdsfactory's saved cross-section snapshot or the port's named
`cross_section` metadata. Without either, it checks default-instantiable
factories in `PDK.cross_sections`. Only visible sections with the same width
and offset as the primary marker section identify the core; wider cladding
and offset sections do not. The actual resolved geometry at each port decides
which vertical bands are present, so a local etch can give two ports different
heights. Material identity is supplied by the LayerStack.

An inferred core must form one abutting, non-overlapping group of a single
material. Equivalent registry aliases are accepted; multiple different cores
or incompatible bands raise `UnsupportedPortError` instead of selecting one.
Unmapped vertical optical ports retain their free-space semantics.

Use `port_layer_map` when a cross-section does not describe an unambiguous core,
or to select particular physical bands explicitly:

```python
sim.geometry(component=waveguide, port_layer_map={(100, 0): "core"})
```

The marker layer is not added to the mesh. The mapped physical entry supplies
the port's material and vertical position, and must contain the port location.
Explicit maps take precedence, followed by existing direct physical-layer
matches. Cross-section inference is a fallback when direct matching fails.
When an unmapped port matches abutting vertical bands of the same material, its
mesh surface includes those bands and its center uses their combined height.
Overlapping entries are not joined; candidates are considered in mesh-priority
order. The saved port `layer` is the primary material and priority reference;
the port's mesh physical group defines its complete cross-section.
An explicit single-entry mapping keeps the port limited to that entry.

For a core represented by several physical bands, select their LayerStack keys
as a tuple:

```python
sim.geometry(component=waveguide, port_layer_map={(100, 0): ("lower", "upper")})
```

All selected bands must have resolved geometry at the port location, use the
same material, and abut without overlaps or gaps. Empty selections, duplicate
keys, and incompatible bands raise `UnsupportedPortError`. The first key stays
the primary material/priority reference; the mesh covers the complete selected
height. Existing string mappings retain their single-band behavior.

### Bounded PDK stacks

Keep the PDK's LayerStack unchanged and specify the physical window directly.
Opt in to trimming material geometry with `clip_geometry=True`:

```python
sim = fdtd.Simulation(pdk=pdk)
sim.domain(z_bounds=(-0.5, 1.0), clip_geometry=True)
sim.geometry(component=waveguide)
```

Set the bounds before resolving geometry. Films wholly outside the vertical
window and footprints outside the XY window are omitted before their materials
are required. Sidewall overhangs are retained conservatively. This includes
films with negative thickness; boundary-only contact has no material volume.
Intersecting layers keep their original sidewall profiles and are clipped by
the mesher; the PDK object is not modified. Missing materials inside the window
still raise an error. `GeometryClippingWarning` identifies layers and axes
where explicit bounds trim or wholly omit material, including truncated
blanket films. Repeated resolution of the same component/window does not
repeat this warning. Empty masks and bounds containing all material do not
warn. Source containment and guided-port face checks still apply.

#### Migration for 0.10.0

`clip_geometry` is deprecated starting with 0.10.0, but remains functional
through that release. Its default remains `False`: explicit bounds must
contain the geometry unless `clip_geometry=True` is supplied. Both values
remain supported in saved domain settings and the JSON schema. Explicit domain
configuration and assignments emit a `DeprecationWarning` attributed to the
calling code. Dumps omit an unspecified switch, so ordinary model round trips,
writes, and previews do not warn about a switch the user never set.

The removal target is **0.12.0 or later**, and only after at least two feature
releases and 12 months since the first released warning in 0.10.0. If those
conditions are not met in 0.12.0, removal and automatic clipping move to a later
incompatible release. Until then, keep the flag wherever clipping is required.
Cloud cache lookup uses generated input files; changing
the physical window can change the mesh, configuration, and cache identity.

Levels marked `background=True` fill the physical simulation domain in XY,
including lateral padding, without requiring a drawn source polygon. Their
`background_exclude_layers` masks still remove material. Patterned levels follow
their masks. Every blanket film intersecting the chosen z interval is retained
and clipped to that interval; films wholly outside it are omitted. Preview and
written Gmsh geometry use the same domain. Automatic padding is applied once,
and neither the component nor the PDK stack is modified.

Display metadata such as `info.visible=False` only changes initial viewer
visibility. Hidden films remain in the simulation and can be enabled for
inspection.

Port centers and diagnostic windows follow the clipped active bands. Automatic
diagnostic planes, including their cladding margins, are intersected with the
physical domain when previewed or written. The requested monitor windows remain
available if the domain is enlarged; the original LayerStack extrusion and
sidewall fields are retained.

Linear PDK sidewall angles are represented as continuous ruled solids between
their exact bottom and top contours. If polygon offsetting changes component,
hole, or edge topology, meshing safely falls back to vertical slices bounded by
`geometry_tolerance_nm` instead of constructing an invalid loft.

Use `write(path)` to generate `mesh.msh` and `config.json` without submitting.
The original flat constructor keywords remain accepted for compatibility.

## Material dispersion

MaterialCards attached to the PDK object or exported by its module are discovered
automatically and take precedence over built-in cards. Schema Lorentz and Drude
models are evaluated using their angular frequencies in rad/s and the passive
`n+ik` convention. Their loss and wavelength dependence follow the same solver
rules as other cards below.

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
component, PDK stack, and padding. With `clip_geometry=True`, explicit bounds
trim material geometry and must contain sources; guided ports must remain on
a domain face. Plane monitor
windows are intersected with the domain and warn if cropped. The backend
adds PML outside these bounds. The resulting box is written to `config.json` as
`grid.non_pml_bounds`, in nanometers.
Do not combine explicit bounds and explicit per-axis padding on the same axis;
the shared padding fallback can still be used for other axes.

```python
sim.domain(
    x_bounds=(0.0, 35.0),
    y_bounds=(-12.0, 12.0),
    z_bounds=(-1.0, 1.55),
    clip_geometry=True,
)
```

## Background default in 0.10.0

The 0.10.0 feature release changes the default simulation background from
`"SiO2"` to `"Air"`, an ideal lossless, nondispersive material with refractive
index 1. This intentional change does not target a 0.9.x patch release.

Stack regions continue to use their own material cards and take priority over
the background. The background fills space not occupied by those regions;
it does not replace BOX, substrate, cladding, or patterned materials. A domain
fully filled by stack regions retains those physical material assignments.

Existing calls that omit a background can change behavior in uncovered space.
Source and receiver references that inherit the background can also change.
Automatic Z sizing can differ because it follows matching background stack
levels when available. Unused Air declarations (such as an undrawn undercut)
do not enlarge the automatic domain. `Air` and its built-in `air` alias use
the same bounds. Explicit domain bounds remain in force.

For example, a bare gpdk silicon component with no drawn oxide simulates silicon
in air when the background is omitted, once its material names resolve.
GDSFactory 9.x uses the lowercase `si` name; with those releases, supply
`sim.materials(overrides={"si": 3.47})` when the PDK has no card for it.
Declaring oxide in the LayerStack does
not create oxide geometry. Select `background="SiO2"` to fill otherwise
unoccupied space with oxide.

To preserve the former environment, add `sim.materials(background="SiO2")`,
or retain `background_material="SiO2"` in the historical constructor API.
Saved settings with an explicit background are unchanged; settings omitting it
use the new default when loaded. Set `z_bounds` or `padding_z` explicitly if
preserving an automatically chosen vertical extent matters. Re-exporting
updates inputs and cache keys; previously generated inputs and results are
unchanged. The existing FDTD notebooks already choose their backgrounds
explicitly.

## Setup visualization

`plot_2d_static()` produces a static Matplotlib cross-section, like the Meep setup view.
Materials are colored by their real refractive index at the source center
wavelength, with a colorbar, physical-domain boundary, hatched PML regions, and source
and monitor annotations where they intersect the slice. The plot uses the FDTD
transfer mesh, including sidewalls and material priorities. All physical groups
are shown by default; `show_groups` and `hide_groups` select physical group names.

```python
ax = sim.plot_2d_static()                               # XY at geometry midpoint
ax = sim.plot_2d_static(axis="z", position_um=0.11)      # XY at z = 0.11 µm
ax = sim.plot_2d_static(axis="x", position_um=0, show_mesh=True)  # YZ with mesh edges
ax.figure.savefig("cross_section.png", dpi=200, bbox_inches="tight")
```

The method returns a Matplotlib `Axes`. Pass `ax` to draw into an existing
subplot, `cmap` to change the index colormap, `legend=False` to hide the legend,
`aspect="auto"` to fill the axes, or `figsize=(8, 4)` to set the figure size.
An omitted `position_um` uses the geometry midpoint along the requested axis.
If explicit bounds clip material on that axis, it uses the bounded domain
midpoint instead; bounds containing the geometry keep the geometry midpoint.
The current geometry and material settings are resolved for every call. A
temporary mesh is generated; the plot does not require selecting a source port
or writing run artifacts. A selected source port must exist, and sources and
monitors must fit any explicit domain bounds, as required by `write()`. The PML
thickness uses the same per-axis alignment as the metadata estimate: z PML is
rounded up to a multiple of 16 cells (for example, 20 becomes 32). Guard cells
are not drawn as PML. Mesh edges describe the tetrahedral transfer mesh,
rather than the solver's Yee grid.

The interactive setup views use the GDSFactory FDTD Three.js viewer and do not
depend on PyVista. Smaller material groups are listed first, followed by the
largest volumetric group and then the ports.

```python
sim.plot_3d()                                   # solid interactive geometry
sim.plot_3d(show_mesh=True)                     # add Gmsh surface edges
sim.plot_3d(zoom_to_cursor=False)               # zoom toward the view center
sim.plot_2d_interactive(axis="z", position_um=0.11)  # movable cross-section
viewer = sim.plot_2d_interactive(show_mesh=True)     # add transfer-mesh edges
viewer.save("mesh.html")
```

The interactive 2D viewer includes a slider that moves the plane through the
mesh. `plot_2d_interactive()` and `plot_3d()` return a standalone `MeshViewer`.
Each call resolves the current settings and generates a temporary mesh, so edits
to geometry, meshing, background material, and domain settings appear immediately.
Previously written simulation artifacts are preserved.
The 3D viewer uses PDK layer-view fill colors, falling back to frame colors when
the fill is absent. Stack `info["color"]` overrides the layer-view color. Colors
can be CSS color strings; their alpha channel does not change surface opacity.
An exactly named layer view matching the stack key, or its declared name,
is preferred over a physical-layer lookup. Otherwise the derived layer view
is preferred, with the source layer view as a fallback. A drawing layer reused
by background films of distinct materials does not supply an individual film
color: those films use explicit stack/named colors or the fallback palette.
Visibility and opacity still follow their existing PDK defaults.
For duplicate colored views of a GDS layer, the first effectively visible view
is preferred, then the first colored view. Missing colors retain the viewer's
fallback palette, including its port and largest-group colors. These display
choices do not change the mesh or solver configuration.

The viewer also uses each PDK stack entry's `info["opacity"]` and layer-view
visibility by default, including hidden parent groups. If several layer views
refer to the same GDS layer, it is initially visible when any of those views
and its enclosing groups are visible, regardless of view order. Stack `info["visible"]`
overrides layer-view visibility. Only stack entries with resolved geometry have
their display metadata applied and validated. A zero opacity starts a layer hidden;
enabling it in the viewer's layer list makes it fully opaque (`opacity=1.0`),
which can obscure enclosed layers. Explicit `show_groups`
and `hide_groups` override these initial visibility defaults; `hide_groups`
takes precedence. Entries without display metadata retain the viewer defaults.
Zooming moves toward the pointer by default; set `zoom_to_cursor=False` to zoom
toward the view center instead. The same choice is available in the viewer's
**Zoom toward** controls.

The viewer also summarizes the physical domain, estimated Yee-grid dimensions
and total cell count (including PML). Grid counts use the same Python calculation
as `metadata.json`, preserving its rounding and per-axis alignment. These values are estimates; the solver
backend is authoritative when it constructs the final grid.

### 2D plotting APIs and compatibility (0.9.0)

Use `plot_2d_static()` for a Matplotlib `Axes` and export it with
`ax.figure.savefig(...)`. Use `plot_2d_interactive()` for an HTML `MeshViewer`
with `.html` and `.save()`.

The original `plot_2d()` remains an alias for the interactive viewer, accepts
all original keywords including `height` and `zoom_to_cursor`, and still returns
`MeshViewer`. Starting with 0.9.0 it emits a `DeprecationWarning` directing callers
to the explicit methods. The alias will remain for at least two feature releases
and roughly 12 months after the first released warning. This change does not
switch its default behavior or remove the old API.

## Sources

`PortSource` requires an explicit `sim.source(port="...")` selection before
artifacts can be written or a job can run. A guided PDK port maps to an
eigenmode source. Selecting a `vertical_te` or `vertical_tm` PDK port instead
derives a Gaussian beam and fiber monitor from the port metadata.

To excite a specific guided mode, set its one-based `mode_order`:

```python
sim.source(port="o1", mode={"kind": "mode_order", "mode_order": 2})
```

The selector is also available as `fdtd.ModeSelection`. `te_like` and `tm_like`
select the first candidate with the requested polarization; those selectors
require a transverse `te_axis` and can set `num_candidate_modes` and
`min_polarization_fraction`. With no selector, the fundamental mode is used.
The implicit guided-port monitors use the same selection as the source.
`te_axis` is a global coordinate axis and must be transverse to every guided
port. Components with both x-normal and y-normal ports, such as a 90-degree
bend, require `te_axis="z"`. Vertical fiber monitors do not use this selector.

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

FDTD input exports include a versioned monitor-memory estimate in
`metadata.json` under `memory.estimated_monitor_memory_bytes`. The estimate
covers explicit plane monitors, automatic guided-port monitors, spectral field
accumulators, reference profiles, retained port-mode data, and sample buffers.
It uses conservative pixel bounds and excludes the main grid, material/PML
state, and temporary eigensolver workspace. More wavelength points increase
spectral storage even when no extra monitor is added.

Cloud preparation sends this byte estimate alongside the unchanged Yee-cell
count as `sizing_requirements.fdtd.estimated_monitor_memory_bytes`. Resource
selection remains server-owned. Older input directories without the estimate
retain their cell-count-only sizing request. Diagnostic metadata remains
excluded from the simulation input hash.

For notebook diagnostics, add center-wavelength field intensity images with one
call after configuring geometry, source and solver:

```python
sim.add_monitors(xy_name="field_xy")                # Optical port images and an XY field image.
sim.add_monitors(xy=False)                          # Alternative: optical port images only.
sim.add_monitors(xy_name="field_xy", z_um=0.11)      # Alternative: set the XY height in micrometers.
sim.add_monitors(xy_name="field_xy", margin_um=1.0)  # Alternative: more cladding around each port.
```

These are alternative calls, not a sequence: monitor names must be unique.
`xy=True` is the default; `z_um=None` chooses the resolved geometry's vertical
midpoint within any opted-in crop and is ignored when `xy=False`. Each optical
port's plane follows its
own width and its selected physical bands' active vertical bounds, including
negative layer thickness and any opted-in geometry crop. `margin_um=0.75` adds
that much cladding on each side of the cross section; set it to zero for only
the active region or increase it for wider field
tails. This is a geometric window, not an automatically mode-converged aperture.
The plane is three grid cells inward along its normal. Port images are named
`port_<port_name>` (for example, `port_o1`); the `field_xy` image spans the component's
bounding box at a constant Z height. All these planes use the source center wavelength, record intensity
without flux, and leave implicit guided-port responses unchanged. They show the
driven field, not independent source eigenmode profiles. Existing custom monitors
are preserved, and name collisions fail before adding any of these planes.
Changing geometry/source/solver later requires recreating the diagnostic planes.
This helper only configures monitors; it does not mesh or submit a job. Dimensional suffixes
follow the existing `center_um`/`position_um` conventions; the domain's `padding`
rename is specific to its directional-padding API.

During `write()` or `run()`, every explicitly configured plane monitor is
intersected with the physical, non-PML domain, whether the domain is automatic
or explicitly bounded. This applies to XY, YZ, and XZ planes, including port
images and custom heatmap, flux, and fiber-overlap monitors. Each cropped
monitor emits one `UserWarning` identifying its name and the original and cropped
extent on every changed axis, in micrometers. Flux and fiber-overlap warnings
also note that the measurement aperture changes. The physical domain and the
requested `sim.monitors` settings remain unchanged; the submitted `config.json`
contains the cropped bounds, and results report the actual sampled extents.
`plot_2d_static()` shows the same cropped monitor windows and warnings.

A plane whose normal position is outside the physical domain, whose window is
disjoint from it, or whose cropping collapses a requested nonzero extent raises
`FDTDConfigError` before meshing or submission. Planes on the domain boundary
and already contained zero-width line or point probes retain their existing
behavior. Sources are never cropped.

### Field-monitor migration (0.9.0 onward)

Starting in 0.9.0, use `sim.add_monitors(xy_name="field_xy")` to choose the clearer
name for the device-wide intensity image. Existing `sim.add_monitors()` calls
still generate `overview_xy`, preserving monitor iteration, serialized settings,
`config.json`, raw result keys, and cache inputs for contained planes. Omitting
`xy_name` when `xy=True` emits a `DeprecationWarning` announcing the new name.
The default will remain `overview_xy` for at least two feature releases and
roughly 12 months after the first released warning. The default change targets
0.11.0 or a later feature release, whichever first satisfies both conditions. Set
`xy_name="overview_xy"` explicitly to retain the old generated name indefinitely.
`xy_name` is ignored when `xy=False`, which does not issue a naming warning.

When opting into `field_xy`, update monitor-name comparisons and code reading
raw configuration or solver JSON to use that name. Existing
`sim.monitors["overview_xy"]`, `sim.monitors.remove("overview_xy")`, and
`result.monitors["overview_xy"]` lookups remain supported as compatibility aliases;
`field_xy` lookups also work on saved results using `overview_xy`. Exact names
take precedence if both names were explicitly configured. Iteration and
serialization retain the stored name without duplicate images. The old lookup
name remains supported, with no removal planned.

Earlier versions checked monitor containment only on explicitly bounded axes:
partially outside monitors either failed locally or reached the solver and
could fail there. The 0.9.0 behavior crops overlapping windows with a warning
and rejects unusable planes locally, avoiding wasted cloud setup. To retain a
requested measurement aperture, enlarge the physical domain where allowed by
guided-port constraints; otherwise reduce the window or move the plane into
the domain. Rewriting artifacts applies these checks to existing configurations.

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

### Fiber reference medium

When `FiberMode.refractive_index` is omitted, the exporter determines the real
index at the source center wavelength from the material covering the receiver's
cropped aperture. It checks the exported tetrahedral mesh, including sidewalls,
holes, overlapping material priorities, and both sides of the plane. A receiver
outside stack volumes uses the simulation background. Different materials with
the same real index can share a reference medium.

Set `FiberMode(refractive_index=...)` to choose the reference medium explicitly.
For an implicit receiver derived from a vertical port, use
`sim.source(vertical_refractive_index=...)`. The selected vertical port's Gaussian
beam and its receiver use the same index. Explicit `GaussianBeamSource` settings
retain their existing behavior.

An aperture spanning different indices, lying on an interface, or lacking a
usable area cannot determine one homogeneous Gaussian reference. The exporter
warns and retains the background index for compatibility; set the relevant
explicit index to define the intended reference.

This numerical correctness fix targets the next patch release after 0.9.0.
Previously, an omitted fiber index always used the global background, even for
a receiver embedded in a different stack material. Such setups can produce
different coupling values after this fix, while their material assignments stay
unchanged. To preserve the previous reference, supply the background's index
explicitly through `FiberMode.refractive_index` or
`PortSource.vertical_refractive_index`. Re-exporting updates solver inputs and
their cache keys; existing downloaded results are unchanged. The simulation
background default is unchanged by this fix.

### Complex-field export

Request complex plane fields with `Fields`. Omit `fields` to retain the
existing output, or pass `Fields()` to save all six E/H components at the
monitor's wavelengths. Components are case-insensitive; explicit selections
must be nonempty and unique. The field request can select its own wavelengths
in micrometers and can be used without flux, heatmaps, or a fiber mode.

```python
sim.monitors.add_plane(
    name="capture",
    center_um=(5, 0, 1),
    size_um=(10, 8, 0),
    normal="+z",
    flux=False,
    fields=fdtd.Fields(
        components=["ex", "ey", "hx", "hy"],
        wavelengths_um=[1.54, 1.55, 1.56],
    ),
)
result = sim.run(check_cache=True)
monitor = result.monitors["capture"]
sample = monitor.field("ey", wavelength_um=1.55)
ey = sample.load()  # Complex NumPy array, shape (v, u).
u_um = sample.coordinates.u_um
v_um = sample.coordinates.v_um
sample.metadata["normalization"]  # "raw_dft" for GDSFactory FDTD output.
```

`monitor.fields` contains the exported component/frequency records. `field()`
selects the closest exported wavelength for the requested component; inspect
the returned `wavelength_um`. Real and imaginary sidecars are loaded lazily,
with shape, dtype, and finite-value checks. A missing file raises an error
when loaded. Actual sample coordinates and the snapped normal position are
available through `monitor.field_coordinates` or `sample.coordinates`, in
micrometers. They can differ from the requested monitor bounds and position.

The plane's `monitor.shape` is `(n_u, n_v)`. Complex-field and heatmap arrays
are `(n_v, n_u)`: rows follow `v_um`, columns follow `u_um`. For example, a
plane with three U samples and two V samples reports `(3, 2)` and loads
arrays of shape `(2, 3)`.

Loading preserves the exported complex values without source normalization,
noise masking, or phase corrections. `sample.metadata` retains the backend's
normalization, DFT convention, phase reference, colocation, field-unit system,
solver time unit, and boundary-interpolation metadata. Raw DFT fields are not
incident-power-normalized fields or fiber coupling efficiencies. Quantitative
overlap postprocessing requires consistent normalization and sampling.

`dft_convention="exp(-i*omega*t)"` describes the Fourier **integration kernel**:
the solver accumulates `sum(field(t) * exp(-i*omega*t) * sample_dt)`.
The corresponding phasor has time dependence **exp(+iωt)**, so a forward
wave's phase decreases along its propagation direction. This agrees with
`result.s_parameters`; the loader does not conjugate either result. Electric
and magnetic fields use their actual Yee sample times before spatial colocation.
Missing or unsupported DFT conventions are rejected; legacy results without
complex-field entries remain supported.

A [recorded straight-waveguide run](https://github.com/gdsfactory/gsim/tree/main/tests/fdtd/data/plane_fields)
checks both conventions on real output. Its `(25, 32)` arrays retain the forward
phase decrease, and the independent `0.5 * Re(E × conj(H))` integral agrees with
the native plane flux within 1.2%. Exact equality is not expected because the
exported fields are colocated while the native flux retains transverse Yee
staggering. The sample includes the result JSON and paired arrays.

Selecting fewer components or wavelengths reduces the saved field payload.
Existing result files without complex fields continue to load normally.

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

### Export to SAX and Touchstone

One run excites one port, so it measures one column of the S-matrix. Run a source at each port and combine the results
with [`s_matrix`][gsim.fdtd.s_matrix]. SAX is not a gsim dependency and is installed separately with `pip install sax`;
`s_matrix` and `write_touchstone` do not need it.

```python
import sax

result_o1 = sim.run(check_cache=True)
sim.source(port="o2", wavelength_um=1.55, wavelength_span_um=0.1, num_wavelengths=101)
result_o2 = sim.run(check_cache=True)

matrix = fdtd.s_matrix([result_o1, result_o2])

model = matrix.sax_model()
circuit, _ = sax.circuit(
    netlist={
        "instances": {"arm": "dut", "delay": "straight"},
        "connections": {"arm,o2": "delay,in0"},
        "ports": {"in": "arm,o1", "out": "delay,out0"},
    },
    models={"dut": model, "straight": sax.models.straight},
)
transmission = circuit(wl=1.55)["in", "out"]

matrix.write_touchstone("device")  # device.s2p
```

With fewer runs than ports, [`s_matrix`][gsim.fdtd.s_matrix] raises and lists the entries no run measured. SAX reads an
absent entry as zero, so you state each assumption explicitly:

```python
matrix = fdtd.s_matrix(
    result_o1,
    assume_reciprocal=True,  # S(o1,o2) copied from the measured S(o2,o1)
    assume_unmeasured_zero=True,  # what is left, here S(o2,o2), set to zero
)
matrix.measured  # False for every assumed entry
```

Runs must share their ports and wavelengths. Samples below the source noise floor are dropped, with a warning. This
often affects a band-edge sample, which narrows the band the model accepts.

The model takes `wl` in micrometers and covers the simulated band only: a wavelength outside it raises. It interpolates
magnitude and unwrapped phase, and unwrapping needs phase to change by less than pi between wavelengths, which means a
wavelength step below λ²/(2·n_g·L). Stay below 3/4 of that step, about 2.6 nm for 100 µm at n_g = 3.4 and 1.55 µm:
[`s_matrix`][gsim.fdtd.s_matrix] warns from 3/4 pi, for each measured entry with at least a tenth of the strongest peak.
It cannot catch every undersampled grid, because a delay can alias to a smaller one with small steps, so choose the step
from the longest path in your device. The model computes with numpy, so it works in `sax.circuit` but cannot be traced by `jax.jit`.

The two exports use different conventions:

| Where | Entry for transmission from `o1` to `o2` | Time convention |
| --- | --- | --- |
| `result.s_parameters`, `matrix.s`, Touchstone | `S(o2,o1)` (receive, source) | exp(+jωt), phase falls with length |
| `sax_model()` | `("o1", "o2")` (in, out) | exp(-iωt), phase rises with length, as `sax.models.straight` |

`sax.parsers.parse_touchstone` keeps the phase sign of the file and reads `s[i, j]` as (in, out), so a file written here
does not compose correctly with SAX's photonic models that way. Use `sax_model()` to reach SAX.

**Time convention of the solver.** gsim takes the solver's values as exp(+jωt). The solver does not state this, so it was
checked on a real run of a 10 µm silicon straight waveguide: phase falls with frequency at every sample, with a delay of
139 fs against 137 fs from the group index of a separate mode solver. That is one device, so
[`s_matrix`][gsim.fdtd.s_matrix] still warns if the strongest transmission has a
negative group delay. Real anomalous dispersion, such as inside a ring notch, can also trigger this warning. A
transmission whose phase does not move across the band gives no sign to judge. On a grid too
coarse to unwrap phase, this check can stay silent even when the sign is wrong, so silence is not proof.

Phases are referenced where the solver's port monitors sit, and gsim does not de-embed them. The `z0` argument of
`write_touchstone` only fills the file header, because the S-parameters are normalized per mode.

## API reference

- [`Simulation`][gsim.fdtd.Simulation]
- [`PortSource`][gsim.fdtd.PortSource]
- [`DipoleSource`][gsim.fdtd.DipoleSource]
- [`LineCurrentSource`][gsim.fdtd.LineCurrentSource]
- [`GaussianBeamSource`][gsim.fdtd.GaussianBeamSource]
- [`PlaneMonitor`][gsim.fdtd.PlaneMonitor]
- [`Fields`][gsim.fdtd.Fields]
- [`ComplexFieldResult`][gsim.fdtd.ComplexFieldResult]
- [`FieldCoordinates`][gsim.fdtd.FieldCoordinates]
- [`FDTDResult`][gsim.fdtd.FDTDResult]
- [`s_matrix`][gsim.fdtd.s_matrix]
