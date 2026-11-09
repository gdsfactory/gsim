# Meep guide

Use [`gsim.meep`][gsim.meep] to configure photonic FDTD simulations and inspect their results.

## Simulation

- [`Simulation`][gsim.meep.Simulation]

## Configuration

For overlapping physical layers, lower `mesh_order` values take precedence
in XY, XZ and 3D. The runner emits those objects last to follow Meep's overlap
convention. Equal priorities retain input order, so the last object wins ties.
Legacy entries without `mesh_order` retain their relative order and are emitted
before explicitly ranked entries. Configurations containing only legacy entries
therefore keep their previous ordering.

- [`Geometry`][gsim.meep.Geometry]
- [`Domain`][gsim.meep.Domain]

`x_bounds`, `y_bounds`, and `z_bounds` set exact PML-inner intervals in
absolute micrometers; PML is added outside them. Leave an axis as `"auto"`
to size it from the component bounding box and that axis's margin:

```python
sim.domain(
    pml=1.0,
    margin_x=1.0,
    y_bounds=(-4.0, 4.0),
    z_bounds=(0.0, 3.0),
)
```

Do not combine an explicit bound with `margin_x` or `margin_y` on the same
axis. In XZ 2D simulations Y is collapsed, so `y_bounds` is not active.

- [`ModeSource`][gsim.meep.ModeSource]

## 3D port placement

Meep automatically places each waveguide source and monitor at the vertical
center of its port's fabrication layer. Its vertical span is that layer's
thickness plus `domain.port_margin` on both sides. Resolution happens before
simulation layers are remapped, so mixed-layer components can use independent
Si and SiN port planes.

Use `port_overrides` for ambiguous fabrication tuples, virtual ports, or
intentional custom mode-plane placement:

```python
sim.domain(z_bounds=(0, 3))
sim.port_overrides = {
    "o1": 1.535,  # shorthand for {"z": 1.535}
    "o2": {"z": 1.535, "z_span": 1.39},
}
```

Override fields take precedence individually; omitted fields remain inferred.
Every inferred or overridden mode plane must fit inside `domain.z_bounds`.
Overrides do not restore a physically drawn layer excluded by those bounds.
They are supported in 3D and XZ simulations, but not in top-down XY 2D where Z
is collapsed. For mixed-height devices, set explicit bounds that contain every
port layer; automatic Z cropping currently follows one optical reference layer.

- [`PortVerticalOverride`][gsim.meep.PortVerticalOverride]

## Results

- [`SParameterResult`][gsim.meep.SParameterResult]
- [`s_matrix`][gsim.meep.s_matrix]

### Export to SAX and Touchstone

[`s_matrix`][gsim.meep.s_matrix] builds an [`SMatrix`][gsim.common.s_matrix.SMatrix] from one result per excited port, in
the same way as the [GDSFactory FDTD export](fdtd.md#export-to-sax-and-touchstone), which describes the assumptions,
conventions and limits.

```python
from gsim.meep import s_matrix

matrix = s_matrix([result_o1, result_o2], port_names=["o1", "o2"])
model = matrix.sax_model()
matrix.write_touchstone("device")
```

Meep's result file numbers its ports, so pass `port_names` in port order, or the entries are named `port_1`, `port_2`.
Meep uses exp(-iωt), and the values are conjugated so that `matrix.s` and Touchstone files hold exp(+jωt). Fiber-source
results (`S10`) are normalized to launched flux, not to a port mode, so they are refused.

Meep's monitors sit inward from each port center, by `source_port_offset` at non-source ports and by
`source_port_offset + distance_source_to_monitors` at the source port. So in a matrix merged from several runs, the
diagonal and off-diagonal entries are referenced to different planes: a reflection's phase differs by
2·beta·`distance_source_to_monitors`, with beta the mode's propagation constant, from what the transmission entries
received at that port imply. This follows from the monitor positions and has not been measured. gsim cannot correct it
from the result file, which holds neither the offsets nor the mode index. Transmission phases are consistent between runs
when the ports share one beta (S21 and S12 agree to about 2e-4 rad on a real 10 µm straight waveguide), but do not rely on
reflection phases in circuits sensitive to them, such as resonators.

[`s_matrix`][gsim.meep.s_matrix] warns about this on every matrix merged from several runs, because the result file
cannot show whether the ports share one beta. The warning names each nonzero measured reflection and each entry copied
with `assume_reciprocal`. A matrix from one run does not warn, because that run gives each port a single plane.

Two more things are unchecked on that real straight waveguide. Its delay is 3.3% longer than a mode solver's group index
predicts over the 9.6 µm between monitors, and one length cannot tell whether this comes from the group index or the
monitor positions. Ports of different widths have not been run.

Samples that repeat a wavelength, from `wavelength_span=0` or a span finer than the result file's 6 decimals, become one
if their values agree, and are refused otherwise. Export supports at most 9 ports, because Meep's entry names are
ambiguous from 10 ports up: `S111` may be port 1 from 11 or port 11 from 1.
