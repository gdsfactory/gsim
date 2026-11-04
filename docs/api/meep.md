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
received at that port imply. gsim cannot correct this from the result file, which holds neither the offsets nor the mode
index. Transmission phases are consistent between runs when the ports share one beta, but do not rely on reflection
phases in circuits sensitive to them, such as resonators.

[`s_matrix`][gsim.meep.s_matrix] warns about this and names the entries affected: each nonzero measured reflection, and
each entry copied with `assume_reciprocal`, in a matrix merged from several runs. A matrix from one run does not warn,
because that run gives each port a single plane.
