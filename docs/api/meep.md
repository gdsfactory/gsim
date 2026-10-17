# Meep guide

Use [`gsim.meep`][gsim.meep] to configure photonic FDTD simulations and inspect their results.

## Simulation

- [`Simulation`][gsim.meep.Simulation]

## Configuration

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
