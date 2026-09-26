# Palace Ports

Palace uses two solver-level port families:

- **Lumped ports** integrate the electric field across one or more prescribed surfaces and apply a circuit impedance.
- **Wave ports** solve a modal field on a simulation-domain boundary.

The lumped-port names in gsim describe conductor topology rather than a fixed global coordinate plane. Port orientation
and layer placement determine whether the generated surface lies in XY, XZ, or YZ.

## Lumped-port geometries

| Geometry     | Use                                       | Surface                                              |
| ------------ | ----------------------------------------- | ---------------------------------------------------- |
| `inplane`    | Two conductors on one layer               | Horizontal rectangle on that layer                   |
| `gap`        | A direct connection across a coplanar gap | Vertical sheet through the conductor thickness       |
| `interlayer` | Conductors on different layers            | Vertical sheet spanning the layers                   |
| `cpw`        | Signal with two coplanar grounds          | Two horizontal gap elements with opposite directions |

All four are surface definitions. They specify where Palace integrates the field; they do not add a physical conductor.

### In-plane

Use `inplane` when one rectangular surface on a conductor layer describes the desired lumped excitation:

```python
sim.add_port(
    "feed",
    layer="topmetal2",
    length=5.0,
    impedance=50.0,
    geometry="inplane",
)
```

The gdsfactory port width sets one rectangle dimension and `length` sets the other. Its orientation determines the
integration direction.

### Gap

Use `gap` to place one vertical sheet directly between two coplanar terminals:

```python
sim.add_port(
    "Pdiff",
    layer="topmetal2",
    impedance=50.0,
    geometry="gap",
)
```

The gdsfactory port must be centered in the gap. Its width spans the gap along its cardinal orientation, and the
generated sheet extends through the selected conductor layer's thickness. Omit `length`.

This geometry can define a floating terminal-to-terminal one-port measurement; it does not require a global ground plane
when the sheet touches both conductors.

### Interlayer

Use `interlayer` when the two conductors are on different layers, such as a microstrip signal referenced to a lower
ground plane:

```python
sim.add_port(
    "feed",
    from_layer="metal1",
    to_layer="topmetal2",
    impedance=50.0,
    geometry="interlayer",
)
```

The gdsfactory port width sets the lateral size. The generated sheet spans the vertical space between the named layers.

The former name `via` is a deprecated alias. Existing code remains functional and emits a `DeprecationWarning`:

```python
sim.add_port(
    "feed",
    from_layer="metal1",
    to_layer="topmetal2",
    geometry="via",  # Deprecated: use "interlayer".
)
```

The lower-level `configure_via_port()` helper is likewise retained as a warning alias for `configure_interlayer_port()`.

### CPW

Use the dedicated CPW method when a coplanar signal is referenced to ground on both sides:

```python
sim.add_cpw_port(
    "feed",
    layer="topmetal2",
    s_width=10.0,
    gap_width=6.0,
    length=5.0,
    impedance=50.0,
)
```

gsim constructs two gap elements and assigns opposite integration directions so they form one Palace lumped port. CPW
has its own method because signal width and gap width are required to construct both surfaces.

## Choosing a geometry

- Choose `gap` for a direct floating measurement between coplanar terminals.
- Choose `interlayer` when signal and reference are on different layers.
- Choose `cpw` when one signal uses two coplanar return gaps.
- Choose `inplane` for other single-surface, same-layer lumped excitations.
- Choose a wave port when the port lies on the domain boundary and modal solving or de-embedding is important.

A lumped-port surface must connect the two conductors that define its voltage. The S-parameter reference impedance
(typically 50 Ω) is a normalization and circuit loading value; it does not mean the reference conductor must be global
ground.
