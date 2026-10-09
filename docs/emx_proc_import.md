# Import an EMX process file

Some foundries deliver their RF back-end stack only as an EMX process file (`.proc`). `load_emx_proc` reads the nominal
stack from such a file and returns a gsim `LayerStack`, so the file can be used without a PDK package:

```python
from gsim.common.stack.emx import load_emx_proc
from gsim.palace import DrivenSim

stack = load_emx_proc("my_process.proc")

sim = DrivenSim()
sim.set_geometry(component)
sim.set_stack(stack)
sim.set_airbox(margin_x=50, margin_y=50, z_above=100)
sim.add_cpw_port("o1", layer="M3", s_width=10, gap_width=6)
```

The substrate of an RF process is often several hundred um thick. Use `substrate_thickness_um` to simulate a thinner (or
thicker) wafer than the one declared in the file:

```python
stack = load_emx_proc("my_process.proc", substrate_thickness_um=100)
```

## What is imported

| In the `.proc` file                                      | In the gsim stack                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------ |
| `define NAME = expr` (numbers, `+ - * /`, other defines) | Evaluated; a table-valued define contributes its first (reference) entry |
| `layer <thickness> <er> [name N]`                        | A region of `stack.dielectrics` with permittivity `er`                   |
| `layer <thickness> <er> [name N] <rho> ohm-cm`           | The substrate region, with conductivity `100 / rho` S/m                  |
| `layer infinity <er>`                                    | Dropped; use `set_airbox()` for the surrounding air                      |
| `conductor <thickness> <rs or value S/m> NAME`           | A `Layer` of type `conductor` named `NAME` (upper case)                  |
| `offset <value>` before a conductor                      | Shifts that conductor from the bottom of the preceding layer             |
| `define NAME = l<L>t<D>`                                 | The GDS layer `(L, D)` of the conductor or via `NAME`                    |
| `via FROM TO { ... <value> S/m ... } NAME`               | A `Layer` of type `via` between conductors `FROM` and `TO`               |

Conventions of the resulting stack:

- Lengths are in um. Layers are read **from the bottom to the top** of the file. `z = 0` is the top of the substrate, so
  the substrate occupies `[-thickness, 0]`.
- An embedded conductor starts at the bottom of the layer that precedes it in the file, plus its optional offset. It
  does not move the layer interface.
- A via spans from the top of the lower conductor it connects to the bottom of the upper one. The connectivity is stored
  in `stack.simulation["emx"]["via_connectivity"]`, for example `{"V1": ["M1", "M2"]}`.
- A conductor given as a sheet resistance `Rs` (ohm/sq) and thickness `t` gets the conductivity `1 / (Rs * t)` in S/m,
  with `t` in m. A conductor given in S/m keeps its nominal value.
- Material names start with `emx_`. Names that match the built-in gsim material database would otherwise be replaced by
  the database values.

## What is ignored

Only the nominal stack is imported. For each of the following features that appears in the file, an `EmxImportWarning`
is emitted and the feature is skipped:

- geometry bias,
- fill and slotting (optional fill streams are not mapped),
- via merge operations,
- temperature dependence (only the part of a conductivity expression before the first `*` is used, evaluated at the
  reference temperature).

Other limits:

- A sheet resistance that depends on width and spacing is reduced to the first table entry.
- A conductor or via drawn on several GDS streams uses the first stream. A conductor or via without a direct GDS stream
  is left out, with a warning.
- gsim matches shapes to stack layers by GDS layer number only. Two layers that share a layer number with different
  datatypes therefore see the same shapes; this is reported as a warning.
- Dielectric layers have no loss tangent in the file, so none is set. Add one to the material entry if needed, for
  example `stack.materials["emx_dielectric_er_3p9"]["loss_tangent"] = 0.001`.
- The file must list layers from the bottom to the top. If the substrate is not the first layer, a warning points at a
  possible top-to-bottom listing.
- Layer and conductor thicknesses and permittivities must be single tokens (a number or the name of a `define`), not
  expressions with spaces. An `offset` line may contain a spaced expression.

## Portable JSON

`parse_emx_proc(path)` returns the parsed process as a plain dictionary (schema `portable-em-stackup-v1`) that can be
saved with `json.dump`. `load_portable_stackup_json(path)` builds the same stack from such a file.
