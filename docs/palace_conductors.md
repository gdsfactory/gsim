# Conductor models in Palace

This page says what gsim writes into `config.json` for a metal layer, and when each choice is a reasonable model. It describes gsim.
What Palace does with a key once it reads it is described in Palace's own configuration reference, which this page does not
reproduce. Statements that were not checked are marked *not verified*, and the list is repeated at the end.

Every number below comes from the gpdk CPW line in the snippet on this page (coarse preset), from the gsim source, or from the
tests named in the text. Nothing on this page was solved: the snippet stops at `write_config()`.

## What gsim writes for a conductor

### Three models

| Model | A layer gets it when | `config.json` gets |
| --- | --- | --- |
| Surface impedance on a shell (default) | It is a conductor layer, 0.02 um thick or more | `Boundaries.Conductivity` entries |
| PEC sheet | `planar_conductors=True`, or the conductor layer is under 0.02 um | `Boundaries.PEC` attributes |
| Lossy volume | It is a via, or a polygon dielectric with a conductivity | `Domains.Materials` with `Conductivity` |

What is meshed and written:

- **Shell.** The metal volume is removed and its faces are kept, so there are no elements inside the metal. Each layer gets one entry for
  its top and bottom faces and one for its side walls. Each entry has `Conductivity`, `Thickness` and `Crack`.
- **PEC sheet.** A zero-thickness surface. It has no `Conductivity` entry and no `Thickness`.
- **Lossy volume.** Elements fill the volume. Doped silicon is the example of a polygon dielectric with a conductivity.

Details that follow from the code:

- A via layer with no conductivity is meshed as a PEC sheet, and gsim logs a warning. A via thinner than 0.02 um is a PEC sheet too.
- A conductor layer is never meshed as a volume, whatever its conductivity. See
  [Meshing a conductor as a lossy volume](#meshing-a-conductor-as-a-lossy-volume).
- `add_pec(gds_layer=, from_layer=, to_layer=)` adds a PEC block, a drawn polygon extruded between two stack layers and treated as a
  PEC boundary (its docstring). It is separate from the layer models above.
- `planar_conductors=True` is refused for a magnetostatic simulation, whose current-source sheets must end on metal volumes. The
  magnetostatic solver treats conductors as perfect conductors anyway; see the [Palace guide](api/palace.md).

### See it on a line

This builds the gpdk CPW line twice, once with the defaults and once with `planar_conductors=True`, and prints the conductor keys of
each `config.json`. It stops at `write_config()`.

```python
import json
from pathlib import Path

import gdsfactory as gf
from gsim.palace import DrivenSim

gf.gpdk.PDK.activate()


@gf.cell
def gsg_line(length: float = 300, s_width: float = 10, g_width: float = 40, gap_width: float = 6) -> gf.Component:
    layer = gf.gpdk.LAYER.M1
    c = gf.Component()
    top = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    top.move((0, (g_width + s_width) / 2 + gap_width))
    c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
    bottom = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    bottom.move((0, -(g_width + s_width) / 2 - gap_width))
    for name, x, orientation in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
        c.add_port(name=name, center=(x, 0), width=s_width, orientation=orientation, port_type="electrical", layer=layer)
    return c


def boundaries(output_dir: str, **mesh_options) -> dict:
    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    sim.set_geometry(gsg_line())
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=50.0, z_above=100.0, z_below=100.0)
    sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
    sim.mesh(preset="coarse", **mesh_options)
    config_path = sim.write_config()  # nothing is solved or submitted
    return json.loads(Path(config_path).read_text())["Boundaries"]


shell = boundaries("conductor-shell")
sheet = boundaries("conductor-sheet", planar_conductors=True)
print(shell["Conductivity"])
print(sheet["Conductivity"], sheet["PEC"])
```

On this line (gsim 0.9.0, coarse preset, one Windows machine):

| | Default | `planar_conductors=True` |
| --- | --- | --- |
| `Boundaries.Conductivity` | Two entries, `Conductivity` 3.77e7, `Thickness` 0.7, `Crack` true | An empty list |
| `Boundaries.PEC` | Absent | One entry with one attribute list |
| Tetrahedra | 7,897 | 6,074 |

The two default entries are the shell's top and bottom faces and its side walls. The second column shows that the sheet needs fewer
elements; it does not show that the answer is the same.

## Conductivity and Thickness

For each shell entry gsim writes:

- `Conductivity`: the conductivity of the layer's material in the stack. The stack of the gpdk `metal1` layer calls it `Aluminum`, with
  3.77e7 S/m (gsim's `aluminum` entry; its `copper` entry is 5.8e7 S/m). If the material has no conductivity, the code falls back to
  5.8e7 S/m without a warning.
- `Thickness`: the layer's `zmax - zmin` in um, here 0.7 for `metal1` (z = 1.1 to 1.8 um). It is the same number that sets the extrusion
  height before the volume is removed, so there is no separate knob for it; change the stack layer.
- `Crack`: see the next section.

To change the conductivity, use `sim.set_material(name, material_type="conductor", conductivity=...)` with the name exactly as the stack
spells it. In the check behind this page, `set_material("Aluminum", conductivity=1e7)` changed both entries to 1e7, and
`set_material("aluminum", conductivity=1e7)` left them at 3.77e7 without a message. Read the spelling from
`sim.stack.layers["metal1"].material` after `mesh()`.

What Palace does with `Thickness` on a cracked boundary, and whether the conductivity enters through a finite-thickness impedance or the
semi-infinite one, is *not verified* here. Read Palace's description of the `Conductivity` boundary for the runtime you use.

## Palace 0.19 and `Crack`

gsim's own comments and tests say this; Palace's release notes were not read:

- Palace 0.19 reads an interior `Boundaries.Conductivity` boundary as a thin film unless it is cracked.
- gsim meshes a 3D conductor as a shell with the dielectric inside it, so every such boundary is interior. It writes `"Crack": true` on
  these entries for driven and eigenmode problems, so they stay thick conductors.
- A native-2D conductor (a `BoundaryModeSim` cross-section) is a hole in the mesh, so its boundary is exterior and gets no key.
  Tests: `tests/palace/test_conductor_crack.py`.
- Palace before 0.19 rejects the key and always cracks. `run_local()` therefore hands such a runtime a copy of the configuration without
  `Crack`, when it knows the version. `config.json` on disk keeps the key, so an upload or a later run on a newer Palace still has it.
  The local auto-downloaded runtime is Palace 0.18.0, see the [Palace guide](api/palace.md).
- Which Palace the cloud runs is not given as a release number in these docs (the [Palace guide](api/palace.md) names a commit), so
  this page does not say whether the cloud reads `Crack`. gsim sends `config.json` as written.

## Wave ports and the conductor

With wave ports, the default `waveport_boundary="pec"` lists the conductor attributes, together with the absorbing and impedance
attributes, under `Boundaries.WavePortPEC`. In the port's 2D mode solve the metal is then a perfect conductor, while the 3D problem keeps
its `Conductivity` boundaries. `waveport_boundary="inherit"` writes no `WavePortPEC`, so the port mode sees the same surface impedance.
Test: `test_wave_port_case_with_waveport_pec`. Details and the outer-wall rule are in the [Palace guide](api/palace.md).

For a `BoundaryModeSim`, `sim.conductor_boundary = "pec"` moves the conductor attributes from `Boundaries.Conductivity` to
`Boundaries.PEC`, so that the 2D modes are comparable with the port modes above. The default, `"conductivity"`, keeps them lossy.

## When to use a PEC sheet

- The metal is superconducting, or its loss does not matter for the question.
- A fast check of the geometry, ports and mesh. A PEC sheet has no elements inside the metal and no `Thickness`, so it is cheaper (7,897
  against 6,074 tetrahedra above).
- The layer is thinner than 0.02 um. gsim makes it a PEC sheet on its own.

The planar qubit-chip notebook (`palace_qpdk_filled_qubit_test_chip`) is built this way: its text says "planar PEC metal on silicon" and it
meshes with `planar_conductors=True`. A PEC sheet has no conductor loss at all, so it cannot be used to read conductor loss.

## Meshing a conductor as a lossy volume

Main has no public switch for this. In the code, `Domains.Materials` gets a `Conductivity` only for vias and for polygon dielectrics, and a
conductor layer goes through the shell or the sheet path whatever its material.

One public-API attempt was made for this page: redeclare `metal1` as a `"via"` layer in a `LayerStack` and pass it to `set_stack()`. For
the CPW above, `mesh()` then stops with `Mesh validation failed: No conductor surfaces in mesh`. So this is not a route.

Work on a route is in [gsim PR #522](https://github.com/gdsfactory/gsim/pull/522), a branch that is not on main. Until it lands, a
meshed-metal reference for a Palace conductor is not available from main alone.

## Validity versus skin depth

Rule of thumb: a surface-impedance condition describes a conductor whose thickness t is much larger than the skin depth delta (a
textbook statement, not something measured here). The skin depth is:

```text
delta = sqrt(2 / (omega * mu0 * sigma)) = 1 / sqrt(pi * f * mu0 * sigma)
```

Skin depth in um, with sigma = 3.77e7 S/m (aluminum, gpdk `metal1`) and 5.8e7 S/m (copper). The last column is the thickness of gpdk
`metal1`, 0.7 um, divided by the aluminum skin depth:

| Frequency (GHz) | delta, Al (um) | delta, Cu (um) | t / delta, Al, t = 0.7 um |
| --- | --- | --- | --- |
| 1 | 2.59 | 2.09 | 0.270 |
| 10 | 0.820 | 0.661 | 0.854 |
| 100 | 0.259 | 0.209 | 2.70 |

These values are computed with `scipy.constants.mu_0` and checked against this table by `tests/palace/test_conductor_docs.py`.

Reading the table:

- The 0.7 um `metal1` is as thick as the aluminum skin depth (t = delta) at about 13.7 GHz. Below that, the layer is thinner than delta;
  a thickness much larger than delta needs a frequency well above it.
- The same applies to any stack: compare `Thickness` with delta at the highest and the lowest frequency of the sweep, before trusting a
  conductor-loss number from the shell model.
- The de-embedding notebook applies this to its own case: it models 3 um conductors with sigma = 3.77e7 S/m, notes a skin depth of about
  2.6 um at 1 GHz (the table gives 2.59 um), and fits only from 5 GHz up
  ([CPW length sweep and de-embedding](nbs/palace_cpw_deembedding.md)).

How large the error of the surface condition is for a given t / delta is not shown on this page. No comparison with a meshed-metal run
is stored in the repository, and main cannot make one (see the previous section), so no number is given.

## Not verified

- Anything Palace does with these keys: no solver was run for this page. In particular what `Thickness` does on a cracked boundary, and
  whether the cloud runtime reads `Crack`.
- The claim that Palace 0.19 reads an interior `Conductivity` boundary as a thin film comes from gsim's code comments and tests.
- The check that `set_material` is case-sensitive was made on the gpdk stack only.
- The `"via"` re-declaration was tried on one CPW with lumped ports.
- The numbers in the line table were produced on one Windows machine with the coarse preset; the tetrahedron counts depend on the mesher.
