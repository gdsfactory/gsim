# Conductor models in Palace

This page says what gsim writes into `config.json` for a metal layer, and when each choice is a reasonable model. It describes gsim.
What Palace does with a key once it reads it is described in Palace's own configuration reference, which this page does not
reproduce. Statements that were not checked are marked *not verified*, and the list is repeated at the end.

Every number below comes from the gpdk CPW line in the snippets on this page (coarse preset), from the gsim source, or from the
tests named in the text. The snippets stop at `write_config()`. The only solved results are the cloud runs in
[Surface model versus volume metal on a CPW](#surface-model-versus-volume-metal-on-a-cpw).

## What gsim writes for a conductor

### Three models

| Model | A layer gets it when | `config.json` gets |
| --- | --- | --- |
| Surface impedance on a shell (default) | It is a conductor layer, 0.02 um thick or more | `Boundaries.Conductivity` entries |
| PEC sheet | `planar_conductors=True`, or the conductor layer is under 0.02 um | `Boundaries.PEC` attributes |
| Lossy volume | It is a via, or a polygon dielectric with a conductivity | `Domains.Materials` with `Conductivity` |

What is meshed and written:

- **Shell.** The metal volume is removed and its faces are kept as surfaces. The faces are not the wall of a hole: on the line below
  (coarse preset) 340 tetrahedra have their centroid inside the signal strip, all 1,074 triangles of the top and bottom faces have a
  tetrahedron on both sides, and so do 364 of the 400 triangles of the side walls. Each layer gets one entry for its top and bottom
  faces and one for its side walls. Each entry has `Conductivity`, `Thickness` and `Crack`.
- **PEC sheet.** A zero-thickness surface with elements on both sides (all 542 triangles of the sheet on that line). It has no
  `Conductivity` entry and no `Thickness`.
- **Lossy volume.** Elements fill the volume. Doped silicon is the example of a polygon dielectric with a conductivity.

Details that follow from the code:

- A via layer with no conductivity is meshed as a PEC sheet, and gsim logs a warning. A via thinner than 0.02 um is a PEC sheet too.
- A layer classified as a conductor is never meshed as a volume, whatever its conductivity. A drawn metal layer classified as a dielectric
  is; see [Meshing a conductor as a lossy volume](#meshing-a-conductor-as-a-lossy-volume).
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

- The loss of the metal does not matter for the question, and neither do its reactive effects. Negligible dissipation, or
  superconductivity, does not mean that the surface, internal or kinetic inductance of the metal is negligible: a metal surface has a
  reactive surface impedance, and a superconducting film adds a kinetic inductance that grows as the film gets thin (textbook statements,
  not measured here). A PEC sheet has none of these. Use it only if the phase, the impedance or the resonant frequency you want does not
  depend on them. For a superconducting layer, see
  [Superconducting layers](#superconducting-layers-london-penetration-depth).
- A fast check of the geometry, ports and mesh. A PEC sheet has no `Thickness` and no tetrahedra inside the metal, so it is cheaper (7,897
  against 6,074 tetrahedra above).
- The layer is thinner than 0.02 um. gsim makes it a PEC sheet on its own.

The planar qubit-chip notebook (`palace_qpdk_filled_qubit_test_chip`) is built this way: its text says "planar PEC metal on silicon" and it
meshes with `planar_conductors=True`. Whether neglecting the reactive effects is justified for the frequencies that notebook reads is not
checked here. A PEC sheet has no conductor loss at all, so it cannot be used to read conductor loss.

## Meshing a conductor as a lossy volume

Main has no dedicated switch for this. In the code, `Domains.Materials` gets a `Conductivity` for vias and for polygon dielectrics, and a
layer classified as a conductor goes through the shell or the sheet path whatever its material. A meshed-metal reference can still be
built through the public API by classifying the drawn metal layer as a dielectric:

```python
from gsim.common.stack import get_stack

stack = get_stack(substrate_thickness=2.0)  # the stack that set_stack(substrate_thickness=2.0) builds above
stack.layers["metal1"].layer_type = "dielectric"  # the material stays Aluminum, 3.77e7 S/m

sim = DrivenSim()
sim.set_output_dir("conductor-volume")
sim.set_geometry(gsg_line())
sim.set_stack(stack)  # a LayerStack object, not substrate_thickness=
sim.set_airbox(margin_x=0, margin_y=50.0, z_above=100.0, z_below=100.0)
sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
sim.mesh(preset="coarse")
config = json.loads(Path(sim.write_config()).read_text())  # nothing is solved or submitted
volume = {"materials": config["Domains"]["Materials"], "conductivity": config["Boundaries"]["Conductivity"]}
print(volume)
```

It continues the first snippet (it reuses `gsg_line` and the imports). For the same CPW line, `metal1` becomes a meshed volume group with its
own `Domains.Materials` entry: `Conductivity` 3.77e7, `Permittivity` 1.0, and no `Thickness`. `Boundaries.Conductivity` is an empty list
and there is no `Boundaries.PEC`. Test: `test_snippet_volume_metal_has_conductivity_and_no_surface_boundary`.

Caveats:

- The stack must not hold a bulk region of the same material over the layer. A dielectric layer whose material and z-range are covered by an
  entry of `stack.dielectrics` is taken as a placeholder for that region and is not extruded (`_is_covered_by_dielectric_box` in
  `gsim/palace/mesh/geometry.py`; test: `test_a_covering_bulk_region_of_the_same_material_hides_the_layer`). The gpdk stack has only an
  oxide region, so `metal1` is not covered.
- The layer needs polygons on its GDS layer, and anything that looks for a conductor layer no longer sees it. The gap-type lumped port
  raises "requires a conductor layer" for it (`gsim/palace/ports/config.py`). Only the CPW ports above were tried.
- Which other features read the layer type (nets, wave-port boundaries, current sources) was not checked here.
- Only the config was written. Nothing was solved, so how Palace treats the conductive volume, and whether the coarse preset above resolves
  the skin depth inside the metal (see the next section), is *not verified*.

One other public-API attempt was made: redeclare `metal1` as a `"via"` layer. For the CPW above, `mesh()` then stops with
`Mesh validation failed: No conductor surfaces in mesh`. That rules out the `"via"` classification, not the dielectric one.

Work on a further route is in [gsim PR #522](https://github.com/gdsfactory/gsim/pull/522), a branch that is not on main; it is not
described here.

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

## Surface model versus volume metal on a CPW

The comparison below is not stored in this repository; it comes from cloud runs of Palace 295645f (a main build after 0.19) on one case
([#539](https://github.com/gdsfactory/gsim/issues/539)). Case: shielded CPW on gpdk `metal3`, aluminum, 2 um thick, sigma = 3.77e7 S/m, 2D
boundary-mode solve. The reference is the same line with the metal meshed as a lossy volume, which agrees with femwell to about 3e-6 in
n_eff.

| Frequency (GHz) | delta, Al (um) | t / delta | Surface model / volume, attenuation |
| --- | --- | --- | --- |
| 10 | 0.82 | 2.4 | 0.71 |
| 25 | 0.52 | 3.8 | 0.74 |
| 50 | 0.37 | 5.4 | 0.80 |

At 25 GHz the surface model gives 26 % less loss than the volume metal. The settings barely move it: without `Thickness` or with `Thickness`
100 um the loss rises by 6 % and is still 22 % low, and field order 3 or a 0.075 um edge mesh change nothing. The deficit follows the skin
depth: with sigma = 1e10 S/m (delta 0.03 um) it disappears, and with sigma / 4 (delta 1.04 um) it grows. 73 % of the conductor loss on the
outline sits within 0.5 um of the 90 degree edges, where the field varies on the scale of delta and a local surface impedance is not exact.
At 1 GHz (t / delta 0.8) no surface setting matches the volume result, in loss or in the effective permittivity.

For conductor loss on a CPW or any line with edges comparable to delta, mesh the metal as a lossy volume ([Meshing a conductor as a lossy
volume](#meshing-a-conductor-as-a-lossy-volume)). The surface model is cheaper and fine for the phase: on this case its eps_eff is within 1
% of the volume result at 25 and 50 GHz and 3 % low at 10 GHz.

## Superconducting layers (London penetration depth)

`sim.set_superconductor(layer, london_depth=..., thickness=None)` makes a conductor layer superconducting. Lengths are in um. gsim then
writes a lossless `Boundaries.Impedance` entry with a surface inductance `Ls` (H per square) on the layer's surfaces, in place of the
layer's `Conductivity` or `PEC` entry. It works for driven, eigenmode and boundary-mode simulations, and is refused for electrostatic and
magnetostatic ones. `thickness` defaults to the layer's thickness in the stack. gsim has no table of London depths: take the value for
your material, film and temperature from a cited source. Nothing on this page was solved in Palace.

```text
sim.set_superconductor("metal1", london_depth=0.1)                 # lambda = 0.1 um, thickness from the stack
sim.set_superconductor("metal1", london_depth=0.2, thickness=0.1)  # film thickness given in um
sim.superconductors.pop("metal1")                                  # normal metal again
```

### What gsim writes

The value of `Ls` depends on how the layer is meshed (see [Three models](#three-models)). With lambda the London depth and t the
thickness:

| Conductor model | Surfaces that get an entry | `Ls` written |
| --- | --- | --- |
| Planar sheet | The one two-sided sheet, `pec_surfaces[layer]` | mu0 (lambda / 2) coth(t / 2 lambda) |
| 3D shell (default) | Each face group, `conductor_surfaces[layer_xy]` and `[layer_z]` | mu0 lambda coth(t / 2 lambda) per face |
| Boundary mode (a hole) | The exterior faces of the hole | The per-face value of the shell |

When each value is exact:

- **Planar sheet** (`planar_conductors=True`, or a layer under 0.02 um): for t << lambda in any field, and at any t when the field is
  antisymmetric across the film.
- **3D shell:** the interior between the faces is field-transparent, so the value is exact for an antisymmetric field at any t, and for
  t << lambda or t >> lambda in any field. For a field on one side only the error is -2% at t / lambda = 1, up to -14% at 4.
- **Boundary mode:** the conductor is a hole, so each face sees only the exterior field. Exact for an antisymmetric field; for a thin film
  in a field on one side only, up to 2 times too high.

Each entry is `{"Attributes": [group], "Ls": value}`. There is no `Crack` key and the value is per square already: it does not go through
the division by a curve length that `add_impedance_boundary` applies. Entries are sorted by group name and come after any user `Impedance`
entries. A planar layer needs `thickness` when the stack gives it zero thickness (QPDK-style stacks). An explicit `thickness` that differs
from the meshed thickness of a shell or a 2D conductor is logged.

### Derivation

A film fills 0 < z < t and carries current along x. In the lossless London limit (T much below Tc), E = i w mu0 lambda^2 J and
J = -dH/dz, so H'' = H / lambda^2. With u = t / lambda, K_b = H(0), K_t = -H(t) and the total sheet current K = K_b + K_t:

```text
E_bottom = i w mu0 lambda [coth(u) K_b + csch(u) K_t]
E_top    = i w mu0 lambda [csch(u) K_b + coth(u) K_t]

field on one side only (K_t = 0):   Ls = mu0 lambda coth(t / lambda)                (the textbook value)
field antisymmetric (K_b = K_t):    E is equal on both faces
    one sheet carrying all of K:    Ls_sheet = mu0 (lambda / 2) coth(t / 2 lambda)
    each of two faces carrying K/2: Ls_face  = mu0 lambda coth(t / 2 lambda) = 2 Ls_sheet
t << lambda:  E = i w mu0 (lambda^2 / t) K for any split between the faces
t >> lambda:  the faces decouple and each has mu0 lambda
```

A finite-difference solve of H'' = H / lambda^2 across the film reproduces the three closed forms, and
`tests/palace/test_superconductor.py` checks them to 1e-5. For a coplanar line without a nearby ground plane the quasi-static magnetic
field is mirror-symmetric about the film plane, so the antisymmetric case applies.

What Palace does on an interior face is an assumption here (*not verified*): that its weak form adds the Impedance term once over the face,
which gives the jump condition n x (H+ - H-) = E_t / Zs, a sheet that carries the total current. The shell is meshed with elements on both
sides of its faces (see [Three models](#three-models)), so each face is such a two-sided surface. With no `Crack` the sheet is read once;
a cracked face would be one-sided, which doubles the kinetic inductance of a thin film for a one-sided field.

The hollow shell is a three-layer model: two face sheets with a field-transparent interior of thickness t between them. Solving it
(Faraday's law across the interior, a jump condition at each face) gives, for a field on one side only, the ratio to the exact film value
c (c + u) / (2 c + u) / coth(u) with c = coth(u / 2). It is exactly 1 for an antisymmetric field. The test class
`TestHollowShellModel` solves this as a linear system and compares:

| t / lambda | 1 | 2 | 4 | 10 |
| --- | --- | --- | --- | --- |
| Hollow shell against the exact one-sided film | -2.1% | -9.3% | -14.0% | -8.3% |

### A quasi-static reference

To choose a formula for each model, a coplanar line (centre strip 10 um, gap 6 um, grounds 50 um wide, no substrate) was solved with
and without London. The solver is a partial-inductance solve of mu0 lambda^2 J + A = const over rectangular cells, giving
L(lambda) / L(0) = 1 + alpha, alpha = L_kin / L_geo. Three models of the same cross-section:

- full London: solid rectangles, the London term in every cell;
- sheet: one row of cells with Lambda = Ls_sheet t;
- hollow shell: thin cells on the faces with the per-face value, relative to the hollow shell with Ls = 0.

The geometric inductance of the reference, about 4.20e-7 H/m, agrees with conformal mapping (4.236e-7 H/m for t = 0 and infinite
grounds) and with scikit-rf (4.13e-7 H/m for t = 0.1 um, with its thickness correction); the test class
`TestQuasiStaticCoplanarReference` checks both within 3%, and reproduces the first row below on a coarse grid (sheet within 2%, hollow
shell within 6% of the full result).

| lambda, t (um) | Full London, alpha | Planar sheet | Hollow shell | Shell with coth(t / lambda) |
| --- | --- | --- | --- | --- |
| 0.2, 0.1 | 0.2168 (two grids agree to 1e-4) | 0.2144 (-1.1%) | 0.2085 (-3.8%) | 0.122 (-44%) |
| 0.1, 0.1 | 0.0731 (two grids agree to 1e-4) | 0.0729 (-0.3%) | 0.0674 (-8%) | 0.044 (-40%) |
| 0.05, 0.1 | 0.027 to 0.028 (not converged) | 0.0293 | 0.0236 | not computed |
| 0.1, 0.2 | 0.045 to 0.051 (not converged) | 0.0591 | 0.0430 | not computed |
| 0.05, 0.2 | 0.019 to 0.023 (not converged) | 0.0293 | 0.0185 | not computed |
| 0.05, 0.5 | 0.014 to 0.017 (not converged) | 0.0497 | 0.0171 | not computed |

Reading it:

- The textbook value mu0 lambda coth(t / lambda) on each face is 40% or more too low for a coplanar line, which is why gsim does not
  write it.
- For t <= lambda the sheet is within about 1%, and the hollow shell 4% to 8% low.
- At large t / lambda the sheet is too high: 3% to 30% at t / lambda = 2, 30% to 51% at 4 and about 3 times at 10. The sheet carries its
  current in one plane, with no side walls. The hollow shell is within about 20% of the full result in the last two rows. The rows with
  t / lambda of 2 and more were not converged between two grids, so these percentages are ranges, not values.
- The reference has no ground plane near the line and no substrate. For a film over a ground plane the field is closer to one-sided and
  the 1D analysis above applies instead: the shell is within the errors listed under "What gsim writes", and the 2D hole is up to 2 times
  too high for a thin film. That case was not run.

### Which model for which film

- t <= lambda: use the sheet (`planar_conductors=True`).
- t >= 4 lambda: use the shell (the default).
- Between: not validated. In the reference the sheet was 3% to 30% high and the shell 12% to 17% low at t / lambda = 2, and both were not
  converged; expect errors up to about 20%.
- A 2D boundary-mode conductor is always a hole, so there is no sheet alternative. Keep t / lambda above 1, or accept the one-sided error
  above.

gsim logs a warning, once per layer and setting, for a planar sheet with t > 2 lambda, a shell or 2D conductor with t < lambda, and an
eigenmode shell with t < lambda (the hollow interior may hold spurious slow-wave modes in large ground planes; *not verified*).

### Interactions

- **Wave ports.** `waveport_boundary="pec"` lists the superconducting attributes under `Boundaries.WavePortPEC` with the other
  `Impedance` attributes, so the port mode solve treats the layer as PEC. The port mode then misses the kinetic inductance that the line
  carries, and port and line differ by about sqrt(1 + alpha) in the propagation constant. With `"inherit"`, no `WavePortPEC` is written;
  whether Palace's port eigenproblem includes the `Ls` term is *not verified*. gsim logs a warning for either policy.
- **Boundary mode.** With `conductor_boundary="pec"` the superconducting layers become PEC like every other conductor, with a warning
  that names them, so the 2D modes stay comparable with the port modes.
- **Symmetry planes, metallic outer wall.** The plane and the wall are merged into `PEC` or `PMC` without touching the `Impedance`
  entries.
- **Refused combinations.** A superconducting surface that also appears in another boundary entry, for example a user `Impedance` entry
  with the same attributes, raises `ValueError`. So do a layer that is not in the stack, a via layer, and a zero-thickness layer with no
  `thickness`.
- **Mesh reuse.** `reuse_mesh` copies the superconducting layers with the other layer properties (a different setting of the receiver is
  replaced, with a warning). They are not a mesh input, so changing them after the mesh never invalidates it.
- **Saved settings and the cache.** `superconductors` is a public field, so `model_dump_json()` keeps it. The input hash covers the written
  files, so a changed London depth changes the key by itself.
- **`generate_mesh(write_config=True)`** writes no superconducting entries, like it writes no impedance boundaries; `sim.write_config()`
  does, and `sim.mesh()` leaves the config to it.
- **Validation.** A chip whose conductors are all superconducting has no `Conductivity` or `PEC` entry. The mesh validation counts the
  superconducting `Impedance` entries as conductor boundaries, and the lumped-port contact check counts their shell faces.
- **`run_local`.** It logs a warning for a Palace older than 0.19 or of unknown version.

### Not done

A two-fluid loss term (`Rs`) is deferred: the same face and sheet conversion applies to it, but the quasi-particle conductivity would need a
published source for its value. The default is lossless.

### Cloud runs

Runs on Palace 295645f, on a coplanar line on sapphire (strip 10 um, gaps 6 um, film 0.1 um). `alpha` is the kinetic over the geometric
inductance of the line. It is measured from the rise of the mode index in 2D and from the fall of the resonance frequency of a 1 mm
half-wave resonator in 3D, each against the PEC case of the same mesh. The expected values are the quasi-static London reference of this
section, which has no substrate.

| Case | lambda (um) | alpha measured | alpha expected |
| --- | --- | --- | --- |
| 2D boundary mode | 0.2 | 0.212 | 0.209 |
| 2D boundary mode | 0.1 | 0.069 | 0.067 |
| 3D resonator, planar sheet | 0.2 | 0.211 | 0.214 |
| 3D resonator, shell | 0.2 | 0.202 | 0.209 |

These runs are not stored in this repository. The 3D sheet result shows that Palace applies an interior `Impedance` once: applied on both
sides, alpha would be about half. In the 3D runs the lowest mode was taken as the coplanar mode without checking its fields.

## Not verified

- Anything Palace does with these keys, except the surface-versus-volume comparison above (one case, cloud runs). In particular
  what `Thickness` does on a cracked boundary, and whether the cloud runtime reads `Crack`.
- The claim that Palace 0.19 reads an interior `Conductivity` boundary as a thin film comes from gsim's code comments and tests.
- The check that `set_material` is case-sensitive was made on the gpdk stack only.
- The `"via"` re-declaration was tried on one CPW with lumped ports. The dielectric classification was tried on the same line only, and
  only as far as `write_config()`.
- That PEC is adequate for a phase, an impedance or a resonant frequency: the page gives the textbook reason to check, not a measurement.
- The numbers in the line table were produced on one Windows machine with the coarse preset; the tetrahedron counts depend on the mesher.
- Everything under [Superconducting layers](#superconducting-layers-london-penetration-depth) about Palace: that `Ls` is read in H per
  square (SI), that an interior `Impedance` is applied once as a sheet and not cracked, what the port eigenproblem does with it, and
  whether the Palace schema accepts the entries as written. The formulas are checked against a 1D finite-difference solve and a
  quasi-static coplanar reference, not against Palace.
- The superconducting reference rows with t / lambda of 2 and more are not converged; no film over a ground plane was run.
- The quasi-static reference solver is reproduced in the tests for one row only; the other rows came from the same solver run outside
  the test suite.
- The scikit-rf comparison uses the installed 1.13.0; the source tool shows 2.1.0, with the same `CPW.analyse_quasi_static` signature.
