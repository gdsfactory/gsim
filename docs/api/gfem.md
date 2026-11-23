# gfem guide

`gsim.gfem` solves the 2D port mode of a cross-section with [gfem](https://github.com/benvial/gfem), a finite-element solver for
Maxwell's equations. It reads the same native-2D mesh that [`BoundaryModeSim`][gsim.palace.BoundaryModeSim] writes for Palace and
that `gsim.femwell` reads, with the stack handled the way the femwell route handles it. What comes back is a
[`LineReading`][gsim.common.modes.LineReading], the object the femwell and Palace routes return too, so the three solvers can be
compared on one line. This first step covers 2D modes and lines; 3D ports are not part of it.

The solve is [`solve_line`][gsim.gfem.solve_line]. [`line_reading`][gsim.gfem.line_reading] reads one frequency of the result as a
`LineReading`, and [`line_impedances`][gsim.gfem.line_impedances] gives the three impedance definitions side by side.

## Install

gfem is optional. Install the `gfem` extra:

```bash
pip install "gsim[gfem]"
```

It brings `gfem` (0.1 series), which in turn needs JAX and NGSolve. `import gsim.gfem` works without them and imports neither;
`solve_line()` raises an error naming the extra. A gfem outside the 0.1 series imports with a warning.

## Solve a line

```python
from gsim.common.modes import Conductor
from gsim.gfem import line_reading, solve_line

# sim: a meshed BoundaryModeSim of a coplanar line, the metal meshed as lossy regions
signal = Conductor("m3_signal", ((-5.0, 5.0), (3.2, 5.2)), "volume")
ground = Conductor("m3_ground", ((11.0, 150.0), (3.2, 5.2)), "volume")
domain = ((-150.0, 150.0), (-203.0, 105.2))  # the meshed window, (h, v) in um

result = solve_line(
    sim.mesh_path, sim.stack, frequency_hz=[1e9, 10e9], metallic_boundaries=True
)
reading = line_reading(result, signal=signal, return_=ground, domain=domain, index=1)
reading.n_eff, reading.z0_ohm, reading.wall_mode, reading.diagnostic
```

`frequency_hz` is one frequency or a sequence. gfem picks the mode at the lowest frequency and follows it through the band, so
`mode=1` is the most confined propagating mode there. `metallic_boundaries`, `absorbing_boundary`, `conductor_boundary` and
`conductor_thickness` mirror the `BoundaryModeSim` settings of the same name.

## What the mesh becomes

The 2D regions of the mesh become gfem materials. The 1D groups the mesher writes become the boundary conditions of the port-mode
problem.

| Mesh group | gfem |
| --- | --- |
| 2D region | `Material(eps_r, tan_delta, conductivity)` from the stack, at the solve frequency |
| `<layer>_conductivity` | `Conductor(conductivity, thickness)`; a `PEC` with `conductor_boundary="pec"` |
| `<layer>_pec` | `PEC` |
| `absorbing` (outer wall) | `PEC` with `metallic_boundaries=True`, else `Absorbing` with `absorbing_boundary=True`, else the natural PMC |
| any other group | ignored |

The region permittivity is the one `gsim.femwell.adapter.epsilon_by_region` returns, `eps_r (1 - i tan_delta) - i sigma / (omega
eps_0)`, so both solvers see one material per region. A region that names a stack layer takes that layer's material; any other
region (the background media) is named after its material, which is how the native-2D mesher writes it. gfem takes the slab of
`Mesh.from_gmsh_cross_section` as the model and puts a wave port on its `front` face. Metal meshed as regions is the model the
frozen references below use; metal left out of the mesh as a hole takes the `Conductor` or `PEC` row.

An unconditioned outer wall is a PMC in gfem as in Palace. `conductor_thickness="thick"` (the default) models a conductor by its
skin effect alone; `"stack"` gives it the layer thickness. gfem counts that thickness twice on a one-sided face, and whether Palace
does the same with the `Thickness` it writes is not checked here, so treat the two as different until that has been compared.

## Conventions

The conventions are gfem's: `exp(+i omega t)`, peak phasors, modes normalised to 1 W. `n_eff = n' - i n''` with `n'' >= 0` for a
lossy line, and `gamma = alpha + i beta` with the attenuation `alpha >= 0`. The attenuation in dB/cm is
`20 / ln(10) * Re(gamma) / 100`.

## Impedance definitions

A line has no single characteristic impedance, and the three that gfem measures differ by the mode's departure from TEM:

- `Z_PV = |V|^2 / (2 conj(P))`, along the voltage path across the gap: Palace's `Z_PV`.
- `Z_PI = 2 P / |I|^2`, around the signal conductor (Marks and Williams).
- `Z_VI = V / I`, with the sign that makes `Re >= 0`.

`LineReading.z0_ohm` is `Z_PI`, the definition that class documents; the diagnostic string quotes `Z_PV` and `Z_VI` next to it. The
voltage path keeps the same hair of clearance from the conductor faces as the Palace postprocessing path. The current loop does not:
gfem's `Z_PI` is a line integral of the finite-element H field and reads the wrong current in the first layer of elements beside
the metal. On a homogeneous-oxide CPW with perfect-conductor holes the Palace loop (0.002 µm out) gave a `Z_PI` 19 % above `Z_PV`,
while loops 1 to 4 µm out agreed within 0.5 %. The loop therefore keeps half the smallest clearance around the signal conductor
(`LOOP_CLEARANCE_FRACTION`, 1 µm for the 2 µm conductor below), and `loop_clearance_fraction` changes it. It wants to be about one
element away from the metal and clear of every other conductor.

## Line mode or wall mode

A shielded line has a second propagating mode: the one on which both electrodes sit at one potential and the metallic wall returns
the current. `line_reading` tells them apart with the rule Palace's reading uses: below `|Z_PV / Z_PI| = 1e-2` the gap voltage
carries none of the mode's power and `wall_mode` is `True`. gfem 0.1.0 does not expose a signed electrode current, so the
electrode-current pairing the femwell reading uses is not available. Without a return conductor `wall_mode` is `None`.

## Limits

- 2D cross-sections only; the mesh must not name a group `front` or `back`.
- Passive media with `Re(eps) > 0`. An optical metal with a negative permittivity is refused with an error naming the region.
- One set of media per solve. A stack entry that changes between the requested frequencies makes `solve_line` raise: call it once
  per frequency.
- A port-mode solve is slow. gfem solves a few frequencies exactly and tracks the rest with a reduced model; on the 10 950-triangle
  mesh above, two exactly solved frequencies took 38.6 s on a Windows laptop.

## Agreement on a lossy coplanar waveguide

The reference is the shielded CPW of benvial/gfem#2: 10 µm signal, 6 µm gaps, 2 µm aluminium at 3.77e7 S/m, silicon below and air
above, 300 µm wide. Frozen numbers from Palace (2D boundary mode, `Z_PV`) and femwell (`Z_PI`) on a 27 522-triangle mesh with
0.15 µm elements in the metal are compared with gfem through `gsim.gfem` on the metal-as-regions model.

On the mesh `BoundaryModeSim` writes (10 950 triangles, 0.5 µm elements in the metal, the default profile of
`tests/gfem/test_lossy_cpw_three_way.py`), relative differences of gfem from the frozen numbers:

| f (GHz) | Re n vs Palace and femwell | alpha vs Palace / femwell | Z_PV vs Palace | Z_PI vs femwell |
| --- | --- | --- | --- | --- |
| 1 | 5.3e-4 | 5.0e-4 / 5.0e-4 | 3.4e-4 | 6.8e-4 |
| 10 | 5.3e-4 | 8.1e-4 / 8.5e-4 | 3.4e-4 | 8.7e-4 |

The second table is from a plain-gmsh stand-in of the same geometry with finer metal (12 534 triangles, 0.25 µm in the metal),
measured before `BoundaryModeSim` meshes could be read here. It covers the four frozen frequencies, which the table above does
not (50 GHz needs the finer metal mesh); re-measure it on the real mesh with
`GSIM_GFEM_CPW_FULL=1 pytest tests/gfem/test_lossy_cpw_three_way.py -s`.

| f (GHz) | Re n: gfem vs Palace | alpha (dB/cm) | Z_PV (ohm, real part) | Z_PI vs femwell |
| --- | --- | --- | --- | --- |
| 1 | 2.1329 vs 2.1330 (3.9e-5) | 1.1185 vs 1.1175 (8.9e-4) | 66.70 vs 66.76 (8.5e-4) | 66.79 vs 66.82 (3.6e-4) |
| 10 | 1.9098 vs 1.9101 (1.5e-4) | 2.3050 vs 2.3039 (4.8e-4) | 59.72 vs 59.78 (1.0e-3) | 59.80 vs 59.84 (5.2e-4) |
| 25 | 1.8732 vs 1.8735 (1.5e-4) | 3.5598 vs 3.5582 (4.5e-4) | 58.56 vs 58.62 (1.0e-3) | 58.66 vs 58.68 (4.1e-4) |
| 50 | 1.8512 vs 1.8515 (1.5e-4) | 5.1354 vs 5.1325 (5.7e-4) | 57.83 vs 57.89 (9.9e-4) | 57.95 vs 57.97 (3.6e-4) |

Relative differences in brackets. The frozen numbers were computed with Palace build 295645f driven by gsim 0.9.0 and femwell
0.1.12; they were copied from a branch of gfem and have not been compared with the merged version.

## Reference

See [`gsim.gfem`][gsim.gfem] for the full API. The conductor models behind `Conductor`, `PEC` and the metal-as-regions choice are
described on the [Conductor Models](../palace_conductors.md) page.
