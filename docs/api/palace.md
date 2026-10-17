# Palace guide

Use [`gsim.palace`][gsim.palace] for driven RF and microwave simulations, eigenmodes, electrostatics, and boundary modes.
This guide covers solver configuration, meshing, and result processing. Class and function links open the API reference.

## Solver settings

Numerical and problem-specific settings live under `sim.solver`. Each simulation
exposes its applicable problem group alongside the common controls:

```python
import gsim.palace as pa

sim = pa.EigenmodeSim()
sim.solver.order = 1
sim.solver.linear.tolerance = 1e-9
sim.solver.linear.max_iterations = 400
sim.solver.eigenmode.num_modes = 2
sim.solver.eigenmode.target = 4e9  # Hz; required before meshing/exporting
sim.solver.eigenmode.tolerance = 1e-9
sim.solver.eigenmode.save = 2
```

The linear tolerance controls each linear solve. The eigenmode tolerance controls
eigenvalue convergence. They are independent. All Palace simulation types default
to **GMRES with complex MUMPS multigrid**, field order **2**, **400** maximum linear
iterations, and `"CPU"`. Linear tolerance defaults to **1e-6**, except for
`EigenmodeSim`, whose linear and eigenvalue tolerances both default to **1e-9**.

The configurable multigrid controls live under `sim.solver.linear`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `solver_type` | `"MUMPS"` | Coarse solver backend; `"Default"` asks Palace to choose. |
| `mg_max_levels` | `100` | Maximum multigrid levels; Palace uses the available hierarchy. Set 1 for a full-system solve. |
| `pc_mat_real` | `False` | Set True to use a real approximation of the preconditioner matrix. |
| `complex_coarse_solve` | `True` | Use a complex coarse solve for complex-valued problems. |

Real-only problems such as electrostatics still solve real equations. The complex
controls do not turn materials or fields into complex quantities in those problems.
MUMPS no longer forces one Krylov iteration, disables multigrid, or disables the
mesh error estimator or divergence-free projection. Explicit settings are exported
without backend-specific overrides.

```python
sim.solver.linear.pc_mat_real = True
sim.solver.linear.complex_coarse_solve = False  # lower-memory real approximation
sim.solver.linear.mg_max_levels = 1  # full-system solve; can need much more RAM
```

`solver_type="SuperLU"`, `"STRUMPACK"`, and `"Default"` remain available.
`preconditioner="AMS"` or `"BoomerAMG"` applies when `solver_type="Default"`.
Use `pc_mat_real=True` and `complex_coarse_solve=False` to request a real
preconditioner with another backend. These settings can also be supplied to
`set_solver(...)` or in the constructor's `solver={"linear": ...}` group.

The defaults are a starting point. Check convergence and mesh resolution for each
problem, especially when extracting quality factors. Driven adaptive-sweep controls
are separate and retain their existing defaults.

You can also supply grouped settings at construction:

```python
sim = pa.EigenmodeSim(
    solver={
        "order": 1,
        "linear": {"tolerance": 1e-6, "max_iterations": 400},
        "eigenmode": {"num_modes": 2, "target": 4e9, "tolerance": 1e-8, "save": 2},
    }
)
```

Other simulation types expose their own problem group:

```python
driven = pa.DrivenSim(solver={"driven": {"fmin": 1e9, "fmax": 10e9, "num_points": 40}})
driven.solver.driven.adaptive_tol = 0.01

electrostatic = pa.ElectrostaticSim()
electrostatic.solver.electrostatic.save_fields = 2

boundary = pa.BoundaryModeSim()
boundary.solver.boundary_mode.freq = 5e9
boundary.solver.boundary_mode.num_modes = 2
```

All four share nine controls: `order`, `device`, and
`linear.{solver_type, preconditioner, tolerance, max_iterations, mg_max_levels,
pc_mat_real, complex_coarse_solve}`.

| Simulation | Problem group | Problem settings | Total settings |
| --- | --- | --- | --- |
| DrivenSim | `solver.driven` | 12 | 21 |
| EigenmodeSim | `solver.eigenmode` | 8 | 17 |
| ElectrostaticSim | `solver.electrostatic` | 1 | 10 |
| BoundaryModeSim | `solver.boundary_mode` | 7 | 16 |

`sim.set_solver(...)` replaces the common controls using the same defaults as
the simulation constructor, including its linear tolerance, while preserving the problem-specific settings. Assign individual
attributes to retain other common values. The generated Palace JSON continues to
use `Solver.Order`, `Solver.Device`, `Solver.Linear`, and the applicable problem block.

### Existing code

`sim.numerical`, `NumericalConfig`, `set_numerical(...)`, the top-level problem
attributes, and legacy constructor arguments remain supported and emit
`DeprecationWarning` messages pointing to the grouped API. Problem-specific setters
such as `set_eigenmode(...)` remain available as convenience methods. Old serialized
inputs still load; simulation serialization uses the grouped `solver` field.
`NumericalConfig` retains its flat serialization format. Migrate direct access as follows:

| Previous access | Grouped access |
| --- | --- |
| `sim.numerical.order` | `sim.solver.order` |
| `sim.numerical.tolerance` | `sim.solver.linear.tolerance` |
| `sim.eigenmode.target` | `sim.solver.eigenmode.target` |
| `sim.driven.fmin` | `sim.solver.driven.fmin` |
| `sim.electrostatic.save_fields` | `sim.solver.electrostatic.save_fields` |
| `sim.boundary_mode.freq` | `sim.solver.boundary_mode.freq` |

Calling `set_numerical()` without an explicit order now uses **2**, matching the
constructor and `set_solver()`. Pass `order=1` to retain the previous setter behavior.
Such calls emit a temporary `FutureWarning` explaining this change. Explicit-order
calls emit the usual API deprecation warning. The new grouped API emits neither.

Update individual controls directly to preserve other settings:

```python
sim.solver.order = 1
sim.solver.linear.tolerance = 1e-8
```

`sim.set_solver(...)` replaces the common controls with the supplied values and
their defaults, preserving the existing problem group, such as the eigenmode
target. Assigning `sim.solver = ...` replaces the entire solver configuration,
including its problem group; a new eigenmode group defaults to `target=None`.

`sim.numerical` is a deprecated alias for `sim.solver`, so its `model_dump()` uses
the grouped format. `NumericalConfig(**sim.numerical.model_dump())` accepts that
format and copies the nine common numerical controls into a flat legacy model.
For new code, use `sim.solver.model_copy(deep=True)` to copy all solver settings.

- [`SolverConfig`][gsim.palace.SolverConfig]
- [`LinearSolverConfig`][gsim.palace.LinearSolverConfig]
- [`DrivenConfig`][gsim.palace.DrivenConfig]
- [`EigenmodeConfig`][gsim.palace.EigenmodeConfig]
- [`ElectrostaticConfig`][gsim.palace.ElectrostaticConfig]
- [`BoundaryModeConfig`][gsim.palace.BoundaryModeConfig]

See [result validation](../palace-validation.md) for independent solver,
adaptive sampling, mode identity, and provenance checks.

## Numeric wave-port boundaries

The default `waveport_boundary="pec"` approximates finite conductivity, impedance
and absorbing boundaries as PEC **inside the port eigenproblem**. Choose
`"inherit"` to retain those physical conditions when solving the port mode:

```python
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_driven(f=50e9, waveport_boundary="inherit")
# After geometry/ports are configured and the mesh is generated:
# config = json.loads(sim.write_config().read_text())
# "WavePortPEC" not in config["Boundaries"]
```

Only the generated `Boundaries.WavePortPEC` override changes. The physical 3D
conductivity, impedance and absorbing boundaries remain identical. Lumped ports
are unaffected. This controls a modeling approximation; it does not fix Palace
mode selection or establish lossy S-parameter normalization accuracy. Check the
port fields and solver convergence for the chosen Palace runtime.

The policies differ only when absorbing, conductivity or impedance boundaries
exist; without these Robin terms, both omit `WavePortPEC`. A later `set_driven()`
call rebuilds the driven settings, so repeat `waveport_boundary="inherit"` when
updating other sweep options if you want to retain it.

Material override precedence is separate; see [PR #277](https://github.com/gdsfactory/gsim/pull/277).

## Result validation

- [`validation`][gsim.palace.validation]

## Simulation Classes

- [`DrivenSim`][gsim.palace.DrivenSim]
- [`EigenmodeSim`][gsim.palace.EigenmodeSim]
- [`ElectrostaticSim`][gsim.palace.ElectrostaticSim]
- [`BoundaryModeSim`][gsim.palace.BoundaryModeSim]

## Floquet eigenmodes

Set the signed cell phase in radians. The wave vector uses the measured donor-to-receiver mesh translation; the eigenvalue
search frequency does not determine the cell length. An optional `periodic_length` in mesh units (micrometers for generated
meshes) checks that the periodic faces have the expected separation:

```python
from gsim.palace import EigenmodeSim

sim = EigenmodeSim()
sim.set_eigenmode(target=40e9, floquet=True, phi_target=-0.4, periodic_length=100.0)
sim.solver.eigenmode.compute_floquet_wave_vector(periodic_axis="x")
# [-0.004, 0.0, 0.0] rad/um
```

After setting geometry and stack, `sim.mesh(periodic_axis="x")` records the actual translation. `sim.write_config()` writes
it as `Boundaries.Periodic.BoundaryPairs[0].Translation`, and computes `FloquetWaveVector` from that measured length. A length
mismatch raises `ValueError`; omit `periodic_length` to use the measured value without an expected-length check. Domain
padding can change the separation of periodic faces. Generated translations use the CAD face planes rather than the
tolerance-padded OCC bounding boxes.

Zero, negative phases, and the Brillouin-zone endpoints `+/-pi` are supported without wrapping. With
[Palace's phase convention](https://awslabs.github.io/palace/stable/guide/boundaries/#Periodic-boundary), the receiver field is
`exp(-1j * phi_target)` times the donor field. The GDS mesher supports periodic axes `x` and `y`; the wave-vector helper and
config generation also support `z` for supplied mesh metadata.

`n_eff_guess` is retained as a deprecated compatibility argument and no longer affects the result. Passing it explicitly to
`set_eigenmode()` emits a `DeprecationWarning`. Direct calls to
`compute_floquet_wave_vector()` now require an explicit length on the config or as a `periodic_length` argument; the old `l0`
argument and frequency-based length estimate are removed. Old mesh results without translation metadata must be regenerated.

## Electrostatic terminals

An `ElectrostaticSim` terminal selects conductors by connected net, by region or by
layer. A net is the metal that touches on one layer plus the metal that vias join on
other layers, so two electrodes on one layer are two nets (see `sim.nets()`).

```python
sim.add_terminal("TOP", net="PLUS")  # net by port name, or by its name in sim.nets()
sim.add_terminal("LEFT", region=(0, 0, 20, 20), layer="m1")  # nets with an m1 shape in the box
sim.add_terminal("SHIELD", net="net3", floating=True)  # solved, then eliminated
sim.add_terminal("T1", layer="metal5")  # a whole layer, as before
sim.unassigned_conductors = "ground"  # the default; or "floating"
print(sim.resolve_terminals())  # check the selection before meshing
cap = sim.load_capacitance(sim.run())  # floating terminals are eliminated
cap.between("TOP", "LEFT")
cap.to_ground("LEFT")
cap.pairs()  # every pair and every terminal-to-ground value, as a table
```

- **Net names** are the names of the ports on the net joined by `+`, or `net<k>`
  (`k` counts from 1 in layout order). `net=` matches the whole name or any one port
  name; if several nets carry the same port name they all go into that terminal.
  `net<k>` changes when the layout is edited, so prefer port names or `region=`.
  GDS text labels are not used.
- **`region=`** is a box `(xmin, ymin, xmax, ymax)` or a list of at least three
  `(x, y)` points, in um. It selects every net with a conductor or via shape that
  overlaps the region, only on `layer` when that is given. It always selects whole
  nets, because part of a conductor cannot be at another potential.
- **`layer=` alone** keeps the old behaviour while no terminal uses `net`, `region` or
  `floating`: every shape on that layer, and vias by z-overlap. Once any terminal uses
  nets, `layer=` means every net with a shape on that layer, including the metal that
  vias join to it on other layers. `net=` cannot be combined with `region=` or `layer=`.
- **Conductors in no terminal** are grounded, as before. With
  `unassigned_conductors="floating"` each becomes an extra terminal named after its
  net. PEC blocks are always grounded. A net in two terminals is an error.
- **Floating conductors** are solved as terminals, one extra Palace excitation each,
  and `load_capacitance` removes them with a Schur complement of the Maxwell matrix,
  `C_tt - C_tf C_ff^-1 C_ft`, which is exact for electrostatics. Pass
  `eliminate_floating=False` to keep them. The rows are the terminals in the order
  they were added, then the automatic floating terminals in net order.
- **Vias and merging.** `mesh()` merges vias closer than `merge_via_distance` (2 um by
  default). If that would join the vias of two nets it raises a `ValueError` naming the
  via layer and the nets; set `merge_via_distance` below the gap between them.
- **Changing the selection style.** A mesh made for layer terminals cannot serve net
  terminals and the other way round; `write_config()` raises and asks you to call
  `mesh()` again. In net mode `write_config()` also writes `terminals.json`, which
  `load_capacitance` reads when the simulation object is new.
- **Reading the results.** `cap.between(a, b)` is the capacitance between two
  electrodes and `cap.to_ground(a)` the one to ground; `cap.pairs()` lists both kinds
  apart. If no conductor is grounded, `to_ground` comes out near zero.

## Capacitance

- [`CapacitanceMatrices`][gsim.palace.CapacitanceMatrices]
- [`load_capacitance`][gsim.palace.load_capacitance]

## Saved fields and S-parameters

`load_fields` accepts a simulation directory, Palace output directory, or results
dictionary. Driven, BoundaryMode, and Eigenmode volume and boundary outputs are
supported. Enable field saving with `save_step >= 1` for driven simulations or
`save >= 1` for eigenmode simulations.

```python
from gsim.palace.results import load_fields

fields = load_fields("eigenmode_sim/output/palace", mode=1)
electric = fields["E_real"] + 1j * fields["E_imag"]
assert electric.shape == (fields.n_points, 3)
surface = load_fields("eigenmode_sim/output/palace", mode=1, boundary=True)
```

`mode=1` selects `Cycle000001` from Eigenmode or BoundaryMode output; a missing
mode or a mode containing only mesh metadata raises an error. Use `cycle=N` to
select an exact ParaView cycle instead, including metadata cycles. Omit both
selectors to load the last cycle containing solution fields, skipping final
`Indicator`, `Rank`, or `attribute` arrays. `mode` and `cycle` are mutually
exclusive. For driven output, `excitation=N` selects the excitation directory.
A missing excitation now raises `FileNotFoundError` instead of selecting another
excitation. Metadata-only output raises `ValueError` by default; explicit `cycle=N`
still permits inspecting that metadata. Mode/cycle selection was also checked
against saved Palace Floquet run `palace-552ac5e8`, in addition to synthetic tests.

`load_sparams` converts frequency, dB magnitude, and degree phase columns to
floating-point arrays. A whitespace-padded `-inf` dB value remains negative
infinity and converts to exact zero, without a magnitude floor:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gsim.palace.results import load_sparams

with TemporaryDirectory() as directory:
    output = Path(directory)
    (output / "port-S.csv").write_text(
        "f (GHz), |S[2][1]| (dB), arg(S[2][1]) (deg.)\n"
        "1.0,   -inf   , 0.0\n"
    )
    parameters = load_sparams(output)
    assert np.isneginf(parameters.s21.db[0])
    assert parameters.s21.complex[0] == 0j
    assert parameters.to_skrf().s[0, 1, 0] == 0j
```

Malformed numeric cells reject the entire CSV, including otherwise valid rows.
Errors report the path, column and one-based **data row**, excluding the header
and blank lines. NaN, positive infinity and empty numeric cells raise `ValueError`;
phase and frequency must be finite, while dB magnitudes permit negative infinity.

## Mesh

`sim.mesh()` also reports **Estimated Field DOFs** for tetrahedral 3D driven and
eigenmode problems, using the configured field order. For order 2, the estimate
is `2 * unique_edges + 2 * unique_triangular_faces`, counting interior and shared
entities once. Geometry order and field order are separate. This is the input
mesh count before Palace splits interior boundaries, applies periodic constraints
or refines the mesh; it is not a bound on the final solve size. Other problem
types and mixed/non-tetrahedral meshes have a `null` estimate.

Mesh results expose a JSON-compatible `result.metadata` dictionary, also written
to `metadata.json` beside the mesh and included in cloud input uploads:

```python
result = sim.mesh()
metadata = result.metadata
metadata["schema_version"]  # 1
metadata["mesh"]["topology"]  # edges and triangular_faces
metadata["mesh"]["field_dofs"]["estimated_field_dofs"]
metadata["mesh"]["kappa"]["max"]  # numerical distortion, not formatted text
```

`sim.write_config()` refreshes the file using the effective config's field order,
problem type, solver settings and refinement controls. The result object is a
snapshot from meshing. Consumers should accept additional metadata keys so new
measurements can be added. Allocation rules remain in DataLab; these are sizing
hints. Compatible SDKs also forward it as a separate submission parameter;
older SDK interfaces retain fixed-resource submission. See
[cloud sizing metadata](cloud.md#sizing-metadata) for the calibration assumptions.
The root-level `metadata.json` name is reserved for these diagnostics. It is
excluded from the Palace result-cache key so measurements can evolve without
rerunning identical mesh/config inputs. The full `compute_dir_digest()` includes
the metadata by default; the cache key is not a complete bundle-integrity hash.

`sim.mesh()` reports **Worst element distortion, κ** in its summary. The value is
also available in `result.mesh_stats["kappa"]["max"]` and `sim.print_mesh_stats()`.
It uses Palace/MFEM's normalized Jacobian condition number: 1 is an ideal
equilateral tetrahedron; larger values indicate greater distortion. Curved
elements are sampled at their centers, so this is not a bound over their interiors.
Use it alongside SICN's signed validity check and solution convergence; κ alone
does not measure simulation accuracy. A numerically singular center is displayed
as infinite and stored as `max=None` with a nonzero `singular_elements` count.

- [`MeshConfig`][gsim.palace.MeshConfig]
- [`generate_mesh`][gsim.palace.generate_mesh]

## Nets

- [`Nets`][gsim.palace.Nets]
- [`Net`][gsim.palace.Net]
- [`extract_nets`][gsim.palace.extract_nets]

## Stack

- [`LayerStack`][gsim.palace.LayerStack]
- [`Layer`][gsim.palace.Layer]

## Circuit Synthesis

- [`CircuitSynthesis`][gsim.palace.CircuitSynthesis]
- [`fit_rlc`][gsim.palace.fit_rlc]
- [`differential_impedance`][gsim.palace.differential_impedance]
- [`initial_guess_rlc`][gsim.palace.initial_guess_rlc]
- [`z_rlc`][gsim.palace.z_rlc]
- [`load_circuit_synthesis`][gsim.palace.load_circuit_synthesis]

## Fitting

- [`RLCFit`][gsim.palace.RLCFit]
- [`VectorFit`][gsim.palace.VectorFit]

## Parameter Conversions

- [`s_to_z`][gsim.palace.s_to_z]
- [`z_to_s`][gsim.palace.z_to_s]
- [`s_to_y`][gsim.palace.s_to_y]
- [`y_to_s`][gsim.palace.y_to_s]
- [`z_to_y`][gsim.palace.z_to_y]
- [`y_to_z`][gsim.palace.y_to_z]
- [`is_complete`][gsim.palace.is_complete]

## Transmission-line analysis

See [Transmission-line analysis](../transmission_line_analysis.md) for using
`load_sparams(...).to_skrf()` with scikit-rf's multiline TRL calibration,
physical impedance normalization and independent-length checks.

## Terminal S-parameters

See [Terminal (multi-pin) Wave Ports](../terminal_wave_ports.md) for the
conventions and a worked example.

::: gsim.palace.terminal
    options:
      show_source: false
