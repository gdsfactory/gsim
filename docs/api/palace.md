# Palace API

## Solver settings

Numerical and problem-specific settings live under `sim.solver`. Each simulation
exposes its applicable problem group alongside the common controls:

```python
import gsim.palace as pa

sim = pa.EigenmodeSim()
sim.solver.order = 1
sim.solver.linear.tolerance = 1e-6
sim.solver.linear.max_iterations = 400
sim.solver.eigenmode.num_modes = 2
sim.solver.eigenmode.target = 4e9  # Hz; required before meshing/exporting
sim.solver.eigenmode.tolerance = 1e-8
sim.solver.eigenmode.save = 2
```

The linear tolerance controls each linear solve. The eigenmode tolerance controls
eigenvalue convergence. They are independent. Field order defaults to **2**, linear
tolerance to **1e-6**, and maximum linear iterations to **400**. The backend and
preconditioner default to `"Default"`, and the device defaults to `"CPU"`.

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

All four share six controls: `order`, `device`, and
`linear.{solver_type, preconditioner, tolerance, max_iterations}`.

| Simulation | Problem group | Problem settings | Total settings |
| --- | --- | --- | --- |
| DrivenSim | `solver.driven` | 11 | 17 |
| EigenmodeSim | `solver.eigenmode` | 7 | 13 |
| ElectrostaticSim | `solver.electrostatic` | 1 | 7 |
| BoundaryModeSim | `solver.boundary_mode` | 7 | 13 |

`sim.set_solver(...)` replaces the common controls using the same defaults as
`SolverConfig()`, while preserving the problem-specific settings. Assign individual
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
format and copies the six common numerical controls into a flat legacy model.
For new code, use `sim.solver.model_copy(deep=True)` to copy all solver settings.

::: gsim.palace.SolverConfig
    options:
      show_source: false
      inherited_members: false
      members:
        - order
        - device
        - linear

::: gsim.palace.LinearSolverConfig
    options:
      show_source: false
      inherited_members: false
      members:
        - tolerance
        - max_iterations
        - solver_type
        - preconditioner

::: gsim.palace.DrivenConfig
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.EigenmodeConfig
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.ElectrostaticConfig
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.BoundaryModeConfig
    options:
      show_source: false
      inherited_members: false

## Simulation Classes

::: gsim.palace.DrivenSim
    options:
      show_source: false
      inherited_members: true
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_driven
        - set_material
        - solver
        - set_solver
        - add_port
        - add_cpw_port
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - write_config
        - run
        - start
        - upload
        - get_status
        - wait_for_results

::: gsim.palace.EigenmodeSim
    options:
      show_source: false
      inherited_members: true
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_eigenmode
        - set_material
        - solver
        - set_solver
        - add_port
        - add_cpw_port
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run

::: gsim.palace.ElectrostaticSim
    options:
      show_source: false
      inherited_members: true
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_electrostatic
        - set_material
        - solver
        - set_solver
        - add_terminal
        - nets
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run
        - load_capacitance

::: gsim.palace.BoundaryModeSim
    options:
      show_source: false
      inherited_members: true
      members:
        - solver
        - set_solver
        - set_boundary_mode
        - set_cross_section
        - set_output_dir
        - set_geometry
        - set_stack
        - set_material
        - mesh
        - validate_config
        - validate_mesh
        - write_config
        - run

## Capacitance

::: gsim.palace.CapacitanceMatrices
    options:
      show_source: false
      inherited_members: false
      members:
        - between
        - to_ground
        - maxwell_frame
        - mutual_frame
        - problems

::: gsim.palace.load_capacitance
    options:
      show_source: false

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
hints. The current cloud API does not accept a separate metadata parameter, so
the metadata travels as an input artifact pending that API integration.
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

::: gsim.palace.MeshConfig
    options:
      show_source: false
      inherited_members: false
      members:
        - coarse
        - default
        - fine

::: gsim.palace.generate_mesh
    options:
      show_source: false

## Nets

::: gsim.palace.Nets
    options:
      show_source: false
      inherited_members: false
      members:
        - net_at

::: gsim.palace.Net
    options:
      show_source: false
      inherited_members: false
      members:
        - layers

::: gsim.palace.extract_nets
    options:
      show_source: false

## Stack

::: gsim.palace.LayerStack
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.palace.Layer
    options:
      show_source: false
      inherited_members: false
      members: false
