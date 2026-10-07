# Palace API

## Simulation Classes

::: gsim.palace.DrivenSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_driven
        - set_material
        - set_numerical
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
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_eigenmode
        - set_material
        - set_numerical
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
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_electrostatic
        - set_material
        - set_numerical
        - add_terminal
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

## Floquet eigenmodes

Set the signed cell phase in radians. The wave vector uses the measured donor-to-receiver mesh translation; the eigenvalue
search frequency does not determine the cell length. An optional `periodic_length` in mesh units (micrometers for generated
meshes) checks that the periodic faces have the expected separation:

```python
from gsim.palace import EigenmodeSim

sim = EigenmodeSim()
sim.set_eigenmode(target=40e9, floquet=True, phi_target=-0.4, periodic_length=100.0)
sim.eigenmode.compute_floquet_wave_vector(periodic_axis="x")
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
