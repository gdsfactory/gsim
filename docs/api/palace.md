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

Malformed numeric cells report the CSV path, row, and column. Phase and frequency
must be finite; dB magnitudes also allow negative infinity.

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
