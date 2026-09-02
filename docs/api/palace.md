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
        - plot_3d
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
        - plot_3d
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
        - plot_3d
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run

## Setup visualization

PALACE meshes can be viewed with the shared, notebook-friendly Three.js
viewer. The existing `plot_mesh()` method remains available for PyVista
rendering.

```python
sim.mesh()
sim.plot_3d()                    # solid interactive physical groups
sim.plot_3d(show_mesh=True)      # add mesh edges
```

The footer reports the exact number of highest-dimensional finite elements in
the Gmsh file. Its heading and count label can be customized with
`footer_title` and `cell_count_label`.

## Mesh

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
