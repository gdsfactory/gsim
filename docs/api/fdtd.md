# FDTD API

`gsim.fdtd` generates the coarse tetrahedral mesh and validated `config.json`
consumed by ZapFDTD. The backend voxelizes this mesh onto its own Yee grid, so
the Gmsh mesh does not need to resolve the electromagnetic fields.

## PDK-native workflow

Pass the PDK module when it exposes project-level `MATERIAL_CARDS`; otherwise,
pass a PDK object or use the active PDK. Material names are resolved exactly,
using the project's cards first and gsim's built-in cards as fallbacks.

```python
import gpdk

from gsim import fdtd

simulation = fdtd.Simulation(pdk=gpdk)
simulation.geometry("mmi1x2")
artifacts = simulation.write("fdtd_output")

print(artifacts.mesh_path)  # fdtd_output/mesh.msh
print(artifacts.config_path)  # fdtd_output/config.json
```

The generated mesh is ASCII Gmsh MSH 2.2 with linear tetrahedra for material
regions and linear triangles for `port_<name>` groups. Geometry and wavelength
values in the artifacts are in nanometers. PML extrusion is left to ZapFDTD.

## Initial geometry limits

The first backend supports axis-aligned optical ports, one connected polygon per
material-bearing layer, and vertical or constant-angle sidewalls. It rejects
ambiguous geometry, unsupported `bias`/`z_to_bias` profiles, and lossy material
snapshots because ZapFDTD config schema version 1 accepts only real scalar
refractive indices.

## Reference

::: gsim.fdtd.Simulation
    options:
      show_source: false

::: gsim.fdtd.SimulationArtifacts
    options:
      show_source: false

::: gsim.fdtd.MeshManifest
    options:
      show_source: false

::: gsim.fdtd.ZapConfig
    options:
      show_source: false
