# Palace modeling roadmap

This file tracks possible extensions to `gsim.palace`. It does not imply an implementation order or timeline.

## Conductor models

- [ ] Allow each conductor layer to select an appropriate Palace model: finite-conductivity shell, PEC surface,
  impedance sheet, conductive volume, or superconducting volume.
- [ ] Document when each model is valid and its computational cost. In particular, conductive volumes require elements
  inside the metal and may require enough resolution to capture skin depth, while surface models are generally more
  efficient for good RF conductors.

Current behavior: ordinary finite-thickness conductor layers use hollow finite-conductivity shells, planar conductors
can use zero-thickness PEC surfaces, and vias use conductive volumes.

## Mesh refinement

- [ ] Add initial mesh-size overrides per layer and physical group, including separate controls for metal edges, vias,
  ports, and dielectric regions.
- [ ] Make Palace region-based and solution-driven adaptive refinement easier to configure, inspect, and reuse in
  convergence studies.

## Simulation-domain faces

- [ ] Configure the domain margin independently on `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, and `zmax`.
- [ ] Assign outer boundary conditions independently on those six faces, including absorbing, PEC, PMC, periodic, and
  port boundaries where valid. Palace supports absorbing boundary conditions, not a true volumetric PML.

## Facet-specific conductor properties

- [ ] Allow horizontal (`*_xy`) and sidewall (`*_z`) conductor surfaces to use different boundary parameters, especially
  effective thickness. Keep these as separate physical groups for Palace even when the mesh viewer combines them into
  one material control.
