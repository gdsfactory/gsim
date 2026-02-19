# gsim.meep Roadmap

Features planned beyond the current S-parameter extraction workflow.

## Planned

- [ ] **Grating coupler simulation** — near-to-far-field transformation, fiber mode overlap, coupling efficiency vs wavelength
- [ ] **Resonator & Q-factor extraction** — Harminv-based resonance finding, Q/FSR extraction, SAX circuit model integration
- [ ] **Parameter sweep framework** — sweep any sim parameter, parallel cloud runs, result aggregation
- [ ] **Convergence testing** — automated resolution/domain sweeps, convergence metric, pass/fail report
- [ ] **Broadband flux monitors** — total transmission/reflection via `add_flux`, for filters/gratings/stacks
- [ ] **Dispersive materials** — Lorentz/Drude models, MEEP built-in material library (~20 fitted materials)
- [ ] **Band structure / photonic crystal** — Bloch-periodic boundaries, k-point sweeps, band diagram plotting
- [ ] **Adjoint optimization / inverse design** — topology optimization via `meep.adjoint`, design region, GDS export
