# Choosing a solver

Pick the row that matches the question you are asking. Each row names the gsim class that answers it, where it runs, what
drives its cost, and where to read more.

## By question

| Question | Solver and class | Runs | Cost grows with | Read |
| --- | --- | --- | --- | --- |
| S-parameters of a photonic splitter, coupler, MMI or crossing | GDSFactory FDTD, `gsim.fdtd.Simulation` | Mesh and inputs local; solve on GDSFactory+ cloud (`run()`) | Yee cells, `grid.cell_counts.total_cells` in `metadata.json` | [Couplers and MMIs](couplers-and-mmis.md) |
| Same, for a feature FDTD does not have yet (for example a 2D XZ grating coupler) | Meep, `gsim.meep.Simulation` | Cloud (`run()`), or local (`run_local()`) with meep installed | See the [Meep guide](../api/meep.md) | [Meep: 2D grating coupler](../nbs/meep_2d_xz_gc.md) |
| Waveguide mode: n<sub>eff</sub>, loss, mode profile | femwell, `gsim.femwell.solve_modes`, on a 2D cross-section mesh | Local | 2D mesh elements; cheap | No guide page yet. See [phase shifters](phase-shifters.md), which uses it |
| Same, as a cross-check on one mesh | Palace, `BoundaryModeSim` | Local with a Palace binary (`run_local()`), or cloud | 2D mesh elements | [Palace guide](../api/palace.md) |
| Ring resonator | FDTD for the coupler section, then compose a circuit in SAX from `fdtd.s_matrix(...).sax_model()` | FDTD on cloud, SAX local | The coupler section only | No guide page yet. Export is described in the [FDTD guide](../api/fdtd.md#export-to-sax-and-touchstone) |
| Carrier-depletion phase shifter: V<sub>π</sub>L, loss | TCAD (`gsim.tcad.Simulation`, DEVSIM) for carriers, then femwell for n<sub>eff</sub>(V) | Local | 2D mesh nodes and bias points | [Phase shifters](phase-shifters.md) |
| Transmission line: S-parameters, Z<sub>0</sub>, loss | Palace `DrivenSim` with lumped CPW or wave ports | Mesh local; solve on cloud (`run()`), or local with a Palace binary | Tetrahedra and estimated field DOFs from `mesh()` | [Palace guide](../api/palace.md), [CPW with lumped ports](../nbs/palace_cpw_lumped.md), [wave ports](../palace_wave_ports.md), [line analysis with scikit-rf](../transmission_line_analysis.md) |
| Z<sub>0</sub> of a 2D line cross-section | Palace `BoundaryModeSim.add_impedance_path`, or femwell `z0_power_current` | Local | 2D mesh elements | [Palace guide](../api/palace.md) |
| Resonant frequency and Q of a resonator | Palace `EigenmodeSim`, plus `DrivenSim` on the same mesh | Mesh local; solve on cloud or local with a Palace binary | Tetrahedra and estimated field DOFs | [Superconducting resonators](superconducting-resonators.md) |
| Capacitance matrix | Palace `ElectrostaticSim` | As above | As above | [Palace guide: capacitance](../api/palace.md#capacitance) |
| Inductance of a coil or transformer | Palace `MagnetostaticSim`, or `DrivenSim` | As above | As above | [Inductor](../nbs/palace_inductor.md), [magnetostatic inductance](../nbs/palace_magnetostatic.md) |
| Thermal tuning | Not covered | None | None | [Thermal tuning](thermal-tuning.md) |

"Cloud" means GDSFactory+ cloud and needs an account. `run(check_cache=True)` reuses a finished job with byte-identical
inputs instead of submitting a new one.

## FDTD or Meep

GDSFactory FDTD is the primary engine for new photonic simulations. Meep stays supported for workflows FDTD does not cover
yet, and it will be deprecated only after FDTD reaches the required feature coverage and a migration path is documented.
The [home page](../index.md) states this, together with the FDTD roadmap.

At the time of writing, material dispersion is not accepted by the deployed FDTD solver: with `dispersion=False` each
material enters at its index at the source centre wavelength, so a broadband result carries that approximation. Read the
[FDTD guide](../api/fdtd.md) for the current state before relying on a wide band.

## What drives the cost

- **FDTD.** The number of Yee cells: the physical domain divided by `cell_size_nm` (default 60 nm), plus the PML and guard
  cells. `write()` estimates it locally. One run excites one port, so an n-port S-matrix is n runs.
- **Palace.** The number of tetrahedra and the estimated field degrees of freedom, which depend on the mesh sizes and the
  field order (default 2). `mesh()` reports both. See [cloud sizing](../api/cloud.md#sizing-metadata). There is no single
  count that guarantees a job fits in memory.
- **femwell and TCAD.** The size of a 2D mesh. Both run on a laptop. TCAD keeps one device live per process.

## Where the answer can be wrong

The solver choice does not remove the need for checks. Each task page lists them. The evidence that exists for each solver is
collected in [validation and benchmarks](validation.md), including what it does not cover.
