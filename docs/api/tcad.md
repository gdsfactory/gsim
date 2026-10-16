# TCAD guide

`gsim.tcad` solves Poisson and electron/hole drift-diffusion on a 2D cross-section of a doped silicon device with
[DEVSIM](https://devsim.org). Each bias point returns the carrier concentrations, the terminal currents and the
small-signal capacitance and admittance of the swept contact. The cross-section is meshed by the same native-2D pipeline
as the Palace mode solver, so the charge solve and the RF and optical solves see the same mesh.

Use [`Simulation`][gsim.tcad.Simulation] to configure and solve, and [`BiasSweepResult`][gsim.tcad.BiasSweepResult] to
read the result. The [PN junction notebook](../nbs/tcad_pn_junction.md) walks through a complete C(V) extraction.

## Install

DEVSIM is optional. Install the `tcad` extra:

```bash
pip install "gsim[tcad]"
```

It brings `devsim` and `devsim-openblas`, which gives DEVSIM a BLAS/LAPACK library on every OS. `import gsim.tcad`
works without DEVSIM; `solve()` and `sweep()` raise an error naming the extra.

## Configure and solve

The simulation is set up by physical concern, like [`gsim.fdtd.Simulation`][gsim.fdtd.Simulation]:

```python
from gsim import tcad

sim = tcad.Simulation()
sim.geometry(component, stack=stack)
sim.domain(plane="x=0", window=(-25.0, -15.0), margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=1.0)
sim.materials(temperature_k=300.0, mobility=tcad.MobilityModel.masetti_silicon())
sim.doping.add(tcad.StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18))
sim.doping.add(tcad.StepDoping(region="n_rib", dopant_type="donor", concentration_cm3=1e18))
sim.contacts.add(name="anode", semiconductor="p_rib", electrode="metal_p")
sim.contacts.add(name="cathode", semiconductor="n_rib", electrode="metal_n")
sim.interfaces.add(name="junction", region_a="p_rib", region_b="n_rib")
sim.solver(mesh_preset="coarse", refined_mesh_size_um=0.05, relative_error=1e-8)

sim.mesh(output_dir="./tcad")
point = sim.solve(1.0, contact="cathode")  # BiasPoint
sweep = sim.sweep([0.0, 0.5, 1.0, 2.0], contact="cathode")  # BiasSweepResult
```

A positive bias on the n-side contact reverse-biases the junction. The other contacts are at 0 V: one swept by an
earlier solve is ramped back first. Between bias points the swept contact is ramped in `solver.bias_step_v` steps, and
a bias must be a finite voltage.

Calling a concern applies several settings at once and validates them together; a failed call changes nothing. Single
attributes can also be assigned, `sim.solver.relative_error = 1e-8`, and a whole concern replaced by a model or a mapping,
`sim.materials = {"temperature_k": 350.0}`. The constructor takes the same concerns as keyword arguments:

```python
sim = tcad.Simulation(
    domain={"plane": "x=0", "window": (-25.0, -15.0)},
    doping=[tcad.StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)],
    contacts=[{"name": "anode", "semiconductor": "p_rib", "electrode": "metal_p"}],
    solver={"mesh_preset": "coarse"},
)
```

## Concerns

Lengths are in µm, concentrations in cm⁻³, voltages in V, frequencies in Hz and temperatures in K. Field names carry
their unit.

- `geometry(component, *, stack=None)`: the gdsfactory component and its layer stack, a `LayerStack`, a stack YAML
  path, or `None` for the active PDK.
- `domain(...)`: `plane` (`"x=<um>"` or `"y=<um>"`); `window` and `window_z`, `(min, max)` clips of the domain in µm;
  the airbox `margin_x`, `margin_y`, `z_above`, `z_below` in µm.
- `materials(...)`: `temperature_k` (300 K); `semiconductor`, the DEVSIM material of the doped Regions (`"Silicon"`);
  `mobility`, a `MobilityModel` (Masetti silicon); `insulators`, Region to static relative permittivity.
- `doping`: the doping profiles, with `add(profile)`, `remove(profile)`, `clear()`, iteration and `len()`.
- `contacts`: `add(name=, semiconductor=, electrode=)`, `remove(name)`, `sim.contacts[name]`, iteration.
- `interfaces`: `add(name=, region_a=, region_b=)`, `remove(name)`, `sim.interfaces[name]`, iteration.
- `solver(...)`: mesh resolution (`mesh_preset`, `refined_mesh_size_um`, `max_mesh_size_um`, `refinement_boxes`),
  Newton convergence (`absolute_error`, `relative_error`, `max_iterations`, `bias_step_v`, `extended_precision`), the
  small-signal frequencies (`small_signal_freq_hz`, `junction_freq_hz`) and `claim_devsim_namespace`.

### Doping

A profile names the mesh Region it applies to and its dopant type. `StepDoping` is uniform inside an optional box,
`GaussianDoping` and `ImplantDoping` are analytic implants, `TableDoping` interpolates a sampled profile (a SIMS depth
profile, a process simulator's output) and `CallableDoping` evaluates a function of `(x, y)` in µm. Profiles on the same
Region add up; donors and acceptors compensate in the net doping and add up as scatterers in the mobility.

### Contacts, interfaces and insulators

Every Region of the device takes one role, and the declarations state it:

- A **doped Region** carries at least one doping profile. It holds the potential and both carrier densities.
- A **Contact** is where an electrode meets a doped Region: an ohmic contact whose bias the solve sets.
  `semiconductor` is the doped Region it injects into; `electrode` is the Region on the other side, the metal or any
  Region that is neither doped nor insulating.
- An **Interface** is where two doped Regions meet. Potential and carriers are continuous across it; it carries no bias.
  Declare one between every pair of touching doped Regions: two touching doped Regions without one are refused, since
  no carrier would cross between them.
- An **insulator** is a Region in `materials.insulators`. It joins the electrostatic solve with its own permittivity and
  holds no carriers. Without any, Poisson is solved in the doped silicon alone and its boundary is a zero-normal-field
  wall; with the oxide declared, the field may fringe around the junction through it. The interfaces an insulator shares
  with its neighbours are bound automatically.

A Region is doped or insulating, never both. A Contact against a declared insulator is refused: one curve cannot be both
a Contact and a semiconductor-insulator interface.

Each Contact and Interface becomes a line group of that name on the mesh, so contact and interface names are unique
across both.

## Mesh, then solve

`mesh(output_dir=...)` writes the native-2D mesh in µm and its copy in cm that DEVSIM loads, and returns a
[`TCADMesh`][gsim.tcad.TCADMesh] with both paths and the mesh's physical groups. `output_dir` is required on the first
call and remembered afterwards. Resolution comes from the `solver` concern.

The DEVSIM device is built on the first solve, and `mesh()` is never called implicitly. What a later change invalidates
depends on the concern:

- **Mesh inputs**: `geometry`, `domain`, `contacts`, `interfaces`, and the solver's mesh fields. `mesh()` records them.
  A solve after any of them changed raises `ValueError("... changed since mesh(); call mesh() again")`.
- **Device inputs**: `materials`, `doping`, and the other solver fields. A change drops the DEVSIM device and its bias
  state; the next solve builds it again on the existing mesh.

## Results

A DEVSIM 2D device is 1 cm deep, so extensive quantities are per cm of device length; each has a per-metre counterpart.

```python
sweep.voltages  # V
sweep.capacitance_f_per_cm  # small-signal C(V)
sweep.capacitance_f_per_m
sweep.admittance_s_per_cm  # complex admittance at solver.junction_freq_hz
sweep.junction_branch()  # series-RC fit: r_s_ohm_m, c_j_f_per_m

point = sweep.points[-1]
point.currents_a_per_cm  # DC terminal current per contact
point.carriers.electrons_cm3, point.carriers.holes_cm3  # per mesh node
point.carriers.x_um, point.carriers.y_um  # in-plane and vertical node coordinates
```

The capacitance comes from an AC solve at the quasi-static `small_signal_freq_hz`. The admittance comes from a second AC
solve at `junction_freq_hz`, high enough that the capacitive current dominates the junction's DC leakage and far below
the branch's own RC rolloff, so `junction_branch()` reads the series resistance of the doped slab and the junction
capacitance a loaded-line RF model needs.

`tcad.compare_capacitance` evaluates the depletion-approximation capacitance (Sze) at the same biases and reports the
relative deviation; `tcad.estimate_depletion_width_um` reads the depletion width off a 1D cut of the carrier map. The
comparison uses the solve's own silicon constants (below) for whatever the `PNJunctionConfig` leaves at its defaults.

## Silicon constants

The solve takes DEVSIM's silicon parameters, with one change: the intrinsic carrier density follows the temperature.

- **Permittivity:** static relative permittivity 11.1, DEVSIM's value
  (`gsim.tcad.silicon.SILICON_RELATIVE_PERMITTIVITY`). The RF and optical stacks use 11.9 for silicon.
- **Intrinsic density:** `n_i(T) = n_i(300 K) (T/300)^1.5 exp(E_g(300)/2kT_300 - E_g(T)/2kT)`, with Varshni's
  band gap and DEVSIM's `n_i(300 K) = 1e10` cm⁻³ (`gsim.tcad.silicon.intrinsic_density_cm3`). A 300 K solve is
  DEVSIM's own; away from 300 K the built-in voltage falls with temperature as it should.
- **Temperature:** `materials.temperature_k` sets the thermal voltage, the intrinsic density and the SRH trap densities.
  The Masetti mobility is its 300 K fit at every temperature.

## Limits

- DEVSIM keeps meshes, devices and its circuit in one process-wide namespace. By default each device setup claims it
  (`solver.claim_devsim_namespace`), so one charge-transport device is live at a time.
- The physics is DEVSIM's Scharfetter-Gummel drift-diffusion for silicon, with Shockley-Read-Hall recombination and the
  doping-dependent mobility of `materials.mobility`. There is no field-dependent mobility (velocity saturation), no
  impact ionisation and no quantum correction.
- Contacts, interfaces and insulators are declared, never inferred from the layer stack.
