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

It brings `devsim` and `scipy-openblas32`, which gives DEVSIM a BLAS/LAPACK library on Linux, Windows and macOS. On
macOS it needs macOS 15 or later on Apple silicon, the only macOS wheel DEVSIM ships. `import gsim.tcad` works without
DEVSIM; `solve()` and `sweep()` raise an error naming the extra.

### BLAS/LAPACK library

DEVSIM loads its BLAS/LAPACK library once, while it is first imported, from the file named by the `DEVSIM_MATH_LIBS`
environment variable. You normally set nothing. Before gsim first imports DEVSIM it sets three variables, so DEVSIM uses the
OpenBLAS that `scipy-openblas32` ships:

- `DEVSIM_MATH_LIBS`: the first `libscipy_openblas*` shared library in the folder that `scipy_openblas32.get_lib_dir()`
  returns (`.dll` on Windows, `.dylib` on macOS, `.so` otherwise).
- `DEVSIM_BLAS_PREFIX`: `scipy_`.
- `DEVSIM_BLAS_SUFFIX`: `_`.

The prefix and suffix are there because `scipy-openblas32` exports its routines under prefixed names, such as `scipy_dgetrf_`.
DEVSIM 2.11.1 and later looks each routine up as prefix + name + suffix.

gsim leaves all three variables alone in these cases:

- DEVSIM is already imported in the process, for example by your own `import devsim`.
- `DEVSIM_MATH_LIBS` is already set. gsim then sets neither the prefix nor the suffix, so those two are yours to set when your
  library exports prefixed or suffixed names.
- `scipy-openblas32` is not installed, or its folder holds no `libscipy_openblas*` library. DEVSIM then runs its own
  library discovery.

If DEVSIM cannot find a BLAS/LAPACK library, gsim raises an `ImportError` that names these remedies. DEVSIM cannot initialise
twice in one process, so restart Python (or the notebook kernel) after you change any of these variables.

On Windows 11 with DEVSIM 2.11.2, a fresh process with only `DEVSIM_MATH_LIBS` set to scipy-openblas32's
`libscipy_openblas.dll` also worked. If DEVSIM fails to initialise after you set `DEVSIM_MATH_LIBS` by hand, set
`DEVSIM_BLAS_PREFIX=scipy_` and `DEVSIM_BLAS_SUFFIX=_` as well, the values gsim uses for this library.

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
  `semiconductor_permittivity`, the static relative permittivity of every doped Region (11.1, DEVSIM's value);
  `mobility`, a `MobilityModel` (Masetti silicon); `insulators`, Region to static relative permittivity.
- `doping`: the doping profiles, with `add(profile)`, `remove(profile)`, `clear()`, iteration and `len()`.
- `contacts`: `add(name=, semiconductor=, electrode=)`, `remove(name)`, `sim.contacts[name]`, iteration.
- `interfaces`: `add(name=, region_a=, region_b=)`, `remove(name)`, `sim.interfaces[name]`, iteration.
- `solver(...)`: mesh resolution (`mesh_preset`, `refined_mesh_size_um`, `max_mesh_size_um`, `refinement_boxes`,
  `mesh_grading`),
  Newton convergence (`absolute_error`, `relative_error`, `max_iterations`, `bias_step_v`, `extended_precision`), the
  small-signal frequencies (`small_signal_freq_hz`, `junction_freq_hz`) and `claim_devsim_namespace`.

Without grading, the mesh refines along every internal curve and the element size grows about 2 per µm from it, whatever
the sizes set; a refinement box holds its size inside and steps to the surrounding size at its edge.
`mesh_grading=MeshGrading(growth_per_um=0.25)` replaces that: the size is `refined_mesh_size_um` on the conductors and
the declared Interfaces, and a box's own size inside each refinement box, and grows from them by `growth_per_um` per µm
of distance up to `max_mesh_size_um`. `sources=` limits the features it grows from (`"conductors"`, `"junctions"`,
`"fine_regions"`). A growth of 0.1 to 0.25 per µm keeps the oxide elements around a small section well shaped at a node
count close to the ungraded mesh. Grading is off by default, so existing meshes and results do not change.

`refined_mesh_size_um` does not set the same size in the two cases. Without grading, the elements on the internal curves
are half of `refined_mesh_size_um`; with grading, the elements on the conductors and Interfaces are
`refined_mesh_size_um` itself. To keep the same size at the contacts and the junction when you turn grading on, halve
`refined_mesh_size_um`.

A layer with a numeric `mesh_resolution` keeps that size with grading whatever `refined_mesh_size_um` is, as long as
`"fine_regions"` is among the sources; without grading it gets its own size only when it is finer than
`refined_mesh_size_um`. Any size above `max_mesh_size_um` is held at `max_mesh_size_um`.

### Doping

A profile names the mesh Region it applies to and its dopant type. `StepDoping` is uniform inside an optional box,
`GaussianDoping` and `ImplantDoping` are analytic implants, `TableDoping` interpolates a sampled profile (a SIMS depth
profile, a process simulator's output) and `CallableDoping` evaluates a function of `(x, y)` in µm. Profiles on the same
Region add up; donors and acceptors compensate in the net doping and add up as scatterers in the mobility.
Box and window edges, the implant surface and the ends of a zero-fill table are inclusive, with a 1e-9 µm slack that
absorbs DEVSIM's µm to cm rounding. Two boxes on the same Region that abut therefore both dope the nodes on their shared
edge, which then carry the sum; one box covering both dopes that edge once.

### Contacts, interfaces and insulators

Every Region of the device takes one role, and the declarations state it:

- A **doped Region** carries at least one doping profile. It holds the potential and both carrier densities.
- A **Contact** is where an electrode meets a doped Region: an ohmic contact whose bias the solve sets.
  `semiconductor` is the doped Region it injects into; `electrode` is the Region on the other side, the metal or any
  Region that is neither doped nor insulating. Where the Contact ends on another solved Region (an Interface to the
  next doped Region, or the oxide), that Region's node touches the same metal and the Contact holds it too: a doped
  Region's node as an ohmic contact of its own doping, an insulator's at the potential of the Contact's node beside it.
  Its charge and current count towards the Contact. A node on another Contact's curve stays with that Contact, and a
  node two Contacts reach at a shared end goes to the one declared first. Where the doping steps at the end, say a
  contact implant beside a plus implant, each side is neutral at the metal's Fermi level, so the potential steps there
  by the step's built-in voltage, `V_t ln(N_high/N_low)`: the drop a real high-low junction makes over a Debye length.
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

The solve takes DEVSIM's silicon parameters, with two changes: the permittivity can be set, and the intrinsic carrier
density follows the temperature.

- **Permittivity:** `materials.semiconductor_permittivity` sets the static relative permittivity of every doped Region.
  It defaults to 11.1, DEVSIM's value (`gsim.tcad.silicon.SILICON_RELATIVE_PERMITTIVITY`), and does not reach the
  insulators, which keep their own. A `BiasSweepResult` records the value it was solved with in
  `semiconductor_permittivity`. The RF and optical stacks use 11.9 for silicon.
- **Intrinsic density:** `n_i(T) = n_i(300 K) (T/300)^1.5 exp(E_g(300)/2kT_300 - E_g(T)/2kT)`, with Varshni's
  band gap and DEVSIM's `n_i(300 K) = 1e10` cm⁻³ (`gsim.tcad.silicon.intrinsic_density_cm3`). A 300 K solve is
  DEVSIM's own; away from 300 K the built-in voltage falls with temperature as it should.
- **Temperature:** `materials.temperature_k` sets the thermal voltage, the intrinsic density and the SRH trap densities.
  The Masetti mobility is its 300 K fit at every temperature.

## Edge couples and node volumes

DEVSIM assembles its equations by finite boxes: each mesh edge carries a couple, the length of the boundary shared by
its two nodes' boxes, and each node a volume. DEVSIM's own couples and volumes take the absolute cotangent of the angle
opposite each edge, which is wrong on every obtuse triangle: the potential equation gains coupling it should not have,
capacitance comes out too high and the node volumes of a Region add up to more than its area. gsim corrects this: each
device is assembled with the signed cotangent couples and the node volumes that go with them, so the potential equation
is the P1 finite-element Laplacian on any triangulation and the node volumes of each Region add up to its area.

The size of the correction depends on the mesh. Measured:

- A carriers-off capacitor on a mesh with obtuse triangles in both dielectrics now matches an independent P1
  finite-element solve (scikit-fem) to 1e-14; DEVSIM's own couples read 35 % high on that mesh.
- On the padded PN diode of the test suite (0, 0.5 and 1 V reverse bias), the small-signal capacitance falls by 9.3 %
  to 12.8 % and the reverse current by up to 19 %, and the capacitance moves closer to the depletion approximation.
  Carrier totals move by 0.2 %.

This correction lands without a transition period and with no option to restore DEVSIM's own couples, because those
give wrong answers on any mesh with obtuse triangles. It is planned for gsim 0.10.0. Up to 0.9.0, every device was
assembled with DEVSIM's own couples and volumes. Anyone with saved `gsim.tcad` results (capacitance, currents,
admittance) should rerun them: the same mesh and settings now give the corrected values. To reproduce an earlier
result exactly, run it with gsim 0.9.0 or older.

Signed couples can be negative on an edge whose opposite angles add up to more than 180°. The Scharfetter-Gummel flux
takes them as they come; fewer obtuse triangles in the doped Regions keep the solve closer to a monotone scheme. Node
volumes can be negative too, on a node at the end of the long edge of strongly obtuse triangles: two 150° triangles
sharing their long edge of length `L` give each end of that edge `-0.18 L²`. A Region's volumes still add up to its
area, but the space charge, recombination and stored charge of a node are weighted by its volume, so a negative volume
flips their sign at that node. On a 1e17/1e17 cm⁻³ junction meshed with 640 triangles, from 0 to 2 V reverse:

- With about 150 negative couples and no negative volume, the depletion width stays within 0.3 % of a right-angle mesh;
  DEVSIM's own couples read 1.5 % to 3.4 % low.
- With strongly obtuse triangles (about 180 negative couples and 3 to 7 negative volumes), it stays within a few
  percent, depending on the mesh: 0.2 % to 9 % over twelve random meshes, not tied to the number of negative volumes.
  DEVSIM's own couples read 17 % to 39 % low on the same meshes.

## Limits

- DEVSIM keeps meshes, devices and its circuit in one process-wide namespace. By default each device setup claims it
  (`solver.claim_devsim_namespace`), so one charge-transport device is live at a time.
- The physics is DEVSIM's Scharfetter-Gummel drift-diffusion for silicon, with Shockley-Read-Hall recombination and the
  doping-dependent mobility of `materials.mobility`. There is no field-dependent mobility (velocity saturation), no
  impact ionisation and no quantum correction.
- Contacts, interfaces and insulators are declared, never inferred from the layer stack.
- Strongly obtuse triangles in a doped Region can stop the Newton solve from converging with the signed couples, on
  meshes where DEVSIM's own couples converge (to a less accurate answer). Keep such triangles out of doped Regions.
