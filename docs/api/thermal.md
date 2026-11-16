# Thermal guide

Use [`gsim.thermal`][gsim.thermal] to find the heater power that shifts a waveguide's phase by π, how fast the heater
switches, and how much it heats its neighbours. It solves heat conduction in a 2D cross-section of an infinitely long
heater over one or more waveguides. Then it re-solves each waveguide's optical mode with the heated refractive index.

The solvers are [scikit-fem](https://scikit-fem.readthedocs.io) (heat) and [femwell](https://helgegehring.github.io/femwell/)
(modes). Install them with the `femwell` extra:

```bash
pip install "gsim[femwell]"
```

Importing `gsim.thermal` works without the extra. Meshing or solving needs it and raises an error that names the extra.

## Configure and solve

The simulation is set up by physical concern, like [`gsim.tcad.Simulation`][gsim.tcad.Simulation] and
[`gsim.fdtd.Simulation`][gsim.fdtd.Simulation]. A TiN heater 2 µm above a 500 × 220 nm silicon strip, 320 µm long:

```python
import gdsfactory as gf

from gsim import thermal

gf.gpdk.PDK.activate()
xs = gf.cross_section.strip_heater_metal(width=0.5, heater_width=2.0)
component = gf.components.straight(length=320.0, cross_section=xs)

sim = thermal.Simulation()
sim.geometry(component, stack=stack)  # a gsim LayerStack: heights and materials of every layer
sim.domain(plane="x=160", window=(-30.0, 30.0))
sim.heater(layers="heater", length_um=320.0)
sim.waveguides.add(name="arm", position_um=0.0)

result = sim.solve(wavelength_um=1.55)
step = sim.solve_transient(power_mw=result.p_pi_mw, duration_us=60.0)
result.report(step)  # P_pi, efficiency, R, V_pi, peak heater rise, tau
```

The [thermo-optic phase shifter tutorial](../nbs/thermal_phase_shifter.md) builds the `stack` for a published heater.
It then compares the result with the paper's simulations and measurements, and checks convergence.

Calling a concern applies several settings at once and validates them together; a failed call changes nothing. Single
attributes can also be assigned, `sim.heater.length_um = 640.0`, and a whole concern replaced by a model or a mapping,
`sim.solver = {"precision": "fast"}`. The constructor takes the same concerns as keyword arguments:

```python
sim = thermal.Simulation(
    geometry={"component": component, "stack": stack},
    domain={"plane": "x=160", "window": (-30.0, 30.0)},
    heater={"layers": "heater", "length_um": 320.0},
    waveguides=[{"name": "arm", "position_um": 0.0}],
    solver={"precision": "fast"},
)
```

The meshes are built on the first solve and kept, with the steady solve, until an input they depend on changes. Editing
a concern, the stack or the component in place is enough: the next solve starts again from it. A change to the
transient's time stepping alone keeps the steady solve.

## Units and conventions

| Quantity                             | Unit                               |
| ------------------------------------ | ---------------------------------- |
| Lengths, positions, Window, depth    | µm                                 |
| Heater power (total over its length) | mW                                 |
| Temperature                          | K, as a rise above ambient         |
| Time                                 | µs                                 |
| Resistance, voltage                  | Ω, V                               |
| Phase                                | rad                                |
| Material properties (`materials`)    | SI: W/m/K, J/kg/K, kg/m³, 1/K, S/m |
| Heat transfer coefficient            | W/m²/K                             |

Field and argument names carry their unit: `length_um`, `power_mw`, `p_pi_mw`, `tau_us`. Material properties and the
heat transfer coefficient are SI, like the stack's material records.

The model is linear in power, because k, ρc_p and dn/dT do not depend on temperature. One solve therefore gives every
power. P_π does not depend on the heater length in 2D. The length only sets the resistance, V_π and the phase at a
given power.

## Concerns

- `geometry(component, *, stack=...)`: the gdsfactory component (heater and waveguides) and its gsim `LayerStack`.
- `domain(...)`: `plane`, the cut (`"x=<um>"` or `"y=<um>"`), across the heater and the waveguides; `window`, the
  in-plane `(min, max)` extent of the thermal Window in µm; `depth_um`, the Modelled depth.
- `heater(...)`: `layers`, the stack layer name(s) of the heater conductors; `length_um`, the heater length.
- `waveguides`: `add(name=, position_um=, layer=None, window=None, polarization="TE")`, `remove(name)`,
  `sim.waveguides[name]`, iteration. The first waveguide added is the phase shifter (`waveguides.phase_shifter`); every
  other one is a crosstalk victim. Each gets its own optical window around its drawn core.
- `materials`: per-run overrides, `sim.materials["tin"] = {"conductivity": 1.6e6}`, keyed on the material name
  (case-insensitive). The fields are `thermal_conductivity`, `specific_heat`, `mass_density`, `dn_dT`, `conductivity`
  (electrical), `permittivity` and `loss_tangent`.
- `boundary(...)`: `top` and `sides`, `"adiabatic"` (default) or `"convective"`; `heat_transfer_coefficient` (10 W/m²/K).
- `solver(...)`: `precision`, `mesh_size_um`, `degree`, `optical_order`, `time_steps` and `theta` (see below).

The cut must be vertical and cross the heater and the waveguides. Two parameters belong to the model, not to the
device, and are worth a convergence check:

- **Window** (`domain.window`): the in-plane extent. Its sides are adiabatic, so make the half-width at least five times
  the oxide thickness between the heater and the substrate.
- **Modelled depth** (`domain.depth_um`): how much substrate lies under the stack. Its bottom is the heat sink, at
  ambient. A 2D section has no steady state without a sink at a finite distance. The temperature rise also grows slowly
  (as a logarithm) with depth, because a 2D model cannot spread heat along the heater the way the real chip does. The
  default is the stack's substrate thickness.

A material override keys on the material name, so every layer made of that material changes. To give the 220 nm core a
thin-film conductivity without changing the bulk substrate, give the core its own material name in the stack. The
tutorial shows how.

Free convection in air (h ≈ 10 W/m²/K) is negligible next to conduction into the substrate, which is why adiabatic is
the default boundary.

## Numerical settings

- [`Solver`][gsim.thermal.Solver]: `precision="fast" | "default" | "accurate"` scales the mesh and sets the number of
  transient time steps. `mesh_size_um` caps the element size per region, and `time_steps` overrides the step count.
  `degree=1 | 2 | 3` sets the Lagrange element degree of the heat solve (default 2). On the same mesh, degree 2 cuts
  the temperature error of degree 1 by about two orders of magnitude for a similar solve time. The optical Mode solve
  does not depend on it. `optical_order=1 | 2` sets the finite element order of the optical Mode solves (default 2, as
  in the femwell adapter). On femwell's TiN heater, P_π at order 2 moves by 0.02 % from `"fast"` to `"accurate"`
  precision, against 0.6 % at order 1, for about 3 times the solve time. `theta` picks the time scheme: 1 (backward
  Euler, default) or 0.5 (Crank-Nicolson).

## Results

- [`ThermalResult`][gsim.thermal.ThermalResult]: `p_pi_mw`, `efficiency_rad_per_mw`, `resistance_ohm`, `v_pi_v`,
  `crosstalk_rad` (on each victim at P_π), `crosstalk_ratio` (victim phase over phase-shifter phase),
  `thermal_resistance_k_per_mw` (at each core), `dneff_per_mw`, `temperature(power_mw)`, `phase(power_mw)` and
  `drive_power(voltage_v=… | current_a=…)`, which turns a drive voltage or current into heater power in mW.
- [`TemperatureMap`][gsim.thermal.TemperatureMap]: the temperature rise at one power, with `at(position_um, z_um)`,
  `peak(layer)` and `plot(xlim=, zlim=)`. Pass `ambient_k=` (e.g. 293.15) to `at` or `peak` for an absolute
  temperature instead of the rise.
- [`StepResponse`][gsim.thermal.StepResponse]: the phase over time after the heater is switched on (`time_us`,
  `phase_rad`, `temperature_k` at each core). It gives `tau_us` (the 10–90 % rise time) and `tau_1e_us` (the time to
  reach 1 − 1/e, about 63 %). In this linear model the fall time equals the rise time.
- [`HeaterReport`][gsim.thermal.HeaterReport]: the figures of merit in one table. It prints as text and renders as
  HTML in notebooks.

## Limits of the 2D model

- No heater ends, contacts, meanders or via spreading. The heater is infinitely long.
- k, ρc_p and dn/dT do not depend on temperature. At a rise of a few tens of kelvin this is a few-percent effect.
- The heater current runs along the waveguide, with uniform Joule heating over each conductor's cross-section.
- No free-carrier or loss change with temperature.

## Functional core

These lower-level building blocks are public for custom pipelines. They work on meshes and arrays, not on components:

- [`build_cross_section_mesh`][gsim.thermal.build_cross_section_mesh] and
  [`CrossSectionMesh`][gsim.thermal.CrossSectionMesh]
- [`solve_steady`][gsim.thermal.solve_steady], [`solve_transient`][gsim.thermal.solve_transient] and
  [`HeatOperator`][gsim.thermal.HeatOperator]
- [`joule_heat_density`][gsim.thermal.joule_heat_density] and
  [`drive_power_per_length`][gsim.thermal.drive_power_per_length]
- [`thermo_optic_permittivity`][gsim.thermal.thermo_optic_permittivity],
  [`solve_thermo_optic_shift`][gsim.thermal.solve_thermo_optic_shift], [`ModeShift`][gsim.thermal.ModeShift] and
  [`sample_temperature`][gsim.thermal.sample_temperature]

A solved temperature rise is a vector of degrees of freedom (DOFs): the mesh vertices first, in `mesh.p` order, then the
edge and interior points of a degree 2 or 3 solve. `sample_temperature` reads it at any point, and works out the
degree from the vector's length.
