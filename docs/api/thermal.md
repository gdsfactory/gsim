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

## The canonical script

A TiN heater 2 µm above a 500 × 220 nm silicon strip, 320 µm long:

```python
import gdsfactory as gf

from gsim.thermal import ThermalSim

gf.gpdk.PDK.activate()
xs = gf.cross_section.strip_heater_metal(width=0.5, heater_width=2.0)
component = gf.components.straight(length=320.0, cross_section=xs)

sim = ThermalSim()
sim.set_geometry(component)
sim.set_stack(stack)  # a gsim LayerStack: heights and materials of every layer
sim.set_cross_section("x=160", window=(-30.0, 30.0))
sim.set_heater("heater", length=320.0)
sim.add_waveguide("arm", position=0.0)

result = sim.solve(wavelength=1.55)
step = sim.solve_transient(power=result.p_pi, duration=60.0)
result.report(step)  # P_pi, efficiency, R, V_pi, peak heater rise, tau
```

The [thermo-optic phase shifter tutorial](../nbs/thermal_phase_shifter.md) builds the `stack` for a published heater.
It then compares the result with the paper's simulations and measurements, and checks convergence.

## Units and conventions

| Quantity                             | Unit                               |
| ------------------------------------ | ---------------------------------- |
| Lengths, positions, Window, depth    | µm                                 |
| Heater power (total over its length) | mW                                 |
| Temperature                          | K, as a rise above ambient         |
| Time                                 | µs                                 |
| Resistance, voltage                  | Ω, V                               |
| Phase                                | rad                                |
| Material properties (`set_material`) | SI: W/m/K, J/kg/K, kg/m³, 1/K, S/m |

The model is linear in power, because k, ρc_p and dn/dT do not depend on temperature. One solve therefore gives every
power. P_π does not depend on the heater length in 2D. The length only sets the resistance, V_π and the phase at a
given power.

## Setting up

- [`ThermalSim`][gsim.thermal.ThermalSim]: `set_geometry`, `set_stack`, `set_cross_section`, `set_heater`,
  `add_waveguide`, `set_material`, `set_boundary`.

The cut must be vertical (`"x=…"` or `"y=…"`) and cross the heater and the waveguides. Two parameters belong to the
model, not to the device, and are worth a convergence check:

- **Window** (`window=`): the in-plane extent. Its sides are adiabatic, so make the half-width at least five times the
  oxide thickness between the heater and the substrate.
- **Modelled depth** (`depth=`): how much substrate lies under the stack. Its bottom is the heat sink, at ambient. A 2D
  section has no steady state without a sink at a finite distance. The temperature rise also grows slowly (as a
  logarithm) with depth, because a 2D model cannot spread heat along the heater the way the real chip does. The default
  is the stack's substrate thickness.

The first waveguide you add is the phase shifter. Every other waveguide is a crosstalk victim. Each waveguide gets its
own optical window around its drawn core.

`set_material(name, **properties)` overrides a material for one simulation. It keys on the material name, so every
layer made of that material changes. To give the 220 nm core a thin-film conductivity without changing the bulk
substrate, give the core its own material name in the stack. The tutorial shows how.

`set_boundary` chooses adiabatic or convective top and sides; settings you leave out keep their current value.
Free convection in air (h ≈ 10 W/m²/K) is negligible next
to conduction into the substrate, which is why adiabatic is the default.

## Numerical knobs

- [`ThermalOptions`][gsim.thermal.ThermalOptions]: `precision="fast" | "default" | "accurate"` scales the mesh and
  sets the number of transient time steps. `max_size` caps the element size per region, and `time_steps` overrides the
  step count. `degree=1 | 2 | 3` sets the Lagrange element degree of the heat solve (default 2). On the same mesh,
  degree 2 cuts the temperature error of degree 1 by about two orders of magnitude for a similar solve time. The
  optical Mode solve does not depend on it.

## Results

- [`ThermalResult`][gsim.thermal.ThermalResult]: `p_pi`, `efficiency`, `resistance`, `v_pi`, `crosstalk` (rad on each
  victim at P_π), `crosstalk_ratio` (victim phase over phase-shifter phase), `thermal_resistance` (K/mW at each core),
  `dneff_per_power`, `temperature(power)`, `phase(power)` and `drive_power(voltage=… | current=…)`, which turns a
  drive voltage or current into heater power in mW.
- [`TemperatureMap`][gsim.thermal.TemperatureMap]: the temperature rise at one power, with `at(position, z)`,
  `peak(layer)` and `plot(xlim=, zlim=)`. Pass `ambient=` (in K, e.g. 293.15) to `at` or `peak` for an absolute
  temperature instead of the rise.
- [`StepResponse`][gsim.thermal.StepResponse]: the phase over time after the heater is switched on. It gives
  `tau` (the 10–90 % rise time) and `tau_1e` (the time to reach 1 − 1/e, about 63 %). In this linear model the fall time
  equals the rise time.
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
