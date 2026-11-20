# Thermal tuning

gsim does not cover thermal tuning yet. This page says what is missing, and what you can do with the tools that exist.

## Status

- gsim has no heat solver. It computes no temperature field for a heater, and no thermo-optic index change from one.
- A thermal solver is under review. Until one is merged and released, treat thermal tuning as unsupported. This page makes
  no statement about when or whether that will happen.
- A search of `src/` and `docs/` for `thermal`, `heater` and `thermo-optic` finds only the items listed below, none of
  which simulates heating.

## What is there, and what it is not

| Item | What it does | What it does not do |
| --- | --- | --- |
| `tcad` `materials(temperature_k=...)` | Sets the temperature of the carrier solve: thermal voltage, intrinsic carrier density and SRH trap densities. Default 300 K. | It does not heat the device. The Masetti mobility is its 300 K fit at every temperature, and no optical index is derived from it. See the [TCAD guide](../api/tcad.md#silicon-constants). |
| `phase_offset_rad` in `gsim.common.twmzm.mzm_transfer` | A static phase delay added to one arm of a Mach-Zehnder transfer. Its docstring calls it "the heater". | It is a number you choose. No power, geometry or temperature enters. |
| A `temperature` input in an analytic material card | Nothing: the card is refused with an error that says the model "only supports wavelength inputs". | A material cannot change its index with temperature inside gsim. |
| Heater layer names in the Cornerstone overlay | A `heater` level is classified as a conductor in one PDK overlay (`gsim.common.pdk.overlays`). | It is a layer role for electrical meshing, not a thermal model. |

## A manual route, with limits

If you have a temperature-shifted refractive index or permittivity from somewhere else (a heat-transfer tool, or a
thermo-optic coefficient from your process documentation), you can feed it into the existing solvers by hand:

- **Mode index against temperature.** Pass a modified permittivity to `gsim.femwell.solve_modes(epsilon=...)`, either as a
  `{region: epsilon}` dict or as per-element values, and read n<sub>eff</sub> per temperature. The
  [phase shifter page](phase-shifters.md) shows the same call with a carrier-dependent permittivity. Only the part of the
  device that changes needs a different value.
- **Coupler or MMI response.** `fdtd.Simulation.materials(overrides={"si": <index>})` sets a constant index at the source
  wavelength, so a coupler can be simulated at two indices. See [couplers and MMIs](couplers-and-mmis.md).

In both cases the temperature profile and the thermo-optic coefficient are your inputs. gsim does not check them, and
this page has no example with results, because no part of the workflow could be validated inside gsim.

## Related

- [Phase shifters (carrier depletion)](phase-shifters.md) for the electro-optic alternative.
- [Choosing a solver](choosing-a-solver.md)
