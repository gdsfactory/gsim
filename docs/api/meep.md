# Meep API

## Material cards

`gsim.meep` resolves each material name from the active project's
`MaterialCard` registry, then falls back to gsim's built-in cards. The model
authored in the card is authoritative: scalar index or permittivity remains
nondispersive, while Sellmeier, Lorentz, and Drude models are serialized as
Meep susceptibility terms. There is no solver-level automatic dispersion
threshold.

The built-in `SiO2` fallback is a lossless two-pole reduction of Malitson's
fused-silica model over 0.4–2.0 µm. The original three-pole Malitson model and
the Arosa and de la Fuente model remain available as `SiO2-Malitson` and
`SiO2-Arosa`, respectively. Project PDK material cards take precedence over
these fallbacks.

Use the public validator before constructing a simulation when accepting cards
from another source:

```python
from gsim.meep import validate_meep_material_card

validate_meep_material_card(card, wavelength_range_um=(1.5, 1.6))
```

The adapter rejects tabulated, polynomial, external-reference, Cauchy, Debye,
and pole-residue dispersion because those representations require a causal
Lorentz/Drude fit before Meep can use them. It also rejects unsupported loss,
magnetic, and perturbation fields rather than silently dropping them.

## Simulation

::: gsim.meep.Simulation
    options:
      show_source: false
      inherited_members: false
      members:
        - geometry
        - source
        - domain
        - solver
        - validate_config
        - write_config
        - plot_2d
        - plot_3d
        - run
        - start
        - upload
        - get_status
        - wait_for_results

## Configuration

::: gsim.meep.validate_meep_material_card
    options:
      show_source: false

::: gsim.meep.Geometry
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.meep.Domain
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.meep.ModeSource
    options:
      show_source: false
      inherited_members: false
      members: false

## Results

::: gsim.meep.SParameterResult
    options:
      show_source: false
      inherited_members: false
      members:
        - from_csv
        - from_directory
        - plot
        - show_animation
        - show_diagnostics
