# Optical material library

`GSIM_MATERIAL_CARDS` contains the built-in fallback optical models. Project and PDK cards take precedence.

<!-- markdownlint-disable MD033 -->

<style>
  .md-typeset__table {
    width: 100%;
  }

  .md-typeset table:not([class]) {
    display: table;
    table-layout: fixed;
    width: 100%;
  }

  .md-typeset table:not([class]) th:nth-child(1) {
    width: 28%;
  }

  .md-typeset table:not([class]) th:nth-child(2) {
    width: 31%;
  }

  .md-typeset table:not([class]) th:nth-child(3) {
    width: 15%;
  }

  .md-typeset table:not([class]) th:nth-child(4) {
    width: 26%;
  }
</style>

<!-- markdownlint-enable MD033 -->

## Built-in cards

### Silicon

<!-- markdownlint-disable MD013 MD033 MD060 -->

| Lookups                       | Supported values                     | Defaults       | Paper                                                             |
| ----------------------------- | ------------------------------------ | -------------- | ----------------------------------------------------------------- |
| `Si` (default), `Si-Salzberg` | `λ = 1.357–11.04 µm`<br>`T` fixed    | `T = 299.15 K` | [Salzberg and Villa 1957](https://doi.org/10.1364/JOSA.47.000244) |
| `Si-Li-293K`                  | `λ = 1.2–14.0 µm`<br>`T = 100–750 K` | `T = 293 K`    | [Li 1980](https://doi.org/10.1063/1.555624)                       |

### Silicon nitride

| Lookups                     | Supported values     | Defaults | Paper                                                    |
| --------------------------- | -------------------- | -------- | -------------------------------------------------------- |
| `SiN` (default), `SiN-Luke` | `λ = 0.310–5.504 µm` | —        | [Luke et al. 2015](https://doi.org/10.1364/OL.40.004823) |

### Fused silica

| Lookups                           | Supported values               | Defaults    | Paper                                                   |
| --------------------------------- | ------------------------------ | ----------- | ------------------------------------------------------- |
| `SiO2` (default), `SiO2-Malitson` | `λ = 0.21–6.7 µm`<br>`T` fixed | `T = 293 K` | [Malitson 1965](https://doi.org/10.1364/JOSA.55.001205) |

### Lithium niobate

| Lookups                         | Supported values                                        | Defaults      | Paper                                                          |
| ------------------------------- | ------------------------------------------------------- | ------------- | -------------------------------------------------------------- |
| `LN` (default), `LiNbO3-Zelmon` | `λ = 0.4–5.0 µm`<br>`axis = o or e`<br>`T` fixed        | `T = 21 °C`   | [Zelmon et al. 1997](https://doi.org/10.1364/JOSAB.14.003319)  |
| `LiNbO3-MgO5-Gayer`             | `λ = 0.5–1.62 µm`<br>`axis = o or e`<br>`T = 20–100 °C` | `T = 24.5 °C` | [Gayer et al. 2008](https://doi.org/10.1007/s00340-008-2998-2) |

<!-- markdownlint-enable MD013 MD033 MD060 -->

## Refractive index

Curves span each model's full wavelength validity and use its default temperature.

<iframe
  src="assets/plots/material-index-vs-wavelength.html"
  title="Refractive index versus wavelength for the built-in optical material models"
  loading="lazy"
  style="width: 100%; height: 500px; border: 0;"
></iframe>

```python
from gsim.common.materials import resolve_material_snapshot

snapshot = resolve_material_snapshot("Si", wavelength_um=1.55)
print(snapshot.refractive_index)
```

Lookup is case-sensitive. A project card with the same name replaces that fallback.

## Palace RF cards

Stack extraction reads `PDK.material_cards` (or a module's `MATERIAL_CARDS`). An RF card matching a layer's exact
material token takes precedence over the legacy material database. Its resolved properties and full card provenance
survive stack YAML export/import. Explicit `sim.set_material(...)` overrides take precedence over both.

The RF adapter supports constant scalar or diagonal relative permittivity and conductivity, including dielectric loss
from `eps_imag / eps_real`. It rejects unsupported dispersion, magnetic models, validity constraints, variation, and
simultaneous nonzero conductivity and dielectric loss instead of substituting generic values. Materials without RF cards
retain the legacy behavior. Constant RF cards are used unchanged across a Palace sweep; this does not establish a
measured validity band.

IHP versions publishing RF cards use layer-specific metal, contact, via, and MIM electrode conductivities from the
[IHP-linked Palace stack](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/blob/df29aa608be858f53fbce46f81238b5534812d9b/workflow/SG13G2_200um.xml).
For example, Metal1 uses 21.64 MS/m and Via1 uses 1.66 MS/m. The substrate card uses 2 S/m (50 ohm cm); it applies when
the substrate is included. Installing an older IHP version without cards retains the generic material database defaults.
