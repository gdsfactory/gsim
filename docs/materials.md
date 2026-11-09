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

### Ideal air

`Air` and its `air` alias provide a lossless, nondispersive approximation with
refractive index 1 and extinction coefficient 0. This is an ideal background
model, not a measured model of atmospheric conditions. A PDK card with the same
name takes precedence.

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
PDKs can declare `material_aliases` (or module-level `MATERIAL_ALIASES`) mapping
compatibility names to canonical card keys. These aliases also apply to layer
materials and FDTD index overrides. No general case folding is performed;
distinct project material names remain distinct. The built-in `air` alias
resolves to `Air` unless an exact project card overrides it.
At gsim's material boundary, alias targets can come from project cards, built-in
fallbacks, or explicit FDTD overrides. Missing targets are reported when used.

Meep mode setup preserves drawn `Air`/`air` voids, including inside silicon with
an air background. These regions keep their declared Z extent instead of being
discarded or extended into the lower margin. Undrawn Air declarations remain
excluded from component cross-sections.
Mode cells and sampling grids use the same non-air Z reference as the reported
domain, so unused Air padding does not enlarge the actual solver cell.
Mode geometry honors PDK `mesh_order`: lower values win overlaps, including Air
voids. Equal priorities retain declaration order; unranked legacy entries keep
their relative order and precede explicitly ranked regions.
Explicit 1D Air slabs are retained even with an Air background, so declaration
order can carve an unranked cutout from an overlapping solid.
Slab mode treats each entry as a uniform band; component cross-section mode
applies the layout masks.

## Analytic PDK cards

Project cards can use `pdk_schema.AnalyticDispersion` with scalar `n` or `n_squared` output, including AN800's modified
Sellmeier equations. The resolver converts the simulation wavelength to each declared input unit (`um`, `nm`, or `m`),
evaluates the stored equation and parameters, and enforces the card's wavelength validity. These models produce a
lossless snapshot with zero extinction coefficient.

Analytic inputs other than wavelength and `eps_real` output are not supported by the scalar index resolver. Invalid or
non-real equation results raise `MaterialModelError`; out-of-range wavelengths raise `WavelengthOutOfRangeError`.
