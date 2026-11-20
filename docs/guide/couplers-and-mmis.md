# Directional couplers and MMIs

How to get the S-parameters of a passive photonic splitter, coupler or MMI with GDSFactory FDTD, and what to check
before using them.

## When to use / solver and why

Use [`gsim.fdtd`](../api/fdtd.md). It is the primary photonic engine in gsim (see the engine table on the
[home page](../index.md)) and it returns S-parameters from guided-port monitors, one source port per run. A directional
coupler, a Y-branch, an MMI and a waveguide crossing are all this kind of device.

Use [`gsim.meep`](../api/meep.md) only for something FDTD does not do yet, for example the 2D XZ grating coupler in
[Meep: 2D Grating Coupler](../nbs/meep_2d_xz_gc.md). Use the
[3D grating coupler tutorial](../nbs/fdtd_focusing_grating_coupler_gpdk.md) for a vertical fibre port.

If the device is a ring or contains a long delay line, simulate the coupler section here and compose the rest as a circuit.
That workflow has no guide page yet.

The simulation runs on GDSFactory+ cloud. Meshing and input export run locally.

## Minimal setup

The example is the generic-PDK 1x2 MMI. It builds the inputs for one source port at a time and writes them to disk. The
cloud call is the last step of each loop pass.

```python
import gdsfactory as gf

from gsim import fdtd

gf.gpdk.PDK.activate()
component = gf.get_component("mmi1x2")

sim = fdtd.Simulation()
sim.materials(background="SiO2", overrides={"si": 3.4757})
sim.geometry(component)
sim.source(port="o1", wavelength_um=1.55, wavelength_span_um=0.1, num_wavelengths=101)
sim.domain(padding=1.0)
sim.solver(cell_size_nm=60)

results = []
for port in ("o1", "o2", "o3"):
    sim.source(port=port)
    sim.write(f"./mmi-{port}")  # mesh.msh, config.json, metadata.json; nothing is submitted
    results.append(sim.run(check_cache=True))  # one cloud job per port

matrix = fdtd.s_matrix(results)  # S(receive, source), one source column per run
model = matrix.sax_model()  # needs no SAX install to build; see the FDTD guide for sax.circuit
matrix.write_touchstone("mmi")  # mmi.s3p
```

Steps, in order:

1. `fdtd.Simulation()` with no arguments uses the active PDK.
2. `materials(background=..., overrides=...)` sets the material around the device and index overrides at the source
   wavelength.
3. `geometry(component)` resolves the component through the PDK layer stack.
4. `source(port=...)` selects the port to excite. Without it, `write()` and `run()` refuse to continue.
5. `domain(...)` and `solver(...)` are optional. Their defaults are below.
6. `write(path)` builds the inputs locally. `run(check_cache=True)` uploads them, or reuses a finished job with identical
   inputs.
7. One run measures one column of the S-matrix. `fdtd.s_matrix([...])` combines runs and raises when an entry was not
   measured, unless you pass `assume_reciprocal=True` or `assume_unmeasured_zero=True`.

## Key settings and defaults

| Concern | Setting | Default |
| --- | --- | --- |
| `source` | `waveform` | `"pulse"` |
| `source` | `wavelength_um` | `1.55` |
| `source` | `wavelength_span_um` | `0.1` |
| `source` | `num_wavelengths` | `101` |
| `materials` | `background` | `"Air"` since 0.10.0 |
| `materials` | `dispersion` | `False` |
| `domain` | `padding` | `1.0` um on every side |
| `domain` | `pml_cells` | `32` |
| `solver` | `cell_size_nm` | `60` |
| `solver` | `energy_decay_fraction` | `1e-6` |
| `solver` | `max_wall_seconds` | `3600` |
| `geometry` | `mesh_size_nm` | `1000` |
| `geometry` | `geometry_tolerance_nm` | `10` |

`cell_size_nm` sets the Yee grid of the FDTD solver. `mesh_size_nm` and `geometry_tolerance_nm` only control the
tetrahedral transfer mesh that carries the geometry to the solver; they do not change the Yee grid.

With `dispersion=False` each material enters the solver as its index at the source wavelength, and a broadband run uses
that one index at every wavelength. gsim warns when a material card varies over the source band. `dispersion=True` is
rejected by the deployed solver.

To excite a higher-order mode, pass `mode={"kind": "mode_order", "mode_order": 2}` to `source()`. `"te_like"` and
`"tm_like"` need a transverse `te_axis`. With no selector the fundamental mode is used, and the guided-port monitors use
the same selection as the source.

## What to check

Run these on the result of every device, not only on the first one.

1. **Cell size.** Run again with a finer `cell_size_nm` (for example 40 instead of 60) and compare the S-parameters you
   report. The [FDTD benchmark](../fdtd-benchmark.md) uses 40 nm nominal. A change in the quantity you care about, such
   as the split ratio at the centre wavelength, is the convergence error of the coarse run.
2. **Power.** For each source port, the power summed over all ports cannot exceed 1 in a passive device. A value clearly
   above 1 points to a setup problem (mode, reference plane, grid), not to gain. A value below 1 is loss, radiation or
   power in a port you did not simulate.
3. **Reciprocity.** A passive device without magnetic or non-reciprocal materials has S(a,b) = S(b,a). With one run per
   port you have both entries measured, so compare them. Do not pass `assume_reciprocal=True` if you want to test this,
   because that flag copies one entry into the other.
4. **Noise floor.** Samples below the source noise floor are flagged. `s_matrix` drops them with a warning, which can narrow
   the band that `sax_model()` accepts. Look at the band edges.
5. **Wavelength step.** The SAX model interpolates unwrapped phase, which needs a phase change below pi between samples:
   a step below λ²/(2·n_g·L) for the longest path L in your circuit. `s_matrix` warns from 3/4 of that limit for entries
   with at least a tenth of the strongest peak. It cannot catch every aliased delay, so choose the step from the longest
   path.
6. **Mode.** If the port is multimode, confirm that the mode you excited is the one you meant (`mode_order` or `te_like`).

A small helper for checks 2 and 3 on the `matrix` built above:

```python
import numpy as np

power_out = (np.abs(matrix.s) ** 2).sum(axis=1)  # [wavelength, source port]
print("largest power out of any source:", power_out.max())
print("largest |S(a,b) - S(b,a)|:", np.abs(matrix.s - matrix.s.transpose(0, 2, 1)).max())
```

`matrix.s[f, i, j]` is S(receive i, source j) in time convention exp(+jωt) (`gsim.common.s_matrix.SMatrix`).

## Known pitfalls

- **Background.** The default background is `"Air"`. A device that is cladded in oxide in the real stack needs
  `background="SiO2"`, as above. Declaring oxide in the layer stack does not create oxide geometry.
- **Material names.** With gdsfactory 9.x the silicon layer is called `si`. If the PDK has no material card for it, pass
  `overrides={"si": <index>}`. The value `3.4757` above is the one the [home page](../index.md) quick start uses; choose
  your own value for your wavelength.
- **Padding at ports.** Explicit axis padding (`padding_x`, `padding_y`, `padding_z`) must be zero on a side that holds a
  guided port. The shared `padding` fallback keeps the automatic port alignment.
- **One column per run.** A single run is not an S-matrix. Without one run per port, `s_matrix` raises and lists the
  missing entries.
- **Phase reference.** Phases are referenced where the solver's port monitors sit. gsim does not de-embed them.
- **SAX conventions.** `sax_model()` keys entries `(in, out)` and conjugates the stored values to SAX's phase convention.
  A Touchstone file keeps gsim's convention. Read the table at the end of the
  [FDTD guide's export section](../api/fdtd.md#export-to-sax-and-touchstone) before mixing the two.
- **Cost.** The run time grows with the number of Yee cells. `metadata.json`, written by `write()`, holds the estimated grid
  in `grid.cell_counts` and the monitor memory in `memory.estimated_monitor_memory_bytes`; read both before you launch a
  finer run. For the snippet above, `write()` reported `total_cells` = 9,969,264 and a monitor-memory estimate of
  810,662,160 bytes (client estimates, from a local `write()` with gsim 0.10.0). They are not measured run times.

## Examples

Published notebooks, all on the generic PDK unless stated:

- [MMI quick start](../nbs/fdtd_mmi_gpdk.md)
- [Directional coupler](../nbs/fdtd_directional_coupler.md)
- [Broadband directional coupler](../nbs/fdtd_broadband_directional_coupler.md)
- [Waveguide crossing](../nbs/fdtd_waveguide_crossing.md)
- [Four-solver directional coupler benchmark](../fdtd-benchmark.md)
- [Meep: 3D directional coupler](../nbs/meep_dc.md)
- Paper reproductions: [Lu broadband coupler](../nbs/fdtd_lu_broadband_coupler.md),
  [Sattari suspended coupler](../nbs/fdtd_sattari_suspended_coupler.md),
  [Romero-García visible MMI](../nbs/fdtd_romero_visible_mmi.md),
  [Sacher visible MMI](../nbs/fdtd_sacher_visible_mmi.md)

See [choosing a solver](choosing-a-solver.md) and [validation](validation.md) for how far these results can be trusted.

## Related API

- [GDSFactory FDTD guide](../api/fdtd.md)
- [`fdtd.Simulation`][gsim.fdtd.Simulation]
- [`fdtd.s_matrix`][gsim.fdtd.s_matrix]
- [`SMatrix`][gsim.common.s_matrix.SMatrix]
