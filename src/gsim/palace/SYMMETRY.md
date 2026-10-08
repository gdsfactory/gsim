# Symmetry planes (PEC / PMC)

Simulate half of a mirror-symmetric structure. You pass the full symmetric component; gsim cuts it at the plane, meshes
the kept half and applies a PEC or PMC boundary on the cut.

```python
sim.add_symmetry_plane(axis="y", position=0.0, kind="pmc", keep="positive")
```

| Argument          | Meaning                                                           |
| ----------------- | ----------------------------------------------------------------- |
| `axis`            | `"x"` or `"y"`: the axis normal to the plane (`"z"` is rejected)  |
| `position`        | Plane position in um, on the 1 nm grid                            |
| `kind`            | `"pmc"` (even / common mode) or `"pec"` (odd / differential mode) |
| `keep`            | Side that is simulated: `"positive"` or `"negative"`              |
| `verify_symmetry` | Check that the layout is mirror-symmetric (default `True`)        |

## Limits (v1)

- One plane only; a second call raises.
- `driven` and `eigenmode` only. `electrostatic` and `boundarymode` raise (`NotImplementedError` on the call, an error
  in `validate_config`, and `ValueError` when a config is generated).
- `mesh(periodic_axis=...)` on the plane's axis raises.
- The symmetry check groups polygons by `(layer, datatype)`. A layer that has content on one side only counts as
  asymmetric and raises; if you deliberately drew only the kept half (a pre-cut layout), pass `verify_symmetry=False`.
- The symmetry check covers layout polygons only. Ports are not checked: keeping the ports symmetric is up to you.

## Which wall gives which mode

| Wall | Field on the plane | Mode in the half model          |
| ---- | ------------------ | ------------------------------- |
| PMC  | tangential H = 0   | even / common (V_a = V_b)       |
| PEC  | tangential E = 0   | odd / differential (V_a = -V_b) |

## What gsim does

- Clips every layout polygon (and PEC-block polygon) on its own to the kept side. `geometry.bbox` stays the bbox of the
  full layout, so the half domain is exactly the full domain cut at the plane (same margins, same substrate extent).
- Clamps the airbox, background dielectrics and `max_size` wave ports to the plane. A plane outside the domain raises
  `ValueError`.
- Drops conductor shell faces that lie on the plane, so the cut face of a conductor does not become a Conductivity
  boundary.
- Moves the faces on the plane out of the `*__None` (absorbing) groups into the group `symmetry_<axis>_<kind>`. The
  group is written to `Boundaries.PMC`, or merged into `Boundaries.PEC`. It is never absorbing.
- Records the plane in `port_information.json` under `"symmetry"` (`axis`, `position`, `kind`, `keep`, `mode`) and marks
  wave ports that were cut with `"cut_by_symmetry_plane"`.

## Port rules

- **Lumped (inplane, gap, interlayer)**
  - Kept side / touching: ok (warning when touching)
  - Straddles the plane: `ValueError`
  - Removed side / in the plane: `ValueError`
- **CPW / two-terminal (any element)**
  - Kept side / touching: ok
  - Straddles the plane: `ValueError` with a hint
  - Removed side / in the plane: `ValueError`
- **Wave port, `max_size=True`**
  - Kept side / touching: clipped with the domain
  - Straddles the plane: n/a
  - Removed side / in the plane: `ValueError` if its face lies on the plane or on the removed side
- **Wave port, not `max_size`**
  - Kept side / touching: ok
  - Straddles the plane: face perpendicular to the plane: clipped to the plane (width halved, `cut_by_symmetry_plane`);
    face parallel to it: `ValueError`
  - Removed side / in the plane: `ValueError`

Ports on the removed side that were never configured are ignored.

In the half model a wave port supports only the modes with the plane's symmetry. Its `Mode=1` is the full model's lowest
even mode (PMC) or odd mode (PEC) as far as the mode ordering allows (Palace orders wave-port modes by decreasing
propagation constant).

## Results and normalisation

A half model's S-parameters are the full structure's modal S-parameters as they are. No scaling is applied.

| Full-model quantity    | From the half models               |
| ---------------------- | ---------------------------------- |
| Differential impedance | `Z_diff = 2 * Z_odd` (PEC half)    |
| Common-mode impedance  | `Z_cm = Z_even / 2` (PMC half)     |
| Effective index, loss  | identical to the full model's mode |

Lumped ports with R on each line:

- PEC half: `S_dd`, referenced to a differential impedance of `2R`.
- PMC half: `S_cc`, referenced to a common-mode impedance of `R/2`.
- A half model cannot show mode conversion (`S_dc`, `S_cd` are zero by construction), and its energies and DOFs are
  halved (ratios such as Q are not).

`SParams.symmetry` carries the plane from `port_information.json` and `repr` shows it in one line. Helpers in
`gsim.palace.symmetry`:

- `mixed_mode_from_halves(even, odd)`: checks that both are matching PMC/PEC half models (same plane, ports,
  frequencies, port R) and returns `{"cc", "dd", "z_ref_cc", "z_ref_dd"}`.
- `combine_even_odd(even, odd, mirror_names=None)`: single-ended 2N-port result for lumped ports, with
  `S_ij = (cc_ij + dd_ij)/2` and `S_ij' = (cc_ij - dd_ij)/2` (default mirror names `f"{name}_mirror"`). Wave ports
  raise: a full model with wave ports is modal.
- `full_model_impedance(z_half, kind)`.

## Not verified

Checked in Palace's `palace/models/waveportoperator.cpp` (main, October 2026): the wave-port mode solve marks only `PEC`
and `WavePortPEC` attributes as Dirichlet; impedance, conductivity and absorbing boundaries enter through their
operators, and any other attribute on the port edge is natural. So a PEC plane is Dirichlet in the port solve and a PMC
plane stays natural as long as it is not listed in `WavePortPEC`, which the config generator guarantees.

- **Multi-element lumped-port convention.** Whether Palace combines the elements of a multi-element port in parallel
  decides the `2 * R` rule for a CPW port cut by a PMC plane. v1 rejects that case; the error hint is provisional.
- **Cloud validation.** A 500 um GSSG coupled line on the generic gpdk PDK (`max_size` wave ports, 5-50 GHz) was run as
  a full model and as both halves. The PEC half matches the full model's mode 1 (odd) with max |dS21| 0.0024 and n_eff
  1.961 vs 1.966. The PMC half matches mode 2 (even) with max |dS21| 0.0020 and n_eff 1.825 vs 1.828. The halves have
  21,580 tetrahedra against 42,155 for the full model.

## Later work

Electrostatics (`ZeroCharge` / `Ground`), native 2D `BoundaryModeSim`, two orthogonal planes, lumped ports touching or
spanning the plane (`R/2` for PEC), CPW ports cut by a PMC plane, field mirroring for visualisation, and a helper that
runs both halves.
