# Terminal (multi-pin) wave ports

A field solver reports **modal** S-parameters: one port per propagating mode of the port plane. Circuit work
(differential and common-mode analysis, a bias line as its own terminal, SPICE export) needs **terminal** S-parameters:
one port per conductor, voltage measured against a common ground, with a reference impedance you choose.
`gsim.palace.terminal` converts one into the other in post-processing, using for each port the terminal voltages and
currents of its modes. It is the standard multiconductor-line approach and it needs no change in Palace.

`gsim.palace.terminal_ports` builds those voltages and currents from a 2D `BoundaryModeSim` of the port plane, reads the
modal S-parameters of a driven run with a **multimode wave port** (several modes on one port face), and joins the two. The
sections below cover the transformation (1 to 3), what Palace allows on one port face (4), where the voltages and currents
come from (5), how the 2D modes are matched to Palace's port modes (6), the choice of conductors (7), the whole workflow
(8), network helpers (9), limits (10), a synthetic example (11) and the validation geometries (12). The notebook
[Terminal wave ports in Palace](nbs/palace_terminal_waveport.md) runs the local part of the workflow and holds the cloud
cells. [Coupled lines with terminal wave ports](nbs/palace_terminal_waveport_gssg.md) runs it end to end on a
coupled GSSG line and checks the result against a uniform line built from the 2D port solve.

## 1. Why modal S is not terminal S

Take a pair of coupled signal lines over a common ground. A two-end simulation with one wave port per end and two modes
returns a 4x4 modal S: (end 1 mode 1, end 1 mode 2, end 2 mode 1, end 2 mode 2). Mode 1 and mode 2 are generally neither
"line A" nor "line B": they are the even-like and odd-like solutions, each with its own propagation constant and with
voltage on both lines. A circuit that connects 50-ohm sources to line A and line B needs the 4x4 S in the order (A1, B1,
A2, B2). The two are related by a linear map that depends on how each mode loads the two terminals. Running one wave
port per signal conductor instead does not replace this map: each port face then only sees part of the cross-section,
and its mode solve finds modes of a truncated problem rather than the coupled modes of the line.

## 2. Conventions and formulas

Time convention exp(+j w t), as in the rest of gsim. A physical port `p` has `n_p` terminals and `n_p` modes kept.

- Column `k` of `T_V[p]` holds the terminal voltages (terminal minus the reference conductor) of mode `k` for a unit
  forward modal wave. Column `k` of `T_I[p]` holds the terminal currents flowing **into** the network for that wave.
- Modal waves: `V = T_V (a + b)` and `I = T_I (a - b)`.
- Terminal power waves for a real reference impedance `Zr`: `a_t = F (V + Zr I)` and `b_t = F (V - Zr I)`, with
  `F = diag(1 / (2 sqrt(Zr)))`. For real `Zr` power waves and pseudo waves coincide.
- With block-diagonal `T = diag(T_V[p])` and `W = diag(T_I[p])`, and `A = F (T + Zr W)`, `B = F (T - Zr W)`:

```text
S_t = (B + A S_m) (A + B S_m)^-1          (modal_to_terminal_s)
S_m = (A - S_t B)^-1 (S_t A - B)          (terminal_to_modal_s)
```

- Port ordering: physical port by physical port, terminals (or modes) in column order inside each port. A two-end line
  is ordered (A1, B1, A2, B2) and its modal S is (end 1 mode 1, end 1 mode 2, end 2 mode 1, end 2 mode 2).
- Scaling column `k` of `T_V` and `T_I` together by any complex constant leaves `S_t` unchanged when `S_m` has no
  cross-mode terms. The solver's modal normalisation therefore cancels; only the voltage/current shape of each mode
  matters. This holds only if the scaling of a mode is the same at both ends of the line and in `S_m` (section 6).
- The modal S of a uniform line is `[[0, E], [E, 0]]` with `E = diag(exp(-gamma_k * length))` (`uniform_line_modal_s`).

`z_ref` may be a scalar, one value per terminal, or one value per frequency and terminal. It must be real and positive.

## 3. Degenerate modes

If several modes share a propagation constant (a homogeneous dielectric, or a symmetric layout), the solver may return
any basis of their subspace, and the mode order and mixing can change from run to run. **Any basis of a degenerate
subspace gives the same terminal S.** A special choice is the *terminal-aligned* basis `R = T_V,cluster^-1`, in which
each mode puts its voltage on one terminal only ("the combination that maximises the field on each terminal"); the
transformation does not need it. `degenerate_mode_groups(n_eff, rtol=...)` lists which modes share a propagation
constant at one frequency.

Checked on synthetic coupled lines (generic values, 700 um, 1-100 GHz, `z_ref = 50` ohm). Errors are the largest
difference of the terminal S against an independent chain-matrix solution:

| case                                                    | bases tried                                   | max abs error |
| ------------------------------------------------------- | --------------------------------------------- | ------------- |
| homogeneous, 2 conductors, lossless                     | 5 random complex bases + aligned + `eig`      | 7.9e-15       |
| homogeneous, 3 conductors, lossless                     | 5 random complex bases + aligned + `eig`      | 1.1e-14       |
| homogeneous, 2 conductors, lossy (`R = aL`, `G = bC`)   | 5 random complex bases                        | 7.2e-15       |
| 3 conductors, one degenerate pair and one distinct mode | random 2x2 rotation inside the pair, 5 trials | 3.4e-14       |

Near degeneracy (`C[0,0]` scaled by `1 + delta`, and a random symmetric perturbation of relative size `delta`): `S_t`
differs from the unperturbed `S_t` by about `1.1 * delta`, i.e. it is continuous, and the modal route still agrees with
the chain matrix to 1e-14. Eigenvectors, by contrast, rotate by 6 to 49 degrees between rows when the perturbation
direction changes (as mesh noise does) while `S_t` moves by about `delta`. Mode identification is unreliable near
degeneracy; the terminal S is not. The same transformation also covers the non-degenerate case (for example even and odd
modes with different velocities), where rotating modes is not allowed.

## 4. Several modes on one Palace port face

Palace has no terminal wave port. What it does allow was read from its source (commit `cfa430a5`, September 2026) and has
been run on a shared face in the validation runs of section 12, where Palace accepted it.

- **One face, several entries.** Several `WavePort` entries may share the same boundary attributes with different `Mode`,
  but only one of them may be active, and the active entry must have the lowest `Index` of its face (the check is on the
  attribute, and the ports are visited in index order). `add_multimode_wave_port` writes exactly this, with the active mode
  on the port's own index and every other mode on an index after all gsim ports, and records the map from Palace index to
  (port, mode, active, excited) in `port_modes.json` next to `config.json`.
- **One 2D solve per entry.** Every entry solves its own 2D port eigenproblem at every frequency and takes the
  eigenvector `Mode`. The solves are separate, so two entries of a degenerate pair can come out in different bases
  (awslabs/palace#328).
- **Only the active entry terminates the face.** The impedance termination and the modal correction are added for active
  ports only (and the line attenuation of the mode is left out in a driven run). A wave that arrives in another mode is not
  matched: it is partly reflected, by roughly `|k_j - k_1| / (k_j + k_1)` for quasi-TEM modes (an estimate, not derived in
  Palace; it is about 7 % for `n_eff` of 2.0 against 2.3, and the validation runs are meant to measure it).
- **Any entry can be excited, and every entry is measured.** An excited inactive mode sees a termination tuned to the
  active one, so its incident amplitude is off by a similar relative amount. `port-S.csv` has one column per entry and
  excitation, each projected onto the entry's own port mode. The `-1` of the reflection is subtracted on the driven entry
  only.
- **No per-terminal voltages or currents come out of a driven run.** Each entry has at most one `VoltagePath`, `port-V.csv`
  is the voltage of the *total* 3D field along it, and `port-Z.csv` holds `Z_PV` only. `T_V` and `T_I` therefore come from
  the 2D solve (section 5).
- **Gauge.** Palace normalises every port mode to unit power, fixes its phase by a rule of its own and, when a
  `VoltagePath` is given, flips the sign so that the voltage along it is positive (section 6).

Two ways to get the modal S-parameters follow, at the same number of 3D solves but different mesh and job counts:

- **Single run**: `excite="all"`, `active_mode=1`, one run with `N x P` excitations. Exact for mode 1; the other modes
  carry the mismatch error above.
- **Cycling**: run `k` has `active_mode=k` and `excite="active"` (`P` excitations), and column `k` of the modal S comes
  from run `k`, so `N` runs. Exact for a mode without conversion; with conversion `c` and face reflection `G`, the error
  is about `c G` (estimate).

(`N` modes per face, `P` faces.) Cycling is the default recommendation; run 1 of the cycle can be the single run, since its
mode-1 columns are exact. `modal_s_from_runs` takes each column from the run in which its entry is active, and falls back
to the first run that excites it (then marked not exact in `ModalS.exact`). A longer-term alternative is a multimode
termination in Palace itself, which is out of scope here.

A symmetry half model (`add_symmetry_plane`) has one mode per face for two terminals over a ground. It does not replace
this method for more terminals: with floating grounds the odd half still carries two modes.

## 5. Where `T_V` and `T_I` come from

- **Layout and paths.** `TerminalPortSpec` names the terminals and the reference conductor as rectangles on the port's
  cross-section (`gsim.common.modes.Conductor`, in the cross-section's own `(h, v)` coordinates in um). `terminal_paths`
  draws one voltage path per terminal, from the terminal to the reference, that never crosses a conductor, and
  `declare_terminal_paths(sim, spec, domain)` registers them on a `BoundaryModeSim` (`add_impedance_path`; the order of the
  calls is the index Palace reports each path under; a `BoundaryModeSim` takes no ports, so `add_port` is refused there).
  Palace then writes the complex voltage of each mode on each path to `mode-V.csv`
  (`PalaceTextResults.mode_voltage(index=..., mode=...)`). Modes come with `k_n` and `n_eff` in `mode-kn.csv`, from which
  `gamma = j * k0 * n_eff`.
- **Voltages.** `port_terminal_matrices` / `port_modes_2d` take `T_V` from line integrals of the saved mode field along the
  paths (`voltage_line_integral`), so that voltages and currents share one normalisation. `mode-V.csv` stays a cross-check.
  The quadrature is 800 sample points per path segment by default (`nsamples`): along a detour path that passes a few
  thousandths of a gap from a metal corner, 200 left 0.9 % in a voltage of a three-terminal port and up to 3.5e-3 in a
  terminal S at 50 GHz. Palace's own integral on the declared paths (`declare_terminal_paths(..., nsamples=)`, default 100)
  is separate; declare 400 or more if `mode-V.csv` is to stay a close cross-check.
- **Currents.** `T_I` comes from Ampere's contour integral of `H` around each terminal's rectangle (`current="contour"`,
  which needs the terminal's outline on mesh edges, as a metal left out of the mesh as a hole has), or from the reaction
  route (`current="reaction"`, no outline needed): with the unconjugated reaction matrix `R = T_V^T T_I`,
  `T_I = T_V^{-T} R`; `currents_from_reaction(t_v, reaction)` implements it. For distinct modes of a reciprocal line `R` is
  diagonal (`r_k = V_k^T I_k`) and the vector `r` is enough. Within a degenerate group any basis is valid, and a mixed
  basis has off-diagonal reactions, so pass the full matrix there (or orthogonalise the group first). A quasi-TEM solve
  gives the same result from `T_I = (G + j w C) T_V Gamma^-1`.
- **Normalisation and orientation.** Each column is scaled to `|2 P| = 1` with `P` the complex power flux of the saved
  mode, and oriented so that `Re(T_V^H T_I) > 0`: the incident wave carries power into the network, whichever way the port
  face points.
- **Warning: do not derive currents from power alone.** If `T_I` is rebuilt from the complex voltages in `mode-V.csv`
  and the *real* power implied by `Z_PV`, i.e. `r_k` is replaced by `2 P_k`, the result is exact only without loss. On
  synthetic asymmetric coupled lines the terminal S error was:

| series resistance                | `r_k = V_k^T I_k` (reaction) | `r_k = 2 P_k` (power only) | quasi-TEM admittance |
| -------------------------------- | ---------------------------- | -------------------------- | -------------------- |
| lossless                         | 4.7e-15                      | 4.7e-15                    | 4.7e-15              |
| about 40 ohm/m                   | 7.0e-15                      | 3.0e-4                     | 6.9e-15              |
| about 2.6 kohm/m (on-chip metal) | 1.3e-14                      | 1.9e-2                     | 1.3e-14              |

A 2 % error is not acceptable for S-parameters, so take complex currents from the fields. (Numbers: synthetic lines, 700
um, 1-100 GHz, `z_ref = 50` ohm; the power-only route needs a phase alignment of each voltage column, here the largest
entry made real.)

Checked on the real 2D mesh of the validation geometries (section 12) without a Palace solve: every sample of every
terminal path lies in a mesh triangle and outside the conductor holes (a linear synthetic field integrates to the exact
potential difference), and the outline of every terminal rectangle lies on mesh edges (the contour integral of a linear
synthetic `H` equals `2 k A` to 1e-16). The outline of a reference conductor that is a union of shapes (the `metal1` plane
with via walls on it) does not, which is why a reference needs no contour.

## 6. Matching the 2D modes to the 3D port modes

`modal_to_terminal_s` is invariant only to a scaling of `T_V` and `T_I` that is the same in `S_m`, so the 2D modes must be
brought into the gauge of Palace's port modes. `align_gauge` (called by `terminal_s`) does this per port:

1. **Pairing.** 2D mode `l` is paired with 3D mode `k` by the effective index, at each 2D frequency, with
   `scipy.optimize.linear_sum_assignment` on the relative `n_eff` difference. The 3D wave numbers come from the Palace log
   (`load_modal_run(..., log=...)`); the 2D ones from `mode-kn.csv`. A pair that differs by more than `rtol` (0.5 % by
   default) is an error. Two 3D modes closer than `degenerate_rtol` (also 0.5 % by default) cannot be told apart, so their
   2D basis need not be Palace's, and that is an error unless a basis is given. The two tolerances are separate on purpose:
   opening `rtol` (for example to 0.15, to pair modes of a lossy 2D solve with the port modes while the cause of the
   difference is being looked at) does not make 3D modes a few percent apart degenerate.
2. **Magnitude.** Every 2D column is already normalised to `|2 P| = 1`, which is Palace's unit power for port modes.
3. **Sign and phase.** Each 3D entry gets a *pilot* `VoltagePath`: the voltage path of the terminal where the 2D mode has
   its largest voltage, on the port plane. Each 2D column is rotated so that its voltage on the pilot terminal is real and
   positive. For a low-loss quasi-TEM mode this should reproduce Palace's own gauge. That is an assumption that only a run
   can check.
4. **Fit to Palace's own port-mode fields (optional).** Given the terminal voltages of the saved 3D port modes
   (`reference_voltages=`), the gauge is fitted instead (`V_3D ~ V_2D M`, a small matrix for a degenerate group), and the
   residual of the fit measures how well the 2D and 3D modes agree. A driven run that saves fields
   (`set_driven(save_fields_at=...)`) writes Palace's own unit-power port-mode field of every `WavePort` entry as
   `E0_<index>_real` / `E0_<index>_imag` in its boundary ParaView output, one cycle per saved frequency in ascending order.
   `port_mode_reference_voltages(source, spec, "port_modes.json", cycles=[...], domain=...)` integrates those fields along
   the terminal paths and returns one matrix per 2D solve (`cycles` is the position of that solve's frequency among the
   saved ones, starting at 1; column `k` is mode `k` of the port); the low-level readers are `load_port_mode_fields` and
   `gsim.palace.mode_fields.port_mode_fields_from_grid`. Save the fields at the frequencies of the 2D solves that feed
   `align_gauge`, since each reference is paired with one 2D solve. The gauge phase of Palace's port modes was checked
   against this route on the floating-ground validation line: the pilot-terminal rule reproduced it to 0.0 degrees there.
5. **Frequency.** The aligned matrices are interpolated between the 2D solves, by default in `1 / f`.

The pilot terminal of a mode is known after the 2D solve, so the order is: 2D solves, then pilot terminals, then the 3D
runs with the matching `VoltagePath`. A degenerate port is refused by default (`allow_degenerate=True` accepts it at the
cost of the basis); the validation geometries are chosen with distinct `n_eff`.

**The 2D solve must share the port eigenproblem's conductors.** With `waveport_boundary="pec"` (the default) the port
eigenproblem makes the outer boundary *and every finite-conductivity conductor surface* perfect conductors: the attributes
of `Boundaries.Conductivity` (and of the absorbing and impedance boundaries) are listed under `WavePortPEC`. The 2D solve
with `metallic_boundaries=True` does the same for the outer wall only. Its conductors stay Robin surface-impedance
conditions, so its modes are lossy, with a complex and frequency-dependent `n_eff`, and they are not the modes the port
solver finds. On the floating-ground validation line the difference was 12 % at 1 GHz, 3.9 % at 20 GHz and 2.7 % at
50 GHz for mode 1 (2D `n_eff` 1.713 - 0.44j, 1.584, 1.566 against 1.5244 in the port solve, which is flat to 0.04 % from
1 to 50 GHz), so the 0.5 % pairing of `align_gauge` cannot succeed with such a solve.

For the 2D solves that supply `T_V`, `T_I` and the `n_eff` to pair with, set `BoundaryModeSim.conductor_boundary = "pec"`
(default `"conductivity"`). The conductor attributes are then written under `Boundaries.PEC` instead of
`Boundaries.Conductivity`, and nothing else in the configuration changes. The modes are lossless, `T_V` and `T_I` hardly
move with frequency, and a few solves are enough; the same solves can save the fields at the frequencies where the 3D run
saves its port-mode fields. Keep one lossy solve per frequency of interest (the default) only for the propagation constant
of the physical line, `gamma = j k0 n_eff`: compare a uniform-line `S21` with that `gamma`, not with the port solve's
lossless wave number. That a 2D solve with PEC conductors reproduces the 3D port `n_eff` within 0.5 % has not been verified
on a run (the 2D and 3D meshes also differ in size); it is the first thing the next 2D solves check.

**The outer wall is one more conductor.** A port with `N + 1` conductors (`N` terminals and the reference) inside a
shielded box has `N + 1` quasi-TEM modes, one of them living between the metal and the wall. Ask the 2D solve for `N + 1`
modes, keep the first `N` by effective index and check `mode-kn.csv`; that the wall mode is the last one has not been
verified on a run.

## 7. Grounds are part of the definition

Terminal voltages are measured against a reference conductor, and a line with `N + 1` conductors has `N` quasi-TEM
modes. Floating outer grounds are therefore extra conductors: a GSSG line with floating grounds has three non-reference
conductors (two signals and one outer ground, with the other outer ground as the reference) and three modes, not two.
Either tie the grounds together (metal strips, a lower plane and continuous via stacks give one ground conductor, and
then two signal terminals and two modes), or declare the extra ground as a terminal. The `T_V`/`T_I` blocks must be
square: as many modes as terminals. Do not add an artificial reference conductor: it adds a mode whose voltage depends on
where the reference is placed, and an ill-conditioned `T_V`.

## 8. Workflow

```python
from gsim.common.modes import Conductor
from gsim.palace import (
    BoundaryModeSim, DrivenSim, TerminalPortSpec, declare_terminal_paths, load_boundary_mode_field,
    load_modal_run, port_mode_reference_voltages, port_modes_2d, terminal_s,
)

spec = TerminalPortSpec(
    name="o1",
    terminals=[Conductor("G1", ((-66, -26), (3.2, 5.2))), Conductor("S1", ((-20, -10), (3.2, 5.2))), ...],
    reference=Conductor("G2", ((26, 66), (3.2, 5.2))),
)  # one spec per port, in the cross-section's (h, v) um

# 1. 2D solves at a few frequencies (cloud), paths declared
sim2d = BoundaryModeSim()  # geometry, stack, set_cross_section("x=0"), metallic_boundaries=True,
sim2d.conductor_boundary = "pec"  # the gauge solves: conductors as in the port eigenproblem (a default solve gives gamma)
sim2d.set_boundary_mode(freq=20e9, num_modes=len(spec.terminals) + 1, save=len(spec.terminals) + 1)
sim2d.mesh(...)
declare_terminal_paths(sim2d, spec, sim2d.mesh_extent)
# ... run, then per frequency: port_modes_2d(fields, spec, freq_ghz=..., n_eff=..., paths=...)

# 2. 3D runs: one multimode port per end; modes (1..N); one run per active mode (cycling)
sim = DrivenSim()
sim.add_multimode_wave_port("o1", layer="metal3", modes=(1, 2, 3), active_mode=2, excite="active",
                            max_size=True, pilot_paths={...})
sim.set_driven(fmin=1e9, fmax=50e9, num_points=50, adaptive_tol=0.0)

# 3. read the runs (S-parameters, sidecar, log) and convert
runs = [load_modal_run(files, sidecar="run/port_modes.json", log=files["palace.log"]) for files in results]
net = terminal_s(runs, [spec_o1, spec_o2], {"o1": modes_2d, "o2": modes_2d}, z_ref=50.0)
# or, with the gauge fitted to Palace's own saved port modes (the run that saved fields at the 2D frequencies):
# refs = {p: port_mode_reference_voltages(files, spec, "run1/port_modes.json", cycles=[1, 2, 3], domain=domain)
#         for p, spec in (("o1", spec_o1), ("o2", spec_o2))}
# net = terminal_s(runs, [spec_o1, spec_o2], {"o1": modes_2d, "o2": modes_2d}, reference_voltages=refs)
# build_terminal_s (same arguments) also returns the modal S, the pairing and the fit residuals.
net.write_touchstone("line_terminal")  # ports "o1:G1", "o1:S1", ...
```

Notes:

- Adaptive frequency sampling interpolates the S-parameters between solved points, but the wave numbers come from the log
  of solved points only. Use `adaptive_tol=0` (a full solve at every point) for the runs that feed `terminal_s`.
- A change of `active_mode` needs a new `mesh()` call: the ports are captured at mesh time.
- Lumped ports (for example one per diode) pass through unchanged: `LumpedTerminal(name, index, resistance)` gives an
  identity block (`A = 1`, `B = 0`), and RC networks and cascading belong in scikit-rf on the terminal Touchstone.
- `terminal_s` refuses a port whose number of 2D modes differs from its terminals, runs without wave numbers, and
  `cond(T_V) > 1e12` ("the modes do not span the terminals").

## 9. Network helpers

`short_terminals(net, names)` ties terminals to the reference (`V = 0`): the Y-matrix sub-block of the remaining ones,
written on S so that it also works when `Y` does not exist. Shorting terminals at the ports of a line ties them at its two
ends only, not along its length; the two agree for an electrically short line. `mixed_mode(net, pairs)` returns
differential ports first, common-mode ports next and unpaired terminals last, with reference impedances twice and half of
the single-ended one, named `d(<first>,<second>)` and `c(<first>,<second>)`. The two terminals of a pair must have the
same reference impedance: for different ones the plain sum and difference of the waves is not the mixed-mode transform
(a matched two-port with `z0 = (50, 75)` ohm and `S = 0` has `S_dd = S_cc = -0.0101` with `se2gmm`, not 0), and
`mixed_mode` raises `ValueError` instead of returning a wrong network. Renormalise the network to one impedance per pair
first (`Network.renormalize`), or give the two terminals of a pair the same `z_ref` in `build_terminal_s`.

Terminal S feeds scikit-rf directly. For a two-end coupled pair ordered (A1, B1, A2, B2), `Network.se2gmm(p=2)` pairs
ports (0, 1) and (2, 3), puts the differential ports first and the common ports after, and defaults the reference
impedances to twice and half of the single-ended value (it works in place and returns `None`: copy first, or use
`mixed_mode`). This replaces the one-wave-port-per-signal plus `se2gmm` workaround of gsim#264 once terminal S is
available. On a symmetric line the differential and common blocks match `DefinedGammaZ0` lines with `2 * Z_odd` (at 100
ohm) and `Z_even / 2` (at 25 ohm) to 1.5e-14 with no mode conversion (`|S_dc| <= 1e-15`).

## 10. Limits

- Real, positive `z_ref` only. Complex reference impedances are later work.
- Cross-mode terms in the modal S (a mode on one end exciting another mode on the other end) must come from a multi-mode
  port solve. `uniform_line_modal_s` has none; with a mirror-symmetric layout they vanish.
- `T_V` and `T_I` from a *separate* BoundaryMode run describe the port modes only when those modes are not degenerate,
  or when the cluster is treated as one group (section 3).
- The blocks must be square and well conditioned: `modal_to_terminal_s` raises `ValueError` if `cond(T_V)` or
  `cond(T_I)` exceeds 1e12 ("the modes do not span the terminals").
- The single-run mode is approximate for every mode but the active one (section 4), and the line attenuation of a port mode
  is not part of Palace's driven termination.
- The full-wave voltage depends a little on the path; for quasi-TEM modes the difference is small. The path rule is fixed in
  code (straight across the gap, else over the metal).
- The gauge rule of section 6 (pilot terminal) is an assumption for low-loss quasi-TEM modes until it is compared with a run.

## 11. Example with synthetic data

A coupled pair over a common ground with series loss. The modes come from the eigenproblem of `Z Y`, the modal S is a
uniform line, and the result is checked against an independent chain-matrix solution and turned into mixed-mode S.

```python
import numpy as np
import skrf
from scipy.linalg import expm
from skrf.network import z2s

from gsim.palace.terminal import modal_to_terminal_s, uniform_line_modal_s

# per-unit-length matrices of a coupled pair over a common ground (generic values)
L = np.array([[4.0e-7, 1.2e-7], [1.2e-7, 4.0e-7]])  # H/m
C = np.array([[1.6e-10, -0.5e-10], [-0.5e-10, 1.6e-10]])  # F/m
R = np.diag([40.0, 40.0])  # ohm/m
G = np.diag([2e-4, 2e-4])  # S/m
freq = np.linspace(1e9, 100e9, 41)
length = 700e-6  # m

# modes at each frequency: columns of t_v are the mode voltages, t_i = Y t_v / gamma
t_v, t_i, gamma = [], [], []
for w in 2 * np.pi * freq:
    z_pul = R + 1j * w * L
    y_pul = G + 1j * w * C
    g2, v = np.linalg.eig(z_pul @ y_pul)
    g = np.sqrt(g2)
    g = np.where(g.real < 0, -g, g)
    t_v.append(v)
    t_i.append(y_pul @ v / g)
    gamma.append(g)
t_v, t_i, gamma = np.array(t_v), np.array(t_i), np.array(gamma)

# modal S of the uniform line -> terminal S in the order (A1, B1, A2, B2)
s_modal = uniform_line_modal_s(gamma, length)
s_term = modal_to_terminal_s(s_modal, [t_v, t_v], [t_i, t_i], z_ref=50.0)

# independent reference: chain matrix expm([[0, -Z], [-Y, 0]] * length) -> Z -> S
z_mats = []
for w in 2 * np.pi * freq:
    z_pul = R + 1j * w * L
    y_pul = G + 1j * w * C
    a = np.block([[np.zeros((2, 2)), -z_pul], [-y_pul, np.zeros((2, 2))]])
    phi = expm(a * length)
    p11, p12, p21, p22 = phi[:2, :2], phi[:2, 2:], phi[2:, :2], phi[2:, 2:]
    p21i = np.linalg.inv(p21)
    z_mats.append(
        np.block([[-p21i @ p22, -p21i], [p12 - p11 @ p21i @ p22, -p11 @ p21i]])
    )
s_ref = z2s(np.array(z_mats), z0=50.0, s_def="power")
print("max |S_terminal - S_chain| =", np.abs(s_term - s_ref).max())

# mixed-mode S with scikit-rf: differential ports first, then common ports
net = skrf.Network(f=freq, s=s_term, z0=50.0, f_unit="Hz")
mm = net.copy()
mm.se2gmm(p=2)
print("|S_dd21| at 50 GHz:", abs(mm.s[20, 1, 0]), " |S_cc21|:", abs(mm.s[20, 3, 2]))
```

## 12. Validation geometries and what is checked

Two coupled GSSG lines on the generic gdsfactory PDK (`metal3`, signal width 10 um, gap 6 um, edge-to-edge signal spacing
20 um, ground width 40 um, length 300 um, 1 to 50 GHz, 50 points): `G_float` with floating outer grounds (terminals G1, S1,
S2; reference G2; three modes) and `G_tied` with the outer grounds tied by a `metal1` plane and continuous via walls
(terminals S1, S2; reference the plane; two modes). The notebook builds both.

Checked locally:

| item                                                        | `G_float` | `G_tied` |
| ----------------------------------------------------------- | --------- | -------- |
| tetrahedra, `refined_mesh_size=2 um`, `max_mesh_size=40 um` | 78,462    | 145,823  |
| tetrahedra, the `default` preset (5 um)                     | 24,844    | 46,169   |
| tetrahedra of the PEC or PMC half model, 2 um mesh          | -         | 77,938   |
| triangles of the 2D cross-section (1 um near metal)         | 2,628     | 3,049    |
| Palace `WavePort` entries per run (single run)              | 6         | 4        |

The generated `config.json` of every variant validates against Palace's JSON schema (0 errors), `validate_config()` passes,
the port surface of every variant is one physical group per face, and the pilot paths lie on the port face.

Measured on the cloud (Palace, order 2, 1 to 50 GHz, 50 points):

| check                                                            | `G_float`     | `G_tied`                              |
| ---------------------------------------------------------------- | ------------- | ------------------------------------- |
| Palace accepts several entries on one face                       | yes           | yes                                   |
| 2D (PEC conductors) against 3D port `n_eff`, all modes           | within 0.05 % | within 0.05 %                         |
| gauge fit residual of `align_gauge`                              | 2.1e-3        | 2.1e-3 to 2.4e-3                      |
| reciprocity, max abs(S - S^T) of the terminal S                  | 5.9e-3        | 7.9e-5                                |
| largest singular value of the terminal S (passivity)             | 1.000         | 1.000                                 |
| uniform-line consistency, S21 phase error per mode (1 to 50 GHz) | -             | <= 0.25 deg diff., <= 0.29 deg common |
| uniform-line consistency, abs(dS21) per mode (1 to 50 GHz)       | -             | <= 4.4e-3 diff., <= 6.2e-3 common     |
| mixed mode against the PEC / PMC half models, S21 phase          | -             | within 0.26 deg                       |

The uniform-line check compares the propagation constant of each mode extracted from the terminal S (ABCD of the
mixed-mode two-ports, independent of the reference impedance) with the lossy 2D solve (`conductor_boundary` left at
`"conductivity"`). That lossy solve needs `target` set just above the largest PEC `n_eff` (the notebook uses 1.02 times
it): with Palace's automatic shift, `sqrt(1.1 max eps_r)`, it returns spurious conductor modes below 50 GHz on this stack.
The floating-ground line is less reciprocal than the tied one because of its air mode, whose `n_eff` depends on the box
size (section 10); use a line whose grounds are tied for validation.
