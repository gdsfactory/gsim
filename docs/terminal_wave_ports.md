# Terminal (multi-pin) wave ports

A field solver reports **modal** S-parameters: one port per propagating mode of the port plane. Circuit work
(differential and common-mode analysis, a bias line as its own terminal, SPICE export) needs **terminal** S-parameters:
one port per conductor, voltage measured against a common ground, with a reference impedance you choose.
`gsim.palace.terminal` converts one into the other in post-processing, using for each port the terminal voltages and
currents of its modes. It is the standard multiconductor-line approach and it needs no change in Palace.

This page covers the transformation. Building the voltage and current matrices from Palace output and a cloud comparison
on a coupled CPW follow in a separate change.

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
  matters.
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

## 4. Where `T_V` and `T_I` come from

- **Voltages.** Run a `BoundaryModeSim` on the port plane with one voltage path per terminal, from the terminal to the
  reference conductor: `add_port("A", voltage_path=[...])` (the order of `add_port` calls is the path index). Palace
  writes the complex voltage of each mode on each path to `mode-V.csv`; `PalaceTextResults.mode_voltage(index, mode)`
  reads it. Modes come with `k_n` and `n_eff` in `mode-kn.csv`, from which `gamma = j * k0 * n_eff`.
- **Currents.** Use complex terminal currents from the saved mode fields (an Ampere contour around each terminal), or
  the reaction route below. For a reciprocal line `T_I^T T_V` is diagonal with entries `r_k = V_k^T I_k` (the
  unconjugated reaction), so `T_I = T_V^{-T} diag(r)`; `currents_from_reaction(t_v, reaction)` implements it. A
  quasi-TEM solve gives the same result from `T_I = (G + j w C) T_V Gamma^-1`.
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

## 5. Grounds are part of the definition

Terminal voltages are measured against a reference conductor, and a line with `N + 1` conductors has `N` quasi-TEM
modes. Floating outer grounds are therefore extra conductors: a GSSG line with floating grounds has three non-reference
conductors (two signals and one outer ground, with the other outer ground as the reference) and three modes, not two.
Either tie the grounds together (metal strips, a lower plane and continuous via stacks give one ground conductor, and
then two signal terminals and two modes), or declare the extra ground as a terminal. The `T_V`/`T_I` blocks must be
square: as many modes as terminals.

## 6. Mixed-mode output with scikit-rf

Terminal S feeds scikit-rf directly. For a two-end coupled pair ordered (A1, B1, A2, B2), `Network.se2gmm(p=2)` pairs
ports (0, 1) and (2, 3), puts the differential ports first and the common ports after, and defaults the reference
impedances to twice and half of the single-ended value. This replaces the one-wave-port-per-signal plus `se2gmm`
workaround of gsim#264 once terminal S is available. On a symmetric line the differential and common blocks match
`DefinedGammaZ0` lines with `2 * Z_odd` (at 100 ohm) and `Z_even / 2` (at 25 ohm) to 1.5e-14 with no mode conversion
(`|S_dc| <= 1e-15`).

## 7. Limits

- Real, positive `z_ref` only. Complex reference impedances are later work.
- Cross-mode terms in the modal S (a mode on one end exciting another mode on the other end) must come from a multi-mode
  port solve. `uniform_line_modal_s` has none; with a mirror-symmetric layout they vanish.
- `T_V` and `T_I` from a *separate* BoundaryMode run describe the port modes only when those modes are not degenerate,
  or when the cluster is treated as one group (section 3).
- The blocks must be square and well conditioned: `modal_to_terminal_s` raises `ValueError` if `cond(T_V)` or
  `cond(T_I)` exceeds 1e12 ("the modes do not span the terminals").

## 8. Example with synthetic data

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
