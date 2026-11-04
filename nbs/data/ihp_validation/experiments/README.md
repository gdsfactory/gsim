# IHP L2n0 exploratory experiment matrix

These are recorded **1 GHz screening runs**, not a mesh-converged validation.
The primary L and Q columns use the differential impedance
`Zdiff = Z11 - Z12 - Z21 + Z22`, as in the pinned IHP OpenEMS two-port script.
The two measured dies give 1.674 and 1.672 nH by this extraction. The best
recorded two-port value below is 0.802 nH, about a factor of two low. The
`ground0_pec` configuration has the current-stack 1–50 GHz sweep used for full complex S comparison;
that comparison is in [the notebook](../../../ihp_l2n0_openems_validation.ipynb).

| Variant | Ports | Differential L at 1 GHz (nH) | Differential Q | Status |
| --- | --- | ---: | ---: | --- |
| baseline | P1, P2 | 0.749 | 0.132 | Exploratory |
| shell_conformal | P1, P2 | 0.754 | 0.122 | Exploratory |
| ground0 | P1, P2 | 0.754 | 0.082 | Exploratory |
| ground0_pec | P1, P2 | 0.802 | 0.087 | Exploratory; outer wall is Palace-default PMC |
| openems_ground | P1, P2 | 0.732 | 0.107 | Exploratory |
| pec | P1, P2 | 0.796 | 0.140 | Exploratory; outer wall is Palace-default PMC |
| mesh1.5 | P1, P2 | 0.754 | 0.122 | Exploratory |
| substrate180 | P1, P2 | 0.749 | 0.132 | Exploratory |
| gap | Pdiff only | 0.780 | 3.061 | One-port diagnostic; not a two-port S comparison |
| planar_pec | P1, P2 | — | — | Invalid numerical output; excluded from comparisons |

The `ground0_pec` historical name records the campaign's intent. Setting
`absorbing_boundary=False` does **not** impose a PEC wall; Palace uses its
default PMC wall. Its apparent 1.921 nH agreement with the measured
1.911/1.932 nH is from `2 Z0 (1-S21)/S21`, which assumes a pure series
element. The two-port has shunt and return-path effects, so this agreement
does not validate the inductor. At 1 GHz its Q is 0.21 by that series
extraction versus about 6.6 measured; the full S21 comparison has about
0.31 complex RMS error and 8.9 dB magnitude RMS error across 11 frequencies.
The recorded mesh error indicator is about 0.18 against a 0.02 tolerance.

The single-frequency `planar_pec` solve exited but produced a negative
inductance of roughly -148,188 nH by the S21 series extraction, so its
`results.json` entry is retained only as failure provenance. Its `ok: true`
means the process returned, while `pass: false` flags a nonphysical result;
neither field is a validation result. The `gap` run has one port and cannot be
directly compared to the measured two-port S-matrix.

The historical 1/5/9/30 GHz verification files used an earlier passivation
model. The 30 GHz point is near or past a measured resonance; its extracted
inductance is not a useful scalar agreement metric. These files remain
exploratory provenance, not evidence of frequency or mesh convergence.

The current 1 GHz screening stack uses PDK layer heights and materials but
replaces its dielectric regions with a 280 µm substrate, 3.75 µm epi, and
SiO2 ending at the PDK passivation bottom (14.03 µm). The default 1.9 µm
blanket SiN passivation is absent. The `shell_conformal` variant adds only a
0.6 µm SiN sidewall ring around TopMetal2 (layer 999/0).

The `openems_ground` variant adds a Metal1 rectangle but retains vertical
Metal1-to-TopMetal1 ports and omits the reference script's central pillar.
The separate `openems_edge` setup adds edge-gap ports and pillar, but has
only been meshed at 5 µm, without a Palace solve. The mesh under-resolves a
2.998 µm feature; no result from that setup is included here.

The scripts `scripts/run_ihp_experiments.py` and
`scripts/run_ihp_verification.py` describe how the historical runs were
created. They require a Palace executable and should not be run as a routine
test on a shared workstation.

For each recorded case, the PR retains the input and resolved configs,
port information, `port-S.csv`, error indicators and metadata needed to
recompute these comparisons and inspect solver settings. Derived `domain-E`,
`port-I`, `port-V` and Palace output-config copies were removed from the PR.
