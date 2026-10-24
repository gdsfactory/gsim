# IHP L2n0 experiment matrix

All runs at **1 GHz** on the 280 µm-substrate stack. `L` is the mandated series extraction; `L diff` is the port-comparison differential extraction. `pass` means only that L is finite and positive, not that it matches measurement.

| variant | L 1GHz nH | Q | L diff nH | pass | notes |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 1.599 | 0.28 | 0.749 | True | {} |
| shell_conformal | 1.628 | 0.26 | 0.754 | True | {'shell_variant': True, 'refined_mesh_size': 1.5} |
| ground0 | 1.882 | 0.20 | 0.754 | True | {'ground_pad_um': 0.0} |
| ground0_pec | 1.921 | 0.21 | 0.802 | True | {'ground_pad_um': 0.0, 'pec_boundary': True} |
| openems_ground | 1.654 | 0.24 | 0.732 | True | {'openems_ground': True} |
| pec | 1.639 | 0.29 | 0.796 | True | {'pec_boundary': True} |
| mesh1.5 | 1.623 | 0.26 | 0.754 | True | {'refined_mesh_size': 1.5} |
| planar_pec | -148188.292 | 10.34 | -2294.982 | False | {'planar_conductors': True} |
| substrate180 | 1.599 | 0.28 | 0.749 | True | {'substrate_thickness_um': 180.0} |
| gap | 0.780 | 3.06 | 0.780 | True | {'port_variant': 'gap'} |

Best series result: `ground0_pec` 1.921 nH vs measured 1.911/1.932 nH. Differential comparison: ground0_pec 0.802 nH and gap 0.780 nH vs measured differential 1.674/1.672 nH — both ~2.1x low. The series match is therefore not a full validation.

Conformal shell: `shell_conformal` adds a 0.6 µm SiN sidewall ring around TopMetal2 (layer 999/0, same z-range as TopMetal2) plus the existing 1.9 µm top passivation; it gives 1.6277 nH at 1 GHz and does not improve the match.

Finite conductors use Palace `Conductivity` + `Thickness` surface-impedance BCs: δ=sqrt(2/(ωμσ)), Rs=1/(σδ), σ=3.77e7 S/m, PDK thicknesses 2/3/3.75 µm. `planar_pec` is numerically invalid.

OpenEMS-like ground: `openems_ground` uses a Metal1 rectangle (-240..240 × -140..10 µm) and Metal1→TopMetal1 via ports. The pinned script instead uses TopMetal1 edge ports with a central Metal1→TopMetal1 PEC pillar; the central pillar is not modeled, and this difference contributes to the remaining mismatch.

## Verification frequencies

| model | 1 GHz L | 5 GHz L | 9 GHz L | 30 GHz L |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.599 | 1.487 | 1.424 | 0.686 |
| best (ground0_pec) | 1.921 | 1.775 | 1.692 | 0.716 |

At 30 GHz both models are far below measured ~3.61/3.71 nH. Low-frequency Q is 0.21–0.28 vs measured ~6.6, and the mesh error Norm is ~0.18 against AdaptiveTol 0.02; the model is not converged.

Reproduce the matrix with `scripts/run_ihp_experiments.py`; reproduce the exact 1/5/9/30 GHz table with `scripts/run_ihp_verification.py`.
