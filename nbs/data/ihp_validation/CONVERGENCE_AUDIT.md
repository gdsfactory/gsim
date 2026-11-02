# L2n0 Palace validation audit

The primary validation observable is the measured 50 Ω complex two-port S-matrix, not the S21-only series-inductance
formula. The recorded best `ground0_pec` sweep has 11 points from 1 to 50 GHz. Against the two pinned measured `.s2p`
files, its S21 magnitude RMS error is 8.86/8.83 dB and its phase RMS error is 44.0/45.4 degrees. S11, S12 and S22 also
disagree (the notebook reports all four). The model is not validated.

<!-- markdownlint-disable MD013 -->

| Item             | Pinned IHP OpenEMS L2n0                                                                                                                                     | Recorded Palace `ground0_pec`                                                               | Implication                                                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Extraction       | Differential `Z11 - Z12 - Z21 + Z22`                                                                                                                        | Headline used `2 Z0 (1-S21)/S21`                                                            | The close 1 GHz series value is not a like-for-like validation. Differential L is about 0.80 nH simulated versus 1.67 nH measured |
| Ground and ports | Metal1 rectangle x=-240..240, y=-140..10; central PEC pillar x=-10..10, y=-10..10; two x-directed gap ports x=-22.2..-10 and 10..22.2, y=0..10 at TopMetal1 | SUBGND pad, vertical interlayer ports centred at x=±34.2, y=0; no central pillar            | Different return current and port parasitics; differential-extraction 1 GHz Q ≈0.087 simulated versus ≈6.6 measured               |
| Boundary         | PEC outer boundary                                                                                                                                          | Disabling `absorbing_boundary` leaves the outer wall unconditioned (Palace default PMC)     | The `ground0_pec` name describes an intended setting, not a PEC wall                                                              |
| Stack            | 280 µm substrate; epi conductivity 5 S/m; SiO2 epsilon 4.1; TopMetal2 3.03e7 S/m, TopMetal1 2.78e7 S/m; no passivation                                      | PDK-derived stack plus local overrides; epi 2 S/m, uniform aluminum conductivity 3.77e7 S/m | The models do not use identical material cards                                                                                    |
| Mesh             | 1.5 µm refined OpenEMS cell; separately published 1 and 0.5 µm variants                                                                                     | 5 µm nominal Palace mesh in the sweep; 1.5 µm single-frequency sensitivity                  | Discretization error remains unresolved; Palace error-indicator Norm ≈0.18 versus AdaptiveTol 0.02                                |

<!-- markdownlint-enable MD013 -->

The new `openems_edge` variant in the benchmark helper places the two gap ports at the pinned coordinates, adds the
central Metal1-to-TopMetal1 PEC pillar, and uses the reference Metal1 ground rectangle. It is a setup for a controlled
comparison, **not a validated numerical result**. It retains the PDK material cards. A one-thread 5 µm coarse mesh
completed with 21,977 nodes, 126,515 tetrahedra and about 807,000 estimated field DOFs. The generated Palace config has
two separate 50 Ω port attributes; their sheets span x = -22.2..-10 and 10..22.2 µm at y = 5 µm and z = 6.23..8.23 µm,
meeting the proposed feed-to-pillar gaps. The sheets lie at y = 5 µm, rather than spanning y = 0..10 µm. The lightweight
mesh/config evidence is in `experiments/openems_edge_mesh/`; the 6.8 MB mesh is kept outside the repository. In the
recorded two-port setup, half of each port (y < 0) touches no metal; that setup has not passed a port-current or
DC-resistance check. A 2.998 µm conductor feature triggered the mesher's under-resolution warning at 5 µm. The model has
**not been solved**. Field/current-density and 1 GHz loss checks, with a finer mesh, are required before a full sweep.

The pinned `L_2n0_simplified.gds` also contains a large polygon on GDS layer 75. The current IHP-to-Palace stack does
not map layer 75, and the recorded `ground0_pec` mesh has no physical group for it. The gray outline in the notebook's
GDS view is therefore not a simulated ground conductor. The recorded model instead adds a separate SUBGND polygon on
layer 250; the OpenEMS reference defines its Metal1 ground and central pillar explicitly.

Three historical variants have unresolved absorbing-boundary provenance. In the archived resolved configs,
`openems_ground` and `planar_pec` assign Absorbing to `si__None` as well as `air__None`; `gap` assigns it to
`sio2__None` as well as `air__None`. The group names and attributes agree with the archived mesh metadata, but we have
not inspected whether these are genuinely exterior faces. These variants remain exploratory or invalid and should be
remeshed with the corrected boundary classifier before use in any accuracy or convergence argument.

Next controlled sequence: (1) verify that the new port sheets meet the TM1 feed and central PEC pillar and that R at 1
GHz is near the measured order of 1–2 Ω; (2) compare all four complex S entries with identical reference planes; (3)
vary conductor model, material cards, boundary margin and mesh one at a time, recording DOF, wall time, peak RAM, Palace
convergence and physical S-error. The source
[IHP OpenEMS L2n0 script](https://github.com/IHP-GmbH/IHP-Open-PDK/blob/943983786b04351feb20e33e8d3d76c0e0d87cce/ihp-sg13g2/libs.tech/openems/testcase/SG13_Octagon_L2n0/OpenEMS_Python/SG13_L2n0_2port_groundpolygon_mesh1.5um.py)
defines the target setup.
