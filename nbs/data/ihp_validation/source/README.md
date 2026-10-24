# IHP SG13G2 L2n0 benchmark source data

Pinned from the IHP Open PDK at commit `943983786b04351feb20e33e8d3d76c0e0d87cce`
(`ihp-sg13g2/libs.tech/openems/testcase/SG13_Octagon_L2n0`), Apache-2.0.

- `L_2n0_simplified.gds` — octagonal 2-turn, nominal 2 nH inductor layout.
- `die_x2y7.s2p` — measured, THRU-de-embedded GSGSG data on die x2y7.
- `die_x5y8.s2p` — measured, THRU-de-embedded GSGSG data on die x5y8.

Reference planes are the TopMetal1 feed ends at y = 0. The IHP openEMS model uses TopMetal2 windings, TopMetal1
feeds/underpass, a 280 um substrate, no passivation, IHP OpenEMS refined_cellsize=1.5 um, max_cellsize=lambda_air/(sqrt(eps_max)*20), and edge ports at the TopMetal1 feed ends referencing the Metal1/common ground pillar.
