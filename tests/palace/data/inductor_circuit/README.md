This test fixture contains the minimal circuit matrices, port loads, reference impedances, port mapping, and FEM
S-parameters from the inductor tutorial's fresh 10–200 GHz circuit-synthesis run on November 3, 2026.

- Palace: `0dd8d015034c0fc1f0cf4c9ff952b533f9cb91d5`
- gsim base: `9f31460946f5c3f036fb91bc32a2ed1bc38245ce`
- Process model: `ihp-gdsfactory==2.0.1`
- Execution: 32 MPI ranks × 2 OpenMP threads, CMake build
- Port order: P1, P2; reference impedances: 50 ohms

The independent reference for the circuit export is Palace's `port-S.csv`. Its SciPy one-pole fit gives R=1.6335 ohms,
L=105.792 pH, C=8.181 fF, f0=171.082 GHz, Q=69.62. The exported circuit agrees within 0.0006241 in complex S. The
passive two-real-pole, two-complex-pair fit agrees with the exported circuit within 1% in S and differential Z, and with
the FEM differential Z within 1.09%. The older three-real-pole, three-complex-pair recipe gives about 62.7% FEM Z error
after passivity enforcement on these inputs.

These files support offline regression tests. The notebook submits fresh jobs and does not read this directory. The old
notebook caches have been removed.
