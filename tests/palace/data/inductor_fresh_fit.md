The CSV records the inductor tutorial's fresh 10–200 GHz circuit-synthesis sweep using Palace
`0dd8d015034c0fc1f0cf4c9ff952b533f9cb91d5`, gsim main `9f31460946f5c3f036fb91bc32a2ed1bc38245ce`, and
`ihp-gdsfactory==2.0.1`. The circuit S matrix uses 50-ohm reference impedances and port order P1, P2. The independent
FEM reference is the differential impedance obtained from Palace's `port-S.csv`, rather than from the exported circuit
matrices.

The original three-real-pole, three-complex-pair recipe becomes inaccurate after passivity enforcement on these inputs.
Two real poles and two complex pairs preserve stability, passivity, and the tutorial's 5% impedance-error limit. These
samples are a regression fixture; the notebook submits fresh solver jobs.
