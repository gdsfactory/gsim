# Palace validation log excerpts

These are verbatim, noncontiguous excerpts of locally archived runs. They are parser fixtures, not complete runs or
claims of physical validation. Line numbers below refer to the original logs, before extraction. ANSI warning colors and
Unicode mode symbols are retained.

- `disabled-estimator.txt`: recorded `palace.json` GitTag `cfa430a`. Source:
  `nbs/palace-sim-waveport-lowfreq-debug/remote/ln-original/palace.log`. Original lines: 76–85, 121–135.
- `adaptive-core-failure.txt`: recorded GitTag `cfa430a`. Source:
  `nbs/palace-sim-waveport-lowfreq-debug/remote/cpw_adaptive/palace.log`. Original lines: 128, 737–740, 1726–1733,
  3714–3721, 3727–3731.
- `offline-disabled-estimator.txt`: recorded GitTag `cfa430a-dirty`. Source:
  `nbs/palace-sim-mzm-ln-chains/adaptive/remote/adaptive-tbar_1cell/palace.log`. Original lines: 69–70, 79, 116–126,
  540–548.
- `eigenmode-disabled-estimator.txt`: recorded GitTag `v0.18.1`. Source: local Palace v0.18.1 run of a gsim-built
  eigenmode config with `Solver.Linear.EstimatorMaxIts=0`. Original lines: 823–825, 827–832, 881–895.
- `boundarymode-disabled-estimator.txt`: recorded GitTag `v0.18.1`. Source: local Palace v0.18.1 run of a gsim-built
  boundary-mode config with `Solver.Linear.EstimatorMaxIts=0`. Original lines: 240–241, 247–278.
- `electrostatic-disabled-estimator.txt`: recorded GitTag `v0.18.1`. Source: local Palace v0.18.1 run of a gsim-built
  electrostatic config with `Solver.Linear.EstimatorMaxIts=0`. Original lines: 66, 148–158, 193–203.

The first and third inputs explicitly set `Solver.Linear.EstimatorMaxIts=0`. The first has an evanescent selected mode
despite GMRES convergence. The second reports successful adaptive sampling independently of an earlier failed GMRES
solve. The dirty suffix is retained; these fixtures do not claim an exact clean build for that run. The eigenmode
excerpt keeps a PETSc `EPS Object` view between the estimator header and the warnings. Tests also use explicitly
constructed representative field-correction failures, missing outcomes, and caller-defined parity samples.
