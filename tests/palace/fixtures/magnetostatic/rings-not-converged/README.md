# Palace `rings` magnetostatic run that did not converge

These files are verbatim outputs of Palace's `examples/rings` magnetostatic example in awslabs/palace (Apache-2.0), run
on its shipped `mesh/rings.msh` with a local Palace v0.18.1 (`palace.json` GitTag `v0.18.1`, single rank), with
`Solver.Order=1` and `Solver.Linear.MaxIts=10`. Both current-source solves stop at 10 iterations, yet Palace exits 0 and
writes a matrix that passes every physics check: symmetric, positive diagonal, |k| < 1, and E = ½LI². It is wrong all
the same; the mutual inductance is −2.66 pH where the converged run gives +1.96 pH.

- `terminal-M.csv`: inductance matrix, in H.
- `terminal-I.csv`: excitation current of each source, in A.
- `domain-E.csv`: stored energy of each excitation; `E_mag (J)` is the magnetic energy.
- `palace.log`: the run's full standard output, less its final blank line, with the `PCG solver did NOT converge`
  warnings (ANSI colour codes retained). Its first line, `Running with Apptainer`, comes from the local Palace wrapper.
