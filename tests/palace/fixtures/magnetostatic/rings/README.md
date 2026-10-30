# Palace `rings` magnetostatic outputs

These files are verbatim outputs of Palace's `examples/rings` magnetostatic example in awslabs/palace (Apache-2.0), run
unchanged on its shipped `mesh/rings.msh` with a local Palace v0.18.1 (`palace.json` GitTag `v0.18.1`, single rank). The
tables match Palace's regression references, `test/data/regression/ref/rings`, to 6e-8 relative. The two current sources
are the inner ring (index 1) and the outer ring (index 2).

- `terminal-M.csv`: inductance matrix, in H.
- `terminal-Minv.csv`: its inverse, in 1/H (UTF-8 header: `M⁻¹`).
- `terminal-Mm.csv`: Palace's current-difference form of the matrix; gsim does not read it.
- `terminal-I.csv`: excitation current of each source, in A.
- `domain-E.csv`: stored energy of each excitation; `E_mag (J)` is the magnetic energy.
- `palace.log`: the run's full standard output, less its final blank line; both current-source solves converge.
