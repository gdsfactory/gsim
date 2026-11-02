# OpenEMS-style L2n0 port mesh check

This is a **mesh-only** check of the `openems_edge` benchmark variant. It does
not include a Palace solve or any S-parameters. `config.json` records two
separate 50 Ω gap-port attributes; `port_information.json` records their
locations; `metadata.json` records mesh size and quality. The generated
`palace.msh` remains in `/tmp/gsim-openems-edge-mesh/` and is not committed.

The one-thread coarse mesh used a 5 µm refined size. It had 21,977 nodes and
126,515 tetrahedra. The mesher warned that a 2.998 µm conductor feature may
be under-resolved and suggested a refined size of at most 1.499 µm for two
cells across it. The coarse mesh establishes that the geometry and two port
sheets can be constructed; it cannot establish numerical convergence or a
correct 1 GHz resistance.
