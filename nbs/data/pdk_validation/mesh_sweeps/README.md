# UBC directional-coupler mesh sweep

`../runs/2d_coupler/` is the archived 30 pixels/µm baseline. `coupler_r50/` and `coupler_r70/` contain the refined
S-parameters, generated Meep configurations, diagnostics and comparison summaries. All three use the same GDS and
asymmetric-slab core permittivity (8.1081139); see `../GEOMETRY_AUDIT.md` for the physical comparison and its limits.

The 50 pixels/µm run used one Meep process and 101 frequencies:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_pdk_validation.py \
  --cases coupler --dimension 2d --resolution 50 --num-freqs 101 \
  --wall-time 90 --max-time 300 --processes 1 \
  --python /path/to/python-with-meep --output /tmp/gsim-coupler-r50-slab
```

The code now pins the 2-D core index to the local slab instead of the full-stack substrate mode. The archived files are
simulation output, not a measurement or a convergence certificate.

The 70 pixels/µm run also used one Meep process and 101 frequencies. Its command used
`--resolution 70 --decay-by 0.0001 --wall-time 300 --max-time 1000`, matching the archived 30 pixels/µm stopping
settings. The 50 pixels/µm run used a tighter decay threshold and a shorter maximum time. Actual Meep stop times were
about 250, 250 and 200 for the 30, 50 and 70 pixels/µm runs.
