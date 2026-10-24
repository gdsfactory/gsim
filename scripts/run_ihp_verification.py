#!/usr/bin/env python
"""Run the exact 1/5/9/30 GHz baseline and best-model verification points."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gsim.common.benchmarks import ihp_l2n0 as b

PALACE = "/home/martin/.cache/palacetoolkit/runtime/palace-cpu-v0.18.0/bin/palace"
OUT = REPO / "nbs" / "data" / "ihp_validation" / "experiments" / "verification"
VARIANTS = {
    "baseline": {},
    "best": {"ground_pad_um": 0.0, "pec_boundary": True},
}
OUT.mkdir(parents=True, exist_ok=True)
rows: dict[str, dict[str, dict[str, float]]] = {}
for name, kwargs in VARIANTS.items():
    rows[name] = {}
    for f_ghz in (5, 9, 30):
        run = b.run_benchmark(
            OUT / f"{name}_{f_ghz}ghz",
            palace_executable=PALACE,
            num_processes=4,
            frequency_hz=f_ghz * 1e9,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        sp = run.sparams
        l_nh, q, _ = b.lq_at(sp, 0)
        z = b.s_to_z(b.s_matrix(sp, 0))
        z_diff = z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1]
        l_diff, q_diff = b.lq_from_impedance(z_diff, f_ghz * 1e9)
        rows[name][str(f_ghz)] = {
            "L_nH": float(l_nh),
            "Q": float(q),
            "L_diff_nH": float(l_diff),
            "Q_diff": float(q_diff),
        }
(OUT / "verification.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
