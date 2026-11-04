#!/usr/bin/env python
"""Run the exact 1/5/9/30 GHz baseline and best-model verification points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import ihp_l2n0_benchmark as b

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--palace", type=Path, required=True)
parser.add_argument("--processes", type=int, default=1)
args = parser.parse_args()
if args.processes < 1:
    parser.error("--processes must be positive")
OUT = REPO / "nbs" / "data" / "ihp_validation" / "experiments" / "verification"
VARIANTS = {
    "baseline": {},
    "best": {"ground_pad_um": 0.0, "use_default_pmc_wall": True},
}
OUT.mkdir(parents=True, exist_ok=True)
rows: dict[str, dict[str, dict[str, float]]] = {}
for name, kwargs in VARIANTS.items():
    rows[name] = {}
    for f_ghz in (5, 9, 30):
        run = b.run_benchmark(
            OUT / f"{name}_{f_ghz}ghz",
            palace_executable=args.palace,
            num_processes=args.processes,
            frequency_hz=f_ghz * 1e9,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        sp = run.sparams
        l_nh, q, _ = b.lq_at(sp, 0)
        z_diff = b.differential_impedance_from_s(b.s_matrix(sp, 0))
        l_diff, q_diff = b.lq_from_impedance(z_diff, f_ghz * 1e9)
        rows[name][str(f_ghz)] = {
            "L_nH": float(l_nh),
            "Q": float(q),
            "L_diff_nH": float(l_diff),
            "Q_diff": float(q_diff),
        }
(OUT / "verification.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
