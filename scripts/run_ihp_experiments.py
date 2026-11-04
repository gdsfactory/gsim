#!/usr/bin/env python
"""Run the IHP L2n0 1 GHz experiment matrix and write results.json.

``pass`` means only that the extracted L is finite and positive; it is not a
match against measured data.  The exact 1/5/9/30 GHz verification points are
produced by ``scripts/run_ihp_verification.py``.
"""

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
F = 1e9
variants = {
    "baseline": {},
    "shell_conformal": {"shell_variant": True, "refined_mesh_size": 1.5},
    "ground0": {"ground_pad_um": 0.0},
    "openems_ground": {"openems_ground": True},
    "pec": {"use_default_pmc_wall": True},
    "ground0_pec": {"ground_pad_um": 0.0, "use_default_pmc_wall": True},
    "mesh1.5": {"refined_mesh_size": 1.5},
    "planar_pec": {"planar_conductors": True},
    "substrate180": {"substrate_thickness_um": 180.0},
    "gap": {"port_variant": "gap"},
}
out = REPO / "nbs" / "data" / "ihp_validation" / "experiments"
out.mkdir(parents=True, exist_ok=True)
results = {}
for name, kw in variants.items():
    print("RUN", name, flush=True)
    try:
        run = b.run_benchmark(
            out / name,
            palace_executable=args.palace,
            num_processes=args.processes,
            frequency_hz=F,
            **kw,  # ty: ignore[invalid-argument-type]
        )
        sp = run.sparams
        if len(sp.port_names) == 2:
            l, q, z = b.lq_at(sp, 0)
            zm = b.s_to_z(b.s_matrix(sp, 0))
            zd = zm[0, 0] - zm[0, 1] - zm[1, 0] + zm[1, 1]
            ld, qd = b.lq_from_impedance(zd, F)
        else:
            n = sp.port_names[0]
            z = b.one_port_impedance_from_s(sp[n, n].complex[0])
            l, q = b.lq_from_impedance(z, F)
            ld, qd = l, q
        results[name] = {
            "ok": True,
            "frequency_ghz": 1.0,
            "L_nH": float(l),
            "Q": float(q),
            "L_diff_nH": float(ld),
            "Q_diff": float(qd),
            "pass": bool(l > 0),
            "ports": sp.port_names,
            "kwargs": kw,
        }
        print("DONE", name, l, q, flush=True)
    except Exception as e:
        results[name] = {
            "ok": False,
            "frequency_ghz": 1.0,
            "error": repr(e),
            "kwargs": kw,
        }
        print("FAIL", name, repr(e), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
