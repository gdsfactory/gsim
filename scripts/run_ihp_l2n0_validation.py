#!/usr/bin/env python
"""Run the IHP SG13G2 L2n0 benchmark and record its S-parameters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import ihp_l2n0_benchmark as _l2n0


def main() -> int:
    """Run the benchmark and write its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_l2n0.RUN_ROOT / "l2n0_30ghz")
    parser.add_argument("--sweep", action="store_true", help="Run 1-50 GHz, 11 points.")
    parser.add_argument("--palace", type=Path, default=None)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")

    run = _l2n0.run_benchmark(
        args.output,
        sweep=args.sweep,
        palace_executable=args.palace,
        num_processes=args.processes,
    )
    f_hz = float(run.frequencies_ghz[0]) * 1e9
    z_diff = _l2n0.differential_impedance_from_s(_l2n0.s_matrix(run.sparams, 0))
    l_nh, q = _l2n0.lq_from_impedance(z_diff, f_hz)
    series_l_nh, series_q, _ = _l2n0.lq_at(run.sparams, 0)
    summary = {
        "case": "ihp_sg13g2_l2n0",
        "frequency_ghz": float(run.frequencies_ghz[0]),
        "differential_inductance_nh": l_nh,
        "differential_q": q,
        "z_diff_ohm": [z_diff.real, z_diff.imag],
        "s21_series_diagnostic_l_nh": series_l_nh,
        "s21_series_diagnostic_q": series_q,
        "ports": run.sparams.port_names,
        "output": str(run.directory),
    }
    (run.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
