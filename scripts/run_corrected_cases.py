#!/usr/bin/env python
"""Re-run the corrected validation cases.

The original MZI used dL=10 um (the measured MZI3 is 214.215 um), the ring
case used a single ring (the measurement is a double-bus ring), and the Bragg
case used a rectangular 20-period grating read at port 0 (the measurement is a
sinusoidal 200-period Set1 #10 read at the reflection port 1).

This runs the corrected devices in 2-D with the ``fast_overrides`` preset and
writes each result to ``nbs/data/pdk_validation/runs/2d_<case>``.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASES = ("mzi", "ring", "bragg")


def _run(name: str) -> tuple[str, dict, float]:
    """Run one corrected case in a worker process."""
    import ubcpdk

    ubcpdk.PDK.activate()

    from tests.pdk_validation import _validation as V  # noqa: N812

    root = REPO / "nbs" / "data" / "pdk_validation" / "runs"
    out = root / f"2d_{name}"
    out.mkdir(parents=True, exist_ok=True)
    overrides = V.fast_overrides()[name]
    outcome = V.run_case(
        name,
        output_dir=str(out),
        python_executable=os.environ.get("GSIM_MEEP_PYTHON"),
        verbose=True,
        dimension="2d",
        force_run=True,
        **overrides,
    )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "wall_seconds": outcome.wall_seconds,
                "meep_time": outcome.meep_time,
                "timesteps": outcome.timesteps,
                "resolution": outcome.resolution,
                "num_freqs": outcome.num_freqs,
                "dimension": outcome.dimension,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return name, outcome.summary(), outcome.wall_seconds


def main() -> int:
    """Run the selected corrected cases in parallel and write a summary."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        choices=CASES,
        help="cases to run (default: all three)",
    )
    args = parser.parse_args()
    selected = tuple(args.cases)

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=len(selected)) as pool:
        futures = {pool.submit(_run, name): name for name in selected}
        for future in as_completed(futures):
            name = futures[future]
            try:
                case, summary, wall = future.result()
                results[case] = summary
                print(f"[done] {case} wall={wall:.0f}s", flush=True)
            except Exception as error:
                results[name] = {"error": repr(error)}
                print(f"[failed] {name}: {error!r}", flush=True)

    summary_path = REPO / "nbs" / "data" / "pdk_validation" / "corrected_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf8")
    print(f"wrote {summary_path}", flush=True)
    return 0 if all("error" not in row for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
