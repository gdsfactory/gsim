#!/usr/bin/env python
"""Record the 2-D validation cases that have no committed spectra.

The committed run data covers 2-D straight, coupler, ring-coupling, Bragg and
an exploratory ring, plus 3-D straight and Bragg.  A complete 2-D render still
needs the MZI and a full ring, so this script runs only those two and writes
``s_parameters.csv`` / ``sim_config.json`` / ``meta.json`` beside the others.

Run with the environment that has MEEP, e.g.::

    GSIM_MEEP_PYTHON=/path/to/meep/python \
        python scripts/run_missing_2d.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.pdk_validation import _validation as V  # noqa: N812

MISSING = ("mzi", "ring")


def main() -> int:
    """Run the missing 2-D cases, skipping any that already have spectra."""
    import ubcpdk

    ubcpdk.PDK.activate()

    root = REPO / "nbs" / "data" / "pdk_validation" / "runs"
    overrides = V.high_resolution_overrides()
    meep_python = os.environ.get("GSIM_MEEP_PYTHON")

    for name in MISSING:
        run_dir = root / f"2d_{name}"
        if (run_dir / "s_parameters.csv").is_file():
            print(f"skip {name}: {run_dir} already recorded", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"running {name} (2-D) -> {run_dir}", flush=True)
        outcome = V.run_case(
            name,
            output_dir=str(run_dir),
            python_executable=meep_python,
            verbose=True,
            dimension="2d",
            force_run=True,
            **overrides.get(name, {}),
        )
        (run_dir / "meta.json").write_text(
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
        print(outcome.summary(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
