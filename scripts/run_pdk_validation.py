#!/usr/bin/env python
"""Run the open-PDK MEEP validation cases and record time + size.

This drives :mod:`tests.pdk_validation._validation`: it builds each matched
device, runs MEEP at a (higher) resolution, de-embeds the S-parameters against
the SAX / measured reference and records, per example:

* the mesh (resolution, frequency samples, 2-D or 3-D),
* the simulation cell extent in centre wavelengths (``lambda = 1.55 um``),
* the number of grid cells, and
* the wall-clock compute time (plus MEEP time and timestep count).

Results are written as ``summary.json`` (machine readable) and ``summary.md``
(a table of compute time alongside the size in wavelengths), and the raw
S-parameters are written per case for later plotting.

Example::

    GSIM_MEEP_PYTHON=/path/to/meep/python \
        python scripts/run_pdk_validation.py --high-res --jobs 6 \
        --output /tmp/opencode/pdk_meep_results
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from tests.pdk_validation import _validation as V  # noqa: N812

ALL_CASES = (
    "straight",
    "coupler",
    "ring_coupling",
    "mzi",
    "ring",
    "bragg",
    "cspdk_coupler",
    "cspdk_ring_coupling",
    "siepic_coupler",
)


def _complex_lists(values: Any) -> list[list[float]]:
    """Return a complex array as JSON-friendly ``[real, imag]`` pairs."""
    array = np.asarray(values, dtype=complex)
    return [[float(value.real), float(value.imag)] for value in array]


def _json_result(outcome: V.ValidationOutcome) -> dict:
    """Return a JSON-serialisable record for one validation outcome."""
    return {
        "case": outcome.case,
        "dimension": outcome.dimension,
        "resolution": outcome.resolution,
        "num_freqs": outcome.num_freqs,
        "wall_seconds": outcome.wall_seconds,
        "meep_time": outcome.meep_time,
        "timesteps": outcome.timesteps,
        "size": None
        if outcome.size is None
        else {
            "cell_um": [
                outcome.size.cell_x_um,
                outcome.size.cell_y_um,
                outcome.size.cell_z_um,
            ],
            "size_lambda": list(outcome.size.size_lambda),
            "area_lambda2": outcome.size.area_lambda2,
            "cells": outcome.size.cells,
        },
        "wavelengths": np.asarray(outcome.wavelengths, dtype=float).tolist(),
        "s_params": {
            name: _complex_lists(values) for name, values in outcome.s_params.items()
        },
        "metrics": {
            port: (
                None
                if metric is None
                else {
                    k: v
                    for k, v in metric.items()
                    if k not in ("wavelengths", "sim_db", "ref_db")
                }
            )
            for port, metric in outcome.metrics.items()
        },
        "best_port": outcome.best_port,
        "summary": outcome.summary(),
    }


def _activate_pdk() -> None:
    """Activate the open PDK used by the validation cases, if available."""
    try:
        import ubcpdk
    except Exception:
        ubcpdk = None
    if ubcpdk is not None:
        try:
            ubcpdk.PDK.activate()
        except Exception:
            pass
        else:
            return
    try:
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
    except Exception:
        pass


def _run_one(args: tuple[str, dict, str | None, str | None, int | None]) -> dict:
    """Run one case in a worker process and return its JSON record."""
    name, override, output_root, python_executable, num_processes = args
    _activate_pdk()
    case = V.get_case(name)
    dimension = override.get("dimension") or case.dimension
    output_dir = None
    if output_root:
        output_dir = str(Path(output_root) / f"{dimension}_{name}")
    outcome = V.run_case(
        name,
        output_dir=output_dir,
        python_executable=python_executable,
        num_processes=num_processes,
        dimension=dimension,
        resolution=override.get("resolution"),
        num_freqs=override.get("num_freqs"),
        decay_by=override.get("decay_by"),
        max_time=override.get("max_time"),
        wall_time=override.get("wall_time"),
        subpixel=override.get("subpixel"),
    )
    return _json_result(outcome)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=ALL_CASES,
        default=list(ALL_CASES),
        help="Validation cases to run (default: all six).",
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use the accuracy-first preset (res=30, 201 freqs, decay_by=1e-7).",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-freqs", type=int, default=None)
    parser.add_argument("--decay-by", type=float, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--wall-time", type=float, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/opencode/pdk_meep_results",  # noqa: S108
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python with pymeep installed (default: $GSIM_MEEP_PYTHON or sys.executable).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of cases to run concurrently.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="MPI processes per case (default: single process).",
    )
    parser.add_argument(
        "--dimension",
        choices=("2d", "3d"),
        default=None,
        help="Override every case to this simulation plane.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected cases and write the summary files."""
    args = parse_args(argv)
    import os

    python = args.python or os.environ.get("GSIM_MEEP_PYTHON") or sys.executable
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    overrides = (
        V.high_resolution_overrides()
        if args.high_res
        else {name: {} for name in ALL_CASES}
    )
    global_override = {
        key: value
        for key, value in {
            "resolution": args.resolution,
            "num_freqs": args.num_freqs,
            "decay_by": args.decay_by,
            "max_time": args.max_time,
            "wall_time": args.wall_time,
            "dimension": args.dimension,
        }.items()
        if value is not None
    }
    jobs = []
    for name in args.cases:
        override = {**overrides.get(name, {}), **global_override}
        jobs.append((name, override, str(output_root), python, args.processes))

    results: dict[str, dict | str] = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_run_one, job): job[0] for job in jobs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    print(f"[done] {name}", flush=True)
                except Exception as error:
                    results[name] = f"{type(error).__name__}: {error}"
                    print(f"[fail] {name}: {error!r}", flush=True)
    else:
        for job in jobs:
            name = job[0]
            try:
                results[name] = _run_one(job)
                print(f"[done] {name}", flush=True)
            except Exception as error:
                results[name] = f"{type(error).__name__}: {error}"
                print(f"[fail] {name}: {error!r}", flush=True)

    # Ordered summary of outcomes vs failures.
    ordered: dict[str, V.ValidationOutcome | Exception] = {}
    for name in args.cases:
        value = results.get(name)
        if isinstance(value, dict):
            ordered[name] = V.ValidationOutcome(
                case=name,
                wavelengths=np.array(value["wavelengths"], dtype=float),
                s_params={},
                reference=V.Reference(np.array([])),
                metrics={},
                best_port=value.get("best_port"),
                resolution=value["resolution"],
                num_freqs=value["num_freqs"],
                dimension=value["dimension"],
                wall_seconds=value["wall_seconds"],
                meep_time=value["meep_time"],
                timesteps=value["timesteps"],
                size=V.SimulationSize(
                    cell_x_um=value["size"]["cell_um"][0],
                    cell_y_um=value["size"]["cell_um"][1],
                    cell_z_um=value["size"]["cell_um"][2],
                    resolution=value["resolution"],
                    num_freqs=value["num_freqs"],
                    dimension=value["dimension"],
                ),
            )
            # Rebuild the summary metrics from the JSON result.
            outcome = ordered[name]
            assert isinstance(outcome, V.ValidationOutcome)
            best = value.get("metrics", {}).get(value.get("best_port"))
            if best:
                outcome.metrics[value["best_port"]] = best
        else:
            ordered[name] = RuntimeError(str(value))

    table = V.format_summary(ordered)
    (output_root / "summary.json").write_text(json.dumps(results, indent=2))
    (output_root / "summary.md").write_text(table)
    print("\n" + table, flush=True)
    print(f"\nWrote {output_root / 'summary.json'} and {output_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
