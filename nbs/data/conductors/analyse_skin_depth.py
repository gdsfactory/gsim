"""Metal loss of the surface model over the volume model, against t / delta.

Reads the ``port-S.csv`` of each job of ``make_skin_depth_jobs.py`` from
``results/<job name>/`` (any depth below that folder, so both a flat download and the
cloud layout ``results/<name>/output/port-S.csv`` work), and writes

* ``conductor_loss_vs_skin_depth.csv`` next to this script, and
* the figure ``docs/assets/plots/conductor_loss_vs_skin_depth.png``.

Absorbed power
    With only the first port excited, ``1 - sum_i |S[i][1]|**2`` is the fraction of the
    incident power that the structure absorbs. With the lossless oxide and air of
    these jobs, that is the metal loss plus whatever the absorbing outer wall and the
    solver error take.
Baseline
    If the ``pec_*`` jobs were run, their absorbed fraction (the same structure with
    perfect-conductor sheets) is subtracted at the same ``t / delta``, which removes
    that non-metal part. Without them the absorbed fraction itself is used, and the
    CSV says so (``baseline`` column).
Metal loss per length
    The (baseline-corrected) absorbed fraction divided by the line length in mm. The
    port mismatch is in it, the same for both models; only the ratio is meant to be read.

Usage (from the repository root)::

    uv run python nbs/data/conductors/analyse_skin_depth.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS_DIR = HERE / "results"
MANIFEST = HERE / "jobs" / "manifest.json"
CSV_PATH = HERE / "conductor_loss_vs_skin_depth.csv"
FIGURE_PATH = REPO / "docs" / "assets" / "plots" / "conductor_loss_vs_skin_depth.png"

_S_COLUMN = re.compile(r"^\|S\[(\d+)\]\[(\d+)\]\| \(dB\)$")
FREQUENCY_TOLERANCE = 1e-3  # relative; the csv prints GHz


def _load_generator():
    """The job script as a module (it lives next to this file and is not a package)."""
    path = HERE / "make_skin_depth_jobs.py"
    spec = importlib.util.spec_from_file_location("make_skin_depth_jobs", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_skin_depth_jobs", module)
    spec.loader.exec_module(module)
    return module


def read_port_s(path: Path) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Read a Palace ``port-S.csv``: frequencies (GHz) and ``{(i, j): |S[i][j]| in dB}``."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    if len(rows) < 2:
        raise ValueError(f"{path}: no data rows")
    header = [cell.strip() for cell in rows[0]]
    if header[0] != "f (GHz)":
        raise ValueError(f"{path}: first column is {header[0]!r}, expected 'f (GHz)'")
    data = np.array(
        [[float(cell) for cell in row] for row in rows[1:] if row], dtype=float
    )
    if data.shape[1] != len(header):
        raise ValueError(
            f"{path}: {data.shape[1]} values per row, {len(header)} columns"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: non-finite values")
    magnitudes = {}
    for index, name in enumerate(header):
        found = _S_COLUMN.match(name)
        if found:
            magnitudes[(int(found.group(1)), int(found.group(2)))] = data[:, index]
    if not magnitudes:
        raise ValueError(f"{path}: no |S[i][j]| (dB) columns")
    return data[:, 0], magnitudes


def absorbed_fraction(magnitudes_db: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """``1 - sum_i |S[i][j]|**2``, averaged over the excited ports ``j`` in the file."""
    sources = sorted({j for _, j in magnitudes_db})
    per_source = []
    for source in sources:
        power = sum(
            10.0 ** (db / 10.0) for (i, j), db in magnitudes_db.items() if j == source
        )
        per_source.append(1.0 - power)
    return np.mean(per_source, axis=0)


def _find_port_s(results_dir: Path, name: str) -> Path | None:
    folder = results_dir / name
    if not folder.is_dir():
        return None
    found = sorted(folder.rglob("port-S.csv"))
    if not found:
        return None
    return min(found, key=lambda p: len(p.parts))  # the shallowest one


def _absorbed_for(job: dict, results_dir: Path) -> float | None:
    """Absorbed power fraction of one job at its frequency; ``None`` if not run."""
    path = _find_port_s(results_dir, job["name"])
    if path is None:
        return None
    frequencies, magnitudes = read_port_s(path)
    expected = job["frequency_ghz"]
    nearest = int(np.argmin(np.abs(frequencies - expected)))
    if abs(frequencies[nearest] - expected) > FREQUENCY_TOLERANCE * expected:
        raise ValueError(
            f"{path}: frequencies {frequencies.tolist()} GHz, job {job['name']} is at {expected} GHz"
        )
    return float(absorbed_fraction(magnitudes)[nearest])


def analyse(results_dir: Path, jobs: list[dict]) -> list[dict]:
    """One row per ``t / delta``: absorbed power and metal loss for each model."""
    ratios = sorted({job["t_over_delta_requested"] for job in jobs})
    rows = []
    missing = []
    for ratio in ratios:
        row: dict = {"t_over_delta_requested": ratio}
        for job in jobs:
            if job["t_over_delta_requested"] != ratio:
                continue
            row.setdefault("t_over_delta", job["t_over_delta"])
            row.setdefault("frequency_ghz", job["frequency_ghz"])
            row.setdefault("skin_depth_um", job["skin_depth_um"])
            row.setdefault("length_mm", job["length_um"] / 1e3)
            row[f"length_mm_{job['model']}"] = job["length_um"] / 1e3
            absorbed = _absorbed_for(job, results_dir)
            row[f"absorbed_{job['model']}"] = absorbed
            if absorbed is None and job["required"]:
                missing.append(job["name"])
        rows.append(row)
    if missing:
        raise FileNotFoundError(
            f"no port-S.csv under {results_dir} for: {', '.join(missing)}"
        )

    for row in rows:
        baseline = row.get("absorbed_pec")
        row["baseline"] = "pec" if baseline is not None else "none"
        for model in ("surface", "volume", "surface_matched"):
            absorbed = row.get(f"absorbed_{model}")
            if absorbed is None:
                row[f"loss_per_mm_{model}"] = math.nan
                continue
            metal = absorbed - (baseline or 0.0)
            length_mm = row.get(f"length_mm_{model}", row["length_mm"])
            row[f"loss_per_mm_{model}"] = metal / length_mm
        for model in ("surface", "surface_matched"):
            volume = row["loss_per_mm_volume"]
            loss = row[f"loss_per_mm_{model}"]
            valid = volume > 0 and math.isfinite(loss)
            row[f"ratio_{model}_over_volume"] = loss / volume if valid else math.nan
    return rows


COLUMNS = (
    "t_over_delta_requested",
    "t_over_delta",
    "frequency_ghz",
    "skin_depth_um",
    "length_mm",
    "baseline",
    "absorbed_surface",
    "absorbed_volume",
    "absorbed_surface_matched",
    "absorbed_pec",
    "loss_per_mm_surface",
    "loss_per_mm_volume",
    "loss_per_mm_surface_matched",
    "ratio_surface_over_volume",
    "ratio_surface_matched_over_volume",
)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the table; a missing optional value is an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            cells = []
            for column in COLUMNS:
                value = row.get(column)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    cells.append("")
                elif isinstance(value, float):
                    cells.append(f"{value:.6g}")
                else:
                    cells.append(str(value))
            writer.writerow(cells)


def plot(rows: list[dict], path: Path) -> None:
    """Loss per mm of each model, and the surface-over-volume ratio, against t / delta."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([row["t_over_delta"] for row in rows])
    fig, (ax_loss, ax_ratio) = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True
    )

    styles = {
        "volume": ("meshed conductive volume", "o-", "tab:blue"),
        "surface": ("surface impedance", "s--", "tab:red"),
        "surface_matched": (
            "surface impedance, volume's mesh sizing",
            "^:",
            "tab:green",
        ),
    }
    for model, (label, style, color) in styles.items():
        y = np.array(
            [row.get(f"loss_per_mm_{model}", math.nan) for row in rows], dtype=float
        )
        if np.all(np.isnan(y)):
            continue
        ax_loss.loglog(x, np.where(y > 0, y, np.nan), style, color=color, label=label)
    ax_loss.set_xlabel("t / skin depth")
    ax_loss.set_ylabel("metal loss per mm (fraction of incident power)")
    ax_loss.grid(visible=True, which="both", alpha=0.3)
    ax_loss.legend(fontsize=8)

    for model, style, color, label in (
        ("surface", "s--", "tab:red", "surface / volume"),
        (
            "surface_matched",
            "^:",
            "tab:green",
            "surface (volume's mesh sizing) / volume",
        ),
    ):
        y = np.array(
            [row.get(f"ratio_{model}_over_volume", math.nan) for row in rows],
            dtype=float,
        )
        if np.all(np.isnan(y)):
            continue
        ax_ratio.semilogx(x, y, style, color=color, label=label)
    ax_ratio.axhline(1.0, color="black", linewidth=0.8)
    ax_ratio.set_xlabel("t / skin depth")
    ax_ratio.set_ylabel("metal loss: surface model / meshed volume")
    ax_ratio.grid(visible=True, which="both", alpha=0.3)
    ax_ratio.legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_jobs(manifest: Path) -> list[dict]:
    """The jobs of the manifest, or the generator's plan when there is no manifest."""
    if manifest.is_file():
        import json

        return json.loads(manifest.read_text(encoding="utf-8"))["jobs"]
    return _load_generator().planned_jobs()


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args(argv)

    rows = analyse(args.results, load_jobs(args.manifest))
    write_csv(rows, args.csv)
    plot(rows, args.figure)
    for row in rows:
        print(
            f"t/delta {row['t_over_delta']:6.3f}  f {row['frequency_ghz']:9.4f} GHz  "
            f"surface/volume {row['ratio_surface_over_volume']:.4f}  (baseline: {row['baseline']})"
        )
    print(f"wrote {args.csv} and {args.figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
