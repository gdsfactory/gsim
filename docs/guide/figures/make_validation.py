"""Result figure for the "Validation and benchmarks" section of docs/guide/index.md.

The stored benchmark suite by family.

Writes docs/guide/images/validation-benchmark-suite.png. One marker per stored case
file of the benchmark results folder (top level ``<family>.<name>.<kind>.json``): the
relative difference |ours - reference| / |reference| of the case's headline value,
coloured by the status stored in the case file. The stored ``deviation`` field has a
different meaning per case (absolute, relative or in units of a tolerance), so the
relative difference is computed here from the stored ``ours`` and ``reference``. The
paper-device ``catalogue`` family is left out: most of its cases are pending and its
headline values are not comparable with the others.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from _common import (
    BENCHMARK_RESULTS,
    LOG,
    apply_style,
    read_stored_case,
    save,
    snapshot,
)
from matplotlib.lines import Line2D

FAMILIES = ("mode", "coupler", "charge", "circuit", "rf")
FAMILY_LABELS = {
    "mode": "Waveguide modes",
    "coupler": "Directional coupler",
    "charge": "Carrier transport",
    "circuit": "Circuit assembly",
    "rf": "RF line (CPW)",
}
# What the reference of a case is, from the last part of the case id.
REFERENCE_KIND = {
    "exact": "analytic or exact",
    "approximate": "approximate formula",
    "convergence": "finer mesh of the same solver",
    "cross-solver": "another solver",
    "meep": "another solver",
    "palace": "another solver",
}
MARKERS = {
    "analytic or exact": "o",
    "approximate formula": "^",
    "finer mesh of the same solver": "D",
    "another solver": "s",
}
STATUS_COLORS = {"pass": "tab:green", "miss": "tab:red"}
FLOOR = 1e-12  # exact agreement is drawn here


def build_snapshot() -> dict:
    """Read the headline value, reference and status of every stored case."""
    cases = []
    for path in sorted(BENCHMARK_RESULTS.glob("*.json")):
        family = path.name.split(".")[0]
        if family not in FAMILIES:
            continue
        case = read_stored_case(path.name)
        cases.append(
            {k: case.get(k) for k in ("case_id", "status", "ours", "reference", "date")}
        )
    return {"cases": cases}


def main() -> None:
    """Draw the stored benchmark cases by family."""
    apply_style()
    cases = snapshot("benchmark_suite", build_snapshot)["cases"]
    dates = sorted(c["date"][:10] for c in cases if c["date"])
    LOG.info("cases: %d | dates: %s to %s", len(cases), dates[0], dates[-1])
    status_counts = Counter((c["case_id"].split(".")[0], c["status"]) for c in cases)
    LOG.info("status by family: %s", dict(sorted(status_counts.items())))

    fig, ax = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    labels: list[str] = []
    for row, family in enumerate(FAMILIES):
        in_family = [c for c in cases if c["case_id"].split(".")[0] == family]
        drawn = [
            c
            for c in in_family
            if isinstance(c["ours"], (int, float))
            and isinstance(c["reference"], (int, float))
            and c["reference"] != 0
            and c["status"] in STATUS_COLORS
        ]
        for i, case in enumerate(drawn):
            relative = max(
                abs(case["ours"] - case["reference"]) / abs(case["reference"]), FLOOR
            )
            kind = REFERENCE_KIND[case["case_id"].split(".")[-1]]
            offset = (i % 5 - 2) * 0.09  # deterministic vertical spread
            ax.plot(
                relative,
                row + offset,
                MARKERS[kind],
                color=STATUS_COLORS[case["status"]],
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
        skipped = len(in_family) - len(drawn)
        labels.append(
            f"{FAMILY_LABELS[family]}\n{len(drawn)} drawn, {skipped} pending or skipped"
        )
    ax.set_xscale("log")
    ax.set_xlim(FLOOR / 3, 1.0)
    ax.set_ylim(len(FAMILIES) - 0.5, -0.5)
    ax.set_yticks(range(len(FAMILIES)), labels)
    ax.grid(axis="y", visible=False)
    # \u2212 is the minus sign, which cp1252 cannot encode (check-cp1252 hook).
    ax.set_xlabel(
        "|ours \u2212 reference| / |reference|   "
        "(1e-12 = agreement to the stored precision)"
    )
    ax.set_title(
        "Stored benchmark cases by family: "
        "difference of the headline value to its reference"
    )
    status_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=c,
            markeredgecolor="black",
            label=f"status {s}",
        )
        for s, c in STATUS_COLORS.items()
    ]
    kind_handles = [
        Line2D(
            [],
            [],
            marker=m,
            linestyle="",
            color="lightgray",
            markeredgecolor="black",
            label=k,
        )
        for k, m in MARKERS.items()
    ]
    ax.legend(
        handles=status_handles + kind_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=3,
        framealpha=1,
    )
    save(fig, "validation-benchmark-suite.png")


if __name__ == "__main__":
    main()
