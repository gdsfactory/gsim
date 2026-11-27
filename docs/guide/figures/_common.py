"""Shared helpers for the guide figure scripts (style, saving, checks, stored results).

Every ``make_<page>.py`` in this folder imports this module. The figures are written
to ``docs/guide/images/``. Run a script from the repository root, for example
``uv run python docs/guide/figures/make_couplers_and_mmis.py``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root
OUT = HERE.parent / "images"
DATA = HERE / "data"
# simulation-benchmarks results folder if present; else the snapshot is used.
BENCHMARK_RESULTS = Path(
    os.environ.get("GSIM_BENCHMARK_RESULTS", "simulation-benchmarks/results")
)

DPI = 110
MAX_BYTES = 300_000

# What the scripts report while they run (sizes, numbers that go into captions).
LOG = logging.getLogger("guide_figures")

# Light figure on a white background, so it reads in the dark theme as well.
STYLE: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    """Install the shared Matplotlib style and make the scripts report progress."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    style.use(STYLE)


def today() -> str:
    """Date of the run, for captions such as "local femwell run, 2026-11-25"."""
    return dt.datetime.now(tz=dt.UTC).astimezone().date().isoformat()


def _drawn_texts(fig):
    """Texts that are drawn.

    Tick labels count only where the tick lies inside the view limits.
    """
    texts = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        # Titles (centre, left, right) and every text or annotation placed on the axes.
        texts += [child for child in ax.get_children() if isinstance(child, Text)]
        texts += [ax.xaxis.label, ax.yaxis.label]
        legend = ax.get_legend()
        if legend is not None:
            texts += legend.get_texts()
        for axis, limits, flag in (
            (ax.xaxis, ax.get_xlim(), "labelbottom"),
            (ax.yaxis, ax.get_ylim(), "labelleft"),
        ):
            if not axis.get_tick_params(which="major").get(flag, True):
                continue  # for example the shared y axis of the second panel
            low, high = sorted(limits)
            for loc, label in zip(
                axis.get_majorticklocs(), axis.get_ticklabels(), strict=False
            ):
                if low - 1e-9 <= loc <= high + 1e-9:
                    texts.append(label)
    return texts


def _text_boxes(fig):
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in _drawn_texts(fig):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        boxes.append((text, box))
    return boxes


def check_layout(fig, name: str) -> list[str]:
    """Return problems found in a figure: text outside the canvas or text on text."""
    fig.canvas.draw()
    fig.canvas.draw()  # a second pass settles layouts of axes with a fixed aspect
    width, height = fig.canvas.get_width_height()
    problems = []
    boxes = _text_boxes(fig)
    for text, box in boxes:
        if box.x0 < -2 or box.y0 < -2 or box.x1 > width + 2 or box.y1 > height + 2:
            problems.append(
                f"{name}: text outside the figure: {text.get_text()[:40]!r}"
            )
    for (a, box_a), (b, box_b) in itertools.combinations(boxes, 2):
        overlap_w = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
        overlap_h = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
        if overlap_w > 1.5 and overlap_h > 1.5:
            problems.append(
                f"{name}: texts overlap: {a.get_text()[:30]!r} / {b.get_text()[:30]!r}"
            )
    return problems


def save(fig, name: str) -> Path:
    """Save a figure as PNG in ``docs/guide/images`` and check it.

    Checks: no text outside the canvas or on other text (reported, not fatal), size
    below 300 kB (the image is reduced to a 256-colour palette when needed), white
    background.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    problems = check_layout(fig, name)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)

    from PIL import Image

    if path.stat().st_size > MAX_BYTES:
        with Image.open(path) as image:
            image.convert("RGB").quantize(colors=256).save(path, optimize=True)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        corner = rgb.getpixel((2, 2))
        size_px = rgb.size
    size = path.stat().st_size
    status = "ok" if size <= MAX_BYTES and corner == (255, 255, 255) else "CHECK"
    LOG.info(
        "[%s] %s: %dx%d px, %.0f kB, corner %s",
        status,
        path.relative_to(ROOT).as_posix(),
        size_px[0],
        size_px[1],
        size / 1000,
        corner,
    )
    for problem in problems:
        LOG.info("   layout: %s", problem)
    if size > MAX_BYTES:
        raise SystemExit(f"{name} is {size} bytes, over the {MAX_BYTES} limit")
    return path


def read_stored_case(name: str) -> dict:
    """Read one stored benchmark case."""
    return json.loads((BENCHMARK_RESULTS / name).read_text(encoding="utf8"))


def snapshot(name: str, build) -> dict:
    """Stored-result data for a figure.

    The data come from the benchmark folder if it is present, else from the committed
    snapshot.

    ``build()`` reads the benchmark folder and returns a small JSON-serialisable dict.
    When the folder exists the snapshot file ``data/<name>.json`` is refreshed from it
    (only when the content changed), so a rerun on a machine without the folder
    redraws the same figure.
    """
    path = DATA / f"{name}.json"
    where = path.relative_to(ROOT).as_posix()
    if BENCHMARK_RESULTS.exists():
        data = build()
        stored = json.loads(path.read_text(encoding="utf8")) if path.exists() else None
        if stored == json.loads(json.dumps(data)):
            LOG.info("snapshot is current: %s", where)
        else:
            DATA.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf8"
            )
            LOG.info("snapshot refreshed from the benchmark folder: %s", where)
        return data
    LOG.info("benchmark folder not found, using the committed snapshot %s", where)
    return json.loads(path.read_text(encoding="utf8"))
