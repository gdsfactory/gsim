#!/usr/bin/env python
"""Build the ``pdk-validation`` gh-pages variant.

Collects the GDS layout for each validation case, previews it, copies the
recorded Meep runs and reference comparisons, and (optionally) adds an executed
notebook HTML/IPYNB.  The result is a self-contained directory suitable for
``_tools/publish_variant.sh``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

import gdstk
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.pdk_validation import _validation as V  # noqa: N812

RUNS = REPO / "nbs" / "data" / "pdk_validation"
GDS_STORE = RUNS / "gds"
MEEP_FALLBACK = Path("/tmp/opencode/pdk_meep_results")  # noqa: S108
DEFAULT_OUT = Path("/tmp/opencode/pdk-validation-variant")  # noqa: S108

CASES = [
    ("straight", "Straight waveguide (UBC)", "L = 20 µm"),
    ("coupler", "Directional coupler (UBC)", "gap = 0.2 µm, L = 10 µm"),
    ("ring_coupling", "Ring coupling area (UBC)", "gap = 0.1 µm, r = 10 µm"),
    ("mzi", "MZI (UBC)", "dL = 214.215 µm"),
    ("ring", "Ring resonator (UBC)", "r = 10 µm, gap = 0.1 µm"),
    ("bragg", "Bragg grating (UBC)", "200 sinusoidal periods"),
    ("cspdk_coupler", "Directional coupler (Cornerstone)", "gap = 0.2 µm, L = 10 µm"),
    (
        "cspdk_ring_coupling",
        "Ring coupling area (Cornerstone)",
        "gap = 0.1 µm, r = 10 µm",
    ),
    (
        "siepic_coupler",
        "SiEPIC measured coupler (generic PDK)",
        "gap = 0.1 µm, r = 10 µm",
    ),
]

PLOTS: dict[str, str] = {}

STATUS = {
    "straight": ("Recorded", "authoritative UBC stack"),
    "coupler": ("Recorded", "authoritative UBC stack"),
    "ring_coupling": ("Recorded", "authoritative UBC stack; 2-D"),
    "mzi": ("Recorded", "authoritative UBC stack; 2-D"),
    "ring": ("Recorded", "authoritative UBC stack; 2-D"),
    "bragg": ("Recorded", "authoritative UBC stack"),
    "cspdk_coupler": ("Recorded", "authoritative Cornerstone stack"),
    "cspdk_ring_coupling": ("Recorded", "authoritative Cornerstone stack; 2-D"),
    "siepic_coupler": ("Recorded", "generic photonic PDK; measured data"),
}


def resolve_gds(name: str) -> Path | None:
    """Return the GDS for a case, importing it into the run store if needed."""
    stored = GDS_STORE / f"{name}.gds"
    if stored.is_file():
        return stored
    run_root = RUNS / "runs"
    candidates = sorted(run_root.glob(f"*_{name}/layout.gds"))
    candidates += sorted(run_root.glob(f"{name}/layout.gds"))
    fallback = MEEP_FALLBACK / name / "layout.gds"
    if fallback.is_file():
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.is_file():
            GDS_STORE.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, stored)
            return stored
    return None


def render_gds(gds_path: Path, out_png: Path, title: str) -> tuple[float, float]:
    """Render a GDS layout to PNG; return its drawn extent in micrometres."""
    library = gdstk.read_gds(str(gds_path))
    top = library.top_level()
    polygons = top[0].get_polygons() if top else []  # ty: ignore[unresolved-attribute]
    xs = [point[0] for polygon in polygons for point in polygon.points]
    ys = [point[1] for polygon in polygons for point in polygon.points]
    fig, axis = plt.subplots(figsize=(9, 4))
    layers = sorted({polygon.layer for polygon in polygons})
    cmap = plt.get_cmap("tab20")
    for index, layer in enumerate(layers):
        patches = [
            Polygon(polygon.points, closed=True)
            for polygon in polygons
            if polygon.layer == layer
        ]
        if patches:
            axis.add_collection(
                PatchCollection(
                    patches,
                    facecolor=cmap(index % 20),
                    edgecolor="none",
                    alpha=0.9,
                )
            )
    if xs and ys:
        pad = 0.03 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        axis.set_xlim(min(xs) - pad, max(xs) + pad)
        axis.set_ylim(min(ys) - pad, max(ys) + pad)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return (
        (max(xs) - min(xs)) if xs else 0.0,
        (max(ys) - min(ys)) if ys else 0.0,
    )


def render_data_plot(name: str, title: str, out_png: Path) -> str:
    """Plot the reference (and a recorded run, when present) for one case.

    Returns ``"comparison"`` when a recorded run was overlaid, otherwise
    ``"reference"`` for a reference/measured-only plot.
    """
    wavelengths = np.linspace(1.50, 1.60, 401)
    case = V.get_case(name)
    reference = case.reference(wavelengths)
    recorded = V.find_recorded_run(
        name, RUNS / "runs", dimension=case.dimension, fallback_to_2d=True
    )
    outcome = V.load_outcome(case, recorded) if recorded is not None else None

    figure, axis = plt.subplots(figsize=(8, 3.5))
    if outcome is not None:
        measured = outcome.reference.transmission_db is not None
        keys = (
            outcome.reference.sim_ports
            or tuple(outcome.reference.s_params)
            or tuple(outcome.s_params)
        )
        for s_name in keys:
            if s_name not in outcome.s_params:
                continue
            values = outcome.s_params[s_name]
            curve = V.to_db(values)
            if measured:
                curve = curve - np.nanmax(curve)
            axis.plot(
                outcome.wavelengths,
                curve,
                "-",
                label=f"Meep {outcome.dimension.upper()} {s_name}",
            )
    if reference.s_params:
        for s_name, values in sorted(reference.s_params.items()):
            axis.plot(
                reference.wavelengths,
                V.to_db(values),
                "--",
                label=f"reference {s_name}",
            )
    elif reference.transmission_db is not None:
        ref_wl, ref_db = reference.transmission_db
        axis.plot(
            ref_wl,
            V.normalize_measured_spectrum(ref_db),
            "k",
            lw=0.8,
            label="measured",
        )
    mode = "Meep vs reference" if outcome is not None else "reference only"
    axis.set(title=f"{title} — {mode}", xlabel="wavelength (µm)", ylabel="|S| (dB)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return "comparison" if outcome is not None else "reference"


def build(
    out: Path,
    *,
    notebook_html: Path | None = None,
    notebook_ipynb: Path | None = None,
) -> None:
    """Assemble the variant directory."""
    with contextlib.suppress(Exception):
        import ubcpdk

        ubcpdk.PDK.activate()

    if out.exists():
        shutil.rmtree(out)
    (out / "gds").mkdir(parents=True)
    (out / "plots").mkdir()
    (out / "runs").mkdir()

    for plot_name in PLOTS.values():
        source = RUNS / plot_name
        if source.is_file():
            shutil.copy2(source, out / "plots" / source.name)
    summary_source = RUNS / "runs" / "summary.json"
    if summary_source.is_file():
        shutil.copy2(summary_source, out / "runs" / "summary.json")
        summary = json.loads(summary_source.read_text(encoding="utf8"))
        metrics = {
            name: {
                "best_port": record.get("best_port"),
                "metrics": record.get("metrics"),
                "summary": record.get("summary"),
            }
            for name, record in summary.items()
            if isinstance(record, dict)
        }
        (out / "runs" / "results_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf8"
        )
    runs_dir = RUNS / "runs"
    if runs_dir.is_dir():
        destination = out / "runs" / "runs"
        destination.mkdir()
        for run in sorted(runs_dir.iterdir()):
            if (run / "s_parameters.csv").is_file():
                shutil.copytree(run, destination / run.name)
            elif run.is_file() and run.name == "README.md":
                shutil.copy2(run, destination / run.name)

    notebook_section = ""
    if notebook_html is not None and notebook_html.is_file():
        shutil.copy2(notebook_html, out / "notebook.html")
        notebook_links = '<a href="notebook.html">rendered HTML</a>'
        if notebook_ipynb is not None and notebook_ipynb.is_file():
            shutil.copy2(notebook_ipynb, out / "notebook.ipynb")
            notebook_links += ' · <a href="notebook.ipynb">executed .ipynb</a>'
        notebook_section = f"""
        <div class="notice ok">
          <strong>Executed notebook is published.</strong> Every available validation case is shown with
          their recorded runs, reference overlays and the compute/size table.
          {notebook_links}
        </div>"""
    else:
        notebook_section = """
        <div class="notice">
          <strong>Executed notebook render pending.</strong> The layouts and recorded
          runs below are the data that exist now.
        </div>"""

    cards = []
    for name, title, subtitle in CASES:
        gds_source = resolve_gds(name)
        if gds_source is None:
            continue
        gds_out = out / "gds" / f"{name}.gds"
        shutil.copy2(gds_source, gds_out)
        png_out = out / "gds" / f"{name}.png"
        width, height = render_gds(gds_source, png_out, title)

        status, detail = STATUS[name]
        run_links = ""
        for prefix in ("3d_", "2d_"):
            run = f"{prefix}{name}"
            if (out / "runs" / "runs" / run / "s_parameters.csv").is_file():
                run_links += (
                    f'<a class="chip" href="runs/runs/{run}/s_parameters.csv">{run}</a>'
                )
        run_links = run_links or '<span class="chip muted">no recorded run</span>'
        plot = PLOTS.get(name)
        if plot and (out / "plots" / plot).is_file():
            plot_html = (
                f'<a href="plots/{plot}"><img class="plot" src="plots/{plot}" '
                f'alt="{title} comparison"></a>'
            )
        else:
            generated = out / "plots" / f"{name}_data.png"
            mode = render_data_plot(name, title, generated)
            caption = (
                "Meep vs reference"
                if mode == "comparison"
                else "reference data only — no simulated run recorded"
            )
            plot_html = (
                f'<a href="plots/{generated.name}"><img class="plot" '
                f'src="plots/{generated.name}" alt="{title} {caption}"></a>'
                f'<p class="muted small">{caption}</p>'
            )
        cards.append(
            f"""
            <article class="card">
              <header class="card-head">
                <div>
                  <h2>{title}</h2>
                  <p class="muted small">{subtitle} · {width:.2f} x {height:.2f} µm</p>
                </div>
                <span class="badge ok">{status}</span>
              </header>
              <a href="gds/{name}.gds"><img class="layout" src="gds/{name}.png" alt="{title} layout"></a>
              <p class="small"><a href="gds/{name}.gds">download GDS</a> · {detail}</p>
              {plot_html}
              <p class="small">runs: {run_links}</p>
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gsim — PDK validation notebook &amp; layouts</title>
<style>
  :root {{ color-scheme: light dark; --fg:#1f2328; --muted:#656d76; --bg:#fff; --card:#f6f8fa; --line:#d0d7de; --accent:#0969da; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6edf3; --muted:#8b949e; --bg:#0d1117; --card:#161b22; --line:#30363d; --accent:#4493f8; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }}
  header.page, main, footer {{ max-width:1100px; margin:0 auto; padding:0 24px; }}
  header.page {{ padding-top:40px; }}
  h1 {{ font-size:1.7rem; margin:0 0 4px; letter-spacing:-.01em; }}
  .small {{ font-size:.85rem; }}
  .muted {{ color:var(--muted); }}
  a {{ color:var(--accent); }}
  main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:20px; padding-top:20px; padding-bottom:48px; }}
  .card {{ border:1px solid var(--line); border-radius:14px; padding:16px; background:var(--card); }}
  .card-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
  .card h2 {{ font-size:1.05rem; margin:0 0 2px; }}
  .badge {{ font-size:.75rem; padding:3px 9px; border-radius:999px; border:1px solid var(--line); white-space:nowrap; }}
  .badge.ok {{ color:#1a7f37; border-color:#1a7f37; }}
  .layout, .plot {{ width:100%; border-radius:8px; background:#fff; border:1px solid var(--line); margin:8px 0; }}
  .chip {{ display:inline-block; font-size:.75rem; padding:2px 8px; border:1px solid var(--line); border-radius:999px; margin:2px 4px 2px 0; text-decoration:none; }}
  .notice {{ border-left:4px solid #9a6700; background:color-mix(in srgb, #9a6700 10%, transparent); padding:12px 16px; border-radius:8px; margin-top:14px; }}
  .notice.ok {{ border-left-color:#1a7f37; background:color-mix(in srgb, #1a7f37 10%, transparent); }}
  code {{ font-size:.85em; }}
</style>
</head>
<body>
<header class="page">
  <p><a href="../">&larr; all variants</a></p>
  <h1>PDK validation — notebook &amp; layouts</h1>
  <p class="muted">Executed notebook plus GDS geometry for each validation case, from branch <code>pdk-validation</code>.</p>
  {notebook_section}
</header>
<main>
{"".join(cards)}
</main>
<footer class="page small muted" style="max-width:1100px;margin:0 auto;padding:0 24px 40px">
  Raw S-parameters under <a href="runs/runs/">runs/</a>; metrics in
  <a href="runs/results_metrics.json">results_metrics.json</a> and
  <a href="runs/summary.json">summary.json</a>.
</footer>
</body>
</html>
"""
    (out / "index.html").write_text(html, encoding="utf8")
    (out / "README.md").write_text(
        "# PDK validation notebook & layouts\n\n"
        "Executed validation notebook, GDS layouts for all validation cases, recorded Meep\n"
        "runs and reference comparisons. Rebuild with\n"
        "`scripts/build_pdk_validation_variant.py`.\n",
        encoding="utf8",
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--notebook-html", type=Path)
    parser.add_argument("--notebook-ipynb", type=Path)
    arguments = parser.parse_args()
    build(
        arguments.out,
        notebook_html=arguments.notebook_html,
        notebook_ipynb=arguments.notebook_ipynb,
    )
    print(f"wrote {arguments.out}")
    print("files:", sum(1 for path in arguments.out.rglob("*") if path.is_file()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
