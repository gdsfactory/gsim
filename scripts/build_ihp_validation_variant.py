#!/usr/bin/env python
"""Build the live IHP-validation documentation variant."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
import skrf as rf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gsim.common.benchmarks import ihp_l2n0


def _measured(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frequency_ghz, L_nH, Q) for one measured die."""
    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    z0 = float(np.real(np.asarray(network.z0).reshape(-1)[0]))
    z = 2.0 * z0 * (1.0 - network.s[:, 1, 0]) / network.s[:, 1, 0]
    l_nh = z.imag / (2 * np.pi * network.f) * 1e9
    q = np.divide(z.imag, z.real, out=np.full_like(z.real, np.nan), where=z.real != 0)
    return network.f / 1e9, l_nh, q


def render_comparison(out: Path) -> dict[str, float]:
    """Render L/Q comparison and return metrics at 1 and 30 GHz."""
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_best")
    f_palace = np.asarray(run.sparams.freq, dtype=float)
    l_palace, q_palace = [], []
    for i in range(len(f_palace)):
        l, q, _ = ihp_l2n0.lq_at(run.sparams, i)
        l_palace.append(l)
        q_palace.append(q)
    l_palace = np.asarray(l_palace)
    q_palace = np.asarray(q_palace)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    measured = {}
    for label, color in (("die_x2y7", "C0"), ("die_x5y8", "C1")):
        f, l, q = _measured(label)
        measured[label] = (f, l, q)
        axes[0].plot(f, l, color=color, lw=1.0, label=f"measured {label}")
        axes[1].plot(f, q, color=color, lw=1.0, label=f"measured {label}")
    axes[0].plot(f_palace, l_palace, "o-", color="C3", label="Palace")
    axes[1].plot(f_palace, q_palace, "o-", color="C3", label="Palace")
    for ax in axes:
        ax.axvline(30, color="k", lw=0.8, ls=":")
        ax.grid(alpha=0.3)
        ax.set(xlabel="frequency (GHz)")
        ax.legend(fontsize=8)
    axes[0].set(title="Extracted series inductance", ylabel="L (nH)")
    axes[1].set(title="Extracted quality factor", ylabel="Q")
    fig.tight_layout()
    fig.savefig(
        out / "l2n0_comparison.png", dpi=160, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    def at(f: np.ndarray, values: np.ndarray, target: float) -> float:
        return float(values[int(np.argmin(np.abs(f - target)))])

    rows = {
        "palace_l_1ghz_nh": at(f_palace, l_palace, 1.0),
        "palace_q_1ghz": at(f_palace, q_palace, 1.0),
        "palace_l_30ghz_nh": at(f_palace, l_palace, 30.0),
        "palace_q_30ghz": at(f_palace, q_palace, 30.0),
    }
    for label, (f, l, q) in measured.items():
        rows[f"{label}_l_1ghz_nh"] = at(f, l, 1.0)
        rows[f"{label}_q_1ghz"] = at(f, q, 1.0)
        rows[f"{label}_l_30ghz_nh"] = at(f, l, 30.0)
        rows[f"{label}_q_30ghz"] = at(f, q, 30.0)
    exact = (
        REPO
        / "nbs"
        / "data"
        / "ihp_validation"
        / "experiments"
        / "verification"
        / "verification.json"
    )
    if exact.is_file():
        point = json.loads(exact.read_text())["best"]["30"]
        rows["palace_l_30ghz_nh"] = float(point["L_nH"])
        rows["palace_q_30ghz"] = float(point["Q"])
    return rows


def build(
    out: Path, *, notebook_html: Path | None = None, notebook_ipynb: Path | None = None
) -> None:
    """Assemble the variant directory."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = render_comparison(out)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    experiments = REPO / "nbs" / "data" / "ihp_validation" / "experiments"
    matrix = json.loads((experiments / "results.json").read_text())
    verify = json.loads(
        (experiments / "verification" / "verification.json").read_text()
    )
    campaign = "".join(
        f"<tr><td>{name}</td><td>{matrix[name]['L_nH']:.3f}</td>"
        f"<td>{matrix[name]['Q']:.2f}</td><td>{matrix[name]['L_diff_nH']:.3f}</td></tr>"
        for name in (
            "baseline",
            "shell_conformal",
            "ground0_pec",
            "openems_ground",
            "gap",
        )
    )

    def point(name: str, frequency: str) -> tuple[float, float]:
        if frequency == "1":
            row = matrix["ground0_pec" if name == "best" else name]
        else:
            row = verify[name][frequency]
        return float(row["L_nH"]), float(row["Q"])

    verification = "".join(
        f"<tr><td>{name}</td>"
        + "".join(
            f"<td>{point(name, f)[0]:.3f}</td><td>{point(name, f)[1]:.2f}</td>"
            for f in ("1", "5", "9", "30")
        )
        + "</tr>"
        for name in ("baseline", "best")
    )
    for name in ("L_2n0_simplified.gds", "README.md", "provenance.json"):
        source = ihp_l2n0.SOURCE / name
        if source.is_file():
            shutil.copy2(source, out / name)
    notebook_section = ""
    if notebook_html is not None and notebook_html.is_file():
        shutil.copy2(notebook_html, out / "notebook.html")
        links = '<a href="notebook.html">rendered notebook</a>'
        if notebook_ipynb is not None and notebook_ipynb.is_file():
            shutil.copy2(notebook_ipynb, out / "notebook.ipynb")
            links += ' · <a href="notebook.ipynb">executed .ipynb</a>'
        notebook_section = f"<p><strong>Notebook:</strong> {links}</p>"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IHP SG13G2 L2n0 — inductance extraction</title>
<style>
:root {{ color-scheme:light dark; --fg:#1f2328; --muted:#656d76; --bg:#fff; --card:#f6f8fa; --line:#d0d7de; --accent:#0969da; }}
@media (prefers-color-scheme:dark) {{ :root {{ --fg:#e6edf3; --muted:#8b949e; --bg:#0d1117; --card:#161b22; --line:#30363d; --accent:#4493f8; }} }}
body {{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }}
main {{ max-width:1050px; margin:0 auto; padding:36px 24px 56px; }}
h1 {{ margin:0 0 6px; font-size:1.8rem; }} .muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; margin:20px 0; }}
img {{ width:100%; border-radius:10px; background:#fff; }}
table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
code {{ font-size:.9em; }}
</style></head><body><main>
<p><a href="../">&larr; all variants</a></p>
<h1>IHP SG13G2 Octagon L2n0 — inductance extraction</h1>
<p class="muted">Palace/gsim versus the two pinned IHP measured dies using the existing
series extraction Z = 2 Z0 (1-S21)/S21. Nominal 2 nH benchmark. Layout and
measured data pinned from IHP-Open-PDK commit <code>9439837</code> (Apache-2.0).</p>
<div class="card"><img src="l2n0_comparison.png" alt="Measured and Palace extracted inductance and Q"></div>
<div class="card"><h2>Extraction summary</h2>
<table><thead><tr><th>case</th><th>L @ 1 GHz (nH)</th><th>Q @ 1 GHz</th><th>L @ 30 GHz (nH)</th><th>Q @ 30 GHz</th></tr></thead><tbody>
<tr><td>Palace</td><td>{rows["palace_l_1ghz_nh"]:.3f}</td><td>{rows["palace_q_1ghz"]:.2f}</td><td>{rows["palace_l_30ghz_nh"]:.3f}</td><td>{rows["palace_q_30ghz"]:.2f}</td></tr>
<tr><td>die_x2y7</td><td>{rows["die_x2y7_l_1ghz_nh"]:.3f}</td><td>{rows["die_x2y7_q_1ghz"]:.2f}</td><td>{rows["die_x2y7_l_30ghz_nh"]:.3f}</td><td>{rows["die_x2y7_q_30ghz"]:.2f}</td></tr>
<tr><td>die_x5y8</td><td>{rows["die_x5y8_l_1ghz_nh"]:.3f}</td><td>{rows["die_x5y8_q_1ghz"]:.2f}</td><td>{rows["die_x5y8_l_30ghz_nh"]:.3f}</td><td>{rows["die_x5y8_q_30ghz"]:.2f}</td></tr>
</tbody></table>
<p class="muted">The measured 1 GHz inductance is 1.91-1.93 nH, confirming the nominal
2 nH benchmark. The IHP guide differential 1-port SRF is ~25.5 GHz; the 2-port series extraction
has a different L zero-crossing (~35.5 GHz). The 30 GHz Palace/measured
disagreement is reported honestly.</p></div>
{notebook_section}
<div class="card"><h2>Campaign matrix (1 GHz)</h2>
<table><thead><tr><th>variant</th><th>L nH</th><th>Q</th><th>L diff nH</th></tr></thead><tbody>{campaign}</tbody></table>
<p class="muted">Best series result: ground0_pec 1.921 nH vs measured 1.911/1.932 nH.
The differential comparison is negative: ground0_pec 0.802 nH and gap 0.780 nH
vs measured differential 1.674/1.672 nH. The 1.921 nH headline is therefore a
series-extraction result, not a full validation. `shell_conformal` adds a 0.6 µm
SiN sidewall shell around TopMetal2 and gives 1.6277 nH.</p>
<table><thead><tr><th>model</th><th>1 GHz L</th><th>1 GHz Q</th><th>5 GHz L</th><th>5 GHz Q</th><th>9 GHz L</th><th>9 GHz Q</th><th>30 GHz L</th><th>30 GHz Q</th></tr></thead><tbody>{verification}</tbody></table>
<p class="muted">30 GHz remains far below measured ~3.61/3.71 nH; Q is also low
(0.21 best at 1 GHz vs measured ~6.6). The mesh error Norm is ~0.18 against
AdaptiveTol 0.02, so the model is not converged.</p></div>
<div class="card"><h2>Model and source</h2>
<p>Palace sweep: 1-50 GHz, 11 points; authoritative IHP SG13G2 stack with a
280 µm substrate, TopMetal2 winding, TopMetal1 feeds/underpass, and a minimal
ground pad with PEC boundaries for the best variant. The 5 µm refined / 70 µm
maximum mesh and Palace Conductivity+Thickness surface-impedance BCs
(sigma=3.77e7 S/m; delta ~2.6 um at 1 GHz) model finite conductors. OpenEMS was not
available; pinned measured S-parameters are the approved fallback.</p>
<p>The IHP OpenEMS reference uses SG13_L2n0_2port_groundpolygon_mesh1.5um.py
with TopMetal1 edge ports, a central Metal1->TopMetal1 PEC pillar, Metal1 ground
box, 1.5 µm refined cells, lambda_air/(sqrt(eps_max)*20) maximum cells, PEC
boundaries and 0-30 GHz/401 points. The `openems_ground` variant approximates
this with a Metal1 rectangle and vertical via ports; it omits the central pillar.</p>
<p><a href="L_2n0_simplified.gds">benchmark GDS</a> ·
<a href="metrics.json">metrics</a> · <a href="provenance.json">provenance</a></p></div>
</main></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf8")
    (out / "README.md").write_text(
        "# IHP SG13G2 L2n0 inductance extraction\n\nPalace vs measured L/Q extraction. See notebook.html and metrics.json.\n"
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notebook-html", type=Path)
    parser.add_argument("--notebook-ipynb", type=Path)
    args = parser.parse_args()
    build(
        args.out, notebook_html=args.notebook_html, notebook_ipynb=args.notebook_ipynb
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
