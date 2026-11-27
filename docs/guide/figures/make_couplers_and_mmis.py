"""Figures for the "Directional couplers and MMIs" section of docs/guide/index.md.

Writes docs/guide/images/couplers-setup.png (static FDTD set-up plots, drawn locally,
no cloud call) and docs/guide/images/couplers-cross-power.png (stored cloud FDTD runs
of the 236 nm gap coupler against a femwell supermode model, read from the benchmark
results folder or from the committed snapshot).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
from _common import (
    BENCHMARK_RESULTS,
    LOG,
    apply_style,
    read_stored_case,
    save,
    snapshot,
)
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator

from gsim import fdtd

COUPLER_GAP_UM = 0.236
COUPLER_LENGTH_UM = 20.0  # set-up figure only; the result figure sweeps the length


def build_simulation(component):
    """Set-up of the page's snippet: oxide background, silicon index, source on o1."""
    sim = fdtd.Simulation()
    sim.materials(background="SiO2", overrides={"si": 3.4757})
    sim.geometry(component)
    sim.source(
        port="o1", wavelength_um=1.55, wavelength_span_um=0.1, num_wavelengths=101
    )
    sim.domain(padding=1.0)
    sim.solver(cell_size_nm=60)
    return sim


def setup_figure() -> None:
    """Draw the FDTD set-up of the page's MMI and of a directional coupler."""
    gf.gpdk.PDK.activate()
    coupler_title = (
        f"Directional coupler, gap {COUPLER_GAP_UM * 1000:.0f} nm, "
        f"L = {COUPLER_LENGTH_UM:.0f} µm"
    )
    devices = (
        ("1x2 MMI (the snippet's device)", gf.get_component("mmi1x2")),
        (
            coupler_title,
            gf.components.coupler(gap=COUPLER_GAP_UM, length=COUPLER_LENGTH_UM),
        ),
    )
    # "compressed" is the constrained layout for axes with a fixed aspect ratio.
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), layout="compressed")
    handles: dict[str, Artist] = {}
    for ax, (title, component) in zip(axes, devices, strict=True):
        sim = build_simulation(component)
        sim.plot_2d_static(axis="z", ax=ax, legend=True)
        # gsim puts the legend under each axes, where it meets the x label: collect
        # them into one figure legend.
        legend = ax.get_legend()
        if legend is not None:
            for handle, text in zip(
                legend.legend_handles, legend.get_texts(), strict=True
            ):
                handles.setdefault(text.get_text(), handle)
            legend.remove()
        ax.grid(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_label_coords(
            -0.06, 0.5
        )  # the layout engine misplaces it on fixed-aspect axes
        ax.set_title(f"{title}: source on o1")
    # The stubs of Matplotlib do not list the "outside ..." legend locations.
    legend_options: dict[str, Any] = {
        "loc": "outside lower center",
        "ncols": 4,
        "framealpha": 1,
    }
    fig.legend(list(handles.values()), list(handles), **legend_options)
    save(fig, "couplers-setup.png")


def build_coupler_snapshot() -> dict:
    """Read the stored cloud FDTD runs and the femwell supermode model."""
    at_1550 = read_stored_case("coupler.cmt_vs_fdtd_1550.cross-solver.json")
    band = read_stored_case("coupler.cmt_vs_fdtd_band.cross-solver.json")
    rows = at_1550["details"]["table_1550"]
    cell_sizes, converged = set(), []
    for run in at_1550["details"]["runs"]:
        sparams = next(
            (BENCHMARK_RESULTS / "cloud" / "coupler_fdtd" / run["tag"]).glob(
                "sim-data-*/sparams.json"
            )
        )
        data = json.loads(sparams.read_text(encoding="utf8"))
        cell_sizes.add(tuple(data["grid"]["cell_size_nm"]))
        converged.append(bool(data["convergence"]["converged"]))
    return {
        "case_1550": at_1550["case_id"],
        "case_band": band["case_id"],
        "date": at_1550["date"],
        "model": at_1550["details"]["model"],
        "length_um": [r["length_um"] for r in rows],
        "fdtd_cross_1550": [r["fdtd"] for r in rows],
        "model_cross_1550": [r["model"] for r in rows],
        "fdtd_through_1550": [r["through"] for r in rows],
        "band_nm": [
            min(band["details"]["wavelengths_compared_nm"]),
            max(band["details"]["wavelengths_compared_nm"]),
        ],
        "worst_band_deviation": {
            str(k): v for k, v in band["details"]["worst_deviation_per_length"].items()
        },
        "cell_size_nm": sorted(cell_sizes),
        "all_runs_converged": all(converged),
        "n_runs": len(rows),
    }


def result_figure() -> None:
    """Draw cross power against coupling length: FDTD against the supermode model."""
    data = snapshot("couplers_cross_power", build_coupler_snapshot)
    length = np.asarray(data["length_um"], dtype=float)
    fdtd_cross = np.asarray(data["fdtd_cross_1550"])
    model_cross = np.asarray(data["model_cross_1550"])
    worst_band = np.array([data["worst_band_deviation"][f"{int(v)}"] for v in length])
    LOG.info(
        "cell size nm (all runs): %s | all runs converged: %s",
        data["cell_size_nm"],
        data["all_runs_converged"],
    )
    LOG.info(
        "largest |FDTD - model| at 1550 nm: %s",
        float(np.abs(fdtd_cross - model_cross).max()),
    )
    LOG.info(
        "largest |FDTD - model| over the band: %s | band nm: %s",
        float(worst_band.max()),
        data["band_nm"],
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    left.plot(
        length, model_cross, "-", color="tab:blue", label="femwell supermode model"
    )
    left.plot(
        length,
        fdtd_cross,
        "o",
        color="tab:red",
        markersize=4.5,
        label="FDTD (stored cloud runs)",
    )
    left.set_xlabel("Straight coupling length L (µm)")
    left.set_ylabel("Cross-port power at 1550 nm")
    left.set_ylim(0, 1.0)
    left.set_title("Cross power against length")
    left.legend(loc="upper left", framealpha=1)

    right.plot(
        length,
        np.abs(fdtd_cross - model_cross),
        "o-",
        color="tab:red",
        markersize=4.5,
        label="at 1550 nm",
    )
    right.plot(
        length,
        worst_band,
        "s--",
        color="tab:gray",
        markersize=4,
        label="largest over 1510 to 1600 nm",
    )
    right.set_xlabel("Straight coupling length L (µm)")
    # \u2212 is the minus sign, which cp1252 cannot encode (check-cp1252 hook).
    right.set_ylabel("|FDTD \u2212 model| (cross power)")
    right.set_ylim(0, None)
    right.set_title("Difference to the model")
    right.legend(loc="upper left", framealpha=1)
    save(fig, "couplers-cross-power.png")


def main() -> None:
    """Draw both figures of the page."""
    apply_style()
    setup_figure()
    result_figure()


if __name__ == "__main__":
    main()
