"""``nbs/data/conductors``: the skin-depth job builder and the loss analysis.

No solver and no cloud: the jobs are meshed and written (``write_config()``), and the
analysis reads synthetic ``port-S.csv`` files whose absorbed power is known.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0, pi

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "nbs" / "data" / "conductors"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def make():
    return _load("make_skin_depth_jobs")


@pytest.fixture(scope="module")
def analyse():
    return _load("analyse_skin_depth")


# ---------------------------------------------------------------------------
# The plan: frequencies, names
# ---------------------------------------------------------------------------


def test_frequency_sets_t_over_delta(make):
    """f is chosen so that t / delta is the requested number (independent formula)."""
    for ratio in make.RATIOS:
        f = make.frequency_for_ratio(ratio)
        delta_um = 1e6 * math.sqrt(2.0 / (2 * pi * f * mu_0 * make.SIGMA_S_PER_M))
        assert make.THICKNESS_UM / delta_um == pytest.approx(ratio, rel=2e-4)


def test_ratios_are_the_planned_five(make):
    assert make.RATIOS == (0.5, 1.0, 2.4, 5.0, 10.0)


def test_plan_has_ten_required_jobs_and_unique_names(make):
    jobs = make.planned_jobs()
    names = [job["name"] for job in jobs]
    assert len(names) == len(set(names))
    required = [job for job in jobs if job["required"]]
    assert len(required) == 10
    assert {job["model"] for job in required} == {"surface", "volume"}
    assert "surface_r2p4" in names
    assert "volume_r10" in names
    assert all(job["name"].startswith(job["model"] + "_r") for job in jobs)


def test_interior_size_is_below_delta_and_resolves_the_thickness(make):
    for job in make.planned_jobs(("volume",)):
        size = make.interior_size_um(job["skin_depth_um"])
        assert size <= job["skin_depth_um"]
        assert size <= make.THICKNESS_UM / make.MIN_CELLS_ACROSS


def test_sigma_is_gsims_aluminum(make):
    from gsim.common.stack.materials import MATERIALS_DB

    assert MATERIALS_DB["aluminum"].conductivity == make.SIGMA_S_PER_M


# ---------------------------------------------------------------------------
# Building two cheap jobs (mesh and write_config only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(make, tmp_path_factory):
    jobs_dir = tmp_path_factory.mktemp("jobs")
    by_name = {job["name"]: job for job in make.planned_jobs()}
    entries = {
        name: make.build_job(by_name[name], jobs_dir=jobs_dir, length_um=20.0)
        for name in ("surface_r1", "volume_r1", "pec_r1")
    }
    return jobs_dir, entries


def _config(jobs_dir: Path, name: str) -> dict:
    return json.loads((jobs_dir / name / "config.json").read_text())


def test_surface_job_writes_the_shell_conditions(built, make):
    jobs_dir, entries = built
    entry = entries["surface_r1"]
    config = _config(jobs_dir, "surface_r1")
    boundaries = config["Boundaries"]
    assert len(boundaries["Conductivity"]) == 2
    for item in boundaries["Conductivity"]:
        assert item["Conductivity"] == make.SIGMA_S_PER_M
        assert item["Thickness"] == pytest.approx(make.THICKNESS_UM)
        assert item["Crack"] is True
    assert entry["validate_config_valid"] is True
    assert entry["tetrahedra"] > 0
    assert entry["estimated_field_dofs"] > entry["tetrahedra"]


def test_volume_job_meshes_the_metal_as_a_conductive_volume(built, make):
    jobs_dir, entries = built
    entry = entries["volume_r1"]
    config = _config(jobs_dir, "volume_r1")
    assert config["Boundaries"]["Conductivity"] == []
    assert "PEC" not in config["Boundaries"]
    materials = [m for m in config["Domains"]["Materials"] if "Conductivity" in m]
    assert len(materials) == 1
    assert materials[0]["Conductivity"] == make.SIGMA_S_PER_M
    assert "Thickness" not in materials[0]
    assert entry["metal_tetrahedra"] > 0
    assert entry["metal_mean_edge_over_delta"] <= make.EDGE_OVER_DELTA_LIMIT
    # the metal is resolved by the box, not left to the conductor-edge sizing
    assert entry["tetrahedra"] > entries["surface_r1"]["tetrahedra"]


def test_pec_job_has_sheets_and_no_conductivity(built):
    jobs_dir, _entries = built
    boundaries = _config(jobs_dir, "pec_r1")["Boundaries"]
    assert boundaries["Conductivity"] == []
    assert "PEC" in boundaries


def test_one_frequency_one_solve_one_excited_port(built, make):
    jobs_dir, _entries = built
    config = _config(jobs_dir, "volume_r1")
    driven = config["Solver"]["Driven"]
    expected = make.frequency_for_ratio(1.0) / 1e9
    assert driven["Samples"] == [{"Type": "Point", "Freq": [expected], "SaveStep": 0}]
    assert driven["AdaptiveTol"] == 0
    excited = [p["Excitation"] for p in config["Boundaries"]["LumpedPort"]]
    assert excited == [1, False]
    assert config["Solver"]["Linear"]["Tol"] == make.SOLVER_TOLERANCE


def test_each_job_folder_is_ready_to_upload(built):
    jobs_dir, _entries = built
    for name in ("surface_r1", "volume_r1", "pec_r1"):
        files = {p.name for p in (jobs_dir / name).iterdir()}
        assert {"config.json", "palace.msh", "metadata.json"} <= files
        for text in ((jobs_dir / name / "config.json").read_text(),):
            assert str(jobs_dir) not in text  # no local path in what is uploaded


def test_rebuilding_is_idempotent(make, built):
    jobs_dir, entries = built
    job = {j["name"]: j for j in make.planned_jobs()}["surface_r1"]
    again = make.build_job(job, jobs_dir=jobs_dir, length_um=20.0)
    assert again["mesh_hash"] == entries["surface_r1"]["mesh_hash"]
    assert again["tetrahedra"] == entries["surface_r1"]["tetrahedra"]


@pytest.mark.usefixtures("built")
def test_the_box_patch_is_removed_after_meshing():
    from gsim.palace.mesh import gmsh_utils

    assert gmsh_utils.finalize_mesh_fields.__qualname__ == "finalize_mesh_fields"


# ---------------------------------------------------------------------------
# The analysis, on synthetic port-S.csv files
# ---------------------------------------------------------------------------

LENGTH_MM = 0.04
TRUE_LOSS = {  # metal loss per mm, volume model, by requested t / delta
    0.5: 0.05,
    1.0: 0.04,
    2.4: 0.03,
    5.0: 0.02,
    10.0: 0.015,
}
TRUE_RATIO = {0.5: 1.0, 1.0: 0.98, 2.4: 0.74, 5.0: 0.95, 10.0: 1.01}
NON_METAL = 0.004  # absorbed by the outer wall and the solver in every job
S11_POWER = 0.05


def _write_port_s(
    folder: Path, frequency_ghz: float, absorbed: float, *, two_ports=False
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    s21_power = 1.0 - S11_POWER - absorbed
    columns = [
        "f (GHz)",
        "|S[1][1]| (dB)",
        "arg(S[1][1]) (deg.)",
        "|S[2][1]| (dB)",
        "arg(S[2][1]) (deg.)",
    ]
    values = [
        frequency_ghz,
        10 * math.log10(S11_POWER),
        -12.0,
        10 * math.log10(s21_power),
        -34.0,
    ]
    if two_ports:  # a symmetric line excited at port 2 as well
        columns += [
            "|S[1][2]| (dB)",
            "arg(S[1][2]) (deg.)",
            "|S[2][2]| (dB)",
            "arg(S[2][2]) (deg.)",
        ]
        values += [10 * math.log10(s21_power), -34.0, 10 * math.log10(S11_POWER), -12.0]
    (folder / "port-S.csv").write_text(
        ", ".join(columns) + "\n" + ", ".join(f"{v:.9e}" for v in values) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def jobs(make):
    return make.planned_jobs()


@pytest.fixture
def results(tmp_path, jobs):
    """All 15 jobs: known volume loss, surface = TRUE_RATIO x volume, a baseline."""
    for job in jobs:
        ratio = job["t_over_delta_requested"]
        assert job["length_um"] / 1e3 == LENGTH_MM
        metal = TRUE_LOSS[ratio] * LENGTH_MM
        if job["model"] == "volume":
            absorbed = NON_METAL + metal
        elif job["model"] == "surface":
            absorbed = NON_METAL + metal * TRUE_RATIO[ratio]
        elif job["model"] == "surface_matched":
            absorbed = NON_METAL + metal * (TRUE_RATIO[ratio] + 0.01)
        else:
            absorbed = NON_METAL
        _write_port_s(tmp_path / job["name"], job["frequency_ghz"], absorbed)
    return tmp_path


def test_absorbed_fraction_from_synthetic_file(analyse, tmp_path):
    _write_port_s(tmp_path / "a", 1.0, 0.0123)
    f, magnitudes = analyse.read_port_s(tmp_path / "a" / "port-S.csv")
    assert f.tolist() == [1.0]
    assert analyse.absorbed_fraction(magnitudes)[0] == pytest.approx(0.0123, abs=1e-9)


def test_two_excitations_are_averaged(analyse, tmp_path):
    _write_port_s(tmp_path / "a", 1.0, 0.02, two_ports=True)
    _f, magnitudes = analyse.read_port_s(tmp_path / "a" / "port-S.csv")
    assert analyse.absorbed_fraction(magnitudes)[0] == pytest.approx(0.02, abs=1e-9)


def test_ratio_is_recovered_with_the_baseline_subtracted(analyse, jobs, results):
    rows = analyse.analyse(results, jobs)
    assert [row["t_over_delta_requested"] for row in rows] == [0.5, 1.0, 2.4, 5.0, 10.0]
    for row in rows:
        ratio = row["t_over_delta_requested"]
        assert row["baseline"] == "pec"
        assert row["loss_per_mm_volume"] == pytest.approx(TRUE_LOSS[ratio], rel=1e-6)
        assert row["ratio_surface_over_volume"] == pytest.approx(
            TRUE_RATIO[ratio], rel=1e-6
        )
        assert row["ratio_surface_matched_over_volume"] == pytest.approx(
            TRUE_RATIO[ratio] + 0.01, rel=1e-6
        )


def test_without_the_baseline_the_ratio_is_pulled_to_one_and_says_so(
    analyse, jobs, results
):
    import shutil

    for job in jobs:
        if job["model"] == "pec":
            shutil.rmtree(results / job["name"])
    rows = analyse.analyse(results, jobs)
    row = next(r for r in rows if r["t_over_delta_requested"] == 2.4)
    assert row["baseline"] == "none"
    assert TRUE_RATIO[2.4] < row["ratio_surface_over_volume"] < 1.0


def test_only_the_ten_required_jobs_are_needed(analyse, jobs, results):
    import shutil

    for job in jobs:
        if not job["required"]:
            shutil.rmtree(results / job["name"])
    rows = analyse.analyse(results, jobs)
    assert all(math.isnan(row["ratio_surface_matched_over_volume"]) for row in rows)
    assert all(math.isfinite(row["ratio_surface_over_volume"]) for row in rows)


def test_missing_required_result_is_an_error_naming_the_job(analyse, jobs, results):
    import shutil

    shutil.rmtree(results / "volume_r5")
    with pytest.raises(FileNotFoundError, match="volume_r5"):
        analyse.analyse(results, jobs)


def test_result_at_the_wrong_frequency_is_refused(analyse, jobs, results):
    job = next(j for j in jobs if j["name"] == "surface_r5")
    _write_port_s(results / "surface_r5", job["frequency_ghz"] * 2, 0.01)
    with pytest.raises(ValueError, match="surface_r5"):
        analyse.analyse(results, jobs)


def test_cloud_layout_with_nested_output_folder_is_found(analyse, jobs, results):
    import shutil

    job = next(j for j in jobs if j["name"] == "volume_r10")
    shutil.rmtree(results / "volume_r10")
    _write_port_s(
        results / "volume_r10" / "output" / "palace",
        job["frequency_ghz"],
        0.004 + 0.015 * LENGTH_MM,
    )
    rows = analyse.analyse(results, jobs)
    assert rows[-1]["loss_per_mm_volume"] == pytest.approx(0.015, rel=1e-6)


def test_nonpositive_volume_loss_gives_nan_not_a_ratio(analyse, jobs, results):
    job = next(j for j in jobs if j["name"] == "volume_r1")
    _write_port_s(results / "volume_r1", job["frequency_ghz"], NON_METAL - 0.001)
    rows = analyse.analyse(results, jobs)
    assert math.isnan(rows[1]["ratio_surface_over_volume"])


def test_malformed_csv_is_refused(analyse, tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB)\n1.0, nan\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-finite"):
        analyse.read_port_s(folder / "port-S.csv")
    (folder / "port-S.csv").write_text("freq, x\n1.0, 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="f \\(GHz\\)"):
        analyse.read_port_s(folder / "port-S.csv")


def test_command_line_writes_csv_and_figure(analyse, jobs, results, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    out_csv = tmp_path / "out" / "loss.csv"
    out_png = tmp_path / "out" / "loss.png"
    code = analyse.main(
        [
            "--results",
            str(results),
            "--manifest",
            str(manifest),
            "--csv",
            str(out_csv),
            "--figure",
            str(out_png),
        ]
    )
    assert code == 0
    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(analyse.COLUMNS)
    assert len(lines) == 6
    column = lines[0].split(",").index("ratio_surface_over_volume")
    ratios = [float(line.split(",")[column]) for line in lines[1:]]
    assert ratios == pytest.approx(
        [TRUE_RATIO[r] for r in sorted(TRUE_RATIO)], rel=1e-4
    )
    assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.isfinite(ratios).all()


# ---------------------------------------------------------------------------
# The stored cloud results, the stored csv and the page
# ---------------------------------------------------------------------------

STORED_RESULTS = SCRIPTS / "results"
STORED_CSV = SCRIPTS / "conductor_loss_vs_skin_depth.csv"
PAGE = ROOT / "docs" / "palace_conductors.md"
SECTION = "### Surface model against meshed volume"


def _stored_rows() -> list[dict[str, str]]:
    import csv

    with STORED_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_stored_port_s_files_reproduce_the_stored_csv(analyse, tmp_path):
    """The kept ``port-S.csv`` files alone give the committed table."""
    jobs = analyse.load_jobs(analyse.MANIFEST)
    rows = analyse.analyse(STORED_RESULTS, jobs)
    out = tmp_path / "again.csv"
    analyse.write_csv(rows, out)
    assert out.read_text(encoding="utf-8") == STORED_CSV.read_text(encoding="utf-8")


def test_manifest_has_job_ids_and_runtimes_of_the_runs_only(analyse):
    jobs = analyse.load_jobs(analyse.MANIFEST)
    for job in jobs:
        if job["required"]:
            assert isinstance(job["job_id"], str)
            assert job["runtime_s"] > 0
            folders = [p.name for p in (STORED_RESULTS / job["name"]).iterdir()]
            assert f"sim-data-palace-{job['job_id']}" in folders
        else:  # the PEC and matched-mesh jobs were not run
            assert job["job_id"] is None
            assert job["runtime_s"] is None


def test_page_table_is_the_stored_csv():
    section = PAGE.read_text(encoding="utf-8").partition(SECTION)[2]
    section = section.split("\n## ")[0]
    cells = [r"([\d.]+)", r"([\d.]+)", r"(\d+)", r"([\d.e-]+)", r"([\d.e-]+)"]
    pattern = r"^\| " + r" \| ".join([*cells, r"([\d.]+)"]) + r" \|$"
    found = re.findall(pattern, section, flags=re.MULTILINE)
    rows = _stored_rows()
    assert len(found) == len(rows) == 5
    for (ratio, f_ghz, length_um, surface, volume, quotient), row in zip(
        found, rows, strict=True
    ):
        assert float(ratio) == float(row["t_over_delta_requested"])
        assert float(f_ghz) == float(row["frequency_ghz"])
        assert float(length_um) == pytest.approx(1e3 * float(row["length_mm"]))
        assert float(surface) == pytest.approx(float(row["absorbed_surface"]), rel=6e-3)
        assert float(volume) == pytest.approx(float(row["absorbed_volume"]), rel=6e-3)
        assert quotient == f"{float(row['ratio_surface_over_volume']):.2f}"


def test_page_has_no_unfilled_placeholder():
    text = PAGE.read_text(encoding="utf-8")
    assert "TODO" not in text
    assert "<!--" not in text
