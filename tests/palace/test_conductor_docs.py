"""``docs/palace_conductors.md`` stays true: snippet, skin-depth table, nav entry.

The page's snippet is run as written. It stops at ``write_config()``, so nothing is
solved or submitted. The skin-depth table is compared with
``delta = sqrt(2 / (omega * mu0 * sigma))`` computed from ``scipy.constants``, and the
conductivities with gsim's own material table.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path

import pytest
from scipy.constants import mu_0, pi

from gsim.common.stack.materials import MATERIALS_DB

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGE = DOCS / "palace_conductors.md"
DEEMBEDDING_NOTEBOOK = ROOT / "nbs" / "palace_cpw_deembedding.ipynb"

SIGMA_AL = 3.77e7  # S/m, gsim "aluminum" (gpdk metal1)
SIGMA_CU = 5.8e7  # S/m, gsim "copper"
METAL1_THICKNESS_UM = 0.7  # gpdk metal1: z = 1.1 to 1.8 um
GRID_TOLERANCE = 0.006  # three printed significant digits: at most 0.5 % of the value


def skin_depth_um(sigma: float, frequency_hz: float) -> float:
    """``sqrt(2 / (omega * mu0 * sigma))`` in um."""
    omega = 2 * pi * frequency_hz
    return 1e6 * math.sqrt(2.0 / (omega * mu_0 * sigma))


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    _, found, rest = text.partition(heading)
    assert found, f"heading {heading!r} not found"
    return re.split(r"^## ", rest, maxsplit=1, flags=re.MULTILINE)[0]


def _table_rows() -> list[tuple[float, float, float, float]]:
    section = _section(_page(), "## Validity versus skin depth")
    rows = re.findall(
        r"^\| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
        section,
        flags=re.MULTILINE,
    )
    assert rows, "the skin-depth table is missing"
    return [(float(a), float(b), float(c), float(d)) for a, b, c, d in rows]


# ---------------------------------------------------------------------------
# Skin depth
# ---------------------------------------------------------------------------


def test_skin_depth_two_forms_agree():
    for sigma in (SIGMA_AL, SIGMA_CU):
        for f in (1e9, 10e9, 100e9):
            other = 1e6 / math.sqrt(pi * f * mu_0 * sigma)
            assert skin_depth_um(sigma, f) == pytest.approx(other, rel=1e-12)


def test_skin_depth_falls_as_one_over_root_f():
    assert skin_depth_um(SIGMA_AL, 1e9) / skin_depth_um(
        SIGMA_AL, 100e9
    ) == pytest.approx(10.0)


def test_table_rows_are_the_three_frequencies():
    assert [row[0] for row in _table_rows()] == [1, 10, 100]


@pytest.mark.parametrize("row", range(3))
def test_table_matches_the_formula(row):
    f_ghz, delta_al, delta_cu, ratio = _table_rows()[row]
    f = f_ghz * 1e9
    assert delta_al == pytest.approx(skin_depth_um(SIGMA_AL, f), rel=GRID_TOLERANCE)
    assert delta_cu == pytest.approx(skin_depth_um(SIGMA_CU, f), rel=GRID_TOLERANCE)
    assert ratio == pytest.approx(
        METAL1_THICKNESS_UM / skin_depth_um(SIGMA_AL, f), rel=GRID_TOLERANCE
    )


def test_conductivities_are_gsims_material_table():
    assert MATERIALS_DB["aluminum"].conductivity == SIGMA_AL
    assert MATERIALS_DB["copper"].conductivity == SIGMA_CU
    text = _page()
    assert "3.77e7" in text
    assert "5.8e7" in text


def test_crossover_frequency_of_metal1():
    """The page says where the 0.7 um layer is as thick as the aluminum skin depth."""
    found = re.search(r"\(t = delta\) at about ([\d.]+) GHz", _page())
    assert found, "the page states the frequency at which t = delta"
    t_m = METAL1_THICKNESS_UM * 1e-6
    f_equal = 1.0 / (pi * mu_0 * SIGMA_AL * t_m**2)
    assert float(found.group(1)) == pytest.approx(f_equal / 1e9, rel=GRID_TOLERANCE)
    assert skin_depth_um(SIGMA_AL, f_equal) == pytest.approx(METAL1_THICKNESS_UM)


def test_deembedding_notebook_note_is_consistent():
    """The notebook quotes sigma = 3.77e7 S/m and about 2.6 um at 1 GHz."""
    notebook = json.loads(DEEMBEDDING_NOTEBOOK.read_text(encoding="utf-8"))
    text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    # \S stands for the Greek sigma and the micro sign of the notebook's text
    found = re.search(
        r"\S = ([\d.e+]+) S/m, (\d+) \S+m thick\), "
        r"and the skin depth at 1 GHz is about ([\d.]+) \S+m",
        text,
    )
    assert found, (
        "the fit-band note of the notebook changed; update the page and this test"
    )
    sigma, _thickness_um, quoted = (float(g) for g in found.groups())
    assert sigma == SIGMA_AL
    assert quoted == pytest.approx(skin_depth_um(sigma, 1e9), abs=0.05)
    # the page repeats the notebook's number next to its own table value
    assert re.search(r"about\s+2\.6 um at 1 GHz", _page())


# ---------------------------------------------------------------------------
# The page's snippet (no solver, no cloud)
# ---------------------------------------------------------------------------


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), flags=re.DOTALL)


@pytest.fixture(scope="module")
def snippet_namespace(tmp_path_factory) -> dict:
    blocks = _python_blocks()
    assert len(blocks) == 1
    assert ".run(" not in blocks[0], "the snippet must not call the cloud"
    workdir = tmp_path_factory.mktemp("conductor-docs")
    namespace: dict = {"__name__": "docs_conductors"}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workdir)
        exec(compile(blocks[0], PAGE.name, "exec"), namespace)  # noqa: S102
    return namespace


def test_snippet_shell_entries(snippet_namespace):
    entries = snippet_namespace["shell"]["Conductivity"]
    assert len(entries) == 2
    for entry in entries:
        assert set(entry) == {"Attributes", "Conductivity", "Thickness", "Crack"}
        assert entry["Crack"] is True
        assert entry["Conductivity"] == SIGMA_AL
        assert entry["Thickness"] == pytest.approx(METAL1_THICKNESS_UM)
    assert "PEC" not in snippet_namespace["shell"]


def test_snippet_sheet_has_pec_and_no_conductivity_entries(snippet_namespace):
    sheet = snippet_namespace["sheet"]
    assert sheet["Conductivity"] == []
    assert len(sheet["PEC"]["Attributes"]) == 1


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def test_nav_has_the_page_once_under_modeling_and_analysis():
    config = tomllib.loads((DOCS / "zensical.toml").read_text(encoding="utf-8"))
    sections = {k: v for entry in config["project"]["nav"] for k, v in entry.items()}
    pages = [
        value for entry in sections["Modeling & Analysis"] for value in entry.values()
    ]
    assert pages.count("palace_conductors.md") == 1
    assert PAGE.is_file()


def test_lines_fit_the_markdown_lint_limit():
    long_lines = [
        n for n, line in enumerate(_page().splitlines(), start=1) if len(line) > 150
    ]
    assert not long_lines, f"lines over 150 characters: {long_lines}"


def test_local_links_resolve():
    text = _page()
    for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", text):
        if target.startswith("nbs/"):
            assert (ROOT / "nbs" / (Path(target).stem + ".ipynb")).is_file(), target
        else:
            assert (DOCS / target).is_file(), target
    anchors = set(re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE))
    slugs = {re.sub(r"[^a-z0-9 -]", "", a.lower()).replace(" ", "-") for a in anchors}
    for anchor in re.findall(r"\]\(#([^)]+)\)", text):
        assert anchor in slugs, anchor


def test_the_page_carries_no_unsourced_loss_error_number():
    """No metal-loss error in percent until its source is committed."""
    assert not re.search(r"\b26\s?%", _page())
