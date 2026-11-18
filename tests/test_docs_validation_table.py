"""The benchmarks page cites its evidence: every cited file and line must exist.

``docs/palace_benchmarks.md`` copies each number from a test assertion, a test
comment, a documented result or a stored audit, and names the source in the
``Where`` column as ``path:line`` or ``path:first-last``. Nothing here runs a
solver; the checks read files only.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PAGE = DOCS / "palace_benchmarks.md"

#: ``path`` or ``path:12`` or ``path:12-20`` in backticks, rooted at a tracked folder.
CITATION = re.compile(
    r"`((?:tests|docs|nbs|src)/[^`\s:]+)(?::(\d+)(?:-(\d+))?)?`",
)
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
MAX_LINE = 150


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _tables(text: str) -> dict[str, list[list[str]]]:
    """Return the rows of each pipe table, keyed by its first header cell sequence."""
    tables: dict[str, list[list[str]]] = {}
    current: list[list[str]] | None = None
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if current is None:
                current = [cells]
                tables["|".join(cells)] = current
            else:
                current.append(cells)
        else:
            current = None
    return tables


def _where_table() -> list[list[str]]:
    for key, rows in _tables(_text()).items():
        if key.endswith("|Where"):
            return rows
    pytest.fail("no table with a 'Where' column")


def _check_citation(path: str, first: str, last: str) -> None:
    """Check a cited file and, when lines are given (``""`` if not), their range."""
    target = ROOT / path
    assert target.is_file(), f"cited file does not exist: {path}"
    if not first:
        return
    start = int(first)
    end = int(last) if last else start
    count = len(target.read_text(encoding="utf-8").splitlines())
    assert 1 <= start <= end <= count, f"{path}:{first}-{last} is outside 1-{count}"


def test_the_table_has_a_where_column_with_the_expected_cases():
    rows = _where_table()
    header, body = rows[0], rows[1:]
    assert header == ["Case", "Solvers", "Reference", "Agreement", "Where"]
    cases = " ".join(row[0] for row in body)
    for expected in ("Coax", "GSSG", "de-embedding", "PN junction", "MIM", "L2n0"):
        assert expected in cases
    assert all(len(row) == len(header) for row in body)


def test_each_where_cell_cites_existing_files_and_lines():
    for row in _where_table()[1:]:
        found = CITATION.findall(row[-1])
        assert found, f"no file cited for {row[0]!r}"
        for path, first, last in found:
            _check_citation(path, first, last)


def test_every_citation_on_the_page_exists():
    found = CITATION.findall(_text())
    assert len(found) >= 10
    for path, first, last in found:
        _check_citation(path, first, last)


def test_the_l2n0_row_says_not_validated_with_the_audit_figures():
    (row,) = (r for r in _where_table()[1:] if "L2n0" in r[0])
    assert "not validated" in row[3].lower()
    assert row[4].startswith("`nbs/data/ihp_validation/CONVERGENCE_AUDIT.md:")
    audit = (ROOT / "nbs/data/ihp_validation/CONVERGENCE_AUDIT.md").read_text(
        encoding="utf-8"
    )
    # The figures the page quotes are the audit's own.
    for figure in ("8.86", "8.83", "44.0", "45.4", "0.087", "6.6", "0.80", "1.67"):
        assert figure in audit
        assert figure in _text()
    assert "not validated" in audit


def test_the_cited_figures_are_in_the_cited_sources():
    """Spot checks that a number in the table is on the lines it cites."""

    def lines(path: str, first: int, last: int) -> str:
        source = (ROOT / path).read_text(encoding="utf-8").splitlines()
        return "\n".join(source[first - 1 : last])

    z0 = lines("tests/femwell/test_z0_extraction.py", 100, 104)
    assert "3.2%" in z0
    assert "0.08" in z0
    pec = lines("tests/femwell/test_z0_extraction.py", 141, 145)
    assert "29%, 17% and 8%" in pec
    assert "0.4%" in pec
    assert "rel=0.01" in lines("tests/femwell/test_z0_extraction.py", 192, 204)
    cross = lines("tests/femwell/test_cross_validation.py", 40, 45)
    for figure in ("4.1e-5", "2.9e-4", "9e-3", "N_EFF_RTOL = 1e-3"):
        assert figure in cross
    terminal = lines("docs/terminal_wave_ports.md", 70, 78)
    assert "3.4e-14" in terminal
    gssg = lines("docs/terminal_wave_ports.md", 416, 427)
    for figure in ("0.05 %", "0.29 deg"):
        assert figure in gssg
    fit = lines("tests/palace/test_cpw_notebook_validation.py", 249, 251)
    assert fit.count("rel=1e-3") == 3
    assert "within(0.35)" in lines("tests/tcad/test_solve_runtime.py", 92, 100)
    assert "rel=1e-9" in lines("tests/tcad/test_edge_couples.py", 227, 229)
    assert "1e-14" in lines("docs/api/tcad.md", 233, 234)


def test_the_page_is_in_the_navigation_and_the_llms_guides():
    config = tomllib.loads((DOCS / "zensical.toml").read_text(encoding="utf-8"))
    nav = {
        next(iter(entry)): next(iter(entry.values()))
        for entry in config["project"]["nav"]
    }
    section = nav["Modeling & Analysis"]
    pages = [next(iter(e.values())) for e in section]
    assert pages.count("palace_benchmarks.md") == 1
    assert (
        pages.index("palace_benchmarks.md") == pages.index("palace-validation.md") + 1
    )
    guides = config["project"]["plugins"]["llmstxt"]["sections"]["Guides"]
    assert "palace_benchmarks.md" in guides


def test_the_l2n0_notebook_is_linked_but_not_in_the_navigation():
    text = _text()
    assert "nbs/ihp_l2n0_openems_validation.ipynb" in text
    assert "work in progress" in text.lower()
    nav = (DOCS / "zensical.toml").read_text(encoding="utf-8")
    assert "ihp_l2n0" not in nav


def test_the_other_benchmarks_section_links_the_fdtd_page():
    text = _text()
    _, found, section = text.partition("## Other benchmarks")
    assert found
    section = section.split("\n## ", 1)[0]
    assert "fdtd-benchmark.md" in section


def test_the_baehr_jones_comparison_is_pointed_to_not_tabled():
    text = _text()
    assert "pull/522" in text
    assert "Baehr-Jones" in text
    assert "Baehr" not in "".join(r[0] for r in _where_table()[1:])


def test_lines_are_at_most_150_characters():
    long = [n for n, line in enumerate(_text().splitlines(), 1) if len(line) > MAX_LINE]
    assert not long, f"lines over {MAX_LINE} characters: {long}"


def test_local_links_resolve():
    for target in LINK.findall(_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        path = target.split("#", 1)[0]
        if path.startswith("nbs/"):
            # Notebook pages are generated at build time from nbs/*.ipynb.
            assert (ROOT / path.replace(".md", ".ipynb")).is_file(), target
        else:
            assert (DOCS / path).is_file(), target
