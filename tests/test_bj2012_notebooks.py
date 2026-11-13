"""Consistency of the Baehr-Jones 2012 notebooks and their stored results.

Baehr-Jones et al., Opt. Express 20(11) 12014 (2012),
https://doi.org/10.1364/OE.20.012014.

Each device has two notebooks: a short guide for users and a long "details"
notebook that holds the full set-up, the fits and the checks. The guides copy
their set-up cells from the details notebooks, and quote numbers that live in
``nbs/data/rf_benchmark/bj2012_*.json``. These tests catch drift between the
copies, numbers in the text that no stored row backs, case files that break
their schema, and cloud calls outside ``if RUN_CLOUD:``. Nothing runs a
simulation.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import pytest

NBS = Path(__file__).resolve().parents[1] / "nbs"
DATA = NBS / "data"
STORE = DATA / "rf_benchmark"

ELECTRODE = "palace_bj2012_electrode"
TCAD = "tcad_bj2012_modulator"
CASES = ("bj2012_electrode", "bj2012_modulator")


def _cells(name):
    """Cells of ``nbs/<name>.ipynb`` as (id, type, source) tuples."""
    notebook = json.loads((NBS / f"{name}.ipynb").read_text(encoding="utf-8"))
    return [(c["id"], c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def _source(name, cell_id):
    """Source of one cell of a notebook."""
    return next(s for cid, _t, s in _cells(name) if cid == cell_id)


def _code(name):
    """All code cells of a notebook joined by blank lines."""
    return "\n\n".join(s for _i, t, s in _cells(name) if t == "code")


def _case(name):
    return json.loads((STORE / f"{name}.json").read_text(encoding="utf-8"))


def _reference(path):
    """Value at a dotted path of the reference file ('|' joins paths: their mean)."""
    if "|" in path:
        values = [_reference(p) for p in path.split("|")]
        return sum(values) / len(values)
    node = json.loads((DATA / "bj2012_reference.json").read_text(encoding="utf-8"))
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


class TestSharedCells:
    """The guides' set-up cells are exact copies of the details notebooks' cells."""

    def test_electrode_builder_is_identical(self):
        marker = "STRIP_WIDTH = REF"
        user = next(s for _i, t, s in _cells(ELECTRODE) if marker in s)
        details = next(s for _i, t, s in _cells(f"{ELECTRODE}_details") if marker in s)
        assert user == details

    def test_electrode_functions_are_copies(self):
        user = _code(ELECTRODE)
        details_5 = _source(f"{ELECTRODE}_details", "5")
        details_19 = _source(f"{ELECTRODE}_details", "19")
        solve_line_mode = details_5.split("\n\n\ndef mode_power")[0]
        boundary_mode, _, rest = details_19.partition("\n\n\nclass PECWallDrivenSim")
        assert solve_line_mode.startswith("def solve_line_mode(")
        assert boundary_mode.startswith("def boundary_mode_sim(")
        for block in (solve_line_mode, boundary_mode, "class PECWallDrivenSim" + rest):
            assert block in user

    def test_tcad_setup_cells_are_copies(self):
        user = _code(TCAD)
        # Cell ids are positions (the nbstripout hook renumbers them): the mesh helper,
        # the charge simulation, the doping profiles and the optical mesh and mode.
        for cell_id in ("3", "5", "7", "20", "24", "41", "43", "45"):
            assert _source(f"{TCAD}_details", cell_id) in user, cell_id


def _guarded(source):
    """(name, under ``if RUN_CLOUD:``) of each ``run`` or ``wait_for_results`` call."""
    found = []

    def visit(node, guarded):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "RUN_CLOUD"
        ):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, guarded)
            return
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name in {"run", "wait_for_results"}:
                found.append((name, guarded))
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(ast.parse(source), False)
    return found


class TestNoCloud:
    """No simulation is submitted unless a person turns ``RUN_CLOUD`` on."""

    @pytest.mark.parametrize("name", [f"{ELECTRODE}_details", f"{TCAD}_details"])
    def test_details_notebooks_submit_only_under_run_cloud(self, name):
        for cell_id, kind, source in _cells(name):
            if kind != "code":
                continue
            for call, guarded in _guarded(source):
                assert guarded, f"{name} cell {cell_id}: {call}() outside RUN_CLOUD"

    def test_run_cloud_is_off_in_the_details_notebook(self):
        flags = re.findall(
            r"^RUN_CLOUD = (\w+)$", _code(f"{ELECTRODE}_details"), re.MULTILINE
        )
        assert flags == ["False"]
        assert "RUN_CLOUD" not in _code(f"{TCAD}_details")

    @pytest.mark.parametrize("name", [ELECTRODE, TCAD])
    def test_guides_never_submit(self, name):
        for cell_id, kind, source in _cells(name):
            if kind == "code":
                assert not _guarded(source), f"{name} cell {cell_id} submits a job"
                assert "RUN_CLOUD" not in source


def _numbers(rows):
    """Every number a row stores, as absolute values."""
    for row in rows:
        for key in ("value", "value_imag", "reference_value", "reference_value_imag"):
            if row.get(key) is not None:
                yield abs(float(row[key]))


def _all_numbers():
    numbers = set()
    for name in CASES:
        numbers.update(_numbers(_case(name)["rows"]))

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            numbers.add(abs(float(node)))

    walk(json.loads((DATA / "bj2012_reference.json").read_text(encoding="utf-8")))
    return numbers


def _matches(text, numbers):
    """True if ``text`` is a stored number rounded to the digits it shows."""
    digits = len(text.replace(".", "").lstrip("0")) or 1
    wanted = float(text)
    for value in numbers:
        if value == 0:
            continue
        shown = float(f"{value:.{digits}g}")
        if math.isclose(shown, wanted, rel_tol=1e-9, abs_tol=1e-12):
            return True
    return False


# Bold numbers (the result boxes) and values in dB/cm, V cm or ohm, which name a stored
# row. A "±" marks an uncertainty and a "(" after a bold number a journal volume:
# skipped.
QUOTED = re.compile(
    r"\*\*(\d+(?:\.\d+)?)(?:\s*(?:dB/cm|V cm))?\*\*(?!\()"
    r"|(?<![±\d.])(\d+(?:\.\d+)?) (?:dB/cm|V cm|ohm)\b"
)


class TestQuotedNumbers:
    """Numbers in the guides' text are in a stored row or in the reference file."""

    @pytest.mark.parametrize("name", [ELECTRODE, TCAD])
    def test_text_numbers_are_stored(self, name):
        numbers = _all_numbers()
        missing = []
        for cell_id, kind, source in _cells(name):
            if kind != "markdown":
                continue
            for match in QUOTED.finditer(source):
                text = match.group(1) or match.group(2)
                if not _matches(text, numbers):
                    missing.append((cell_id, text))
        assert not missing, f"not backed by a stored row: {missing}"


class TestStoredResults:
    """The case files follow the schema and the rules of the folder's README."""

    @pytest.mark.parametrize(
        "name", [*CASES, "ihp_l2n0", "cpw_gssg_tied", "cst_cpw", "gfem_cpw"]
    )
    def test_case_file_follows_the_schema(self, name):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads((STORE / "schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(_case(name), schema)

    @pytest.mark.parametrize(
        "name", [*CASES, "ihp_l2n0", "cpw_gssg_tied", "cst_cpw", "gfem_cpw"]
    )
    def test_status_rules(self, name):
        case = _case(name)
        rows = case["rows"]
        assert len({r["id"] for r in rows}) == len(rows), "duplicate row ids"
        if case["status"] == "planned":
            assert not rows
        for row in rows:
            if row["status"] == "published":
                assert row["provenance_missing"] == [], row["id"]
            else:
                assert row["provenance_missing"], row["id"]
        statuses = {r["status"] for r in rows}
        if rows:
            expected = "published" if statuses == {"published"} else "reported"
            assert case["status"] == expected

    @pytest.mark.parametrize("name", CASES)
    def test_both_copies_of_a_reference_agree(self, name):
        rows = {r["id"]: r for r in _case(name)["rows"]}
        for row in rows.values():
            if row.get("reference_key"):
                expected = _reference(row["reference_key"])
                got = row["reference_value"]
                assert got == pytest.approx(expected, rel=1e-9), row["id"]
            if row.get("reference_key_imag"):
                expected = _reference(row["reference_key_imag"])
                assert row["reference_value_imag"] == pytest.approx(expected, rel=1e-9)
            if row.get("reference_row"):
                other = rows[row["reference_row"]]
                assert row["freq_ghz"] == other["freq_ghz"], row["id"]
                assert row["unit"] == other["unit"], row["id"]

    @pytest.mark.parametrize("name", CASES)
    def test_cloud_rows_are_reported_not_published(self, name):
        """A cloud row lists exactly the provenance fields it lacks (never all set)."""
        fields = (
            "solver_version",
            "gsim_version",
            "gsim_commit",
            "hardware",
            "run_seconds",
            "date",
        )
        for row in _case(name)["rows"]:
            if row["origin"] == "cloud":
                assert row["status"] == "reported", row["id"]
                assert row["provenance_missing"], row["id"]
                for field in fields:
                    listed = field in row["provenance_missing"]
                    assert listed == (not row.get(field)), (row["id"], field)

    @pytest.mark.parametrize("name", CASES)
    def test_rows_of_a_notebook_exist_in_it(self, name):
        for row in _case(name)["rows"]:
            notebook = row["source"]["notebook"]
            if not notebook.endswith(".ipynb"):
                continue
            ids = {cid for cid, _t, _s in _cells(notebook.removesuffix(".ipynb"))}
            assert row["source"]["cell_id"] in ids, row["id"]
