"""The femwell guide (``docs/api/femwell.md``) stays true to the code it cites.

Four kinds of check, none of which touches Palace or the cloud:

- every ``path:line`` citation on the page points at a line that exists, and the
  lines the page leans on still hold the text it describes (the page cites only
  files of the repository);
- the defaults the page states are the defaults in the signatures;
- the page's runnable snippets run as written (the one that builds a mesh and
  solves it needs femwell and gmsh, the others need neither);
- the coax table is recomputed on two meshes at both element orders, against the
  lossless closed form and an independent skin-effect reference, and the
  boundary field ratio and ``n_eff`` of the page's solve snippet are checked.

The page cites this file for those numbers. No test here takes over 10 s.
"""

from __future__ import annotations

import inspect
import re
import tomllib
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import mu_0 as MU0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common import modes as modes_module
from gsim.common.modes import propagating_modes, select_line_mode
from gsim.femwell import boundary_field_ratio, conductor_index_bound, solve_modes

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGE = DOCS / "api" / "femwell.md"

CITATION = re.compile(
    r"(?P<path>(?:src|tests|nbs|docs)/[\w./\-]+\.\w+|pyproject\.toml)"
    r":(?P<spec>\d+(?:-\d+)?(?:, \d+(?:-\d+)?)*)"
)
FENCE = re.compile(r"```python\n(.*?)```", flags=re.DOTALL)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _expand(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for part in spec.split(", "):
        first, _, last = part.partition("-")
        ranges.append((int(first), int(last or first)))
    return ranges


def _lines(path: str) -> list[str]:
    return (ROOT / path).read_text(encoding="utf-8").splitlines()


def _citations() -> dict[str, set[int]]:
    """Every file the page cites with a line, and the lines cited in it."""
    cited: dict[str, set[int]] = {}
    for match in CITATION.finditer(_page()):
        lines = cited.setdefault(match["path"], set())
        for first, last in _expand(match["spec"]):
            lines.update(range(first, last + 1))
    return cited


def _snippets() -> dict[str, str]:
    """Python blocks of the page, keyed by their first line (a comment)."""
    blocks = {}
    for source in FENCE.findall(_page()):
        first = source.splitlines()[0]
        assert first.startswith("# "), "every python block opens with a comment"
        assert first not in blocks, f"two blocks open with {first!r}"
        blocks[first] = source
    return blocks


SOLVE = "# Mesh a gpdk cross-section and solve it with femwell"
ORDER_ONE = "# The same solve at element order 1"
MANY_MODES = "# Many modes around a lossy conductor: the metal modes beat the line mode"
DEGENERATE = "# Degenerate candidates and no candidate"


# --- citations ----------------------------------------------------------------


def test_every_cited_line_exists():
    cited = _citations()
    assert cited, "the page cites nothing"
    for path, lines in cited.items():
        assert (ROOT / path).is_file(), f"{path} is cited but is not a file"
        count = len(_lines(path))
        assert max(lines) <= count, f"{path} cited at line {max(lines)} of {count}"
    for match in CITATION.finditer(_page()):
        for first, last in _expand(match["spec"]):
            assert first <= last, f"empty range in {match.group(0)}"


def test_the_page_cites_only_files_that_are_in_the_repository():
    """Scratch scripts are not committed; a number must come from a test."""
    page = _page()
    assert "plan-docs" not in page
    assert "scratch" not in page.lower()
    assert "tests/femwell/test_femwell_docs.py" in _citations()


# (file, first cited line of the range, last line, text the lines must hold)
ANCHORS = [
    ("src/gsim/femwell/adapter.py", 319, 319, "order: int = 2"),
    ("src/gsim/femwell/adapter.py", 335, 347, "pass ``order=1``"),
    ("src/gsim/femwell/adapter.py", 335, 347, "from about 2.5 times"),
    ("src/gsim/femwell/adapter.py", 217, 222, "fraction: float = 0.1"),
    ("src/gsim/femwell/adapter.py", 217, 222, "margin: float = 10.0"),
    ("src/gsim/femwell/adapter.py", 287, 295, "fraction * np.sqrt"),
    ("src/gsim/femwell/adapter.py", 310, 310, "return float(bounds.min())"),
    ("src/gsim/femwell/adapter.py", 297, 309, "returns no bound"),
    ("src/gsim/femwell/adapter.py", 278, 281, "must be positive"),
    ("src/gsim/femwell/adapter.py", 349, 353, "n_guess"),
    (
        "src/gsim/femwell/adapter.py",
        354,
        356,
        "max_index=conductor_index_bound(epsilon)",
    ),
    ("src/gsim/femwell/adapter.py", 466, 485, "bounding box"),
    ("src/gsim/femwell/adapter.py", 744, 752, "wall_mode_from_currents"),
    ("src/gsim/common/modes.py", 150, 150, "class NoLineModeError"),
    ("src/gsim/common/modes.py", 138, 138, "MAX_GAIN_RATIO: float = 1e-3"),
    ("src/gsim/common/modes.py", 330, 335, "n_eff.real > min_index"),
    ("src/gsim/common/modes.py", 353, 353, "max_index: float | None = None"),
    ("src/gsim/common/modes.py", 412, 417, "raise NoLineModeError("),
    ("src/gsim/common/modes.py", 419, 437, "warnings.warn("),
    ("src/gsim/palace/results.py", 236, 238, "def mode_voltage"),
    ("tests/femwell/test_z0_extraction.py", 100, 100, "3.2%"),
    ("tests/femwell/test_z0_extraction.py", 26, 33, "FREQ_HZ = 100e9"),
    ("tests/femwell/test_z0_extraction.py", 97, 104, "< 0.08 * Z0_ANALYTIC"),
    ("tests/femwell/test_series_resistance.py", 1, 19, "8.0 %"),
    ("tests/femwell/test_series_resistance.py", 139, 147, "order=1"),
    ("tests/femwell/test_series_resistance.py", 150, 157, "1e4 to 5e4"),
    ("tests/femwell/test_series_resistance.py", 150, 157, "about 1e9 off"),
    (
        "tests/femwell/test_series_resistance.py",
        159,
        165,
        "conductor_index_bound(_epsilon())",
    ),
    ("tests/femwell/test_cross_validation.py", 40, 45, "4.1e-5"),
    ("tests/femwell/test_staircase_convergence.py", 105, 111, "order=1"),
    ("tests/femwell/test_adapter.py", 165, 197, "returns no bound"),
    ("tests/common/test_mode_selection.py", 115, 126, "max_loss_ratio=0.5"),
    ("nbs/tcad_twmzm.ipynb", 766, 766, "boundary_field_ratio(mode) < 0.01"),
    ("nbs/tcad_twmzm.ipynb", 1005, 1005, "num_modes=4"),
    ("nbs/tcad_twmzm.ipynb", 1010, 1010, "select_line_mode(modes)"),
    ("pyproject.toml", 41, 44, "femwell>=0.1.11"),
    # The page cites this file for the numbers it computes (first line of each).
    ("tests/femwell/test_femwell_docs.py", 265, 270, "ratio = boundary_field_ratio"),
    ("tests/femwell/test_femwell_docs.py", 271, 273, 'namespace["n_eff_order1"]'),
    ("tests/femwell/test_femwell_docs.py", 325, 349, "def test_the_reference_numbers"),
    ("tests/femwell/test_femwell_docs.py", 431, 450, "def test_the_coax_table_row"),
    ("tests/femwell/test_femwell_docs.py", 469, 477, "def test_against_the_lossless"),
    ("tests/femwell/test_femwell_docs.py", 453, 466, "def test_order_two_is_converged"),
    ("tests/femwell/test_femwell_docs.py", 276, 287, "def test_the_many_modes_snippet"),
    ("tests/femwell/test_femwell_docs.py", 309, 316, "def test_a_bound_of_none"),
    ("tests/femwell/test_femwell_docs.py", 298, 306, "def test_the_rule_on_the_page"),
]


@pytest.mark.parametrize(
    ("path", "first", "last", "text"),
    ANCHORS,
    ids=[f"{a[0].rsplit('/', 1)[-1]}:{a[1]}-{a[2]}" for a in ANCHORS],
)
def test_the_cited_lines_hold_what_the_page_says(path, first, last, text):
    assert first in _citations().get(path, set()), (
        f"the page no longer cites {path}:{first}; update the page or this table"
    )
    held = "\n".join(_lines(path)[first - 1 : last])
    assert text in held, f"{path}:{first}-{last} no longer holds {text!r}"


# --- defaults the page states ------------------------------------------------


def test_the_defaults_on_the_page_are_the_defaults_in_the_code():
    solve = inspect.signature(solve_modes).parameters
    assert solve["order"].default == 2
    assert solve["num_modes"].default == 1
    assert solve["metallic_boundaries"].default is False
    assert solve["n_guess"].default is None

    select = inspect.signature(select_line_mode).parameters
    assert select["max_index"].default is None
    assert select["min_index"].default == 1.0
    assert select["max_loss_ratio"].default == 1.0
    assert select["degeneracy_rtol"].default == 0.03

    bound = inspect.signature(conductor_index_bound).parameters
    assert bound["fraction"].default == 0.1
    assert bound["margin"].default == 10.0
    assert modes_module.MAX_GAIN_RATIO == 1e-3

    page = _page()
    for needle in (
        "`order=1`",
        "**The default is order 2**",
        "`max_index=None`",
        "0.03",
    ):
        assert needle in page


def test_the_public_names_the_page_uses_are_exported():
    import gsim.femwell as femwell

    for name in (
        "solve_modes",
        "epsilon_by_region",
        "elementwise_epsilon",
        "boundary_field_ratio",
        "conductor_index_bound",
        "z0_power_current",
        "line_reading",
        "field_fraction_outside",
    ):
        assert name in femwell.__all__
        assert name in _page()
    for name in ("select_line_mode", "propagating_modes", "NoLineModeError"):
        assert name in modes_module.__all__


# --- snippets -------------------------------------------------------------------


def _run(source: str, namespace: dict) -> dict:
    exec(compile(source, "docs/api/femwell.md", "exec"), namespace)  # noqa: S102
    return namespace


def test_no_snippet_submits_a_job():
    for source in _snippets().values():
        assert ".run(" not in source
        assert "run_local(" not in source


def test_every_block_on_the_page_is_run_by_this_file():
    assert set(_snippets()) == {SOLVE, ORDER_ONE, MANY_MODES, DEGENERATE}


def test_the_mesh_and_solve_snippets_run(tmp_path, monkeypatch):
    pytest.importorskip("femwell")
    pytest.importorskip("skfem")
    pytest.importorskip("gmsh")
    monkeypatch.chdir(tmp_path)
    snippets = _snippets()
    namespace: dict = {"__name__": "docs_snippet"}
    _run(snippets[SOLVE], namespace)
    n_eff = namespace["n_eff"]
    # The page quotes 2.4449 (order 2) and 2.4231 (order 1) for this mesh, and a
    # boundary field ratio of the order of 3e-4.
    assert n_eff.real == pytest.approx(2.44485, rel=2e-3)
    assert n_eff.real > 2.0
    ratio = boundary_field_ratio(namespace["mode"])
    assert 1e-4 < ratio < 1e-3
    _run(snippets[ORDER_ONE], namespace)
    assert namespace["n_eff_order1"].real == pytest.approx(2.42305, rel=2e-3)
    assert namespace["n_eff_order1"].real < n_eff.real


def test_the_many_modes_snippet_picks_the_line_mode():
    namespace = _run(_snippets()[MANY_MODES], {"__name__": "docs_snippet"})
    n_eff = namespace["n_eff"]
    assert len(n_eff) == 12
    # The numbers the page prints.
    assert namespace["bound"] == pytest.approx(2613.5, rel=1e-3)
    assert namespace["unbounded"] == 26057.6 - 7991.57j
    assert namespace["line"] == 5.12968 - 4.67966j
    assert min(abs(n) for n in n_eff[1:]) > 1.2e4
    assert abs(np.sqrt(namespace["epsilon"]["metal"])) == pytest.approx(
        2.61e4, rel=2e-3
    )


def test_the_warning_and_error_snippet_runs():
    namespace = _run(_snippets()[DEGENERATE], {"__name__": "docs_snippet"})
    assert namespace["picked"] == 3.0 - 0.01j
    message = str(namespace["caught"][0].message)
    assert "2.95" in message
    assert "3.0%" in message


def test_the_rule_on_the_page_keeps_the_modes_in_a_band():
    modes = [SimpleNamespace(n_eff=n) for n in (4.2 - 0.1j, 3.1 - 0.05j, 1.6 + 0j)]
    assert "if m.n_eff.real < 4.0]" in _page()

    def rule(solved):
        return [m for m in propagating_modes(solved) if m.n_eff.real < 4.0]

    assert select_line_mode(modes) is modes[0]
    assert select_line_mode(modes, rule=rule) is modes[1]


def test_a_bound_of_none_leaves_the_metal_mode_selectable():
    """What the page says about ``conductor_index_bound`` returning ``None``."""
    weak = {"si": 11.7 - 12j, "oxide": 3.9 + 0j}
    with pytest.warns(UserWarning, match="returns no bound"):
        assert conductor_index_bound(weak) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert select_line_mode([3.0 + 0j, 40.0 + 0j], max_index=None) == 40.0 + 0j


# --- the coax row -----------------------------------------------------------------

R_INNER, R_OUTER, EPS_R, SIGMA, FREQ_HZ = 1.0, 3.0, 2.25, 5.8e7, 100e9
Z0_LOSSLESS = MU0 * C0 / (2 * np.pi * np.sqrt(EPS_R)) * np.log(R_OUTER / R_INNER)


def _z0_with_skin_effect() -> complex:
    """sqrt(Z'/Y') with the inner wire's exact internal (Bessel) impedance."""
    from scipy.special import jv

    omega = 2 * np.pi * FREQ_HZ
    radius = R_INNER * 1e-6
    k = np.sqrt(-1j * omega * MU0 * SIGMA)
    z_internal = (
        k * jv(0, k * radius) / (2 * np.pi * radius * SIGMA * jv(1, k * radius))
    )
    inductance = MU0 / (2 * np.pi) * np.log(R_OUTER / R_INNER)
    capacitance = 2 * np.pi * EPS0 * EPS_R / np.log(R_OUTER / R_INNER)
    return complex(
        np.sqrt((1j * omega * inductance + z_internal) / (1j * omega * capacitance))
    )


def test_the_reference_numbers_on_the_page():
    skin = _z0_with_skin_effect()
    assert pytest.approx(43.914, abs=5e-4) == Z0_LOSSLESS
    assert skin.real == pytest.approx(45.988, abs=5e-4)
    assert (skin.real - Z0_LOSSLESS) / Z0_LOSSLESS == pytest.approx(0.0472, abs=5e-5)
    page = _page()
    for text in ("43.914", "45.988", "4.72 %"):
        assert text in page


def _write_coax_mesh(path, far_um: float) -> None:
    """The coax of ``test_z0_extraction.py``; ``far_um=0.35`` is that test's mesh."""
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        inner = occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        occ.fragment([(2, outer)], [(2, inner)])
        occ.synchronize()
        by_area = sorted(
            (occ.getMass(2, tag), tag) for _d, tag in gmsh.model.getEntities(2)
        )
        core = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, core, "conductor")
        rest = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, rest, "dielectric")
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", far_um)
        field = gmsh.model.mesh.field
        ball = field.add("Ball")
        field.setNumber(ball, "Radius", R_INNER + 0.15)
        field.setNumber(ball, "VIn", 0.07)
        field.setNumber(ball, "VOut", far_um)
        field.setAsBackgroundMesh(ball)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


@pytest.fixture(scope="module")
def coax_z0(tmp_path_factory):
    """``coax_z0(far_um, order)``: Re Z0 in ohm, each solve done once."""
    pytest.importorskip("femwell")
    pytest.importorskip("skfem")
    pytest.importorskip("gmsh")
    from gsim.common.modes import Conductor
    from gsim.femwell import z0_power_current

    folder = tmp_path_factory.mktemp("femwell-docs-coax")
    inner_conductor = Conductor(
        "conductor", ((-R_INNER, R_INNER), (-R_INNER, R_INNER)), "volume"
    )
    omega = 2 * np.pi * FREQ_HZ
    epsilon = {"conductor": 1.0 - 1j * SIGMA / (omega * EPS0), "dielectric": EPS_R + 0j}
    meshes: dict[float, object] = {}
    solved: dict[tuple[float, int], float] = {}

    def z0(far_um: float, order: int) -> float:
        if far_um not in meshes:
            meshes[far_um] = folder / f"coax_{far_um}.msh"
            _write_coax_mesh(meshes[far_um], far_um)
        if (far_um, order) not in solved:
            mode = solve_modes(
                meshes[far_um],
                epsilon=epsilon,
                wavelength_um=C0 / FREQ_HZ * 1e6,
                num_modes=1,
                order=order,
                metallic_boundaries=True,
            )[0]
            solved[far_um, order] = z0_power_current(
                mode,
                frequency_hz=FREQ_HZ,
                conductor=inner_conductor,
                mesh=meshes[far_um],
            ).real
        return solved[far_um, order]

    return z0


# The page's coax table: (max element size away from the core in um, order,
# Re Z0 in ohm, error against the lossless closed form, error against the
# skin-effect reference). 0.35 is the mesh of ``test_z0_extraction.py``.
COAX_TABLE = [
    (0.35, 1, 45.323, 0.0321, -0.0145),
    (0.35, 2, 45.957, 0.0465, -0.0007),
    (0.15, 1, 45.825, 0.0435, -0.0035),
    (0.15, 2, 45.996, 0.0474, 0.0002),
]


@pytest.mark.parametrize(
    ("far_um", "order", "z0_ohm", "vs_lossless", "vs_skin"),
    COAX_TABLE,
    ids=[f"far{row[0]}-order{row[1]}" for row in COAX_TABLE],
)
def test_the_coax_table_row(coax_z0, far_um, order, z0_ohm, vs_lossless, vs_skin):
    z0 = coax_z0(far_um, order)
    skin = _z0_with_skin_effect().real
    # 0.3 ohm is the margin the existing coax test leaves for gmsh versions.
    assert z0 == pytest.approx(z0_ohm, abs=0.3)
    assert (z0 - Z0_LOSSLESS) / Z0_LOSSLESS == pytest.approx(vs_lossless, abs=0.007)
    assert (z0 - skin) / skin == pytest.approx(vs_skin, abs=0.007)


def test_order_two_is_converged_on_the_coax_and_order_one_is_not(coax_z0):
    """Against the skin-effect reference, order 2 has settled and order 1 has not."""
    skin = _z0_with_skin_effect().real
    for far_um in (0.35, 0.15):
        # Order 2 sits on the skin-effect reference, order 1 below it.
        assert abs(coax_z0(far_um, 2) - skin) < 0.003 * skin
        assert -0.025 < (coax_z0(far_um, 1) - skin) / skin < -0.002
        assert abs(coax_z0(far_um, 2) - skin) < abs(coax_z0(far_um, 1) - skin)
    # The two meshes agree to 0.2 % at order 2; order 1 moves by more than that.
    move_1 = abs(coax_z0(0.15, 1) - coax_z0(0.35, 1))
    move_2 = abs(coax_z0(0.15, 2) - coax_z0(0.35, 2))
    assert move_2 < 0.002 * skin
    assert move_1 > 0.004 * skin
    assert move_1 > 3 * move_2


def test_against_the_lossless_closed_form_order_one_looks_closer_on_the_tests_mesh(
    coax_z0,
):
    """A coincidence: the closed form leaves out the skin effect the solve includes."""
    error_1 = abs(coax_z0(0.35, 1) - Z0_LOSSLESS)
    error_2 = abs(coax_z0(0.35, 2) - Z0_LOSSLESS)
    assert error_1 < error_2
    # Both are inside the 8 % that test_z0_extraction.py accepts.
    assert error_2 < 0.08 * Z0_LOSSLESS


# --- the page as a page ---------------------------------------------------------


def test_the_page_is_in_the_nav_once_under_api_guides_and_in_llms_guides():
    config = tomllib.loads((DOCS / "zensical.toml").read_text(encoding="utf-8"))
    nav = config["project"]["nav"]
    guides = next(entry["API Guides"] for entry in nav if "API Guides" in entry)
    assert [e for e in guides if "api/femwell.md" in e.values()] == [
        {"femwell": "api/femwell.md"}
    ]
    flat = str(nav)
    assert flat.count("api/femwell.md") == 1
    # llms.txt is generated from these sections; ``api/*.md`` picks the page up.
    assert "api/*.md" in config["project"]["plugins"]["llmstxt"]["sections"]["Guides"]
    assert (DOCS / "api" / "femwell.md").is_file()


def test_the_home_page_links_the_guide():
    assert "(api/femwell.md)" in (DOCS / "index.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["api/femwell.md", "index.md"])
def test_markdown_lines_are_at_most_150_characters(name):
    long = [
        (number, len(line))
        for number, line in enumerate(
            (DOCS / name).read_text(encoding="utf-8").splitlines(), 1
        )
        if len(line) > 150
    ]
    assert long == []


def test_local_links_resolve():
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", _page()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (PAGE.parent / target).exists(), f"broken link {target}"


def test_the_page_makes_no_claim_it_cannot_support():
    page = _page()
    # The table of the page is from this checkout's own runs; it names no cloud job.
    assert "cloud job" not in page.lower()
    assert "26 %" not in page
