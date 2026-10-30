"""Real local Palace runs of the stacked IHP coil pair (magnetostatic).

Two checks on one coarse Order-1 mesh, both sides on the same Palace build:

- gsim's ``config.json`` against a Palace config written here by hand from
  the Palace configuration docs and the mesh's physical groups, so that what
  gsim adds (attributes, PEC set, source directions, outer wall, solver and
  loader) is checked against an independent reading of the docs;
- reversing source B (plus and minus ports swapped) flips the sign of the
  mutual inductance and leaves the self inductances alone.

Accuracy against mesh refinement is not the point here; both sides share the
mesh and the order. Run with ``pytest -m palace_local -k magnetostatic``;
runs are single rank and sequential.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from gsim.palace.inductance import InductanceMatrix, load_inductance

pytestmark = pytest.mark.palace_local

#: Coarse mesh: half the 2 um trace width at the metal, coarse air.
MESH = {"refined_mesh_size": 1.0, "max_mesh_size": 30.0}
SOURCES = ("A", "B")


@pytest.fixture(scope="module")
def palace_binary() -> Path:
    """The local Palace binary, or skip; never downloaded."""
    from gsim.palace.runtime import resolve_palace_runtime

    found = resolve_palace_runtime(download_if_missing=False)
    if found is None:
        pytest.skip("No local Palace binary resolves.")
    return found.binary


def _checked(inductance: InductanceMatrix) -> InductanceMatrix:
    """The matrix, after asserting it holds together."""
    assert inductance.sources == SOURCES
    assert inductance.problems() == []
    return inductance


def _load(load, results) -> InductanceMatrix:
    """Load with the convergence check on, refusing an unchecked log."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*convergence not checked.*")
        return _checked(load(results, check_convergence=True))


def _run_gsim(binary: Path, output_dir: Path, *, reverse_b: bool = False):
    """Mesh the stacked pair, write gsim's config, run it.

    Returns:
        The inductance matrix, the sim, and the hash of its mesh.
    """
    from tests._cloud_fixtures import make_magnetostatic_pair_sim

    sim = make_magnetostatic_pair_sim(output_dir, stacked=True)
    if reverse_b:
        sim.add_current_source("B", plus_port="B_P2", minus_port="B_P1")
    sim.set_solver(order=1)
    mesh_hash = sim.mesh(**MESH, verbose=False).mesh_stats["mesh_hash"]
    sim.write_config()
    results = sim.run_local(
        palace_executable=binary, num_processes=1, num_threads=4, verbose=False
    )
    return _load(sim.load_inductance, results), sim, mesh_hash


@pytest.fixture(scope="module")
def forward(palace_binary, tmp_path_factory):
    """gsim's run of the stacked pair: (matrix, sim, mesh hash)."""
    pytest.importorskip("ihp")
    return _run_gsim(palace_binary, tmp_path_factory.mktemp("forward"))


# ---------------------------------------------------------------------------
# (a) standalone Palace config
# ---------------------------------------------------------------------------


def _physical_groups(mesh_path: Path) -> dict[str, tuple[int, int, np.ndarray]]:
    """Physical name -> (dimension, tag, node coordinates in um)."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(mesh_path))
        groups = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            _, coords = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
            groups[name] = (dim, tag, np.reshape(coords, (-1, 3)))
        return groups
    finally:
        gmsh.finalize()


def _sheet_at(groups, port_centres) -> int:
    """The tag of the one surface group whose nodes all lie between two ports.

    A source sheet bridges the gap between its two side-by-side leads: its
    nodes span the segment between the port centres in the plane, at the
    ports' y.
    """
    (x1, y1), (x2, _) = port_centres
    matches = [
        tag
        for dim, tag, nodes in groups.values()
        if dim == 2
        and np.allclose(nodes[:, 1], y1, atol=1e-6)
        and np.all(nodes[:, 0] >= min(x1, x2) - 1e-6)
        and np.all(nodes[:, 0] <= max(x1, x2) + 1e-6)
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _direction(plus: tuple[float, float], minus: tuple[float, float]) -> str:
    """Palace ``Direction`` of the sheet current for a source between two leads.

    The unit current enters the coil at the plus lead, so on the sheet that
    closes the loop it flows from the minus lead towards the plus lead. The
    leads here sit side by side along x.
    """
    return "+X" if plus[0] > minus[0] else "-X"


def _standalone_config(groups, ports, mesh_name: str) -> dict:
    """A magnetostatic Palace config written from the Palace docs.

    - ``Problem.Type: "Magnetostatic"``; results in ``Problem.Output``.
    - ``Model.L0 = 1e-6``: mesh coordinates are in um.
    - Every volume is non-magnetic (``Permeability`` 1).
    - ``Boundaries.PEC``: the metal surfaces (``topmetal*``) and the outer
      wall of the air box (the air face with no neighbour, ``air__None``).
    - ``Boundaries.SurfaceCurrent``: one entry per source, numbered in source
      order, on its sheet, with the current direction on the sheet.
    - ``Solver``: Order 1, the default AMS + CG, tolerance 1e-8.
    """
    volumes = [tag for dim, tag, _ in groups.values() if dim == 3]
    pec = sorted(
        tag
        for name, (dim, tag, _) in groups.items()
        if dim == 2 and (name.startswith("topmetal") or name == "air__None")
    )
    currents = []
    for index, source in enumerate(SOURCES, start=1):
        plus, minus = ports[f"{source}_P1"], ports[f"{source}_P2"]
        currents.append(
            {
                "Index": index,
                "Attributes": [_sheet_at(groups, (plus, minus))],
                "Direction": _direction(plus, minus),
            }
        )
    return {
        "Problem": {"Type": "Magnetostatic", "Verbose": 2, "Output": "postpro"},
        "Model": {"Mesh": mesh_name, "L0": 1e-6},
        "Domains": {
            "Materials": [{"Attributes": volumes, "Permeability": 1.0}],
        },
        "Boundaries": {"PEC": {"Attributes": pec}, "SurfaceCurrent": currents},
        "Solver": {
            "Order": 1,
            "Linear": {"Type": "AMS", "KSPType": "CG", "Tol": 1e-8, "MaxIts": 400},
        },
    }


@pytest.fixture(scope="module")
def standalone(palace_binary, forward, tmp_path_factory) -> InductanceMatrix:
    """The hand-written config run on gsim's mesh of the stacked pair."""
    from gsim.palace.runtime import resolve_launch, run_palace

    _, sim, _ = forward
    run_dir = tmp_path_factory.mktemp("standalone")
    mesh = shutil.copy(sim.mesh_path, run_dir / "pair.msh")
    ports = {
        port.name: (float(port.center[0]), float(port.center[1]))
        for port in sim.component.ports
    }
    config = _standalone_config(_physical_groups(Path(mesh)), ports, "pair.msh")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    postpro = run_dir / "postpro"
    postpro.mkdir()
    run_palace(
        resolve_launch(executable=palace_binary),
        cwd=run_dir,
        num_processes=1,
        num_threads=4,
        verbose=False,
        emit=lambda _message: None,
        log_path=postpro / "palace.log",
    )
    return _load(
        lambda results, **kw: load_inductance(results, source_names=SOURCES, **kw),
        postpro,
    )


def test_gsim_matches_a_hand_written_palace_config(forward, standalone) -> None:
    gsim_matrix = forward[0].matrix
    reference = standalone.matrix
    scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
    assert np.all(np.abs(gsim_matrix - reference) <= 1e-2 * scale), (
        gsim_matrix,
        reference,
    )


def test_the_stacked_pair_couples_positively(forward) -> None:
    assert 0.2 < forward[0].coupling("A", "B") < 0.5  # k ~ 0.34 (05 sec. 4.3)


# ---------------------------------------------------------------------------
# (e2) reversing a source
# ---------------------------------------------------------------------------


def test_reversing_source_b_flips_the_mutual_and_keeps_the_selfs(
    palace_binary, forward, tmp_path_factory
) -> None:
    original, _, mesh_hash = forward
    reversed_, _, reversed_hash = _run_gsim(
        palace_binary, tmp_path_factory.mktemp("reversed"), reverse_b=True
    )
    assert reversed_hash == mesh_hash
    assert abs(reversed_.coupling("A", "B") + original.coupling("A", "B")) <= 1e-6
    for source in SOURCES:
        assert reversed_.self_inductance(source) == pytest.approx(
            original.self_inductance(source), rel=1e-6, abs=0
        )
