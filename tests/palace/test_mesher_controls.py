"""Tests for the Gmsh mesher controls: the 3D algorithm and the thread counts.

The thread counts and the 3D algorithm decide which mesh Gmsh produces (see
gsim#283), so they are separate from the Palace solver settings. The defaults
keep meshing single-threaded with Delaunay, as before.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace

import gdsfactory as gf
import gmsh
import pytest
from pydantic import ValidationError

from gsim.palace import DrivenSim
from gsim.palace.mesh import generator
from gsim.palace.mesh.generator import apply_mesher_options
from gsim.palace.models import MeshConfig

GMSH_LOGGER = "gsim.palace.mesh.generator"


@pytest.fixture
def _gmsh_session() -> Iterator[None]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    gmsh.finalize()


def _cpw() -> gf.Component:
    """A small GSG electrode, as in the mesh integration tests."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def gsg_electrode(length: float = 500) -> gf.Component:
        layer = gf.gpdk.LAYER.M1
        c = gf.Component()
        r1 = c << gf.c.rectangle((length, 50), centered=True, layer=layer)
        r1.move((0, 38))
        c << gf.c.rectangle((length, 10), centered=True, layer=layer)
        r3 = c << gf.c.rectangle((length, 50), centered=True, layer=layer)
        r3.move((0, -38))
        for name, x, angle in (("o1", -length / 2, 0), ("o2", length / 2, 180)):
            c.add_port(
                name=name,
                center=(x, 0),
                width=10,
                orientation=angle,
                port_type="electrical",
                layer=layer,
            )
        return c

    return gsg_electrode()


def _sim(tmp_path, name: str = "palace-sim") -> DrivenSim:
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / name))
    sim.set_geometry(_cpw())
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    for port in ("o1", "o2"):
        sim.add_cpw_port(port, layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
    return sim


def _options(result) -> dict[str, float]:
    return result.mesh_stats["gmsh_options"]


def test_defaults_keep_meshing_single_threaded_with_delaunay() -> None:
    config = MeshConfig()
    assert config.algorithm_3d == "delaunay"
    assert config.threads == 1
    assert config.surface_threads == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threads": 0},
        {"surface_threads": 0},
        {"threads": -2},
        {"algorithm_3d": "octree"},
    ],
)
def test_invalid_values_are_rejected(kwargs) -> None:
    with pytest.raises(ValidationError):
        MeshConfig(**kwargs)


@pytest.mark.usefixtures("_gmsh_session")
@pytest.mark.parametrize(("algorithm", "code"), [("delaunay", 1), ("hxt", 10)])
def test_apply_sets_the_gmsh_options(algorithm: str, code: int) -> None:
    apply_mesher_options(algorithm_3d=algorithm, threads=4, surface_threads=2)
    assert gmsh.option.getNumber("Mesh.Algorithm3D") == code
    assert gmsh.option.getNumber("General.NumThreads") == 4
    assert gmsh.option.getNumber("Mesh.MaxNumThreads1D") == 2
    assert gmsh.option.getNumber("Mesh.MaxNumThreads2D") == 2
    assert gmsh.option.getNumber("Mesh.MaxNumThreads3D") == 4


@pytest.mark.usefixtures("_gmsh_session")
@pytest.mark.parametrize(
    ("algorithm", "threads", "surface_threads"),
    [("delaunay", 1, 1), ("delaunay", 8, 1), ("hxt", 1, 1)],
)
def test_combinations_that_keep_the_mesh_do_not_warn(
    caplog, algorithm: str, threads: int, surface_threads: int
) -> None:
    with caplog.at_level(logging.WARNING, logger=GMSH_LOGGER):
        apply_mesher_options(
            algorithm_3d=algorithm, threads=threads, surface_threads=surface_threads
        )
    assert not caplog.records


@pytest.mark.usefixtures("_gmsh_session")
def test_parallel_surface_meshing_warns_that_the_mesh_is_not_reproducible(
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING, logger=GMSH_LOGGER):
        apply_mesher_options(algorithm_3d="delaunay", threads=4, surface_threads=4)
    assert len(caplog.records) == 1
    assert "surface_threads=4" in caplog.text
    assert "reproducible" in caplog.text


@pytest.mark.usefixtures("_gmsh_session")
def test_hxt_with_several_threads_warns_that_the_mesh_depends_on_the_count(
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING, logger=GMSH_LOGGER):
        apply_mesher_options(algorithm_3d="hxt", threads=4, surface_threads=1)
    assert len(caplog.records) == 1
    assert "threads=4" in caplog.text
    assert "number of threads" in caplog.text


def test_sim_mesh_applies_and_records_the_controls(tmp_path) -> None:
    result = _sim(tmp_path).mesh(
        preset="coarse", algorithm_3d="delaunay", threads=2, surface_threads=1
    )
    options = _options(result)
    assert options["Mesh.Algorithm3D"] == 1
    assert options["General.NumThreads"] == 2
    assert options["Mesh.MaxNumThreads3D"] == 2
    assert options["Mesh.MaxNumThreads1D"] == 1
    assert options["Mesh.MaxNumThreads2D"] == 1


def test_sim_mesh_defaults_are_single_threaded(tmp_path) -> None:
    options = _options(_sim(tmp_path).mesh(preset="coarse"))
    assert options["Mesh.Algorithm3D"] == 1
    assert options["General.NumThreads"] == 1
    for dim in (1, 2, 3):
        assert options[f"Mesh.MaxNumThreads{dim}D"] == 1


def test_a_configured_sim_keeps_its_controls_unless_overridden(tmp_path) -> None:
    sim = _sim(tmp_path)
    sim.mesh_config = MeshConfig(threads=2)
    assert _options(sim.mesh(preset="coarse"))["General.NumThreads"] == 2
    assert _options(sim.mesh(preset="coarse", threads=1))["General.NumThreads"] == 1


def test_delaunay_with_serial_surfaces_gives_the_same_mesh_for_any_thread_count(
    tmp_path,
) -> None:
    """The documented way to get the same mesh whatever the thread count."""
    hashes = {
        threads: _sim(tmp_path, f"sim-{threads}")
        .mesh(preset="coarse", threads=threads)
        .mesh_stats["mesh_hash"]
        for threads in (1, 4)
    }
    assert hashes[1] == hashes[4]


def test_preview_forwards_the_controls(monkeypatch, tmp_path) -> None:
    """preview() takes the same mesher options as mesh()."""
    captured = {}

    def fake_generate_mesh(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(generator, "generate_mesh", fake_generate_mesh)
    _sim(tmp_path).preview(
        preset="coarse", algorithm_3d="hxt", threads=3, surface_threads=2
    )
    assert captured["algorithm_3d"] == "hxt"
    assert captured["threads"] == 3
    assert captured["surface_threads"] == 2


def test_options_go_through_the_gmsh_that_generate_mesh_uses(monkeypatch) -> None:
    """generate_mesh sets them on its own gmsh, so tests that fake it keep working."""
    calls: list[tuple[str, float]] = []
    fake = SimpleNamespace(
        option=SimpleNamespace(
            setNumber=lambda name, value: calls.append((name, value))
        )
    )
    monkeypatch.setattr(generator, "gmsh", fake)
    apply_mesher_options(algorithm_3d="hxt", threads=3, surface_threads=2)
    assert calls == [
        ("Mesh.Algorithm3D", 10),
        ("General.NumThreads", 3),
        ("Mesh.MaxNumThreads1D", 2),
        ("Mesh.MaxNumThreads2D", 2),
        ("Mesh.MaxNumThreads3D", 3),
    ]
