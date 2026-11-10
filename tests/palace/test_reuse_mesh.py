"""Reusing the mesh of one simulation in another (gsim#389), without Gmsh.

The source is meshed by the real ``mesh()``: real ``validate_config``, real
port extraction and the fingerprint ``mesh()`` records. Only
``generate_mesh`` is replaced, by a fake that writes a small ``.msh`` and
returns a real ``MeshResult``. Once the source has meshed, the fake can be
told to fail, which shows that ``reuse_mesh`` never meshes.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from unittest.mock import Mock

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim import gcloud
from gsim.common.stack import LayerStack
from gsim.hashing import compute_dir_digest, compute_input_hash
from gsim.palace import BoundaryModeSim, DrivenSim, EigenmodeSim, ElectrostaticSim
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.metadata import update_field_dofs, write_metadata
from gsim.palace.models.results import SimulationResult
from tests._electrode_layouts import stack as generic_stack

# o1 and o2: lumped ports on m1 at either end of the m1 strip
TWO_PORTS = {
    "o1": {"layer": "m1", "length": 1.0},
    "o2": {"layer": "m1", "length": 1.0},
}
NODES, TETS = 1234, 4321


class FakeGmsh:
    """Stands in for ``generate_mesh``; counts its calls.

    Writes ``<model_name>.msh`` (bytes that change with every call) and
    ``metadata.json``, and returns a real ``MeshResult`` with one ``P{i}``
    port surface per extracted port, of the port's type (lumped or wave).
    """

    def __init__(self) -> None:
        """Start with no calls, meshing allowed."""
        self.calls = 0
        self.allowed = True

    def __call__(self, **kwargs) -> MeshResult:
        """Write the fake mesh files and return what Gmsh would."""
        if not self.allowed:
            raise AssertionError("generate_mesh was called")
        self.calls += 1
        output_dir = Path(kwargs["output_dir"])
        model_name = kwargs["model_name"]
        ports = kwargs["ports"]

        mesh_path = output_dir / f"{model_name}.msh"
        mesh_path.write_bytes(
            f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n% fake {self.calls}\n".encode()
        )
        groups: dict = {
            "volumes": {"airbox": {"phys_group": 1}, "sio2": {"phys_group": 2}},
            "conductor_surfaces": {"m1_xy": {"phys_group": 3}},
            "port_surfaces": {
                f"P{i}": {"phys_group": 10 + i, "type": port.port_type.value}
                for i, port in enumerate(ports, start=1)
            },
            "boundary_surfaces": {"absorbing": {"phys_group": 4}},
        }
        symmetry = None
        if (plane := kwargs.get("symmetry_plane")) is not None:
            symmetry = {
                "phys_group": 5,
                "axis": plane.axis,
                "position": plane.position,
                "kind": plane.kind,
                "keep": plane.keep,
            }
            groups["boundary_surfaces"]["symmetry"] = dict(symmetry)
        periodic_axis = kwargs.get("periodic_axis")
        if periodic_axis is not None:
            groups["boundary_surfaces"]["periodic_donor"] = {"phys_group": 20}
            groups["boundary_surfaces"]["periodic_receiver"] = {"phys_group": 21}
        port_info = [
            {
                "portnumber": i,
                "name": port.name,
                "Z0": port.impedance,
                "type": port.port_type.value,
            }
            for i, port in enumerate(ports, start=1)
        ]
        stats = {"nodes": NODES, "elements": 5000, "tetrahedra": TETS}
        numerical = kwargs.get("numerical_config")
        update_field_dofs(
            stats,
            field_order=numerical.order if numerical else 2,
            problem_type=kwargs["simulation_type"],
        )
        return MeshResult(
            mesh_path=mesh_path,
            port_info=port_info,
            mesh_stats=stats,
            metadata=write_metadata(stats, output_dir, None),
            groups=groups,
            output_dir=output_dir,
            model_name=model_name,
            fmax=kwargs["fmax"],
            periodic_axis=periodic_axis,
            periodic_translation=None if periodic_axis is None else (10.0, 0.0, 0.0),
            symmetry_plane=symmetry,
        )


@pytest.fixture
def stack() -> LayerStack:
    """Conductors m1 and m2, a conductive via v1, and an oxide at eps 3.9."""
    return generic_stack()


def _layout(m1_height: float = 1.0) -> gf.Component:
    """A strip on m1 and one on m2, with ports o1 and o2 at the ends of m1."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, m1_height), (0, m1_height)], layer=(1, 0))
    c.add_polygon([(0, 3), (10, 3), (10, 4), (0, 4)], layer=(2, 0))
    for name, x, orientation in (("o1", 0, 180), ("o2", 10, 0)):
        c.add_port(
            name=name,
            center=(x, 0.5),
            width=1.0,
            orientation=orientation,
            layer=(1, 0),
            port_type="electrical",
        )
    return c


@pytest.fixture
def component() -> gf.Component:
    """A strip on m1 and one on m2, with ports o1 and o2 at the ends of m1."""
    return _layout()


@pytest.fixture
def fake_gmsh(monkeypatch) -> FakeGmsh:
    """Replace Gmsh meshing with :class:`FakeGmsh`."""
    fake = FakeGmsh()
    monkeypatch.setattr("gsim.palace.mesh.generator.generate_mesh", fake)
    return fake


@pytest.fixture
def make_sim(stack, component):
    """Build a driven or eigenmode sim of the layout, ready to mesh.

    Each sim gets its own copy of the component and the stack. ``ports`` maps
    each declared lumped port to its ``add_port`` settings.
    """

    def _make(cls, folder: Path | None = None, ports=TWO_PORTS):
        sim = cls(absorbing_boundary=False)
        if folder is not None:
            sim.set_output_dir(folder)
        sim.set_geometry(component.copy())
        sim.set_stack(stack.model_copy(deep=True))
        for name, settings in ports.items():
            sim.add_port(name, **settings)
        if cls is EigenmodeSim:
            sim.set_eigenmode(target=5e9)
        else:
            sim.set_driven(fmin=7e9, fmax=10e9)
        return sim

    return _make


@pytest.fixture
def meshed_eigen(make_sim, fake_gmsh, tmp_path) -> EigenmodeSim:
    """An eigenmode sim with two ports, meshed by the real ``mesh()``."""
    sim = make_sim(EigenmodeSim, tmp_path / "eigen")
    sim.mesh(verbose=False)
    assert fake_gmsh.calls == 1
    return sim


def _without_dofs(stats: dict) -> dict:
    """Mesh stats without the DOF estimate, which depends on the problem type."""
    return {k: v for k, v in stats.items() if k != "field_dofs"}


def _listing(folder: Path) -> list[str]:
    """Every path under a folder, relative to it."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


_STATE = (
    "geometry",
    "stack",
    "materials",
    "mesh_config",
    "ports",
    "cpw_ports",
    "wave_ports",
    "two_terminal_ports",
    "_stack_kwargs",
    "_airbox_config",
    "_pec_blocks",
    "_symmetry_planes",
    "_last_mesh_result",
    "_last_ports",
    "_mesh_fingerprint",
    "_output_dir",
)


def _state(sim) -> dict:
    """The objects a sim holds, and the values that could change in place."""
    return {
        "objects": {name: id(getattr(sim, name)) for name in _STATE},
        "ports": [p.model_dump() for p in sim.ports],
        "last_ports": copy.deepcopy(sim._last_ports),
        "mesh_stats": copy.deepcopy(sim._last_mesh_result.mesh_stats),
        "fingerprint": dict(sim._mesh_fingerprint or {}),
        "flags": (sim._mesh_reused, sim._configured_ports),
    }


def _excitations(config: dict) -> list:
    """The ``Excitation`` of each lumped port of a config, in index order."""
    return [port["Excitation"] for port in config["Boundaries"]["LumpedPort"]]


# ----- case 1: the issue's eigenmode-to-driven workflow ------------------------------


def test_one_eigenmode_mesh_drives_two_receivers_with_their_own_excitations(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    """gsim#389: independent excitations and reflecting outer walls, no Gmsh."""
    source_config = json.loads(
        meshed_eigen.write_config(validate_mesh=False).read_text()
    )
    source_digest = compute_dir_digest(tmp_path / "eigen")
    fake_gmsh.allowed = False

    configs = {}
    for excite, other in (("o1", "o2"), ("o2", "o1")):
        receiver = make_sim(
            DrivenSim,
            tmp_path / excite,
            ports={
                excite: {"layer": "m1", "length": 1.0, "excited": True},
                other: {"layer": "m1", "length": 1.0, "excited": False},
            },
        )
        receiver.set_driven(fmin=7e9, fmax=10e9, excitation_port=excite)
        receiver.reuse_mesh(meshed_eigen)
        config_path = receiver.write_config(validate_mesh=False)
        assert config_path == tmp_path / excite / "config.json"
        configs[excite] = json.loads(config_path.read_text())

    assert fake_gmsh.calls == 1
    assert _excitations(configs["o1"]) == [1, False]
    assert _excitations(configs["o2"]) == [False, 2]
    for config in configs.values():
        assert config["Problem"]["Type"] == "Driven"
        # absorbing_boundary=False: no Absorbing block and no outer PEC wall
        assert "Absorbing" not in config["Boundaries"]
        assert "PEC" not in config["Boundaries"]
    # the two configs differ only in which port is excited
    for config in configs.values():
        for port in config["Boundaries"]["LumpedPort"]:
            del port["Excitation"]
    assert configs["o1"] == configs["o2"]

    # the source keeps its config and its folder
    assert compute_dir_digest(tmp_path / "eigen") == source_digest
    assert source_config["Problem"]["Type"] == "Eigenmode"
    assert _excitations(source_config) == [False, False]
    hashes = {
        compute_input_hash(tmp_path / folder, "palace")
        for folder in ("eigen", "o1", "o2")
    }
    assert len(hashes) == 3


# ----- case 2: the four allowed pairs ------------------------------------------------


@pytest.mark.parametrize("source_cls", [DrivenSim, EigenmodeSim])
@pytest.mark.parametrize("receiver_cls", [DrivenSim, EigenmodeSim])
def test_all_four_pairs_return_the_receivers_mesh(
    source_cls, receiver_cls, make_sim, fake_gmsh, tmp_path, caplog
) -> None:
    source = make_sim(source_cls, tmp_path / "source")
    source.mesh(verbose=False)
    receiver = make_sim(receiver_cls, tmp_path / "receiver")
    fake_gmsh.allowed = False
    caplog.set_level(logging.INFO)
    caplog.clear()

    result = receiver.reuse_mesh(source)

    src = source._last_mesh_result
    own = receiver._last_mesh_result
    assert isinstance(result, SimulationResult)
    assert result.mesh_path == tmp_path / "receiver" / "palace.msh"
    assert result.output_dir == tmp_path / "receiver"
    assert result.config_path is None
    assert receiver.has_mesh
    assert receiver.mesh_path == result.mesh_path
    assert receiver.mesh_groups == src.groups
    assert own is not src
    assert own.config_path is None
    # equal to the source's, but copies
    assert result.port_info == src.port_info
    assert own.port_info is not src.port_info
    assert own.port_info[0] is not src.port_info[0]
    assert _without_dofs(result.mesh_stats) == _without_dofs(src.mesh_stats)
    assert own.mesh_stats is not src.mesh_stats
    assert result.mesh_stats["field_dofs"]["problem_type"] == receiver.simulation_type
    assert result.metadata == {
        "schema_version": 1,
        "solver": "palace",
        "mesh": result.mesh_stats,
    }
    assert _without_dofs(result.metadata["mesh"]) == _without_dofs(src.metadata["mesh"])
    assert own.metadata is not src.metadata
    assert receiver._mesh_reused is True
    assert receiver._mesh_fingerprint == source._mesh_fingerprint
    assert receiver._mesh_fingerprint is not source._mesh_fingerprint
    reuse_lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "Reusing mesh" in r.getMessage()
    ]
    assert reuse_lines == [
        f"Reusing mesh of {source_cls.__name__} ({src.mesh_path}): "
        f"{NODES:,} nodes \u00b7 {TETS:,} tets"
    ]


# ----- case 3: refusals --------------------------------------------------------------


@pytest.mark.parametrize("cls", [ElectrostaticSim, BoundaryModeSim])
def test_reuse_refuses_a_receiver_of_another_class(cls, meshed_eigen, tmp_path) -> None:
    receiver = cls()
    receiver.set_output_dir(tmp_path / "other")

    with pytest.raises(ValueError, match=rf"{cls.__name__}.*mesh\(\)"):
        receiver.reuse_mesh(meshed_eigen)


@pytest.mark.parametrize("cls", [ElectrostaticSim, BoundaryModeSim])
def test_reuse_refuses_a_source_of_another_class(cls, make_sim, tmp_path) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    with pytest.raises(ValueError, match=rf"{cls.__name__}.*mesh\(\)"):
        receiver.reuse_mesh(cls())


@pytest.mark.parametrize("kind", ["str", "path", "mesh_result"])
def test_reuse_refuses_a_file_or_a_mesh_result(
    kind, meshed_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    mesh_result = meshed_eigen._last_mesh_result
    source = {
        "str": str(mesh_result.mesh_path),
        "path": Path(mesh_result.mesh_path),
        "mesh_result": mesh_result,
    }[kind]

    with pytest.raises(TypeError, match="physical-group or port mappings"):
        receiver.reuse_mesh(source)


def test_reuse_refuses_an_unmeshed_source(make_sim, tmp_path) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    with pytest.raises(
        ValueError,
        match=r"The source EigenmodeSim has no mesh\. Call mesh\(\) on it first\.",
    ):
        receiver.reuse_mesh(source)


def test_reuse_refuses_its_own_mesh(meshed_eigen) -> None:
    with pytest.raises(ValueError, match="its own mesh"):
        meshed_eigen.reuse_mesh(meshed_eigen)


def test_reuse_refuses_a_receiver_without_output_dir(meshed_eigen, make_sim) -> None:
    receiver = make_sim(DrivenSim)

    with pytest.raises(
        ValueError,
        match=r"Output directory not set\. Call set_output_dir\(\) first\.",
    ):
        receiver.reuse_mesh(meshed_eigen)


def test_reuse_refuses_a_driven_receiver_of_a_mesh_without_ports(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """R1: a driven sim can never gain a port surface on a reused mesh."""
    source = make_sim(EigenmodeSim, tmp_path / "eigen", ports={})
    source.mesh(verbose=False)
    fake_gmsh.allowed = False

    with pytest.raises(ValueError, match="no port surfaces"):
        make_sim(DrivenSim, tmp_path / "driven", ports={}).reuse_mesh(source)

    # an eigenmode receiver needs no port
    receiver = make_sim(EigenmodeSim, tmp_path / "eigen2", ports={})
    receiver.reuse_mesh(source)
    assert receiver.has_mesh


# ----- case 4: a refused call changes nothing ----------------------------------------


def _same_folder(source, receiver, _tmp_path) -> None:
    receiver.set_output_dir(source.output_dir)


def _same_folder_once_resolved(_source, receiver, tmp_path) -> None:
    receiver.set_output_dir(tmp_path / "driven" / ".." / "eigen")


def _folder_inside_the_source_folder(source, receiver, _tmp_path) -> None:
    receiver.set_output_dir(source.output_dir / "driven")


def _folder_containing_the_source_folder(_source, receiver, tmp_path) -> None:
    receiver.set_output_dir(tmp_path)


def _missing_source_mesh_file(source, _receiver, _tmp_path) -> None:
    source.mesh_path.unlink()


def _receiver_only_port(_source, receiver, _tmp_path) -> None:
    receiver.add_port("o3", layer="m1", length=1.0)


def _port_of_another_kind(_source, receiver, _tmp_path) -> None:
    receiver.ports = [p for p in receiver.ports if p.name != "o1"]
    receiver.add_cpw_port("o1", layer="m1", s_width=1.0, gap_width=1.0)


def _port_of_another_geometry(_source, receiver, _tmp_path) -> None:
    receiver.add_port("o1", layer="m2", length=1.0)


def _source_edited_after_mesh(source, _receiver, _tmp_path) -> None:
    source.component.add_polygon([(0, 6), (2, 6), (2, 7), (0, 7)], layer=(1, 0))


def _source_without_fingerprint(source, _receiver, _tmp_path) -> None:
    source._mesh_fingerprint = None


def _mesh_without_port_surfaces(source, _receiver, _tmp_path) -> None:
    source._last_mesh_result.groups["port_surfaces"] = {}


@pytest.mark.parametrize(
    ("cause", "error", "match"),
    [
        (_same_folder, ValueError, "is the output directory"),
        (_same_folder_once_resolved, ValueError, "is the output directory"),
        (_folder_inside_the_source_folder, ValueError, "is inside the output"),
        (_folder_containing_the_source_folder, ValueError, "contains the output"),
        (_missing_source_mesh_file, FileNotFoundError, "does not exist"),
        (_receiver_only_port, ValueError, "'o3' has no surface in the reused mesh"),
        (_port_of_another_kind, ValueError, "Port 'o1' .*does not match"),
        (_port_of_another_geometry, ValueError, "Port 'o1' .*does not match"),
        (
            _source_edited_after_mesh,
            ValueError,
            r"changed after its mesh was made \(geometry \(layout polygons\)\)",
        ),
        (_source_without_fingerprint, ValueError, "has no record of the inputs"),
        (_mesh_without_port_surfaces, ValueError, "no port surfaces"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_refused_call_leaves_the_receiver_unchanged(
    cause, error, match, make_sim, fake_gmsh, tmp_path
) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    cause(source, receiver, tmp_path)
    fake_gmsh.allowed = False
    before = _state(receiver)
    listing = _listing(receiver.output_dir)

    with pytest.raises(error, match=match):
        receiver.reuse_mesh(source)

    assert _state(receiver) == before
    # no file written, and no staging folder left behind
    assert _listing(receiver.output_dir) == listing


def _case_blind_samefile(a, b) -> bool:
    """``os.path.samefile`` as a case-insensitive file system answers it."""
    return str(Path(a)).casefold() == str(Path(b)).casefold()


@pytest.mark.parametrize(
    ("folder", "match"),
    [
        (("OUTER", "EIGEN"), "is the output directory"),
        (("OUTER", "EIGEN", "driven"), "is inside the output"),
        (("OUTER",), "contains the output"),
    ],
    ids=["same", "inside", "contains"],
)
def test_reuse_refuses_the_source_folder_spelled_another_way(
    folder, match, make_sim, fake_gmsh, monkeypatch, tmp_path
) -> None:
    """``./EIGEN`` is ``./eigen`` on macOS and Windows, though not equal paths."""
    source = make_sim(EigenmodeSim, tmp_path / "outer" / "eigen")
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    receiver.set_output_dir(tmp_path.joinpath(*folder))
    fake_gmsh.allowed = False
    before = _state(receiver)
    source_listing = _listing(source.output_dir)
    monkeypatch.setattr(os.path, "samefile", _case_blind_samefile)

    with pytest.raises(ValueError, match=match):
        receiver.reuse_mesh(source)

    assert _state(receiver) == before
    assert _listing(source.output_dir) == source_listing


# ----- case 5: last call wins, chaining ----------------------------------------------


def test_last_call_wins(make_sim, fake_gmsh, tmp_path) -> None:
    first = make_sim(EigenmodeSim, tmp_path / "first")
    first.mesh(verbose=False)
    second = make_sim(DrivenSim, tmp_path / "second")
    second.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    own_bytes = receiver.mesh_path.read_bytes()
    fake_gmsh.allowed = False

    receiver.reuse_mesh(first)
    assert receiver.mesh_path.read_bytes() == first.mesh_path.read_bytes()
    assert receiver.mesh_path.read_bytes() != own_bytes
    assert receiver._mesh_reused is True

    receiver.reuse_mesh(second)
    assert receiver.mesh_path.read_bytes() == second.mesh_path.read_bytes()
    assert receiver._last_mesh_result.mesh_stats["field_dofs"]["problem_type"] == (
        "driven"
    )


def test_chain_eigen_to_driven_to_driven(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    driven = make_sim(DrivenSim, tmp_path / "driven")
    driven.reuse_mesh(meshed_eigen)
    driven2 = make_sim(DrivenSim, tmp_path / "driven2")

    driven2.reuse_mesh(driven)
    config_path = driven2.write_config(validate_mesh=False)

    assert config_path == tmp_path / "driven2" / "config.json"
    config = json.loads(config_path.read_text())
    assert config["Problem"]["Type"] == "Driven"
    assert [p["Excitation"] for p in config["Boundaries"]["LumpedPort"]] == [1, 2]
    assert driven2.mesh_path.read_bytes() == meshed_eigen.mesh_path.read_bytes()
    # the chain carries the fingerprint of the original mesh
    assert driven2._mesh_fingerprint == meshed_eigen._mesh_fingerprint


# ----- case 6: mesh() after reuse, source re-mesh ------------------------------------


def test_mesh_after_reuse_remeshes_the_receiver(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    calls = fake_gmsh.calls

    receiver.mesh(verbose=False)

    assert fake_gmsh.calls == calls + 1
    assert receiver._mesh_reused is False
    assert receiver._mesh_fingerprint == receiver._current_mesh_fingerprint(
        receiver.stack
    )
    assert receiver.mesh_path.read_bytes() != meshed_eigen.mesh_path.read_bytes()


def test_source_remesh_leaves_the_receiver_intact(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    snapshot = receiver.mesh_path.read_bytes()
    groups = copy.deepcopy(receiver.mesh_groups)
    calls = fake_gmsh.calls

    meshed_eigen.mesh(verbose=False)

    assert fake_gmsh.calls == calls + 1
    assert meshed_eigen.mesh_path.read_bytes() != snapshot
    assert receiver.mesh_path.read_bytes() == snapshot
    assert receiver.mesh_groups == groups
    assert receiver._mesh_reused is True
    assert receiver.write_config(validate_mesh=False).parent == tmp_path / "driven"


# ----- case 7: nothing is shared -----------------------------------------------------


def test_nothing_is_shared_after_reuse(make_sim, fake_gmsh, tmp_path) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.set_material("sio2", permittivity=3.9)
    source.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=0)
    source.add_pec(gds_layer=(9, 0), from_layer="m1", to_layer="m2")
    source.add_symmetry_plane(axis="x", position=5.0, verify_symmetry=False)
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    fake_gmsh.allowed = False

    receiver.reuse_mesh(source)

    pairs = [
        (receiver.geometry, source.geometry),
        (receiver.component, source.component),
        (receiver.stack, source.stack),
        (receiver.stack.layers["m1"], source.stack.layers["m1"]),
        (receiver.stack.materials, source.stack.materials),
        (receiver.materials, source.materials),
        (receiver.materials["sio2"], source.materials["sio2"]),
        (receiver._stack_kwargs, source._stack_kwargs),
        (receiver._airbox_config, source._airbox_config),
        (receiver.mesh_config, source.mesh_config),
        (receiver._pec_blocks[0], source._pec_blocks[0]),
        (receiver._symmetry_planes[0], source._symmetry_planes[0]),
        *zip(receiver.ports, source.ports, strict=True),
        *zip(receiver._last_ports, source._last_ports, strict=True),
    ]
    own, src = receiver._last_mesh_result, source._last_mesh_result
    pairs += [
        (own, src),
        (own.port_info, src.port_info),
        (own.port_info[0], src.port_info[0]),
        (own.mesh_stats, src.mesh_stats),
        (own.metadata, src.metadata),
        (own.groups, src.groups),
        (own.groups["port_surfaces"], src.groups["port_surfaces"]),
        (own.symmetry_plane, src.symmetry_plane),
    ]
    for mine, theirs in pairs:
        assert mine is not theirs
    assert own.symmetry_plane == src.symmetry_plane
    assert receiver._airbox_config == source._airbox_config

    # the receiver changes; the source does not see it
    source_stats = copy.deepcopy(src.mesh_stats)
    source_fingerprint = dict(source._mesh_fingerprint)
    receiver.set_material("sio2", permittivity=4.1)
    receiver.ports[0].impedance = 25.0
    receiver.ports[0].excited = True
    receiver.write_config(validate_mesh=False)
    receiver.component.add_polygon([(0, 6), (2, 6), (2, 7), (0, 7)], layer=(1, 0))
    assert source.materials["sio2"].permittivity == 3.9
    assert source.ports[0].excited is False
    assert source.stack.materials["sio2"]["permittivity"] == 3.9
    assert source.ports[0].impedance == 50.0
    assert src.mesh_stats == source_stats
    assert source._current_mesh_fingerprint(source.stack) == source_fingerprint

    # the source changes; the receiver does not see it
    source.ports[1].impedance = 75.0
    source.stack.layers["m1"].zmax = 1.5
    source._airbox_config["margin_x"] = 9.0
    src.groups["volumes"]["extra"] = {"phys_group": 99}
    assert receiver.ports[1].impedance == 50.0
    assert receiver.stack.layers["m1"].zmax == 1.0
    assert receiver._airbox_config["margin_x"] == 5
    assert "extra" not in receiver.mesh_groups["volumes"]


# ----- case 8: port merge ------------------------------------------------------------


def test_port_merge_copies_undeclared_ports_and_keeps_declared_ones(
    meshed_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(
        DrivenSim,
        tmp_path / "driven",
        ports={"o2": {"layer": "m1", "length": 1.0, "impedance": 25.0}},
    )
    own_o2 = receiver.ports[0]

    receiver.reuse_mesh(meshed_eigen)

    # in the source's order; o1 is a copy of the source's, o2 the receiver's own
    assert [p.name for p in receiver.ports] == ["o1", "o2"]
    assert receiver.ports[1] is own_o2
    assert own_o2.impedance == 25.0
    assert own_o2.excited is True
    copied = receiver.ports[0]
    assert copied is not meshed_eigen.ports[0]
    assert copied.model_dump() == meshed_eigen.ports[0].model_dump()
    assert copied.excited is False  # the eigenmode source's flag
    # the port settings reach the receiver's own extracted ports and Z0
    by_name = {p.name: p for p in receiver._last_ports}
    assert (by_name["o1"].excited, by_name["o1"].impedance) == (False, 50.0)
    assert (by_name["o2"].excited, by_name["o2"].impedance) == (True, 25.0)
    z0 = {e["name"]: e["Z0"] for e in receiver._last_mesh_result.port_info}
    assert z0 == {"o1": 50.0, "o2": 25.0}
    # the source keeps its own
    assert [(p.excited, p.impedance) for p in meshed_eigen._last_ports] == [
        (False, 50.0),
        (False, 50.0),
    ]
    assert [e["Z0"] for e in meshed_eigen._last_mesh_result.port_info] == [50.0, 50.0]


def test_port_merge_gives_an_empty_receiver_the_source_ports(
    meshed_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})

    receiver.reuse_mesh(meshed_eigen)

    assert [p.model_dump() for p in receiver.ports] == [
        p.model_dump() for p in meshed_eigen.ports
    ]
    assert not any(p.excited for p in receiver.ports)
    # equal declarations: the settings sync changes nothing
    assert receiver._last_ports == meshed_eigen._last_ports
    assert (
        receiver._last_mesh_result.port_info == meshed_eigen._last_mesh_result.port_info
    )


def test_reuse_again_takes_a_drifted_port_geometry_back_and_keeps_its_settings(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    receiver.reuse_mesh(meshed_eigen)
    receiver.ports[1].excited = True
    receiver.ports[1].impedance = 25.0
    receiver.ports[1].length = 2.0
    with pytest.raises(ValueError, match=re.escape("port 'o2' geometry")):
        receiver.write_config(validate_mesh=False)

    receiver.reuse_mesh(meshed_eigen)

    o2 = receiver.ports[1]
    assert (o2.name, o2.length, o2.excited, o2.impedance) == ("o2", 1.0, True, 25.0)
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())
    assert _excitations(config) == [False, 2]
    assert [p["R"] for p in config["Boundaries"]["LumpedPort"]] == [50.0, 25.0]
    assert meshed_eigen.ports[1].length == 1.0


def test_reuse_again_takes_the_port_kind_back_and_keeps_shared_settings(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    receiver.reuse_mesh(meshed_eigen)
    receiver.ports = receiver.ports[:1]
    receiver.add_cpw_port(
        "o2", layer="m1", s_width=1.0, gap_width=1.0, impedance=30.0, excited=True
    )

    receiver.reuse_mesh(meshed_eigen)

    # o2 is a lumped port again, as in the mesh, with the CPW port's settings
    assert receiver.cpw_ports == []
    o2 = receiver.ports[1]
    assert o2.model_dump() == {
        **meshed_eigen.ports[1].model_dump(),
        "excited": True,
        "impedance": 30.0,
    }
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())
    assert _excitations(config) == [False, 2]


def test_reuse_again_drops_a_port_added_after_reuse(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    receiver.add_port("o3", layer="m1", length=1.0)
    with pytest.raises(ValueError, match=re.escape("port 'o3' (added)")):
        receiver.write_config(validate_mesh=False)

    receiver.reuse_mesh(meshed_eigen)

    assert [p.name for p in receiver.ports] == ["o1", "o2"]
    assert receiver.write_config(validate_mesh=False).is_file()


def test_self_meshed_receiver_with_a_mismatching_port_is_still_refused(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    receiver.ports[0].length = 2.0
    fake_gmsh.allowed = False
    before = _state(receiver)

    with pytest.raises(ValueError, match=r"Port 'o1' .*does not match"):
        receiver.reuse_mesh(meshed_eigen)

    assert receiver._mesh_reused is False
    assert _state(receiver) == before


# ----- case 9: Port settings reach the config ----------------------------------------


def _excite_o2_before_reuse(receiver, source) -> None:
    receiver.add_port("o2", layer="m1", length=1.0, excited=True)
    receiver.reuse_mesh(source)


def _excite_the_copied_o2_after_reuse(receiver, source) -> None:
    receiver.reuse_mesh(source)
    receiver.ports[1].excited = True


def _declare_o2_again_after_reuse(receiver, source) -> None:
    receiver.reuse_mesh(source)
    receiver.add_port("o2", layer="m1", length=1.0, excited=True)


@pytest.mark.parametrize(
    "excite",
    [
        _excite_o2_before_reuse,
        _excite_the_copied_o2_after_reuse,
        _declare_o2_again_after_reuse,
    ],
    ids=lambda value: value.__name__,
)
def test_excitation_set_before_or_after_reuse_reaches_the_config(
    excite, meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})

    excite(receiver, meshed_eigen)
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())

    assert _excitations(config) == [False, 2]
    assert [(p.name, p.excited) for p in receiver._last_ports] == [
        ("o1", False),
        ("o2", True),
    ]
    assert not any(p.excited for p in meshed_eigen._last_ports)


def test_impedance_change_after_reuse_reaches_port_information(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)

    receiver.ports[1].impedance = 25.0
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())

    info = json.loads((tmp_path / "driven" / "port_information.json").read_text())
    assert [(p["name"], p["Z0"]) for p in info["ports"]] == [
        ("o1", 50.0),
        ("o2", 25.0),
    ]
    assert [p["R"] for p in config["Boundaries"]["LumpedPort"]] == [50.0, 25.0]
    assert [e["Z0"] for e in meshed_eigen._last_mesh_result.port_info] == [50.0, 50.0]


def test_lumped_offset_never_reaches_the_de_embedding_offset(
    make_sim, fake_gmsh, tmp_path
) -> None:
    ports = {
        "o1": {"layer": "m1", "length": 1.0, "offset": 0.5},
        "o2": {"layer": "m1", "length": 1.0},
    }
    source = make_sim(EigenmodeSim, tmp_path / "eigen", ports=ports)
    source.mesh(verbose=False)
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports=ports)

    receiver.reuse_mesh(source)
    receiver.write_config(validate_mesh=False)

    assert receiver.ports[0].offset == 0.5
    # the offset moved the port surface; it is no de-embedding distance
    assert [p.offset for p in receiver._last_ports] == [0.0, 0.0]
    assert receiver._last_ports[0].center == source._last_ports[0].center


@pytest.mark.parametrize("cls", [DrivenSim, EigenmodeSim])
def test_settings_sync_changes_nothing_for_equal_declarations(
    cls, make_sim, fake_gmsh, tmp_path
) -> None:
    source = make_sim(cls, tmp_path / "source")
    source.mesh(verbose=False)
    source_config = json.loads(source.write_config(validate_mesh=False).read_text())
    fake_gmsh.allowed = False
    receiver = make_sim(cls, tmp_path / "receiver")

    receiver.reuse_mesh(source)
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())

    assert receiver._last_ports == source._last_ports
    assert receiver._last_mesh_result.port_info == source._last_mesh_result.port_info
    assert config == source_config


# ----- case 10: files ----------------------------------------------------------------


def test_files_copy_the_mesh_and_regenerate_metadata(
    make_sim, fake_gmsh, tmp_path
) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(verbose=False, model_name="chip")
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    fake_gmsh.allowed = False

    receiver.reuse_mesh(source)

    folder = tmp_path / "driven"
    assert _listing(folder) == ["chip.msh", "metadata.json"]
    assert receiver.mesh_path == folder / "chip.msh"
    assert receiver._last_mesh_result.model_name == "chip"
    assert (folder / "chip.msh").read_bytes() == (
        tmp_path / "eigen" / "chip.msh"
    ).read_bytes()
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata == receiver._last_mesh_result.metadata
    assert "simulation" not in metadata
    assert metadata["mesh"]["field_dofs"]["problem_type"] == "driven"
    source_metadata = json.loads((tmp_path / "eigen" / "metadata.json").read_text())
    assert source_metadata["mesh"]["field_dofs"]["problem_type"] == "eigenmode"


def test_files_of_the_source_stay_unchanged_when_the_receiver_writes(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    digest = compute_dir_digest(tmp_path / "eigen")
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    fake_gmsh.allowed = False

    receiver.reuse_mesh(meshed_eigen)
    receiver.write_config(validate_mesh=False)

    assert (tmp_path / "driven" / "config.json").is_file()
    assert compute_dir_digest(tmp_path / "eigen") == digest


def test_files_of_identical_receivers_hash_equal(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    for name in ("a", "b"):
        receiver = make_sim(DrivenSim, tmp_path / name)
        receiver.reuse_mesh(meshed_eigen)
        receiver.write_config(validate_mesh=False)

    assert compute_input_hash(tmp_path / "a", "palace") == compute_input_hash(
        tmp_path / "b", "palace"
    )


# ----- case 11: run on the cloud, mocked ---------------------------------------------


def test_run_with_cache_check_uploads_only_the_receiver_folder(
    meshed_eigen, make_sim, fake_gmsh, monkeypatch, tmp_path
) -> None:
    meshed_eigen.write_config(validate_mesh=False)
    source_digest = compute_dir_digest(tmp_path / "eigen")
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    # the fake .msh is no Gmsh mesh: skip the two checks that open it
    for check in ("check_port_surfaces_in_domain", "check_lumped_port_geometry"):
        monkeypatch.setattr(
            f"gsim.palace.mesh.validation.{check}", lambda **_kwargs: []
        )
    staging = tmp_path / "upload"
    staging.mkdir()
    monkeypatch.setattr(
        "gsim.palace.base.tempfile.mkdtemp", lambda **_kwargs: str(staging)
    )
    hashed: list[Path] = []

    def check_cache_for_dir(directory, job_type):
        hashed.append(Path(directory))
        return compute_input_hash(directory, job_type), None

    upload = Mock(return_value="new-job")
    monkeypatch.setattr(gcloud, "check_cache_for_dir", check_cache_for_dir)
    monkeypatch.setattr(gcloud, "upload", upload)
    monkeypatch.setattr(gcloud, "start", Mock())

    job_id = receiver.run(check_cache=True, wait=False, verbose="quiet")

    folder = tmp_path / "driven"
    assert job_id == "new-job"
    assert hashed == [staging]
    assert upload.call_args.args[0] == staging
    assert upload.call_args.kwargs["input_hash"] == compute_input_hash(folder, "palace")
    # the upload is a copy of the receiver's folder, and only of it
    assert _listing(staging) == _listing(folder)
    assert compute_dir_digest(staging) == compute_dir_digest(folder)
    assert json.loads((staging / "config.json").read_text())["Problem"]["Type"] == (
        "Driven"
    )
    assert (staging / "palace.msh").read_bytes() == meshed_eigen.mesh_path.read_bytes()
    assert compute_dir_digest(tmp_path / "eigen") == source_digest


# ----- case 12: the guard ------------------------------------------------------------


def _edit_a_polygon(sim) -> None:
    sim.component.add_polygon([(0, 6), (2, 6), (2, 7), (0, 7)], layer=(1, 0))


def _set_other_shapes(sim) -> None:
    sim.set_geometry(_layout(m1_height=2.0))


def _move_a_component_port(sim) -> None:
    sim.component.ports["o2"].center = (9.0, 0.5)


def _raise_a_layer(sim) -> None:
    sim.stack.layers["m2"].zmax = 3.5


def _change_the_dielectric(sim) -> None:
    sim.stack.dielectrics[0]["zmax"] = 6.0


def _make_the_oxide_air_like(sim) -> None:
    sim.set_material("sio2", permittivity=1.0)


def _make_the_via_insulating(sim) -> None:
    sim.set_material("tungsten", material_type="conductor", conductivity=0.0)


def _set_an_airbox(sim) -> None:
    sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=0)


def _add_a_pec_block(sim) -> None:
    sim.add_pec(gds_layer=(9, 0), from_layer="m1", to_layer="m2")


def _add_a_symmetry_plane(sim) -> None:
    sim.add_symmetry_plane(axis="x", position=5.0, verify_symmetry=False)


def _change_a_port_layer(sim) -> None:
    sim.ports[0].layer = "m2"


def _change_a_port_length(sim) -> None:
    sim.ports[0].length = 2.0


def _change_a_port_offset(sim) -> None:
    sim.ports[0].offset = 0.5


def _add_a_port(sim) -> None:
    sim.add_port("o3", layer="m1", length=1.0)


def _remove_a_port(sim) -> None:
    sim.ports = sim.ports[:1]


@pytest.mark.parametrize(
    ("change", "label"),
    [
        (_edit_a_polygon, "geometry (layout polygons)"),
        (_set_other_shapes, "geometry (layout polygons)"),
        (
            _move_a_component_port,
            "component ports (names, positions, orientations, widths)",
        ),
        (_raise_a_layer, "stack layer 'm2'"),
        (_change_the_dielectric, "stack dielectric regions"),
        (_make_the_oxide_air_like, "material 'sio2' (air-like or not)"),
        (_make_the_via_insulating, "stack layer 'v1'"),
        (_set_an_airbox, "airbox (set_airbox)"),
        (_add_a_pec_block, "PEC blocks (add_pec)"),
        (_add_a_symmetry_plane, "symmetry plane"),
        (_change_a_port_layer, "port 'o1' geometry"),
        (_change_a_port_length, "port 'o1' geometry"),
        (_change_a_port_offset, "port 'o1' geometry"),
        (_add_a_port, "port 'o3' (added)"),
        (_remove_a_port, "port 'o2' (removed)"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_guarded_change_is_refused_at_write_config(
    change, label, meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    assert receiver._current_mesh_fingerprint(receiver._resolve_stack()) == (
        receiver._mesh_fingerprint
    )

    change(receiver)  # the setter itself stays quiet

    with pytest.raises(ValueError, match=re.escape(label)) as error:
        receiver.write_config(validate_mesh=False)
    message = str(error.value)
    assert message.startswith("This DrivenSim reuses a mesh")
    assert "call mesh()" in message
    assert "reuse_mesh(source) again" in message
    assert not (tmp_path / "driven" / "config.json").exists()


def _unexcite_a_port(sim) -> None:
    sim.ports[1].excited = False


def _change_an_impedance(sim) -> None:
    sim.ports[1].impedance = 25.0


def _set_r_l_c(sim) -> None:
    sim.ports[1].resistance = 10.0
    sim.ports[1].inductance = 1e-9
    sim.ports[1].capacitance = 1e-15


def _raise_the_oxide_permittivity(sim) -> None:
    sim.set_material("sio2", permittivity=4.1)


def _set_a_loss_tangent(sim) -> None:
    sim.set_material("sio2", permittivity=3.9, loss_tangent=1e-3)


def _change_a_conductor_conductivity(sim) -> None:
    sim.set_material("aluminum", conductivity=3.0e7)


def _change_the_mesh_settings(sim) -> None:
    sim.mesh_config.refined_mesh_size = 0.5


def _declare_a_port_again(sim) -> None:
    sim.add_port("o1", layer="m1", length=1.0, excited=True)


@pytest.mark.parametrize(
    "change",
    [
        _unexcite_a_port,
        _change_an_impedance,
        _set_r_l_c,
        _raise_the_oxide_permittivity,
        _set_a_loss_tangent,
        _change_a_conductor_conductivity,
        _change_the_mesh_settings,
        _declare_a_port_again,
    ],
    ids=lambda value: value.__name__,
)
def test_allowed_change_passes_the_guard(
    change, meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)

    change(receiver)

    assert receiver.write_config(validate_mesh=False).is_file()


# ----- case 13: the mesh must hold the receiver's settings ---------------------------


def test_driven_receiver_with_no_excited_port_is_refused(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    """R2: ports copied from an eigenmode source are not excited."""
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    receiver.reuse_mesh(meshed_eigen)

    with pytest.raises(ValueError, match="excites none of its ports") as error:
        receiver.write_config(validate_mesh=False)

    message = str(error.value)
    assert "with excited=True" in message
    assert "set excited=True on its copied port config" in message
    assert not (tmp_path / "driven" / "config.json").exists()


def test_excited_cpw_port_of_an_eigenmode_source_is_not_excited_on_the_receiver(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """A CPW port keeps excited=True on an eigenmode source; a receiver
    that declares nothing does not inherit that excitation."""
    source = make_sim(EigenmodeSim, tmp_path / "eigen", ports={})
    source.add_cpw_port("o1", layer="m1", s_width=1.0, gap_width=1.0)
    assert source.cpw_ports[0].excited
    source.mesh(verbose=False)
    fake_gmsh.allowed = False

    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    receiver.reuse_mesh(source)

    assert [(p.name, p.excited) for p in receiver.cpw_ports] == [("o1", False)]
    assert not any(p.excited for p in receiver._last_ports)
    with pytest.raises(ValueError, match="excites none of its ports"):
        receiver.write_config(validate_mesh=False)
    assert source.cpw_ports[0].excited

    receiver.cpw_ports[0].excited = True
    assert receiver.write_config(validate_mesh=False).is_file()


def test_driven_receiver_does_not_inherit_the_driven_sources_excitation(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """Source excites o1, receiver declares only o2: only o2 is excited."""
    source = make_sim(
        DrivenSim,
        tmp_path / "source",
        ports={
            "o1": {"layer": "m1", "length": 1.0, "excited": True},
            "o2": {"layer": "m1", "length": 1.0, "excited": False},
        },
    )
    source.mesh(verbose=False)
    fake_gmsh.allowed = False

    receiver = make_sim(
        DrivenSim,
        tmp_path / "receiver",
        ports={"o2": {"layer": "m1", "length": 1.0, "excited": True}},
    )
    receiver.reuse_mesh(source)
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())

    assert _excitations(config) == [False, 2]
    assert [(p.name, p.excited) for p in receiver.ports] == [
        ("o1", False),
        ("o2", True),
    ]
    assert [p.excited for p in source.ports] == [True, False]


def test_receiver_mesh_config_holds_the_sizes_the_source_mesh_used(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """mesh(preset=...) sizes are not stored on source.mesh_config."""
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(preset="coarse", verbose=False)
    used = source._last_mesh_config
    assert used.refined_mesh_size != source.mesh_config.refined_mesh_size
    fake_gmsh.allowed = False

    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(source)

    assert receiver.mesh_config == used
    assert receiver.mesh_config is not used
    assert receiver.mesh_config.refined_mesh_size == used.refined_mesh_size
    assert receiver.mesh_config.max_mesh_size == used.max_mesh_size


def test_driven_receiver_exciting_a_port_without_surface_is_refused(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    """R2: every excited port needs its P{i} surface in the mesh."""
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    del receiver._last_mesh_result.groups["port_surfaces"]["P2"]

    with pytest.raises(ValueError, match=r"no surface in the reused mesh: 'o2' \(P2\)"):
        receiver.write_config(validate_mesh=False)


def test_eigenmode_receiver_without_target_is_refused(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(EigenmodeSim, tmp_path / "eigen2")
    receiver.set_eigenmode()
    receiver.reuse_mesh(meshed_eigen)

    with pytest.raises(ValueError, match="target frequency is required") as error:
        receiver.write_config(validate_mesh=False)

    assert str(error.value).startswith("Invalid configuration:\n")


def test_floquet_receiver_of_a_mesh_that_is_not_periodic_is_refused(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(EigenmodeSim, tmp_path / "floquet")
    receiver.set_eigenmode(target=5e9, floquet=True)
    receiver.reuse_mesh(meshed_eigen)

    with pytest.raises(ValueError, match="Floquet eigenmode requires a periodic axis"):
        receiver.write_config(validate_mesh=False)


def _meshed_wave_source(make_sim, folder: Path, *, multimode: bool) -> DrivenSim:
    """A driven sim with wave ports o1 and o2, meshed by the fake."""
    sim = make_sim(DrivenSim, folder, ports={})
    if multimode:
        sim.add_multimode_wave_port("o1", modes=(1, 2), layer="m1")
    else:
        sim.add_wave_port("o1", layer="m1")
    sim.add_wave_port("o2", layer="m1")
    sim.mesh(verbose=False)
    assert set(sim.mesh_groups["port_surfaces"]) == {"P1", "P2"}
    return sim


def test_eigenmode_receiver_of_multimode_wave_ports_is_refused(
    make_sim, fake_gmsh, tmp_path
) -> None:
    source = _meshed_wave_source(make_sim, tmp_path / "driven", multimode=True)
    fake_gmsh.allowed = False
    receiver = make_sim(EigenmodeSim, tmp_path / "eigen", ports={})
    receiver.reuse_mesh(source)
    assert list(receiver.wave_ports[0].modes) == [1, 2]

    with pytest.raises(
        ValueError, match="Multimode wave ports are only supported for driven"
    ) as error:
        receiver.write_config(validate_mesh=False)

    assert str(error.value).startswith("Invalid configuration:\n")


def test_wave_port_modes_can_change_on_a_reused_mesh(
    make_sim, fake_gmsh, tmp_path
) -> None:
    source = _meshed_wave_source(make_sim, tmp_path / "source", multimode=False)
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    receiver.reuse_mesh(source)

    receiver.add_multimode_wave_port("o1", modes=(1, 2), layer="m1")
    config = json.loads(receiver.write_config(validate_mesh=False).read_text())

    assert [
        (entry["Index"], entry["Mode"], entry.get("Active", True))
        for entry in config["Boundaries"]["WavePort"]
    ] == [(1, 1, True), (2, 1, True), (3, 2, False)]
    by_name = {port.name: port for port in receiver._last_ports}
    assert by_name["o1"].modes == (1, 2)
    assert by_name["o2"].modes == ()
    assert [port.modes for port in source._last_ports] == [(), ()]


def test_default_driven_receiver_writes_an_absorbing_boundary(
    meshed_eigen, fake_gmsh, tmp_path
) -> None:
    """absorbing_boundary is not copied: DrivenSim's unset default absorbs."""
    fake_gmsh.allowed = False
    receiver = DrivenSim()
    receiver.set_output_dir(tmp_path / "driven")
    receiver.add_port("o1", layer="m1", length=1.0, excited=True)
    receiver.reuse_mesh(meshed_eigen)
    receiver.set_driven(fmin=7e9, fmax=10e9)

    config = json.loads(receiver.write_config(validate_mesh=False).read_text())

    assert meshed_eigen.absorbing_boundary is False
    assert receiver.absorbing_boundary is None
    outer = meshed_eigen.mesh_groups["boundary_surfaces"]["absorbing"]["phys_group"]
    assert config["Boundaries"]["Absorbing"]["Attributes"] == [outer]
    assert _excitations(config) == [1, False]


# ----- PDK-authoritative stacks ------------------------------------------------------


def _build_from_options(resolved: dict) -> LayerStack:
    """The generic stack, with the options a PDK resolver would apply."""
    built = generic_stack()
    if resolved["dielectrics"] is not None:
        built.dielectrics = [dict(region) for region in resolved["dielectrics"]]
    for layer, role in (resolved["role_overrides"] or {}).items():
        built.layers[layer].layer_type = role
    return built


@pytest.fixture
def fake_pdk(monkeypatch) -> list:
    """Patch the PDK-authoritative chain to build the generic stack.

    Returns the list of components the resolver received.
    """
    components: list = []

    def _resolve(component=None, **options):
        components.append(component)
        return options

    monkeypatch.setattr("gsim.common.pdk.resolve_layer_stack", _resolve)
    monkeypatch.setattr(
        "gsim.palace.stack.build_palace_layer_stack", _build_from_options
    )
    return components


@pytest.fixture
def authoritative_eigen(make_sim, fake_pdk, fake_gmsh, tmp_path) -> EigenmodeSim:
    """An eigenmode sim on a PDK-authoritative stack, meshed by the real ``mesh()``."""
    sim = make_sim(EigenmodeSim, tmp_path / "eigen")
    sim.set_stack(pdk_authoritative=True)
    sim.mesh(verbose=False)
    assert fake_gmsh.calls == 1
    assert fake_pdk  # mesh() resolved through the authoritative path
    return sim


CHANGED_PDK_OPTIONS = [
    (
        {
            "dielectrics": [
                {"name": "ox", "zmin": -3.0, "zmax": 5.0, "material": "sio2"}
            ]
        },
        "stack dielectric regions",
    ),
    ({"role_overrides": {"v1": "conductor"}}, "stack layer 'v1'"),
]


def test_authoritative_source_is_reused_without_gmsh(
    authoritative_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    receiver.reuse_mesh(authoritative_eigen)

    assert receiver._stack_kwargs == authoritative_eigen._stack_kwargs
    assert receiver._stack_kwargs is not authoritative_eigen._stack_kwargs
    assert receiver._stack_kwargs["pdk_authoritative"] is True
    assert receiver.write_config(validate_mesh=False).is_file()
    assert fake_gmsh.calls == 1


@pytest.mark.parametrize(("options", "label"), CHANGED_PDK_OPTIONS)
def test_changed_pdk_option_on_the_receiver_is_refused_at_write_config(
    options, label, authoritative_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(authoritative_eigen)

    receiver.set_stack(pdk_authoritative=True, **options)

    with pytest.raises(ValueError, match=re.escape(label)):
        receiver.write_config(validate_mesh=False)
    assert not (tmp_path / "driven" / "config.json").exists()


@pytest.mark.parametrize(("options", "label"), CHANGED_PDK_OPTIONS)
def test_changed_pdk_option_on_the_source_refuses_reuse(
    options, label, authoritative_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    authoritative_eigen.set_stack(pdk_authoritative=True, **options)

    with pytest.raises(ValueError, match="changed after its mesh was made") as error:
        receiver.reuse_mesh(authoritative_eigen)
    assert label in str(error.value)
    assert receiver._last_mesh_result is None


def test_activating_another_pdk_is_refused_at_write_config(
    authoritative_eigen, make_sim, fake_gmsh, tmp_path, monkeypatch
) -> None:
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(authoritative_eigen)

    def _build_from_another_pdk(resolved: dict) -> LayerStack:
        built = _build_from_options(resolved)
        built.layers["m2"].zmax = 3.5
        return built

    monkeypatch.setattr(
        "gsim.palace.stack.build_palace_layer_stack", _build_from_another_pdk
    )

    with pytest.raises(ValueError, match=re.escape("stack layer 'm2'")):
        receiver.write_config(validate_mesh=False)


def test_reuse_resolves_against_the_receivers_own_component(
    authoritative_eigen, fake_pdk, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    receiver.reuse_mesh(authoritative_eigen)

    assert fake_pdk[-1] is receiver.geometry.component
    assert fake_pdk[-1] is not authoritative_eigen.geometry.component


@pytest.mark.usefixtures("fake_pdk")
def test_airbox_only_authoritative_source_is_reused(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """No dielectric regions: the resolver accepts the copied airbox settings."""
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.set_stack(pdk_authoritative=True, dielectrics=[])
    _set_an_airbox(source)
    source.mesh(verbose=False)
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    receiver.reuse_mesh(source)

    assert receiver.write_config(validate_mesh=False).is_file()


def _minimal_pdk(name: str, m2_zmin: float) -> gf.Pdk:
    """A real PDK with the layers of the layout, and no reviewed overlay."""
    return gf.Pdk(
        name=name,
        layer_stack=GfLayerStack(
            layers={
                "m1": LayerLevel(
                    layer=(1, 0), thickness=1.0, zmin=0.0, material="aluminum"
                ),
                "m2": LayerLevel(
                    layer=(2, 0), thickness=1.0, zmin=m2_zmin, material="aluminum"
                ),
            }
        ),
    )


@pytest.fixture
def _restores_active_pdk(component):  # noqa: ARG001 - activates the layout's PDK first
    """Activate the previous PDK again after the test."""
    previous = gf.get_active_pdk()
    yield
    previous.activate()


@pytest.mark.usefixtures("_restores_active_pdk")
def test_real_pdk_stack_is_reused_and_another_pdk_is_refused(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """The real resolver rebuilds the same stack, until another PDK is active."""
    _minimal_pdk("reuse_mesh_a", m2_zmin=2.0).activate()
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.set_stack(
        pdk_authoritative=True,
        dielectrics=[{"name": "ox", "material": "sio2", "zmin": -2.0, "zmax": 5.0}],
    )
    source.mesh(verbose=False)
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    receiver.reuse_mesh(source)

    assert receiver.write_config(validate_mesh=False).is_file()

    _minimal_pdk("reuse_mesh_b", m2_zmin=2.5).activate()

    with pytest.raises(ValueError, match=re.escape("stack layer 'm2'")):
        receiver.write_config(validate_mesh=False)


# ----- hierarchical layouts ----------------------------------------------------------


def _hierarchical_layout() -> tuple[gf.Component, gf.Component]:
    """The ``_layout`` ports over nested, arrayed, rotated and mirrored cells.

    Returns the layout and the unlocked cell holding its m1 strip, two levels down.
    """
    gf.gpdk.PDK.activate()
    strip = gf.Component()
    strip.add_polygon([(0, 0), (10, 0), (10, 1), (0, 1)], layer=(1, 0))
    nested = gf.Component()
    nested.add_ref(strip)

    pad = gf.Component()
    pad.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(2, 0))
    m2 = pad.kcl.layer(2, 0)
    pad.shapes(m2).insert(gf.kdb.DBox(0, 1.5, 1, 2))
    pad.shapes(m2).insert(
        gf.kdb.DPath([gf.kdb.DPoint(0, 2.5), gf.kdb.DPoint(1, 3.5)], 0.2)
    )

    c = gf.Component()
    c.add_ref(nested)
    c.add_ref(pad, columns=3, rows=2, column_pitch=2.0, row_pitch=4.0).move((0, 3))
    c.add_ref(pad).rotate(30).move((8, 3))
    c.add_ref(pad).rotate(90).move((14, 3))
    c.add_ref(pad).mirror_y().move((16, 3))
    for name, x, orientation in (("o1", 0, 180), ("o2", 10, 0)):
        c.add_port(
            name=name,
            center=(x, 0.5),
            width=1.0,
            orientation=orientation,
            layer=(1, 0),
            port_type="electrical",
        )
    return c, strip


def test_hierarchical_source_is_reused_as_a_flat_copy(
    make_sim, fake_gmsh, tmp_path
) -> None:
    layout, _strip = _hierarchical_layout()
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.set_geometry(layout)
    source.mesh(verbose=False)
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    receiver.reuse_mesh(source)

    assert receiver.write_config(validate_mesh=False).is_file()
    assert receiver.component.kdb_cell.child_instances() == 0
    assert source.component.kdb_cell.child_instances() == 5
    # 10 um x 1 um, in square database units of 1 nm
    assert receiver.component.get_region((1, 0)).area() == 10_000_000
    assert sorted(receiver.component.ports.get_all_named()) == ["o1", "o2"]


def test_editing_a_child_cell_of_the_source_does_not_reach_the_receiver(
    make_sim, fake_gmsh, tmp_path
) -> None:
    layout, strip = _hierarchical_layout()
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.set_geometry(layout)
    source.mesh(verbose=False)
    fake_gmsh.allowed = False
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(source)

    strip.add_polygon([(0, 6), (2, 6), (2, 7), (0, 7)], layer=(1, 0))

    # the snapshot still matches its mesh; the source no longer does
    assert receiver.component.get_region((1, 0)).area() == 10_000_000
    assert receiver.write_config(validate_mesh=False).is_file()
    late = make_sim(DrivenSim, tmp_path / "late")
    with pytest.raises(ValueError, match=re.escape("geometry (layout polygons)")):
        late.reuse_mesh(source)


# ----- failures while installing the files -------------------------------------------


@pytest.mark.parametrize("failing", ["metadata.json", "palace.msh"])
def test_failed_install_keeps_the_mesh_and_state_of_the_receiver(
    failing, make_sim, fake_gmsh, monkeypatch, tmp_path
) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    fake_gmsh.allowed = False
    own_mesh = receiver.mesh_path.read_bytes()
    assert own_mesh != source.mesh_path.read_bytes()
    # the staged metadata would have the same content; this tells them apart
    metadata_path = receiver.output_dir / "metadata.json"
    metadata_path.write_text('{"own": true}')
    before = _state(receiver)
    listing = _listing(receiver.output_dir)
    real_replace = Path.replace

    def _replace(path, target):
        if Path(target).name == failing:
            raise OSError("disk full")
        return real_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", _replace)
        with pytest.raises(OSError, match="disk full"):
            receiver.reuse_mesh(source)

    assert _state(receiver) == before
    assert receiver.mesh_path.read_bytes() == own_mesh
    assert metadata_path.read_text() == '{"own": true}'
    assert _listing(receiver.output_dir) == listing
    assert receiver.write_config(validate_mesh=False).is_file()


def test_failed_mesh_install_of_a_new_receiver_leaves_no_metadata(
    make_sim, fake_gmsh, monkeypatch, tmp_path
) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    fake_gmsh.allowed = False
    real_replace = Path.replace

    def _replace(path, target):
        if Path(target).name == "palace.msh":
            raise OSError("disk full")
        return real_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", _replace)
        with pytest.raises(OSError, match="disk full"):
            receiver.reuse_mesh(source)

    assert not receiver.has_mesh
    assert _listing(tmp_path / "driven") == []


# ----- periodic meshes ---------------------------------------------------------------


@pytest.fixture
def periodic_eigen(make_sim, fake_gmsh, tmp_path) -> EigenmodeSim:
    """A Floquet eigenmode source with a mesh periodic along x."""
    sim = make_sim(EigenmodeSim, tmp_path / "eigen")
    sim.set_eigenmode(target=5e9, floquet=True)
    sim.mesh(verbose=False, periodic_axis="x")
    fake_gmsh.allowed = False
    return sim


def _boundaries(sim) -> dict:
    """The ``Boundaries`` block of the config the sim writes."""
    path = sim.write_config(validate_mesh=False)
    return json.loads(path.read_text())["Boundaries"]


@pytest.mark.parametrize(
    ("receiver_cls", "absorbing"),
    [(DrivenSim, True), (DrivenSim, None), (EigenmodeSim, True)],
    ids=["driven", "driven-unset", "eigenmode"],
)
def test_absorbing_receiver_of_a_periodic_mesh_is_refused_at_write_config(
    receiver_cls, absorbing, periodic_eigen, make_sim, tmp_path
) -> None:
    """The periodic faces left the absorbing group; only Floquet conditions them.

    A driven sim's unset absorbing_boundary absorbs when there are no wave
    ports.
    """
    receiver = make_sim(receiver_cls, tmp_path / "receiver")
    receiver.absorbing_boundary = absorbing
    receiver.reuse_mesh(periodic_eigen)

    with pytest.raises(ValueError, match=r"periodic_axis='x'.*absorbing_boundary"):
        receiver.write_config(validate_mesh=False)
    assert not (tmp_path / "receiver" / "config.json").exists()


def test_receiver_turning_on_absorption_after_reuse_is_refused(
    periodic_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(periodic_eigen)
    assert receiver.write_config(validate_mesh=False).is_file()

    receiver.absorbing_boundary = True

    with pytest.raises(ValueError, match=r"periodic_axis='x'.*absorbing_boundary"):
        receiver.write_config(validate_mesh=False)


def test_reflecting_receiver_of_a_periodic_mesh_has_one_outer_wall(
    periodic_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(periodic_eigen)

    boundaries = _boundaries(receiver)

    assert "Absorbing" not in boundaries
    assert "Periodic" not in boundaries


def test_floquet_receiver_of_a_periodic_mesh_conditions_the_periodic_faces(
    periodic_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(EigenmodeSim, tmp_path / "floquet")
    receiver.absorbing_boundary = True
    receiver.set_eigenmode(target=5e9, floquet=True)
    receiver.reuse_mesh(periodic_eigen)

    boundaries = _boundaries(receiver)

    assert boundaries["Periodic"]["BoundaryPairs"][0]["DonorAttributes"] == [20]
    assert boundaries["Periodic"]["BoundaryPairs"][0]["ReceiverAttributes"] == [21]
    assert boundaries["Absorbing"]["Attributes"] == [4]


def test_direct_periodic_mesh_of_a_driven_sim_is_not_refused(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """The refusal is for reused meshes; a sim's own ``mesh()`` keeps its behaviour."""
    sim = make_sim(DrivenSim, tmp_path / "driven")
    sim.absorbing_boundary = True
    sim.mesh(verbose=False, periodic_axis="x")

    assert fake_gmsh.calls == 1
    assert "Absorbing" in _boundaries(sim)
