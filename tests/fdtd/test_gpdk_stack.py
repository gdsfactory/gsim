"""A generic-PDK component meshes with its whole stack and no template."""

import json
import warnings

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim import fdtd


@pytest.fixture(autouse=True)
def gpdk_active():
    """Use the generic PDK and restore the previous process state."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.gpdk.PDK.activate(force=True)
    yield
    if previous is not None:
        previous.activate(force=True)


def _simulation(name="mmi1x2"):
    simulation = fdtd.Simulation()
    simulation.domain()
    simulation.source(port="o1")
    simulation.geometry(name, mesh_size_nm=700)
    return simulation


def _notebook_simulation(**domain):
    """The setup of the published gpdk notebooks: SiO2 background, ``si`` override."""
    simulation = fdtd.Simulation()
    simulation.materials(background="SiO2", overrides={"si": 3.47})
    simulation.domain(**domain)
    simulation.source(port="o1")
    simulation.geometry("mmi1x2", mesh_size_nm=700)
    return simulation


def _z_extent_nm(simulation) -> tuple[float, float]:
    bounds = simulation._background_bounds_nm()
    return bounds[2], bounds[5]


def _layer_z(simulation) -> dict[str, tuple[float, float]]:
    return {
        key: (round(layer.zmin, 9), round(layer.zmax, 9))
        for key, layer in simulation.resolved.layers.items()
    }


def test_gpdk_stack_reaches_the_mesh_and_config(tmp_path) -> None:
    simulation = _simulation()
    resolved = simulation.resolved
    # The substrate lies 3 um below the core, outside the 1 um padded domain.
    assert list(resolved.layers) == ["box", "core", "clad"]
    assert resolved.wafer_films == {"substrate", "box", "clad"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    for name in ("box", "core", "clad"):
        assert name in mesh.field_data
    assert "substrate" not in mesh.field_data

    config = json.loads(artifacts.config_path.read_text())
    assert {"Si", "SiO2"} <= set(config["materials"])

    # Ports stay on the domain faces: the blanket films end where the ports do.
    port_x = sorted({port.center[0] for port in resolved.ports.values()})
    clad_tag = mesh.field_data["clad"][0]
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    clad_points = mesh.points[np.unique(mesh.cells_dict["tetra"][tags == clad_tag])]
    assert clad_points[:, 0].min() == pytest.approx(port_x[0] * 1000)
    assert clad_points[:, 0].max() == pytest.approx(port_x[-1] * 1000)
    # The film fills the padded domain (core top 0.22 um plus 1 um padding).
    assert clad_points[:, 2].min() == pytest.approx(0)
    assert clad_points[:, 2].max() == pytest.approx(1220)


def test_wafer_films_do_not_size_the_padded_domain() -> None:
    """The gpdk notebooks keep a domain of core plus padding, not the wafer stack."""
    simulation = _notebook_simulation(padding=1)
    assert _z_extent_nm(simulation) == pytest.approx((-1000, 1220))
    # Patterned layers alone give the device bounds.
    (_, _, z_low), (_, _, z_high) = simulation.resolved.bounds
    assert (z_low, z_high) == pytest.approx((0, 0.22))
    assert _layer_z(simulation) == {
        "box": (-1.0, 0.0),
        "core": (0.0, 0.22),
        "clad": (0.0, 1.22),
    }


def test_explicit_z_bounds_work_without_clip_geometry() -> None:
    simulation = _notebook_simulation(z_bounds=(-1, 1.5))
    assert _z_extent_nm(simulation) == pytest.approx((-1000, 1500))
    assert _layer_z(simulation) == {
        "box": (-1.0, 0.0),
        "core": (0.0, 0.22),
        "clad": (0.0, 1.5),
    }


def test_explicit_z_bounds_wider_than_the_stack_keep_every_film() -> None:
    simulation = _notebook_simulation(z_bounds=(-20, 5))
    assert _layer_z(simulation) == {
        "substrate": (-13.0, -3.0),
        "box": (-3.0, 0.0),
        "core": (0.0, 0.22),
        "clad": (0.0, 3.0),
    }


def test_explicit_z_bounds_still_must_contain_the_patterned_layers() -> None:
    with pytest.raises(fdtd.FDTDGeometryError, match="must contain"):
        _notebook_simulation(z_bounds=(0.1, 1.5))


def test_default_air_background_domain_is_also_core_plus_padding() -> None:
    simulation = _simulation()
    assert _z_extent_nm(simulation) == pytest.approx((-1000, 1220))


def test_padded_gpdk_domain_does_not_warn_about_clipping() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", fdtd.GeometryClippingWarning)
        _notebook_simulation(padding=1)
        _notebook_simulation(z_bounds=(-1, 1.5))


def test_ambiguous_material_names_raise_a_config_error() -> None:
    from types import SimpleNamespace

    from gsim.common.materials import GSIM_MATERIAL_CARDS

    cards = {
        "Core": GSIM_MATERIAL_CARDS["Si"].model_copy(update={"name": "Core"}),
        "core": GSIM_MATERIAL_CARDS["SiO2"].model_copy(update={"name": "core"}),
    }
    module = SimpleNamespace(PDK=gf.gpdk.PDK, MATERIAL_CARDS=cards)
    simulation = fdtd.Simulation(pdk=module)
    simulation.materials(background="CORE")
    with pytest.raises(fdtd.FDTDConfigError, match="differ only by letter case"):
        _ = simulation._background_material_name

    simulation = fdtd.Simulation(pdk=module)
    simulation.materials(overrides={"CORE": 2.0})
    with pytest.raises(fdtd.FDTDConfigError, match="differ only by letter case"):
        simulation._canonical_material_overrides()


def test_films_are_dropped_with_a_pdk_copy_without_those_levels() -> None:
    """The documented way to simulate the core alone: copy the PDK's stack."""
    pdk_stack = gf.gpdk.PDK.layer_stack
    assert pdk_stack is not None
    layer_stack = pdk_stack.model_copy(deep=True)
    for key in ("substrate", "box", "clad"):
        del layer_stack.layers[key]
    pdk = gf.gpdk.PDK.model_copy(update={"layer_stack": layer_stack})
    simulation = fdtd.Simulation(pdk=pdk)
    simulation.materials(background="SiO2", overrides={"si": 3.47})
    simulation.domain(padding=1)
    simulation.source(port="o1")
    simulation.geometry("mmi1x2", mesh_size_nm=700)
    assert list(simulation.resolved.layers) == ["core"]
    assert simulation.resolved.wafer_films == frozenset()
    assert _z_extent_nm(simulation) == pytest.approx((-1000, 1220))
    assert "substrate" in pdk_stack.layers  # the PDK is untouched


def test_explicit_blanket_marking_keeps_sizing_the_domain() -> None:
    """A PDK that declares ``background=True`` itself opts in to sizing the domain.

    Only wafer-layer levels that the PDK does not mark are treated as films that
    fill the domain without sizing it.
    """
    plain = _notebook_simulation(padding=1)
    stack = gf.gpdk.PDK.layer_stack
    assert stack is not None
    for key in ("substrate", "box", "clad"):
        stack.layers[key].background = True
    try:
        marked = _notebook_simulation(padding=1)
        marked_extent = _z_extent_nm(marked)
    finally:
        for key in ("substrate", "box", "clad"):
            stack.layers[key].background = False
    assert _z_extent_nm(plain) == pytest.approx((-1000, 1220))
    assert marked_extent == pytest.approx((-13000, 3000))
    assert marked.resolved.wafer_films == frozenset()
    assert list(marked.resolved.layers) == ["substrate", "box", "core", "clad"]
    for key, layer in plain.resolved.layers.items():
        assert marked.resolved.layers[key].material == layer.material
