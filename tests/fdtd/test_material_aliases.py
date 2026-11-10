"""Declared material aliases stay consistent through public FDTD exports."""

import json
from types import SimpleNamespace

import pytest

from gsim import fdtd
from gsim.common.materials import GSIM_MATERIAL_CARDS


@pytest.fixture
def aliased_pdk(fdtd_pdk_module):
    stack = fdtd_pdk_module.PDK.layer_stack.model_copy(deep=True)
    stack.layers["core"].material = "legacy_si"
    pdk = fdtd_pdk_module.PDK.model_copy(update={"layer_stack": stack})
    return SimpleNamespace(
        PDK=pdk,
        MATERIAL_CARDS={
            "Si": GSIM_MATERIAL_CARDS["Si"],
            "Air": GSIM_MATERIAL_CARDS["Air"],
        },
        MATERIAL_ALIASES={"legacy_si": "Si", "air": "Air"},
    )


def test_legacy_and_canonical_overrides_write_identical_inputs(tmp_path, aliased_pdk):
    documents = []
    meshes = []
    for background, silicon in (("Air", "Si"), ("air", "legacy_si")):
        simulation = fdtd.Simulation(pdk=aliased_pdk, source={"port": "o1"})
        simulation.materials(
            background=background, overrides={silicon: 3.4, background: 1.1}
        )
        simulation.geometry("straight", mesh_size_nm=750)
        assert simulation.resolved.layers["core"].material == "Si"
        artifacts = simulation.write(tmp_path / background)
        documents.append(json.loads(artifacts.config_path.read_text()))
        meshes.append(artifacts.mesh_path.read_bytes())
    assert documents[0] == documents[1]
    assert meshes[0] == meshes[1]
    assert documents[0]["materials"]["Si"]["refractive_index"] == 3.4
    assert documents[0]["materials"]["Air"]["refractive_index"] == 1.1


def test_conflicting_alias_overrides_are_rejected(aliased_pdk):
    simulation = fdtd.Simulation(pdk=aliased_pdk)
    simulation.materials(overrides={"legacy_si": 3.4, "Si": 3.5})
    with pytest.raises(fdtd.FDTDConfigError, match="Conflicting material overrides"):
        simulation.geometry("straight")


@pytest.mark.parametrize("overrides", [{}, {"Si": 3.4}])
def test_module_aliases_survive_automatic_domain_sizing(aliased_pdk, overrides):
    aliased_pdk.MATERIAL_ALIASES["legacy_oxide"] = "SiO2"
    aliased_pdk.MATERIAL_CARDS["SiO2"] = GSIM_MATERIAL_CARDS["SiO2"]
    aliased_pdk.PDK.layer_stack.layers["buried_oxide"].material = "legacy_oxide"
    simulation = fdtd.Simulation(pdk=aliased_pdk)
    simulation.materials(background="legacy_oxide", overrides=overrides)
    simulation.geometry("straight")
    if not overrides:
        assert simulation.resolved.pdk_module is aliased_pdk
    assert simulation._background_bounds_nm()[2::3] == pytest.approx((-1000, 1000))
    simulation.domain(z_bounds=(-0.4, 0.6))
    assert simulation._background_bounds_nm()[2::3] == pytest.approx((-400, 600))


def test_domain_keeps_distinct_module_card_names(fdtd_pdk_module):
    fdtd_pdk_module.MATERIAL_CARDS.update(
        {
            "Air": GSIM_MATERIAL_CARDS["Air"],
            "air": GSIM_MATERIAL_CARDS["air"].model_copy(
                update={"info": {"distinct": True}}
            ),
        }
    )
    fdtd_pdk_module.PDK.layer_stack.layers["buried_oxide"].material = "air"
    for background, expected in (("Air", (-1000, 1220)), ("air", (-1000, 1000))):
        simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
        simulation.materials(background=background)
        simulation.geometry("straight")
        assert simulation._background_bounds_nm()[2::3] == pytest.approx(expected)


def test_layer_alias_can_use_builtin_fallback(aliased_pdk):
    del aliased_pdk.MATERIAL_CARDS["Si"]
    simulation = fdtd.Simulation(pdk=aliased_pdk)
    simulation.geometry("straight")
    assert simulation.resolved.materials["Si"].source == "gsim"


@pytest.mark.parametrize("name", ["Custom", "legacy_si"])
def test_layer_alias_can_target_an_override_only_material(aliased_pdk, name):
    aliased_pdk.MATERIAL_ALIASES["legacy_si"] = "Custom"
    simulation = fdtd.Simulation(pdk=aliased_pdk)
    simulation.materials(overrides={name: 3.4})
    simulation.geometry("straight")
    snapshot = simulation.resolved.materials["Custom"]
    assert snapshot.refractive_index == pytest.approx(3.4)


@pytest.mark.parametrize("override", [None, 1.3])
def test_exact_card_collision_keeps_its_geometry_and_override(aliased_pdk, override):
    from pdk_schema import Index, ScalarValue

    custom = GSIM_MATERIAL_CARDS["Air"].model_copy(deep=True, update={"name": "air"})
    assert custom.optical is not None
    assert isinstance(custom.optical.permittivity, Index)
    custom.optical.permittivity.n = ScalarValue(value=1.2, unit="")
    aliased_pdk.MATERIAL_CARDS["air"] = custom
    aliased_pdk.PDK.layer_stack.layers["core"].material = "air"
    simulation = fdtd.Simulation(pdk=aliased_pdk)
    if override is not None:
        simulation.materials(overrides={"air": override})
    simulation.geometry("straight")
    assert simulation.resolved.layers["core"].material == "air"
    snapshots = simulation._material_snapshots()
    assert snapshots["air"].refractive_index == pytest.approx(override or 1.2)
    assert snapshots["Air"].refractive_index == 1


@pytest.mark.parametrize("background", ["Air", "air"])
def test_merged_vertical_source_override_uses_canonical_background(
    tmp_path, fdtd_pdk_module, background
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.materials(background=background)
    simulation.geometry("vertical_coupler", mesh_size_nm=750)
    simulation.source(port="o2", vertical_refractive_index=1.33)
    document = json.loads(simulation.write(tmp_path).config_path.read_text())
    assert document["excitation"]["gaussian_beam"]["refractive_index"] == 1.33
    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.33
