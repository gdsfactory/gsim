"""Tests for patterned dielectric mesh support and stack dielectric toggles."""

from __future__ import annotations

from types import SimpleNamespace

from gdsfactory.technology import LayerLevel

from gsim.common.stack.extractor import Layer, LayerStack, extract_layer_stack
from gsim.palace.mesh.geometry import build_entities


def _fake_gf_stack():
    """Build a minimal gdsfactory-like stack for extractor tests."""
    return SimpleNamespace(
        layers={
            "core": LayerLevel(layer=(1, 0), thickness=0.22, zmin=0.0, material="si"),
            "metal1": LayerLevel(
                layer=(2, 0), thickness=1.0, zmin=1.0, material="aluminum"
            ),
        }
    )


def test_extract_layer_stack_can_disable_synthetic_dielectrics():
    """Synthetic oxide/passivation can be disabled explicitly."""
    stack = extract_layer_stack(
        _fake_gf_stack(),
        pdk_name="test-pdk",
        add_oxide_dielectric=False,
        add_passivation_dielectric=False,
    )

    names = [d["name"] for d in stack.dielectrics]
    assert names == []
    assert stack.simulation["add_oxide_dielectric"] is False
    assert stack.simulation["add_passivation_dielectric"] is False


def test_extract_layer_stack_defaults_keep_synthetic_dielectrics():
    """Defaults preserve synthetic oxide/passive dielectric regions."""
    stack = extract_layer_stack(_fake_gf_stack(), pdk_name="test-pdk")

    names = [d["name"] for d in stack.dielectrics]
    assert "oxide" in names
    assert "passive" in names
    assert "air_box" not in names


def test_build_entities_prioritizes_patterned_dielectrics_over_background_boxes():
    """Patterned dielectric volumes must be cut before blanket boxes."""
    stack_layers = {
        "via1": Layer(
            name="via1",
            gds_layer=(10, 0),
            zmin=0.0,
            zmax=1.0,
            thickness=1.0,
            material="tungsten",
            layer_type="via",
        )
    }
    stack = LayerStack(layers=stack_layers)

    metal_tags = {
        "via1": {
            "volumes": [41],
            "surfaces_xy": [],
            "surfaces_z": [],
        }
    }
    dielectric_tags = {"SiO2": [31], "airbox": [32]}
    patterned_dielectric_tags = {"core": [21]}
    port_tags = {"P1": [11]}
    port_info = [{"portnumber": 1}]

    entities = build_entities(
        metal_tags=metal_tags,
        dielectric_tags=dielectric_tags,
        patterned_dielectric_tags=patterned_dielectric_tags,
        port_tags=port_tags,
        port_info=port_info,
        stack=stack,
    )

    orders = {entity.name: entity.mesh_order for entity in entities}
    assert orders["via1"] == 1
    assert orders["P1"] == -1
    assert orders["core"] == 2
    assert orders["SiO2"] == 3
    assert orders["air"] == 4


def test_add_patterned_dielectrics_skips_covered_dielectric_layers():
    """Layers covered by bulk dielectric boxes must not be re-extruded."""
    import gdsfactory as gf
    import klayout.db as kdb

    from gsim.common.stack.materials import MATERIALS_DB
    from gsim.palace.mesh.geometry import (
        add_dielectrics,
        add_metals,
        add_patterned_dielectrics,
        build_entities,
        extract_geometry,
    )

    gf.gpdk.PDK.activate()

    c = gf.Component("covered_dielectric_test")
    # Put polygons on layers 1, 2, 3
    for layernum in (1, 2, 3):
        idx = c.kdb_cell.layout().layer(layernum, 0)
        box = kdb.DBox(0, 0, 100, 50).to_itype(c.kcl.dbu)
        c.kdb_cell.shapes(idx).insert(box)

    stack = LayerStack(pdk_name="test")
    stack.layers["SUBSTRATE"] = Layer(
        name="SUBSTRATE",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=500.0,
        thickness=500.0,
        material="sapphire",
        layer_type="dielectric",
    )
    stack.layers["METAL"] = Layer(
        name="METAL",
        gds_layer=(2, 0),
        zmin=500.0,
        zmax=500.0,
        thickness=0.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.layers["VACUUM"] = Layer(
        name="VACUUM",
        gds_layer=(3, 0),
        zmin=500.0,
        zmax=1000.0,
        thickness=500.0,
        material="vacuum",
        layer_type="dielectric",
    )
    # Bulk dielectric boxes cover the same z-ranges as SUBSTRATE and VACUUM
    stack.dielectrics = [
        {"name": "substrate", "zmin": 0.0, "zmax": 500.0, "material": "sapphire"},
        {"name": "vacuum", "zmin": 500.0, "zmax": 1000.0, "material": "vacuum"},
    ]
    stack.materials = {
        "sapphire": MATERIALS_DB["sapphire"].to_dict(),
        "aluminum": MATERIALS_DB["aluminum"].to_dict(),
        "vacuum": MATERIALS_DB["vacuum"].to_dict(),
    }

    geometry = extract_geometry(c, stack)

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("test")
    kernel = gmsh.model.occ

    metal_tags = add_metals(kernel, geometry, stack, planar_conductors=True)
    dielectric_tags = add_dielectrics(kernel, geometry, stack, margin_x=50, margin_y=50)
    patterned_tags = add_patterned_dielectrics(kernel, geometry, stack)

    # Patterned tags must be empty because SUBSTRATE and VACUUM are fully
    # covered by the bulk sapphire and vacuum dielectric boxes.
    assert not patterned_tags, (
        f"Patterned dielectrics should be empty when layers are covered by "
        f"bulk boxes, got: {patterned_tags}"
    )

    entities = build_entities(metal_tags, dielectric_tags, patterned_tags, {}, [])
    entity_names = {e.name for e in entities}
    assert "sapphire" in entity_names, (
        f"Bulk sapphire box should be present, got entities: {entity_names}"
    )
    assert "vacuum" in entity_names, (
        f"Bulk vacuum box should be present, got entities: {entity_names}"
    )
    assert "SUBSTRATE" not in entity_names, (
        f"SUBSTRATE patterned volume should be skipped, got entities: {entity_names}"
    )
    assert "VACUUM" not in entity_names, (
        f"VACUUM patterned volume should be skipped, got entities: {entity_names}"
    )
