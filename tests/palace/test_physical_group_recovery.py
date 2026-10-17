"""Recover conductor refinement curves after boolean operations renumber entities."""

from typing import Any

import gmsh
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.gmsh_utils import Entity
from gsim.palace.mesh.groups import assign_physical_groups


@pytest.mark.parametrize(
    "recovery",
    ["live", "mixed-curves", "no-hints", "merged-face", "stale-face", "missing-groups"],
)
def test_group_assignment_preserves_ports_materials_and_recovers_refinement(
    kernel, recovery
):
    volume = kernel.addBox(0, 0, 0, 2, 2, 2)
    surface = kernel.addRectangle(0, 0, 3, 2, 2)
    kernel.synchronize()
    curves = [
        tag
        for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False)
        if dim == 1
    ]
    names = [
        "oxide",
        "air",
        "via",
        "device",
        "m1_pec",
        "m1_xy",
        "m1_z",
        "block_xy",
        "block_z",
        "P1",
        "P2_E0",
        "P2_E1",
    ]
    entities = [
        Entity(
            name,
            3 if name in names[:4] else 2,
            1,
            [volume if name in names[:4] else surface],
        )
        for name in names
    ]
    groups = {}
    for entity in entities:
        groups[entity.name] = gmsh.model.addPhysicalGroup(
            entity.dim, [tag for _, tag in entity.dimtags], name=entity.name
        )
    groups.update(
        {
            name: gmsh.model.addPhysicalGroup(2, [surface], name=name)
            for name in ["oxide__via", "m1__via", "via__None", "air__None"]
        }
    )
    metadata: dict[str, Any] = {
        "m1": {
            "surfaces_xy": [surface],
            "volumes": [volume],
            "refinement_lines": curves,
        },
        "shell": {"refinement_lines": [*curves, 999999]},
        "__shaped_dielectrics__": {"device"},
        "__pec_surface_bboxes__": {"m1": [gmsh.model.getBoundingBox(2, surface)]},
    }
    if recovery == "mixed-curves":
        metadata["m1"]["refinement_lines"] = [*curves, 999999]
    elif recovery in {"no-hints", "merged-face", "stale-face"}:
        metadata["m1"]["refinement_lines"] = []
    if recovery == "merged-face":
        entities = [entity for entity in entities if entity.name != "m1_pec"]
    elif recovery == "stale-face":
        next(entity for entity in entities if entity.name == "m1_pec").dimtags = [
            (2, 999999)
        ]
    elif recovery == "missing-groups":
        groups = {}
    stack = LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(index, 0),
                zmin=0,
                zmax=2,
                thickness=2,
                material="aluminum",
                layer_type=kind,
            )
            for index, (name, kind) in enumerate(
                [("m1", "conductor"), ("via", "via")], 1
            )
        }
    )
    result = assign_physical_groups(
        kernel,
        metadata,
        {"oxide": [], "airbox": [], "absent": []},
        {"P1": [surface], "P2": [surface, surface]},
        [
            {"portnumber": 1},
            {
                "portnumber": 2,
                "type": "cpw",
                "elements": [{"direction": "+X"}, {"direction": "-X"}],
            },
        ],
        entities,
        groups,
        stack,
        pec_block_tags={"block": []},
    )
    assert result["refinement_lines"]["m1"]["tags"] == sorted(curves)
    assert result["refinement_lines"]["shell"]["tags"] == sorted(curves)
    if recovery == "missing-groups":
        assert (
            result["volumes"]
            == result["conductor_surfaces"]
            == result["pec_surfaces"]
            == {}
        )
        assert result["boundary_surfaces"] == {}
        assert result["port_surfaces"]["P2"]["elements"] == []
    else:
        assert set(result["volumes"]) == {"oxide", "air", "via", "device"}
        assert result["volumes"]["via"]["is_via"]
        assert result["volumes"]["device"]["is_shaped_dielectric"]
        assert result["port_surfaces"]["P1"]["tags"] == [surface]
        assert [
            entry["direction"] for entry in result["port_surfaces"]["P2"]["elements"]
        ] == ["+X", "-X"]
        assert result["via_boundary_surfaces"] == {"via": [groups["oxide__via"]]}
