"""Check explicit mesh options and impedance boundary normalization."""

from unittest.mock import Mock

import pytest

from gsim.common.stack import LayerStack
from gsim.palace import DrivenSim
from gsim.palace.mesh.config_generator import (
    _resolve_impedance_boundaries,
    generate_palace_config,
)
from gsim.palace.models.ports import ImpedanceBoundaryConfig


@pytest.mark.parametrize(
    "target", ["attributes", "interface", "reverse", "missing", "zero-length"]
)
def test_impedance_boundary_normalization(target):
    values = {"resistance": 2, "inductance": 3e-9, "capacitance": 4e-12}
    boundary = ImpedanceBoundaryConfig(
        **values,
        **(
            {"attributes": [2, 1]}
            if target == "attributes"
            else {"layer_a": "a", "layer_b": "b"}
        ),
    )
    groups = {
        "interface_surfaces": {}
        if target == "missing"
        else {
            "interface_b_a" if target == "reverse" else "interface_a_b": {
                "phys_group": 7,
                "length_um": 0 if target == "zero-length" else 2,
            }
        }
    }
    entries = _resolve_impedance_boundaries(
        {"_impedance_boundaries": [object(), boundary]}, groups
    )
    if target == "missing":
        assert entries == []
    else:
        length = (
            1 if target == "attributes" else (1e-6 if target == "zero-length" else 2e-6)
        )
        assert entries == [
            {
                "Attributes": [1, 2] if target == "attributes" else [7],
                "Rs": 2 / length,
                "Ls": 3e-9 / length,
                "Cs": 4e-12 / length,
            }
        ]


@pytest.mark.parametrize("preset", ["default", "coarse", "fine"])
def test_explicit_mesh_options_survive_presets(preset):
    sim = DrivenSim()
    options = dict(
        fmax=3e9,
        curve_fit_mode="spline",
        curve_fit_layers=["core"],
        curve_fit_tolerance_um=0.02,
        curve_fit_min_points=10,
        curve_fit_corner_angle_deg=60,
        high_order_elements=True,
        high_order_order=3,
        high_order_optimize=False,
    )
    config = sim._build_mesh_config(
        preset=preset,
        refined_mesh_size=1,
        max_mesh_size=2,
        margin=3,
        airbox_margin=None,
        planar_conductors=True,
        show_gui=False,
        **options,
    )
    for name, expected in options.items():
        assert getattr(config, name) == expected
    with pytest.raises(ValueError, match="Unknown preset"):
        sim._build_mesh_config(
            preset="invalid",
            refined_mesh_size=None,
            max_mesh_size=None,
            margin=None,
            airbox_margin=None,
            fmax=None,
            planar_conductors=None,
            show_gui=False,
        )
    with pytest.raises(ValueError, match="airbox_margin has been removed"):
        sim._build_mesh_config(
            preset=None,
            refined_mesh_size=1,
            max_mesh_size=None,
            margin=None,
            airbox_margin=1,
            fmax=None,
            planar_conductors=None,
            show_gui=False,
        )


@pytest.mark.parametrize("interactive", [False, True])
def test_mesh_plot_paths_are_validated(tmp_path, monkeypatch, interactive):
    import gsim.viz

    sim = DrivenSim()
    with pytest.raises(ValueError, match="Output directory not set"):
        sim.plot_mesh()
    sim.set_output_dir(tmp_path)
    with pytest.raises(ValueError, match="Mesh file not found"):
        sim.plot_mesh()
    path = tmp_path / "palace.msh"
    path.touch()
    display = Mock()
    monkeypatch.setattr(gsim.viz, "plot_mesh", display)
    sim.plot_mesh(interactive=interactive)
    assert display.call_args.kwargs["msh_path"] == path
    assert display.call_args.kwargs["output"] == (
        None if interactive else tmp_path / "mesh.png"
    )


def test_invalid_solver_type_is_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="Unsupported simulation type"):
        generate_palace_config(
            {}, [], [], LayerStack(), tmp_path, "model", 1e9, simulation_type="invalid"
        )


@pytest.mark.parametrize("wall", ["scalar", "list", "absent", "absorbing"])
@pytest.mark.parametrize("shaped", [False, True])
def test_metallic_wall_and_anisotropic_material_configuration(tmp_path, wall, shaped):
    import json

    properties = {
        "permittivity": [4, 5, 6],
        "permeability": [1, 2, 3],
        "material_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    stack = LayerStack(materials={"core": properties})
    groups = {
        "volumes": {"core": {"phys_group": 1, "is_shaped_dielectric": shaped}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {}
        if wall == "absent"
        else {"absorbing": {"phys_group": [2, 3] if wall == "list" else 2}},
    }
    options = dict(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="model",
        fmax=1e9,
        simulation_type="boundarymode",
        absorbing_boundary=wall == "absorbing",
        hints={"_metallic_boundaries": True},
    )
    if wall in {"absent", "absorbing"}:
        with pytest.raises(ValueError, match="outer"):
            generate_palace_config(**options)
    else:
        config = json.loads(generate_palace_config(**options).read_text())
        assert config["Boundaries"]["PEC"]["Attributes"] == (
            [2, 3] if wall == "list" else [2]
        )
        material = config["Domains"]["Materials"][0]
        assert material["Permittivity"] == [4, 5, 6]
        assert material["Permeability"] == [1, 2, 3]
        assert material["MaterialAxes"] == properties["material_axes"]
