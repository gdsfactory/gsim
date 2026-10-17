"""Reject malformed solver meshes while preserving a valid MSH 2.2 file."""

import meshio
import numpy as np
import pytest

from gsim.fdtd.mesh_validation import validate_mesh
from gsim.fdtd.models import FDTDGeometryError
from tests.fdtd.test_config import _manifest


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "sections",
        "format",
        "numbering",
        "names",
        "volume-tag",
        "port-tag",
        "volume-dimension",
        "port-dimension",
    ],
)
def test_mesh_preflight_rejects_incompatible_group_contracts(tmp_path, invalid):
    manifest = _manifest()
    names = {"background": [11, 3], "core": [17, 3], "port_o1": [23, 2]}
    if invalid == "names":
        names["unexpected"] = [24, 2]
    if invalid in {"volume-dimension", "port-dimension"}:
        names["core" if invalid == "volume-dimension" else "port_o1"][1] = (
            2 if invalid == "volume-dimension" else 3
        )
    path = tmp_path / "mesh.msh"
    mesh = meshio.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float),
        cells=[("tetra", [[0, 1, 2, 3], [0, 2, 1, 4]]), ("triangle", [[0, 1, 2]])],
        field_data=names,
        cell_data={
            "gmsh:physical": [
                np.array([11, 99 if invalid == "volume-tag" else 17]),
                np.array([99 if invalid == "port-tag" else 23]),
            ],
            "gmsh:geometrical": [np.array([1, 2]), np.array([3])],
        },
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    if invalid in {"sections", "format", "numbering"}:
        text = path.read_text()
        text = (
            text.replace("$Nodes", "$Other", 1)
            if invalid == "sections"
            else text.replace("2.2 0 8", "4.1 0 8", 1)
            if invalid == "format"
            else text.replace("\n1 0", "\n9 0", 1)
        )
        path.write_text(text)
    if invalid is None:
        validate_mesh(path, manifest)
    else:
        with pytest.raises(FDTDGeometryError):
            validate_mesh(path, manifest)
