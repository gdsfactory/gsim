"""Tetrahedron distortion using the Palace/MFEM geometric convention."""

from __future__ import annotations

import gmsh
import numpy as np

# Columns are the edges of an equilateral reference tetrahedron. MFEM's
# Mesh::GetElementJacobian applies this same perfect-element normalization.
# https://docs.mfem.org/4.8/mesh_8cpp_source.html#l00062
_IDEAL_JACOBIAN_INVERSE = np.linalg.inv(
    np.array(
        [
            [1.0, 0.5, 0.5],
            [0.0, np.sqrt(3.0) / 2, np.sqrt(3.0) / 6],
            [0.0, 0.0, np.sqrt(2.0 / 3)],
        ]
    )
)
_BATCH_SIZE = 16_384


def tetrahedron_distortion(
    blocks: list[tuple[int, np.ndarray, np.ndarray]],
    node_tags: np.ndarray,
    coordinates: np.ndarray,
) -> dict:
    """Return the worst normalized Jacobian condition number at tet centers.

    Each block holds a Gmsh element type, element tags and flattened node tags.
    Coordinates correspond to ``node_tags``, which need not be sorted or dense.
    Gmsh must be initialized to evaluate the geometry basis derivatives.

    Kappa is sigma_max / sigma_min after mapping an equilateral reference tet
    to the physical element. It is 1 for an ideal tet and grows with distortion.
    Linear tets have constant Jacobians; curved tets are sampled only at their
    reference centers, as in MFEM, not searched for their worst interior value.
    This unsigned shape metric does not replace signed validity checks (SICN).

    Temporary coordinate/Jacobian arrays are bounded by ``_BATCH_SIZE``. A
    numerically singular center gives ``max=None`` and a nonzero
    ``singular_elements`` count, keeping the result strict-JSON compatible.
    """
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]
    points = np.asarray(coordinates).reshape(-1, 3)
    maximum = 0.0
    worst_tag = None
    singular_count = 0

    for element_type, element_tags, connectivity in blocks:
        components, gradients, orientations = gmsh.model.mesh.getBasisFunctions(
            int(element_type), [0.25, 0.25, 0.25], "GradLagrange"
        )
        if components != 3 or orientations != 1:
            raise ValueError("Expected scalar Lagrange geometry basis gradients")
        gradients = np.asarray(gradients).reshape(-1, 3)
        connectivity = connectivity.reshape(-1, len(gradients))
        for start in range(0, len(element_tags), _BATCH_SIZE):
            tags = connectivity[start : start + _BATCH_SIZE]
            indices = np.searchsorted(sorted_tags, tags)
            if np.any(indices >= len(sorted_tags)) or not np.array_equal(
                sorted_tags[indices], tags
            ):
                raise ValueError("Tetrahedron references an unknown mesh node")
            element_points = points[order[indices]]
            # Translation does not change a Jacobian. Subtracting one point
            # also avoids cancellation from large absolute coordinate offsets.
            element_points -= element_points[:, :1].copy()
            jacobians = np.einsum("eni,nj->eij", element_points, gradients)
            jacobians = jacobians @ _IDEAL_JACOBIAN_INVERSE
            scales = np.max(np.abs(jacobians), axis=(1, 2), keepdims=True)
            np.divide(jacobians, scales, out=jacobians, where=scales > 0)
            singular_values = np.linalg.svd(jacobians, compute_uv=False)
            largest, smallest = singular_values[:, 0], singular_values[:, -1]
            singular = smallest <= 3 * np.finfo(float).eps * largest
            values = np.full(len(largest), np.inf)
            np.divide(largest, smallest, out=values, where=~singular)
            singular_count += int(np.count_nonzero(singular))
            index = int(np.argmax(values))
            if values[index] > maximum:
                maximum = float(values[index])
                worst_tag = int(element_tags[start + index])

    return {
        "max": None if singular_count else maximum,
        "worst_element_tag": worst_tag,
        "singular_elements": singular_count,
        "sample_location": "tetrahedron center",
    }
