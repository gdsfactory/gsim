"""Physical groups per electrical net for electrostatic terminals.

The mesher makes one physical group per entity name, which is one per
conductor layer, and one group for all the boundary faces of a via layer.
Two electrodes on one layer, or the vias of two nets, then cannot be told
apart in the Palace config.

When terminals select nets, ``generate_mesh`` runs these steps:

1. before meshing, :func:`check_via_merging` makes sure that merging nearby vias
   does not join the vias of two nets, and, with ``clip_merged_vias``, that no
   clipped via volume lies nearer to the vias of another net than to its own;
2. after meshing and before the file is written, :func:`split_groups_by_net`
   replaces each conductor, planar-conductor and via-boundary group with one
   group per net, named ``<old name>@<net name>``. A surface is assigned to the
   net whose shape its mesh elements lie on. The mesh itself is not changed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gsim.common.stack import LayerStack
    from gsim.palace.mesh.geometry import GeometryData
    from gsim.palace.mesh.nets import Nets

logger = logging.getLogger(__name__)

# um: how far a mesh element of a conductor face may be from the drawn shape
_CONDUCTOR_TOLERANCE_UM = 0.05
# um: slack of a via face on top of half the merge distance (vias are merged by
# growing them by merge_distance / 2 + 0.01 and shrinking back)
_VIA_TOLERANCE_MARGIN_UM = 0.06
# a merged via polygon overlaps an original via when they share at least this area
_MIN_OVERLAP_UM2 = 1e-9
# um: spacing of the points tested along the outline of a clipped via volume, and
# the most points per outline (the spacing grows for a long outline)
_OUTLINE_STEP_UM = 0.02
_MAX_OUTLINE_POINTS = 4000
# um: a point is nearer to another net when it is closer to it by more than this
_NEAREST_NET_SLACK_UM = 1e-9
_HINT = (
    "Curve fitting (curve_fit_mode) and polygon decimation (decimate_tolerance) "
    'move edges; try curve_fit_mode="line" or no decimation'
)


def net_labels(nets: Nets) -> list[str]:
    """One unique label per net: its name, with ``#<position>`` when names repeat.

    Two nets can carry the same port name; their labels still differ, so the
    physical groups and group keys of the nets stay apart.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for k, net in enumerate(nets.nets):
        label = net.name
        if label in seen:
            label = f"{net.name}#{k + 1}"
        seen.add(label)
        labels.append(label)
    return labels


def _polygon(
    xs: Sequence[float], ys: Sequence[float], holes: Sequence
) -> shapely.Polygon:
    """A shapely polygon from the coordinate lists of the mesher."""
    return shapely.Polygon(
        list(zip(xs, ys, strict=True)),
        [list(zip(hx, hy, strict=True)) for hx, hy in holes],
    )


class _LayerVias:
    """The drawn vias of each net on one via layer, with an index per net.

    ``split_groups_by_net`` gives a via face to the net with the nearest drawn
    via. Clipping can keep a lobe of the landing area that reaches towards the
    vias of another net without covering them: it overlaps the vias of one net
    only, but its faces would go to the wrong net, or fail late.
    """

    def __init__(
        self, name: str, nets: Nets, by_net: dict[int, list[shapely.Polygon]]
    ) -> None:
        """Index the drawn vias of each net.

        Args:
            name: The via layer.
            nets: The nets of the geometry.
            by_net: Position of a net -> its drawn via polygons on the layer.
        """
        self.name = name
        self.nets = nets
        self.by_net = by_net
        self.trees = {k: shapely.STRtree(shapes) for k, shapes in by_net.items()}

    def _distances(self, k: int, points: Any) -> Any:
        """Distance of each point to the nearest drawn via of net ``k``."""
        _, distance = self.trees[k].query_nearest(
            points, return_distance=True, all_matches=False
        )
        return distance

    def check(
        self, outline: shapely.Polygon, k: int, overlapped: list[shapely.Polygon]
    ) -> None:
        """Fail when a clipped via volume of net ``k`` lies nearer to another net.

        Every point of the outline (sampled along it) must be at least as near
        to the vias of net ``k`` as to the vias of every other net.

        Args:
            outline: The clipped via volume as seen from above.
            k: Position of the one net whose vias it overlaps.
            overlapped: The drawn vias of that net it overlaps.

        Raises:
            ValueError: If a point of the outline is nearer to another net.
        """
        if len(self.by_net) < 2:
            return
        if outline.difference(shapely.union_all(overlapped)).area <= _MIN_OVERLAP_UM2:
            return  # the drawn cuts themselves; nothing was added
        boundary = outline.boundary
        step = max(_OUTLINE_STEP_UM, boundary.length / _MAX_OUTLINE_POINTS)
        coordinates = shapely.get_coordinates(shapely.segmentize(boundary, step))
        points = shapely.points(coordinates)
        own = self._distances(k, points)
        for j in sorted(self.by_net):
            if j == k:
                continue
            other = self._distances(j, points)
            excess = own - other
            if excess.max() <= _NEAREST_NET_SLACK_UM:
                continue
            worst = int(np.argmax(excess))
            x, y = coordinates[worst]
            _, gaps = self.trees[j].query_nearest(
                self.by_net[k], return_distance=True, all_matches=False
            )
            gap = float(gaps.min())
            msg = (
                f"Via layer '{self.name}': with clip_merged_vias the via volume "
                f"of net {self.nets[k].name!r} reaches ({x:.4g}, {y:.4g}) um, "
                f"which is {other[worst]:.3g} um from the vias of net "
                f"{self.nets[j].name!r} but {own[worst]:.3g} um from its own, so "
                "its faces cannot be assigned to a net. Set merge_via_distance "
                f"below the gap between the vias of the two nets ({gap:.3g} um; "
                "0 turns merging off)."
            )
            raise ValueError(msg)


def check_via_merging(
    geometry: GeometryData,
    stack: LayerStack,
    nets: Nets,
    merge_via_distance: float,
    clip_merged_vias: bool = False,
) -> None:
    """Fail when merging nearby vias would join the vias of two nets.

    ``add_metals`` merges the polygons of a via layer that are closer than
    ``merge_via_distance`` into one volume. If those vias belong to different
    nets, that volume shorts them.

    Args:
        geometry: The polygons of the layout, as the mesher reads them.
        stack: The layer stack.
        nets: The nets of that geometry.
        merge_via_distance: The merge distance of the mesh, in um.
        clip_merged_vias: Whether the mesh clips merged vias to the metal
            above and below them. The check then looks at the clipped vias, as
            ``add_metals`` meshes them, and also fails when a clipped via
            volume lies nearer to the vias of another net than to its own.

    Raises:
        ValueError: If a merged via polygon covers vias of more than one net,
            or a clipped one lies nearer to another net than to its own.
    """
    from gsim.palace.mesh.geometry import (
        _merge_via_layer,
        _polygons_by_gds_layer,
        _via_landing_area,
    )

    if merge_via_distance <= 0:
        return
    polygons_by_gds = _polygons_by_gds_layer(geometry.polygons)
    for name, layer in stack.layers.items():
        if layer.layer_type != "via":
            continue
        number = layer.gds_layer[0]
        indices = [
            i for i, polygon in enumerate(geometry.polygons) if polygon[0] == number
        ]
        if len(indices) < 2:
            continue
        net_of_index = {
            index: k
            for k, net in enumerate(nets.nets)
            for member_layer, index in net.members
            if member_layer == name
        }
        polys = [geometry.polygons[i][1:] for i in indices]
        originals = {
            i: _polygon(xs, ys, holes)
            for i, (xs, ys, holes) in zip(indices, polys, strict=True)
        }
        landing = (
            _via_landing_area(polygons_by_gds, stack, name)
            if clip_merged_vias
            else None
        )
        layer_vias: _LayerVias | None = None
        if clip_merged_vias:
            by_net: dict[int, list[shapely.Polygon]] = {}
            for i, original in originals.items():
                if i in net_of_index:
                    by_net.setdefault(net_of_index[i], []).append(original)
            layer_vias = _LayerVias(name, nets, by_net)
        merged = _merge_via_layer(
            [(list(xs), list(ys), holes) for xs, ys, holes in polys],
            merge_via_distance,
            landing,
            label=repr(name),
        )
        for xs, ys, holes in merged:
            outline = _polygon(xs, ys, holes)
            shapes: dict[int, list[shapely.Polygon]] = {}
            for i, original in originals.items():
                k = net_of_index.get(i)
                if k is not None and (
                    outline.intersection(original).area > _MIN_OVERLAP_UM2
                ):
                    shapes.setdefault(k, []).append(original)
            members = sorted(shapes)
            if layer_vias is not None and len(members) == 1:
                layer_vias.check(outline, members[0], shapes[members[0]])
            if len(members) < 2:
                continue
            gap = min(
                a.distance(b)
                for x, k1 in enumerate(members)
                for k2 in members[x + 1 :]
                for a in shapes[k1]
                for b in shapes[k2]
            )
            names = ", ".join(repr(nets[k].name) for k in members)
            msg = (
                f"Via layer '{name}': merge_via_distance={merge_via_distance:g} um "
                f"would merge vias of different nets ({names}) into one volume and "
                f"short them; the closest are {gap:.3g} um apart. Set "
                "merge_via_distance below that gap (0 turns merging off)."
            )
            raise ValueError(msg)


def _net_of_points(
    nets: Nets,
    layer: str,
    points: Sequence[tuple[float, float]],
    tolerance: float,
) -> int:
    """Position of the net whose shape on a layer the points lie on.

    Each point takes the nearest net with a shape on the layer within
    ``tolerance`` (um). All points must pick the same net.

    Raises:
        ValueError: If a point is farther than ``tolerance`` from every shape of
            the layer, or the points pick different nets.
    """
    position = {id(net): k for k, net in enumerate(nets.nets)}
    found: dict[int, tuple[float, float]] = {}
    for x, y in points:
        near = nets.nets_near(x, y, layer, tolerance)
        if not near:
            nearest = nets.nets_near(x, y, layer, 1e3)
            far = f"{nearest[0][0]:.3g} um" if nearest else "no shape at all"
            msg = (
                f"A conductor surface on layer '{layer}' has a mesh element at "
                f"({x:.4g}, {y:.4g}) um that is not within {tolerance:g} um of "
                f"any drawn shape (the nearest is {far} away). {_HINT}."
            )
            raise ValueError(msg)
        found.setdefault(position[id(near[0][1])], (x, y))
    if len(found) != 1:
        names = ", ".join(repr(nets[k].name) for k in sorted(found))
        x, y = next(iter(found.values()))
        msg = (
            f"A surface on layer '{layer}' lies on more than one net ({names}); "
            f"the mesh element at ({x:.4g}, {y:.4g}) um is one of them. {_HINT}."
        )
        raise ValueError(msg)
    return next(iter(found))


def _sample_points(surface: int, samples: int) -> list[tuple[float, float]]:
    """Up to ``samples`` (x, y) barycentres of the mesh elements of a surface."""
    import gmsh

    points: list[tuple[float, float]] = []
    for element_type in gmsh.model.mesh.getElementTypes(2, surface):
        centres = np.asarray(
            gmsh.model.mesh.getBarycenters(int(element_type), surface, False, True)
        ).reshape(-1, 3)
        if len(centres) == 0:
            continue
        step = np.unique(
            np.linspace(0, len(centres) - 1, min(len(centres), samples)).round()
        ).astype(int)
        points.extend((float(x), float(y)) for x, y, _ in centres[step])
    return points


def _layer_of(name: str, stack: LayerStack, nets: Nets, *, planar: bool) -> str | None:
    """The conductor or via layer a group key belongs to, None if it has none.

    Thick conductors are keyed ``<layer>_xy`` and ``<layer>_z``; planar
    conductors and thin vias are keyed by the layer name. PEC blocks have
    neither and give None.
    """
    if planar:
        layer = name
    elif name.endswith(("_xy", "_z")):
        layer = name.rsplit("_", 1)[0]
    else:
        layer = name
    info = stack.layers.get(layer)
    if info is None or info.layer_type not in ("conductor", "via"):
        return None
    return layer if nets.has_layer(layer) else None


def split_groups_by_net(
    groups: dict[str, Any],
    nets: Nets,
    stack: LayerStack,
    *,
    merge_via_distance: float,
    samples: int = 10,
) -> None:
    """Replace the layer groups of a meshed model with one group per net.

    Must be called while gmsh holds the meshed model and before it is written.
    Rewrites ``groups`` in place:

    - each entry of ``conductor_surfaces`` and ``pec_surfaces`` that belongs to
      a stack layer, and each via-boundary group, is replaced by entries
      ``"<old>@<net>"`` with the new ``phys_group``, the surface ``tags``, and
      the ``layer`` and ``net`` they belong to. Via entries are
      ``"<via>@<net>" -> [phys groups]``, as before;
    - ``groups["nets"]`` lists every net in ``Nets`` order with its name,
      label, ports, layers and the physical groups of its surfaces (empty for a
      net with no surface, such as one that is all vias).

    PEC blocks are left alone. Physical groups are numbered after the highest
    tag in use, in sorted old-name order and net order, so the result does not
    depend on set order.

    Args:
        groups: The ``groups`` of the mesh result.
        nets: The nets of the geometry that was meshed.
        stack: The layer stack.
        merge_via_distance: The merge distance used for the vias, in um.
        samples: Mesh elements tested per surface.

    Raises:
        ValueError: If a surface cannot be assigned to exactly one net.
    """
    import gmsh

    labels = net_labels(nets)
    via_tolerance = max(merge_via_distance, 0.0) / 2 + _VIA_TOLERANCE_MARGIN_UM
    next_tag = max((tag for _, tag in gmsh.model.getPhysicalGroups()), default=0) + 1
    net_groups: list[list[int]] = [[] for _ in nets.nets]

    def split(key: str, pg: int, layer: str) -> dict[int, tuple[int, list[int]]]:
        """Split one physical group; ``{net position: (new group, surfaces)}``."""
        nonlocal next_tag
        # the name in the mesh file, e.g. "m1_pec" for the planar layer "m1"
        old_name = gmsh.model.getPhysicalName(2, pg) or key
        tolerance = (
            via_tolerance
            if stack.layers[layer].layer_type == "via"
            else _CONDUCTOR_TOLERANCE_UM
        )
        by_net: dict[int, list[int]] = {}
        for surface in gmsh.model.getEntitiesForPhysicalGroup(2, pg):
            points = _sample_points(int(surface), samples)
            if not points:
                # a surface without mesh elements cannot be assigned to a net
                msg = f"Surface {int(surface)} of group '{old_name}' has no mesh."
                raise ValueError(msg)
            k = _net_of_points(nets, layer, points, tolerance)
            by_net.setdefault(k, []).append(int(surface))
        gmsh.model.removePhysicalGroups([(2, pg)])
        result: dict[int, tuple[int, list[int]]] = {}
        for k in sorted(by_net):
            tag = gmsh.model.addPhysicalGroup(
                2, by_net[k], tag=next_tag, name=f"{old_name}@{labels[k]}"
            )
            next_tag += 1
            net_groups[k].append(tag)
            result[k] = (tag, by_net[k])
        return result

    for section, planar in (("conductor_surfaces", False), ("pec_surfaces", True)):
        entries = groups.get(section, {})
        for key in sorted(entries):
            layer = _layer_of(key, stack, nets, planar=planar)
            if layer is None:
                continue
            parts = split(key, entries[key]["phys_group"], layer)
            del entries[key]
            for k, (tag, surfaces) in parts.items():
                entries[f"{key}@{labels[k]}"] = {
                    "phys_group": tag,
                    "tags": surfaces,
                    "layer": layer,
                    "net": labels[k],
                }

    via_boundary = groups.get("via_boundary_surfaces")
    if via_boundary:
        rebuilt: dict[str, list[int]] = {}
        for via in sorted(via_boundary):
            if not nets.has_layer(via):
                rebuilt[via] = via_boundary[via]
                continue
            for pg in sorted(via_boundary[via]):
                old_name = gmsh.model.getPhysicalName(2, pg)
                for k, (tag, _) in split(old_name, pg, via).items():
                    rebuilt.setdefault(f"{via}@{labels[k]}", []).append(tag)
        groups["via_boundary_surfaces"] = rebuilt

    groups["nets"] = [
        {
            "name": net.name,
            "label": labels[k],
            "ports": list(net.ports),
            "layers": list(net.layers),
            "phys_groups": sorted(net_groups[k]),
        }
        for k, net in enumerate(nets.nets)
    ]


__all__ = ["check_via_merging", "net_labels", "split_groups_by_net"]
