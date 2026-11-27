"""Prepare the Palace jobs for the conductor-model-versus-skin-depth figure.

One gpdk coplanar line, one frequency per job. The two required models give the
10 jobs of the figure:

``surface``
    The default shell with a surface-impedance (``Boundaries.Conductivity``) condition,
    meshed as gsim meshes it (no elements are asked for inside the shell).
``volume``
    The metal layer classified as a dielectric, so that it is meshed as a conductive
    volume (the route described in ``docs/palace_conductors.md``), with elements about
    the skin depth or smaller inside the metal.

Two optional models, five jobs each, remove confounders:

``pec``
    PEC sheets (``planar_conductors=True``). It carries every loss that is not conductor
    loss (radiation, absorbing wall, solver error), so that the analysis can subtract it.
``surface_matched``
    The ``surface`` model on the very mesh sizing of ``volume``, so that the only
    difference between the two is the conductor model, not the elements around the metal.

The metal thickness is fixed; t / delta is set by the frequency:
``f = (t / delta)**2 / (pi * mu0 * sigma * t**2)``.

Each job is meshed, checked and written with ``write_config()`` into
``jobs/<name>/``, ready to be uploaded by a person. Nothing is solved here, and
nothing is submitted: this script never calls ``run()``, an upload or a start.

Usage (from the repository root)::

    uv run python nbs/data/conductors/make_skin_depth_jobs.py            # all jobs
    uv run python nbs/data/conductors/make_skin_depth_jobs.py --list     # plan only
    uv run python nbs/data/conductors/make_skin_depth_jobs.py --only volume_r10

The script is idempotent: a job folder is rebuilt from scratch, and the mesh is
deterministic for a given gmsh version (the manifest records ``mesh_hash``).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import shutil
import sys
from pathlib import Path

from scipy.constants import mu_0, pi

HERE = Path(__file__).resolve().parent
JOBS_DIR = HERE / "jobs"
MANIFEST = JOBS_DIR / "manifest.json"

# --- the line (um) -----------------------------------------------------------------
THICKNESS_UM = 5.0  # raised from the gpdk 0.7 um so that t / delta = 10 is below 30 GHz
LENGTH_UM = 40.0
SIGNAL_UM = 10.0
GAP_UM = 6.0
GROUND_UM = 10.0
PORT_LENGTH_UM = 5.0
SUBSTRATE_UM = 2.0  # oxide fill below z = 0, as in docs/palace_conductors.md
OXIDE_ABOVE_METAL_UM = 3.4  # oxide above the metal, as in the gpdk stack (1.8 to 5.2)
MARGIN_XY_UM = 50.0
MARGIN_Z_UM = 50.0

# --- the mesh (um) -----------------------------------------------------------------
REFINED_UM = 1.0  # conductor edges and port edges, all models
MAX_UM = 30.0  # far field, all models
INTERIOR_OVER_DELTA = 0.75  # box size in the metal; gmsh edges come out ~1.3x the size
MIN_CELLS_ACROSS = 4  # and at least this many box sizes across the thickness
EDGE_OVER_DELTA_LIMIT = 1.2  # measured mean edge in the metal must stay below this

# --- the jobs ----------------------------------------------------------------------
SIGMA_S_PER_M = 3.77e7  # gsim "aluminum", the gpdk metal1 material
RATIOS = (0.5, 1.0, 2.4, 5.0, 10.0)
MODELS = ("surface", "volume", "pec", "surface_matched")
OPTIONAL_MODELS = ("pec", "surface_matched")
INTERIOR_MODELS = ("volume", "surface_matched")  # models with the box inside the metal
SOLVER_ORDER = 2
SOLVER_TOLERANCE = 1e-8  # the metal loss is a small difference; default is 1e-6


def skin_depth_um(frequency_hz: float, sigma: float = SIGMA_S_PER_M) -> float:
    """``1 / sqrt(pi f mu0 sigma)`` in um."""
    return 1e6 / math.sqrt(pi * frequency_hz * mu_0 * sigma)


def frequency_for_ratio(
    ratio: float, thickness_um: float = THICKNESS_UM, sigma: float = SIGMA_S_PER_M
) -> float:
    """Frequency (Hz, four significant digits) at which ``thickness / delta == ratio``."""
    t = thickness_um * 1e-6
    exact = ratio**2 / (pi * mu_0 * sigma * t**2)
    return float(f"{exact:.4g}")


def job_name(model: str, ratio: float) -> str:
    """Folder name, e.g. ``volume_r2p4``."""
    return f"{model}_r{ratio:g}".replace(".", "p")


def planned_jobs(models: tuple[str, ...] = MODELS) -> list[dict]:
    """The job list without building anything (no gsim, no gmsh)."""
    jobs = []
    for model in models:
        for ratio in RATIOS:
            f_hz = frequency_for_ratio(ratio)
            jobs.append(
                {
                    "name": job_name(model, ratio),
                    "model": model,
                    "required": model not in OPTIONAL_MODELS,
                    "t_over_delta_requested": ratio,
                    "frequency_hz": f_hz,
                    "frequency_ghz": f_hz / 1e9,
                    "skin_depth_um": skin_depth_um(f_hz),
                    "t_over_delta": THICKNESS_UM / skin_depth_um(f_hz),
                    "thickness_um": THICKNESS_UM,
                    "length_um": LENGTH_UM,
                    "sigma_s_per_m": SIGMA_S_PER_M,
                }
            )
    return jobs


def interior_size_um(delta_um: float) -> float:
    """Element size asked for inside the metal: about delta, and 4 across the thickness."""
    return min(INTERIOR_OVER_DELTA * delta_um, THICKNESS_UM / MIN_CELLS_ACROSS)


# -------------------------------------------------------------------------------------
# Building (needs gsim, gdsfactory, gmsh)
# -------------------------------------------------------------------------------------


@functools.cache
def _line_cell():
    """The cell function of the page's gpdk CPW line (``docs/palace_conductors.md``)."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()

    @gf.cell
    def skin_depth_line(length: float, ground: float) -> gf.Component:
        layer = gf.gpdk.LAYER.M1
        c = gf.Component()
        top = c << gf.c.rectangle((length, ground), centered=True, layer=layer)
        top.move((0, (ground + SIGNAL_UM) / 2 + GAP_UM))
        c << gf.c.rectangle((length, SIGNAL_UM), centered=True, layer=layer)
        bottom = c << gf.c.rectangle((length, ground), centered=True, layer=layer)
        bottom.move((0, -(ground + SIGNAL_UM) / 2 - GAP_UM))
        for name, x, orientation in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
            c.add_port(
                name=name,
                center=(x, 0),
                width=SIGNAL_UM,
                orientation=orientation,
                port_type="electrical",
                layer=layer,
            )
        return c

    return skin_depth_line


def _line(length: float, ground: float):
    """The line, ``length`` long, with grounds ``ground`` wide (um)."""
    return _line_cell()(length, ground)


def _stack(*, as_volume: bool, thickness_um: float):
    """The gpdk stack with ``metal1`` made ``thickness_um`` thick."""
    import gdsfactory as gf

    from gsim.common.stack import get_stack

    gf.gpdk.PDK.activate()
    stack = get_stack(substrate_thickness=SUBSTRATE_UM)
    metal = stack.layers["metal1"]
    top = metal.zmin + thickness_um
    metal.zmax = top
    metal.thickness = thickness_um
    # the oxide region must still cover the metal, with the gpdk cladding above it
    stack.dielectrics[0]["zmax"] = top + OXIDE_ABOVE_METAL_UM
    if as_volume:
        metal.layer_type = "dielectric"  # the material stays Aluminum
    return stack


@contextlib.contextmanager
def _interior_refinement(
    box: tuple[float, ...] | None, size_um: float, outer_um: float
):
    """Hold the elements inside ``box`` to ``size_um`` for the meshes made in the block.

    gsim sizes the mesh from the conductor edges only, so the inside of a thick metal
    volume would get elements of several um. This adds one gmsh ``Box`` field to the
    list that gsim hands to ``finalize_mesh_fields``. ``box`` is
    ``(xmin, xmax, ymin, ymax, zmin, zmax)``; ``None`` leaves the mesh as gsim makes it.
    """
    import gmsh

    from gsim.palace.mesh import gmsh_utils

    original = gmsh_utils.finalize_mesh_fields

    def with_box(field_ids: list[int]) -> None:
        if box is not None:
            field_id = max(field_ids) + 100
            xmin, xmax, ymin, ymax, zmin, zmax = box
            field = gmsh.model.mesh.field
            field.add("Box", field_id)
            field.setNumber(field_id, "VIn", size_um)
            field.setNumber(field_id, "VOut", outer_um)
            field.setNumber(field_id, "Thickness", 3 * size_um)
            for key, value in (
                ("XMin", xmin),
                ("XMax", xmax),
                ("YMin", ymin),
                ("YMax", ymax),
                ("ZMin", zmin),
                ("ZMax", zmax),
            ):
                field.setNumber(field_id, key, value)
            field_ids = [*field_ids, field_id]
        original(field_ids)

    gmsh_utils.finalize_mesh_fields = with_box
    try:
        yield
    finally:
        gmsh_utils.finalize_mesh_fields = original


def _tets_in_group(mesh_path: Path, phys_group: int) -> tuple[int, float, float]:
    """Tetrahedra of one physical group: count, mean edge and longest edge (um)."""
    import gmsh
    import numpy as np

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(mesh_path))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        xyz = coords.reshape(-1, 3)
        count, total, longest = 0, 0.0, 0.0
        for entity in gmsh.model.getEntitiesForPhysicalGroup(3, phys_group):
            _types, _tags, nodes = gmsh.model.mesh.getElements(3, entity)
            for block in nodes:
                corners = xyz[index[np.asarray(block, dtype=np.int64).reshape(-1, 4)]]
                for a in range(4):
                    for b in range(a + 1, 4):
                        lengths = np.linalg.norm(corners[:, a] - corners[:, b], axis=1)
                        total += float(lengths.sum())
                        longest = max(longest, float(lengths.max()))
                count += len(corners)
        return count, total / (6 * count), longest
    finally:
        gmsh.finalize()


def build_job(
    job: dict, *, jobs_dir: Path = JOBS_DIR, length_um: float | None = None
) -> dict:
    """Build, mesh, check and write one job. Returns its manifest entry."""
    from gsim.common.stack.materials import MATERIALS_DB
    from gsim.palace import DrivenSim

    if MATERIALS_DB["aluminum"].conductivity != SIGMA_S_PER_M:
        raise RuntimeError("gsim's aluminum conductivity changed; update SIGMA_S_PER_M")

    length = LENGTH_UM if length_um is None else length_um
    model = job["model"]
    out_dir = jobs_dir / job["name"]
    if out_dir.exists():
        shutil.rmtree(
            out_dir
        )  # idempotent: rebuilt from scratch, only this job's folder

    stack = _stack(as_volume=model == "volume", thickness_um=job["thickness_um"])
    metal = stack.layers["metal1"]
    half_y = SIGNAL_UM / 2 + GAP_UM + GROUND_UM

    sim = DrivenSim()
    sim.set_output_dir(out_dir)
    sim.set_geometry(_line(length, GROUND_UM))
    sim.set_stack(stack)
    sim.set_airbox(
        margin_x=MARGIN_XY_UM,
        margin_y=MARGIN_XY_UM,
        z_above=MARGIN_Z_UM,
        z_below=MARGIN_Z_UM,
    )
    sim.add_cpw_port(
        "o1", layer="metal1", s_width=SIGNAL_UM, gap_width=GAP_UM, length=PORT_LENGTH_UM
    )
    sim.add_cpw_port(
        "o2",
        layer="metal1",
        s_width=SIGNAL_UM,
        gap_width=GAP_UM,
        length=PORT_LENGTH_UM,
        excited=False,
    )
    sim.set_solver(order=SOLVER_ORDER, tolerance=SOLVER_TOLERANCE)
    sim.set_driven(f=job["frequency_hz"], adaptive_tol=0.0)

    delta = job["skin_depth_um"]
    box = None
    size = None
    if model in INTERIOR_MODELS:
        size = interior_size_um(delta)
        tol = 1e-3
        box = (
            -length / 2 - tol,
            length / 2 + tol,
            -half_y - tol,
            half_y + tol,
            metal.zmin - tol,
            metal.zmax + tol,
        )

    with _interior_refinement(box, size or 0.0, MAX_UM):
        result = sim.mesh(
            refined_mesh_size=REFINED_UM,
            max_mesh_size=MAX_UM,
            planar_conductors=model == "pec",
            verbose=False,
        )
    config_path = sim.write_config()  # nothing is solved or submitted
    valid = sim.validate_config().valid

    metadata = json.loads((out_dir / "metadata.json").read_text())["mesh"]
    entry = {
        **job,
        "length_um": length,
        "folder": f"jobs/{job['name']}",
        "tetrahedra": metadata["tetrahedra"],
        "estimated_field_dofs": metadata["field_dofs"]["estimated_field_dofs"],
        "field_order": metadata["field_dofs"]["field_order"],
        "validate_config_valid": bool(valid),
        "mesh_hash": metadata.get("mesh_hash"),
        "versions": metadata.get("versions"),
        "mesh_bytes": result.mesh_path.stat().st_size,
        "config_conductivity_entries": None,
    }
    config = json.loads(Path(config_path).read_text())
    entry["config_conductivity_entries"] = len(
        config["Boundaries"].get("Conductivity", [])
    )
    entry["config_has_pec"] = "PEC" in config["Boundaries"]

    if model == "volume":
        group = sim.mesh_groups["volumes"]["metal1"]["phys_group"]
        group = group if isinstance(group, int) else group[0]
        tets, mean_edge, longest = _tets_in_group(result.mesh_path, group)
        entry.update(
            {
                "interior_size_um": size,
                "metal_tetrahedra": tets,
                "metal_mean_edge_um": mean_edge,
                "metal_longest_edge_um": longest,
                "metal_mean_edge_over_delta": mean_edge / delta,
            }
        )
        if mean_edge > EDGE_OVER_DELTA_LIMIT * delta and size < delta:
            raise RuntimeError(
                f"{job['name']}: mean edge in the metal is {mean_edge:.3g} um, more than "
                f"{EDGE_OVER_DELTA_LIMIT} x delta = {delta:.3g} um"
            )

    try:
        from gsim.gcloud import estimate_runtime_seconds

        entry["runtime_estimate_s"] = round(estimate_runtime_seconds(out_dir), 1)
    except Exception:  # the heuristic needs the cloud client importable; it is optional
        entry["runtime_estimate_s"] = None
    return entry


def _order(entries: list[dict]) -> list[dict]:
    rank = {model: i for i, model in enumerate(MODELS)}
    return sorted(
        entries, key=lambda e: (rank[e["model"]], e["t_over_delta_requested"])
    )


def write_manifest(entries: list[dict], path: Path = MANIFEST) -> None:
    """Write ``manifest.json``: one entry per job plus the shared line description."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "description": "Palace driven jobs: conductor model versus skin depth (gpdk line)",
        "line": {
            "thickness_um": THICKNESS_UM,
            "length_um": LENGTH_UM,
            "signal_um": SIGNAL_UM,
            "gap_um": GAP_UM,
            "ground_um": GROUND_UM,
            "port_length_um": PORT_LENGTH_UM,
            "sigma_s_per_m": SIGMA_S_PER_M,
        },
        "mesh": {
            "refined_um": REFINED_UM,
            "max_um": MAX_UM,
            "interior_over_delta": INTERIOR_OVER_DELTA,
            "min_cells_across_thickness": MIN_CELLS_ACROSS,
        },
        "solver": {
            "order": SOLVER_ORDER,
            "tolerance": SOLVER_TOLERANCE,
            "adaptive_tol": 0.0,
        },
        "jobs": _order(entries),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _read_manifest(path: Path = MANIFEST) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["jobs"]


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--list", action="store_true", help="print the job plan and exit"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="build only these jobs"
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="skip the optional models (pec, surface_matched): 10 jobs",
    )
    args = parser.parse_args(argv)

    models = tuple(
        m for m in MODELS if not (args.required_only and m in OPTIONAL_MODELS)
    )
    jobs = planned_jobs(models)
    if args.only:
        unknown = set(args.only) - {j["name"] for j in jobs}
        if unknown:
            parser.error(f"unknown job(s): {', '.join(sorted(unknown))}")
        jobs = [j for j in jobs if j["name"] in args.only]

    if args.list:
        for j in jobs:
            print(
                f"{j['name']:22s} f = {j['frequency_ghz']:9.4f} GHz  "
                f"delta = {j['skin_depth_um']:.4f} um  required = {j['required']}"
            )
        return 0

    built = {e["name"]: e for e in _read_manifest()} if args.only else {}
    for job in jobs:
        entry = build_job(job)
        built[entry["name"]] = entry
        print(
            f"{entry['name']:22s} tets {entry['tetrahedra']:>9,}  "
            f"DOFs {entry['estimated_field_dofs']:>11,}  valid {entry['validate_config_valid']}",
            flush=True,
        )
        write_manifest(list(built.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
