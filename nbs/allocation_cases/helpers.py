"""Measure notebook inputs and submit sizing-aware cloud allocation checks."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import meshio
import numpy as np


def fdtd_grid_bounds(mesh_path: Path) -> dict:
    """Measure bounds using only vertices referenced by the written mesh."""
    mesh = meshio.read(mesh_path)
    vertices = np.unique(np.concatenate([block.data.ravel() for block in mesh.cells]))
    points = mesh.points[vertices]
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


def write_fdtd_grid_bounds(mesh_path: Path, config_path: Path) -> None:
    """Add the explicit non-PML bounds required by the current cloud FDTD build."""
    config = json.loads(config_path.read_text())
    config["grid"]["non_pml_bounds"] = fdtd_grid_bounds(mesh_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def fdtd_requirements(mesh_path: Path, config_path: Path) -> dict:
    """Count the requested Yee cells, including PML, from written inputs."""
    bounds = fdtd_grid_bounds(mesh_path)
    extent_nm = np.asarray(bounds["max"]) - np.asarray(bounds["min"])
    grid = json.loads(config_path.read_text())["grid"]
    dimensions = [
        math.ceil(float(extent) / grid["nanometers_per_cell"]) + 2 * grid["pml_cells"]
        for extent in extent_nm
    ]
    print(f"Measured Yee grid including PML: {dimensions}")
    return {"fdtd": {"cells": math.prod(dimensions)}}


def palace_requirements(metadata_path: Path) -> dict:
    """Read the tetrahedra and input-mesh Field DOF estimate emitted by gsim."""
    mesh = json.loads(metadata_path.read_text())["mesh"]
    tetrahedra = mesh["tetrahedra"]
    field_dofs = mesh["field_dofs"]["estimated_field_dofs"]
    if not tetrahedra or not field_dofs:
        raise ValueError("This case requires a tetrahedral 3D EM mesh and Field DOFs")
    return {"palace": {"tetrahedra": tetrahedra, "field_dofs": field_dofs}}


def write_request(case_dir: Path, case_name: str, requirements: dict) -> dict:
    """Write the sizing sidecar and prepare request separately from solver inputs."""
    request = {
        "job_name": f"allocation-{case_name}",
        "job_definition_name": next(iter(requirements)),
        "sizing_requirements": requirements,
    }
    for filename, document in (
        ("metadata.json", requirements),
        ("prepare.json", request),
    ):
        (case_dir / filename).write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(request, indent=2))
    return request


def submit_case(
    case_dir: Path,
    request: dict,
    input_files: list[Path],
    *,
    expect_rejection: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    """Prepare, upload and submit a case, or verify an oversized prepare rejection.

    SIMULATION_API_URL is the API base immediately before /job/prepare, for
    example the simulation service's /v1 URL. No server or credentials are
    selected implicitly. This notebook adapter sends sizing_requirements
    directly until the regular SDK upload path supports that field.
    """
    base_url = os.environ["SIMULATION_API_URL"].rstrip("/")
    headers = {"X-Api-Key": os.environ["GFP_API_KEY"]}
    if client is None:
        with httpx.Client(timeout=300) as owned_client:
            return _submit_case(
                owned_client,
                base_url,
                headers,
                case_dir,
                request,
                input_files,
                expect_rejection=expect_rejection,
            )
    return _submit_case(
        client,
        base_url,
        headers,
        case_dir,
        request,
        input_files,
        expect_rejection=expect_rejection,
    )


def _submit_case(
    client: httpx.Client,
    base_url: str,
    headers: dict,
    case_dir: Path,
    request: dict,
    input_files: list[Path],
    *,
    expect_rejection: bool,
) -> dict:
    response = client.post(f"{base_url}/job/prepare", headers=headers, json=request)
    if expect_rejection and response.status_code == 400:
        message = response.json().get("message", "")
        if "largest supported sizing profile" in message:
            outcome = {"rejected": True, "message": message}
            (case_dir / "cloud-result.json").write_text(
                json.dumps(outcome, indent=2) + "\n"
            )
            return outcome
    response.raise_for_status()
    prepared = response.json()["data"]
    job_id = prepared["job_id"]
    (case_dir / "job-id.txt").write_text(job_id + "\n")
    if expect_rejection:
        raise RuntimeError(
            f"Oversized case was accepted as {job_id}; it has not been submitted. "
            "Check measured counts and the server's sizing support/profiles."
        )
    upload_url = prepared.get("upload_url")
    if not upload_url:
        raise ValueError(
            "These examples require managed storage with an input upload URL"
        )
    archive_path = case_dir / "source.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for path in input_files:
            archive.write(path, path.name)
    with archive_path.open("rb") as content:
        upload = client.put(
            upload_url,
            content=content,
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(archive_path.stat().st_size),
            },
        )
    upload.raise_for_status()
    submitted = client.post(
        f"{base_url}/job/submit", headers=headers, json={"job_id": job_id}
    )
    submitted.raise_for_status()
    job = submitted.json()["data"]
    # Retain only job diagnostics, never presigned URLs or API credentials.
    outcome = {
        "job_id": job_id,
        "status": job.get("status"),
        "requested_cpu": job.get("requested_cpu"),
        "requested_memory_mb": job.get("requested_memory_mb"),
    }
    (case_dir / "cloud-result.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome
