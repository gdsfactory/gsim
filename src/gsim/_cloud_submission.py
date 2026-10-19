"""Send measured sizing requirements through DataLab's deployed prepare API."""

from __future__ import annotations

import json
import secrets
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import httpx
from gdsfactoryplus import sim


def _positive_count(value: object, name: str) -> int:
    """Require a positive count representable by DataLab's unsigned integers."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**64:
        raise ValueError(f"Sizing metadata {name} must be a positive 64-bit integer")
    return value


def read_sizing_requirements(
    input_dir: Path, job_type: str
) -> dict[str, dict[str, int]] | None:
    """Translate the diagnostic sidecar into the deployed API's problem counts.

    Missing metadata and Palace meshes without a supported 3D field-DOF
    estimate retain the SDK's fixed-resource path. Unbounded AMR and invalid
    counts are errors, rather than silently submitting with fixed resources.
    """
    solver = job_type.casefold()
    metadata_path = input_dir / "metadata.json"
    if solver not in {"fdtd", "palace"} or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    if not isinstance(metadata, dict) or metadata.get("solver") != solver:
        raise ValueError("Sizing metadata solver must match the submitted job type")
    try:
        if solver == "fdtd":
            return {
                solver: {
                    "cells": _positive_count(
                        metadata["grid"]["cell_counts"]["total_cells"], "total_cells"
                    )
                }
            }
        mesh = metadata["mesh"]
        refinement = metadata.get("simulation", {}).get("refinement", {})
        max_field_dofs = None
        if refinement.get("MaxIts", 0) > 0:
            try:
                max_field_dofs = _positive_count(refinement.get("MaxSize"), "MaxSize")
            except ValueError as error:
                raise ValueError(
                    "Palace cloud sizing requires a positive 64-bit integer "
                    "Model.Refinement.MaxSize when AMR is enabled (MaxIts > 0). "
                    "Set sim.set_refinement(max_dofs=...) to a finite bound, "
                    "or disable AMR with max_its=0."
                ) from error
        estimate = mesh["field_dofs"]
        if estimate.get("estimated_field_dofs") is None and estimate.get(
            "unavailable_reason"
        ):
            return None
        field_dofs = _positive_count(estimate["estimated_field_dofs"], "field_dofs")
        if max_field_dofs is not None:
            field_dofs = max(field_dofs, max_field_dofs)
        return {
            solver: {
                "tetrahedra": _positive_count(mesh["tetrahedra"], "tetrahedra"),
                "field_dofs": field_dofs,
            }
        }
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Incomplete {solver} sizing metadata") from error


def _input_files(input_dir: Path) -> list[Path]:
    """List staged files and reject symlinks before preparing a cloud job."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Upload input contains a symbolic link: {path}")
        if path.is_file():
            files.append(path)
    return files


def _write_archive(input_dir: Path, files: list[Path], archive_path: Path) -> None:
    """Archive staged inputs without reopening a Windows-locked temporary file."""
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(input_dir))


def upload_sized_simulation(
    input_dir: Path,
    job_type: str,
    requirements: dict[str, dict[str, int]],
    *,
    input_hash: str | None = None,
) -> Any:
    """Prepare and upload a sized job without requiring a new SDK release.

    Reuse the SDK's configured API host, authentication and PreJob result across
    its module and package layouts. Start, polling and downloads stay in the SDK.
    No resource tiers or CPU/memory policy are duplicated in the client.
    """
    transport: Any = getattr(sim, "web", sim)
    job_name = f"{job_type}-{secrets.token_hex(4)}"
    payload: dict[str, Any] = {
        "job_name": job_name,
        "job_definition_name": job_type,
        "original_file_name": "source.zip",
        "sizing_requirements": requirements,
    }
    if input_hash is not None:
        payload["input_hash"] = input_hash
    files = _input_files(input_dir)
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{transport._base_url()}/job/prepare",  # noqa: SLF001
            headers=transport._headers(),  # noqa: SLF001
            json=payload,
        )
        response.raise_for_status()
        prepared = response.json()["data"]
        with tempfile.TemporaryDirectory(prefix="gsim-upload-") as temporary_dir:
            archive_path = Path(temporary_dir) / "source.zip"
            _write_archive(input_dir, files, archive_path)
            with archive_path.open("rb") as archive:
                upload = client.put(
                    prepared["upload_url"],
                    content=archive,
                    headers={
                        "Content-Type": "application/zip",
                        "Content-Length": str(archive_path.stat().st_size),
                    },
                )
                upload.raise_for_status()
    return sim.PreJob(job_id=prepared["job_id"], job_name=job_name)
