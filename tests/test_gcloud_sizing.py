"""Exercise the deployed prepare/upload contract without an SDK upgrade."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from gsim import _cloud_submission, gcloud, hashing


def _inputs(tmp_path, solver="fdtd"):
    tmp_path.joinpath("config.json").write_text("{}")
    metadata = {"schema_version": 1, "solver": solver}
    if solver == "fdtd":
        metadata["grid"] = {"cell_counts": {"total_cells": 60480960}}
    else:
        metadata["mesh"] = {
            "tetrahedra": 34862,
            "field_dofs": {"estimated_field_dofs": 223152},
        }
    tmp_path.joinpath("metadata.json").write_text(json.dumps(metadata))
    return metadata


@pytest.fixture
def upload_transport(monkeypatch):
    recorded: dict[str, Any] = {
        "prepare": [],
        "archive": [],
        "prepare_status": 200,
        "upload_status": 200,
    }
    original_client = httpx.Client
    transport = SimpleNamespace(
        _base_url=lambda: "https://api.example.com/sdk/v1",
        _headers=lambda: {"X-Api-Key": "test-api-key"},
    )
    pre_job = gcloud.sim.PreJob
    monkeypatch.setattr(
        _cloud_submission,
        "sim",
        SimpleNamespace(
            _base_url=transport._base_url,
            _headers=transport._headers,
            PreJob=pre_job,
        ),
    )

    def handle(request):
        if request.method == "POST":
            assert str(request.url) == "https://api.example.com/sdk/v1/job/prepare"
            assert request.headers["X-Api-Key"] == "test-api-key"
            recorded["prepare"].append(json.loads(request.content))
            return httpx.Response(
                recorded["prepare_status"],
                json=recorded.get(
                    "prepare_error",
                    {
                        "data": {
                            "job_id": "submitted",
                            "upload_url": "https://storage.example.com/source.zip",
                        }
                    },
                ),
            )
        assert request.method == "PUT"
        assert str(request.url) == "https://storage.example.com/source.zip"
        assert "X-Api-Key" not in request.headers
        assert request.headers["Content-Type"] == "application/zip"
        contents = request.read()
        assert int(request.headers["Content-Length"]) == len(contents)
        recorded["archive"].append(contents)
        return httpx.Response(recorded["upload_status"])

    monkeypatch.setattr(
        _cloud_submission.httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(handle), **kwargs
        ),
    )

    def legacy_upload(path, job_definition):
        assert path.is_dir()
        assert job_definition is not None
        pytest.fail("Sized submission must work with an SDK lacking sizing arguments")

    monkeypatch.setattr(gcloud.sim, "upload_simulation", legacy_upload)
    return recorded


@pytest.mark.parametrize("entry_point", ["upload", "upload_simulation_dir", "run"])
@pytest.mark.parametrize("solver", ["palace", "fdtd"])
def test_submission_sends_deployed_sizing_contract(
    tmp_path, monkeypatch, upload_transport, entry_point, solver
):
    metadata = _inputs(tmp_path, solver)
    if entry_point == "run":
        job = SimpleNamespace(
            job_id="submitted", job_name="submission", status="completed", exit_code=0
        )
        monkeypatch.setattr(gcloud.sim, "start_simulation", lambda _job: job)
        monkeypatch.setattr(gcloud.sim, "wait_for_simulation", lambda _job: job)
        monkeypatch.setattr(
            gcloud.sim, "download_results", lambda *_args, **_kwargs: {}
        )
        gcloud.run_simulation(
            tmp_path, solver, verbose=False, parent_dir=tmp_path.parent
        )
    elif entry_point == "upload":
        assert gcloud.upload(tmp_path, solver, verbose=False) == "submitted"
    else:
        assert gcloud.upload_simulation_dir(tmp_path, solver).job_id == "submitted"
    expected = (
        {"fdtd": {"cells": 60480960}}
        if solver == "fdtd"
        else {
            "palace": {"tetrahedra": 34862, "field_dofs": 223152},
        }
    )
    assert len(upload_transport["prepare"]) == 1
    payload = upload_transport["prepare"][0]
    assert payload["sizing_requirements"] == expected
    assert payload["job_definition_name"] == solver
    assert set(payload) == {
        "job_name",
        "job_definition_name",
        "original_file_name",
        "sizing_requirements",
    }
    with zipfile.ZipFile(BytesIO(upload_transport["archive"][0])) as archive:
        assert set(archive.namelist()) == {"config.json", "metadata.json"}
        assert json.loads(archive.read("metadata.json")) == metadata


def test_package_sdk_layout_uses_existing_authentication(
    tmp_path, monkeypatch, upload_transport
):
    _inputs(tmp_path)
    current = _cloud_submission.sim
    monkeypatch.setattr(
        _cloud_submission, "sim", SimpleNamespace(web=current, PreJob=current.PreJob)
    )
    assert gcloud.upload(tmp_path, "fdtd", verbose=False) == "submitted"
    assert len(upload_transport["prepare"]) == 1


@pytest.mark.parametrize("solver", ["palace", "fdtd"])
def test_archive_is_windows_safe_and_preserves_staged_files(
    tmp_path, monkeypatch, upload_transport, solver
):
    _inputs(tmp_path, solver)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/mesh.msh").write_bytes(b"mesh\x00bytes")
    (tmp_path / ".hidden").write_bytes(b"included")
    monkeypatch.setattr(
        _cloud_submission.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: pytest.fail("Windows-locked temporary file"),
    )
    gcloud.upload(tmp_path, solver, verbose=False)
    with zipfile.ZipFile(BytesIO(upload_transport["archive"][0])) as archive:
        assert archive.read("nested/mesh.msh") == b"mesh\x00bytes"
        assert archive.read(".hidden") == b"included"


def test_symbolic_link_rejected_before_preparation(tmp_path, upload_transport):
    _inputs(tmp_path)
    try:
        (tmp_path / "link.json").symlink_to(tmp_path / "config.json")
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
        pytest.skip("Windows requires Developer Mode or symlink privileges")
    with pytest.raises(ValueError, match="symbolic link"):
        gcloud.upload(tmp_path, "fdtd", verbose=False)
    assert upload_transport["prepare"] == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100", 2**64])
def test_invalid_counts_fail_before_preparation(tmp_path, upload_transport, value):
    metadata = _inputs(tmp_path)
    metadata["grid"]["cell_counts"]["total_cells"] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="positive 64-bit integer"):
        gcloud.upload(tmp_path, "fdtd", verbose=False)
    assert upload_transport["prepare"] == []


@pytest.mark.parametrize("metadata", [{"solver": "palace"}, {"solver": "fdtd"}, []])
def test_malformed_metadata_fails_before_preparation(
    tmp_path, upload_transport, metadata
):
    _inputs(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        gcloud.upload(tmp_path, "fdtd", verbose=False)
    assert upload_transport["prepare"] == []


def test_amr_uses_explicit_field_dof_limit(tmp_path, upload_transport):
    metadata = _inputs(tmp_path, "palace")
    metadata["simulation"] = {"refinement": {"MaxIts": 3, "MaxSize": 8000000}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    gcloud.upload(tmp_path, "palace", verbose=False)
    assert (
        upload_transport["prepare"][0]["sizing_requirements"]["palace"]["field_dofs"]
        == 8000000
    )


@pytest.mark.parametrize("solver", ["palace", "fdtd"])
def test_oversized_rejection_does_not_upload_or_fall_back(
    tmp_path, monkeypatch, upload_transport, solver
):
    _inputs(tmp_path, solver)
    upload_transport["prepare_status"] = 400
    message = "Problem size exceeds the largest supported resource profile"
    upload_transport["prepare_error"] = {
        "message": message,
        "unrelated_field": "private-response-data",
    }
    monkeypatch.setattr(
        _cloud_submission,
        "_write_archive",
        lambda *_args: pytest.fail("Rejected jobs must not compress inputs"),
    )
    with pytest.raises(httpx.HTTPStatusError) as error:
        gcloud.upload(tmp_path, solver, verbose=False)
    assert error.value.response.status_code == 400
    assert error.value.request.method == "POST"
    assert message in str(error.value)
    assert "private-response-data" not in str(error.value)
    assert len(upload_transport["prepare"]) == 1
    assert upload_transport["archive"] == []


@pytest.mark.parametrize(
    "body",
    [
        b"<html>private-response-data</html>",
        b'["private-response-data"]',
        b'{"message": {"detail": "private-response-data"}}',
        b'{"message": "  "}',
        b'{"detail": "private-response-data"}',
    ],
)
def test_prepare_errors_without_api_message_keep_http_diagnostics(body):
    response = httpx.Response(
        400,
        content=body,
        request=httpx.Request("POST", "https://example.test/job/prepare"),
    )
    with pytest.raises(httpx.HTTPStatusError) as error:
        _cloud_submission._raise_for_prepare_status(response)
    assert error.value.response is response
    assert error.value.request is response.request
    assert "400 Bad Request" in str(error.value)
    assert "DataLab:" not in str(error.value)
    assert "private-response-data" not in str(error.value)


def test_prepare_forbidden_preserves_cloud_access_error(tmp_path, upload_transport):
    _inputs(tmp_path)
    upload_transport["prepare_status"] = 403
    with pytest.raises(gcloud.CloudSimulationNotEnabledError):
        gcloud.upload(tmp_path, "fdtd", verbose=False)


def test_upload_error_is_propagated(tmp_path, upload_transport):
    _inputs(tmp_path)
    upload_transport["upload_status"] = 503
    with pytest.raises(httpx.HTTPStatusError):
        gcloud.upload(tmp_path, "fdtd", verbose=False)
    assert len(upload_transport["prepare"]) == 1


@pytest.mark.parametrize("missing_metadata", [False, True])
def test_unavailable_sizing_preserves_sdk_submission(
    tmp_path, monkeypatch, missing_metadata
):
    metadata = _inputs(tmp_path, "palace")
    if missing_metadata:
        (tmp_path / "metadata.json").unlink()
    else:
        metadata["mesh"]["field_dofs"] = {
            "estimated_field_dofs": None,
            "unavailable_reason": "Not a supported 3D EM mesh",
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        gcloud.sim,
        "upload_simulation",
        lambda **_kwargs: SimpleNamespace(job_id="legacy"),
    )
    assert gcloud.upload(tmp_path, "palace", verbose=False) == "legacy"


@pytest.mark.parametrize("cached", [False, True])
def test_fdtd_cache_reuses_results_or_sends_sizing(
    tmp_path, monkeypatch, upload_transport, cached
):
    from gsim import fdtd

    simulation = fdtd.Simulation()
    _inputs(tmp_path)
    monkeypatch.setattr(simulation, "_prepare_upload_dir", lambda: tmp_path)
    monkeypatch.setattr(
        gcloud.sim,
        "check_cache",
        lambda **_kwargs: SimpleNamespace(cached=cached, job_id="cached"),
        raising=False,
    )
    starts = []
    monkeypatch.setattr(
        gcloud, "start", lambda job_id, **_kwargs: starts.append(job_id)
    )
    assert simulation.run(check_cache=True, wait=False, verbose="quiet") == (
        "cached" if cached else "submitted"
    )
    if cached:
        assert upload_transport["prepare"] == []
        assert starts == []
    else:
        payload = upload_transport["prepare"][0]
        assert payload["sizing_requirements"] == {"fdtd": {"cells": 60480960}}
        assert payload["input_hash"].startswith("sha256:")
        assert starts == ["submitted"]


@pytest.mark.parametrize("solver", ["palace", "fdtd"])
def test_metadata_does_not_change_physics_cache_key(tmp_path, solver):
    (tmp_path / "config.json").write_text("{}")
    original_hash = hashing.compute_input_hash(tmp_path, solver)
    _inputs(tmp_path, solver)
    assert hashing.compute_input_hash(tmp_path, solver) == original_hash
