"""Check sizing measurements and the cloud transport used by the example notebooks."""

import importlib.util
import io
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import meshio
import numpy as np
import pytest

HELPERS_PATH = Path(__file__).parents[1] / "nbs" / "allocation_cases" / "helpers.py"
spec = importlib.util.spec_from_file_location("allocation_helpers", HELPERS_PATH)
assert spec is not None
assert spec.loader is not None
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_fdtd_counts_pml_and_ignores_unreferenced_vertices(tmp_path):
    mesh = meshio.Mesh(
        points=np.array(
            [[0, 0, 0], [100, 0, 0], [0, 200, 0], [0, 0, 300], [9999, 9999, 9999]]
        ),
        cells=[("tetra", np.array([[0, 1, 2, 3]]))],
    )
    mesh_path = tmp_path / "mesh.vtu"
    meshio.write(mesh_path, mesh)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"grid": {"nanometers_per_cell": 50, "pml_cells": 2}})
    )

    assert helpers.fdtd_requirements(mesh_path, config_path) == {"fdtd": {"cells": 480}}


def test_palace_uses_measured_counts_not_target_counts(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "mesh": {
                    "tetrahedra": 321,
                    "field_dofs": {"estimated_field_dofs": 2345},
                }
            }
        )
    )
    assert helpers.palace_requirements(metadata) == {
        "palace": {"tetrahedra": 321, "field_dofs": 2345}
    }


@pytest.fixture
def cloud_case(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATION_API_URL", "https://simulation.example/v1")
    monkeypatch.setenv("GFP_API_KEY", "test-key")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = [inputs / "mesh.msh", inputs / "config.json"]
    for path in files:
        path.write_text("test input")
    request = helpers.write_request(tmp_path, "fdtd-small", {"fdtd": {"cells": 12345}})
    return tmp_path, request, files


def test_supported_case_sends_json_then_uploads_only_inputs_then_submits(cloud_case):
    case_dir, request, files = cloud_case
    calls = []

    def handler(incoming):
        calls.append(incoming.url.path)
        if incoming.url.path == "/v1/job/prepare":
            assert incoming.headers["X-Api-Key"] == "test-key"
            assert json.loads(incoming.content) == request
            return httpx.Response(
                200,
                json={
                    "data": {
                        "job_id": "test-job",
                        "upload_url": "https://storage.example/source.zip?token=secret",
                    }
                },
            )
        if incoming.url.path == "/source.zip":
            assert "X-Api-Key" not in incoming.headers
            with ZipFile(io.BytesIO(incoming.content)) as archive:
                assert set(archive.namelist()) == {"mesh.msh", "config.json"}
            return httpx.Response(200)
        assert incoming.url.path == "/v1/job/submit"
        assert json.loads(incoming.content) == {"job_id": "test-job"}
        return httpx.Response(
            200,
            json={
                "data": {
                    "status": "submitted",
                    "requested_cpu": 2,
                    "requested_memory_mb": 2048,
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = helpers.submit_case(case_dir, request, files, client=client)

    assert calls == ["/v1/job/prepare", "/source.zip", "/v1/job/submit"]
    assert outcome["job_id"] == "test-job"
    assert outcome["requested_memory_mb"] == 2048
    assert "secret" not in (case_dir / "cloud-result.json").read_text()
    assert (
        json.loads((case_dir / "metadata.json").read_text())
        == request["sizing_requirements"]
    )


def test_oversized_case_checks_the_specific_error_and_does_not_upload(cloud_case):
    case_dir, request, files = cloud_case
    calls = []

    def handler(incoming):
        calls.append(incoming.url.path)
        return httpx.Response(
            400,
            json={
                "message": (
                    "job requirements exceed the largest supported sizing profile"
                )
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = helpers.submit_case(
            case_dir, request, files, expect_rejection=True, client=client
        )

    assert outcome["rejected"] is True
    assert calls == ["/v1/job/prepare"]
    assert not (case_dir / "source.zip").exists()


@pytest.mark.parametrize("status", [400, 401, 403, 500])
def test_unrelated_errors_are_not_treated_as_successful_oversized_checks(
    cloud_case, status
):
    case_dir, request, files = cloud_case
    transport = httpx.MockTransport(
        lambda _: httpx.Response(status, json={"message": "unrelated error"})
    )
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        helpers.submit_case(
            case_dir, request, files, expect_rejection=True, client=client
        )


def test_accepted_oversized_case_is_never_submitted(cloud_case):
    case_dir, request, files = cloud_case
    calls = []

    def handler(incoming):
        calls.append(incoming.url.path)
        return httpx.Response(
            200,
            json={
                "data": {
                    "job_id": "unexpected-job",
                    "upload_url": "https://storage.example/source.zip",
                }
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="has not been submitted"),
    ):
        helpers.submit_case(
            case_dir, request, files, expect_rejection=True, client=client
        )

    assert calls == ["/v1/job/prepare"]
    assert (case_dir / "job-id.txt").read_text() == "unexpected-job\n"


def test_failed_upload_does_not_submit_a_job(cloud_case):
    case_dir, request, files = cloud_case
    calls = []

    def handler(incoming):
        calls.append(incoming.url.path)
        if incoming.url.path == "/v1/job/prepare":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "job_id": "test-job",
                        "upload_url": "https://storage.example/source.zip",
                    }
                },
            )
        return httpx.Response(503)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        helpers.submit_case(case_dir, request, files, client=client)
    assert calls == ["/v1/job/prepare", "/source.zip"]
