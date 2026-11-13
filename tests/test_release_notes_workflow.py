"""Verify that publishing uses only the draft notes for the version being released."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash") or not shutil.which("jq"),
    reason="The publishing workflow runs on Ubuntu with Bash and jq",
)


def select_notes(tmp_path, pages):
    workflow_path = Path(__file__).parents[1] / ".github/workflows/release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    step = next(
        step
        for step in workflow["jobs"]["release"]["steps"]
        if step.get("id") == "draft"
    )
    # Execute the workflow's actual shell step with a local GitHub API stand-in.
    script = """
    gh() {
      if [ "$2" = "-X" ]; then
        printf '%s\\n' "$4" >> "$DELETED_RELEASES"
      else
        printf '%s\\n' "$RELEASE_PAGES"
      fi
    }
    """ + step["run"]
    output = tmp_path / "output"
    deleted = tmp_path / "deleted"
    bash_executable = shutil.which("bash")
    assert bash_executable is not None
    subprocess.run(  # noqa: S603
        [bash_executable, "-e", "-o", "pipefail", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GITHUB_REPOSITORY": "example/project",
            "GITHUB_OUTPUT": str(output),
            "DELETED_RELEASES": str(deleted),
            "RELEASE_PAGES": json.dumps(pages),
            "RELEASE_TAG": "0.11.0",
        },
    )
    return output.read_text(), deleted.read_text() if deleted.exists() else ""


def test_consumes_only_matching_draft_across_pages(tmp_path):
    output, deleted = select_notes(
        tmp_path,
        [
            [{"id": 12, "tag_name": "0.12.0", "draft": True, "body": "Future notes"}],
            [{"id": 11, "tag_name": "0.11.0", "draft": True, "body": "This release"}],
        ],
    )

    assert output == "body<<EOFBODY\nThis release\nEOFBODY\n"
    assert deleted == "repos/example/project/releases/11\n"


def test_unrelated_draft_is_preserved_when_no_matching_draft_exists(tmp_path):
    output, deleted = select_notes(
        tmp_path,
        [[{"id": 12, "tag_name": "0.12.0", "draft": True, "body": "Future notes"}]],
    )

    assert output == "body=\n"
    assert deleted == ""
