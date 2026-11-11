"""Exercise Linux CI setup failures without root access or network requests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Linux Bash installer")
SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/install-system-deps.sh"


@pytest.fixture
def run_installer(tmp_path):
    """Replace privileged commands with a recorder and controlled APT failures."""
    bash = shutil.which("bash")
    assert bash is not None
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    shim = (
        f"#!{sys.executable}\n"
        """
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["APT_TEST_DIRECTORY"])
name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "sudo":
    if args == ["tee", "/etc/apt/apt-mirrors.txt"]:
        (root / "mirrors").write_text(sys.stdin.read())
        sys.exit(0)
    assert args[0] == "timeout", args
    os.execvp(args[0], args)
if name == "timeout":
    assert args[:2] == ["--kill-after=10s", "90s"]
    os.execvp(args[2], args[2:])
operation = next(arg for arg in args if arg in {"update", "install"})
with (root / "calls").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
counter = root / operation
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
mode = os.environ["APT_TEST_MODE"]
if mode == "always_fail" or (
    mode == "update_once" and operation == "update" and count == 1
):
    sys.exit(100)
if mode == "install_once" and operation == "install" and count == 1:
    sys.exit(100)
if mode == "timeout_once" and operation == "update" and count == 1:
    sys.exit(124)
if mode == "fallback_install_fails" and (operation == "install" or count == 1):
    sys.exit(100)
"""
    )
    for command in ("sudo", "timeout", "apt-get"):
        executable = binary_directory / command
        executable.write_text(shim)
        executable.chmod(0o755)

    def run(mode, packages=("libglu1-mesa", "xvfb")):
        environment = dict(
            os.environ,
            PATH=f"{binary_directory}{os.pathsep}{os.environ['PATH']}",
            APT_TEST_DIRECTORY=str(tmp_path),
            APT_TEST_MODE=mode,
        )
        result = subprocess.run(  # noqa: S603 -- fixed repository script and test packages.
            [bash, str(SCRIPT), *packages],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        calls = tmp_path / "calls"
        arguments = (
            [json.loads(line) for line in calls.read_text().splitlines()]
            if calls.exists()
            else []
        )
        return result, arguments, tmp_path / "mirrors"

    return run


def test_success_preserves_packages_and_default_mirrors(run_installer):
    result, calls, mirrors = run_installer("success")
    assert result.returncode == 0
    assert [call[-1] for call in calls[:1]] == ["update"]
    assert calls[1][-4:] == ["install", "-y", "libglu1-mesa", "xvfb"]
    assert "APT::Update::Error-Mode=any" in calls[0]
    assert not mirrors.exists()


@pytest.mark.parametrize("mode", ["update_once", "install_once", "timeout_once"])
def test_failed_setup_recovers_using_archive_mirrors(run_installer, mode):
    result, calls, mirrors = run_installer(mode)
    assert result.returncode == 0
    assert "retrying with alternate Ubuntu mirrors" in result.stdout
    assert mirrors.read_text().splitlines() == [
        "https://mirrors.edge.kernel.org/ubuntu/\tpriority:1",
        "https://archive.ubuntu.com/ubuntu/\tpriority:2",
        "https://security.ubuntu.com/ubuntu/\tpriority:3",
    ]
    assert sum(call[-1] == "update" for call in calls) == 2
    assert calls[-1][-4:] == ["install", "-y", "libglu1-mesa", "xvfb"]


@pytest.mark.parametrize("mode", ["always_fail", "fallback_install_fails"])
def test_failed_fallback_fails_the_job(run_installer, mode):
    result, calls, _mirrors = run_installer(mode)
    assert result.returncode == 100
    assert sum(call[-1] == "update" for call in calls) == 2
    if mode == "always_fail":
        assert all(call[-1] == "update" for call in calls)


def test_missing_packages_fail_before_running_apt(run_installer):
    result, calls, mirrors = run_installer("success", packages=())
    assert result.returncode != 0
    assert not calls
    assert not mirrors.exists()
