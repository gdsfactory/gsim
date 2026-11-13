"""Tag a tested version bump and dispatch the existing publishing workflow."""

# Standalone workflow command with messages intended for the Actions log.
# ruff: noqa: INP001, T201

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib


def run(*command: str) -> str:
    """Run a command, preserving failures for the workflow log."""
    return subprocess.run(  # noqa: S603
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def read_file(revision: str, filename: str) -> str:
    """Read release metadata from the commit being tested."""
    return run("git", "show", f"{revision}:{filename}")


def version_tuple(version: str) -> tuple[int, ...]:
    """Require the stable three-part versions used by the release workflow."""
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError(
            f"Expected a stable major.minor.patch version, got {version!r}"
        )
    return tuple(map(int, version.split(".")))


def validate_metadata(revision: str, version: str) -> None:
    """Refuse to publish inconsistent package, runtime, or release versions."""
    release_config = tomllib.loads(read_file(revision, "tbump.toml"))
    runtime = ast.parse(read_file(revision, "src/gsim/__init__.py"))
    runtime_versions = [
        ast.literal_eval(statement.value)
        for statement in runtime.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        )
    ]
    if (
        release_config["version"]["current"] != version
        or runtime_versions != [version]
        or read_file(revision, "README.md").splitlines()[0] != f"# gsim {version}"
    ):
        raise ValueError(
            "Version mismatch between pyproject.toml, tbump.toml, runtime, or README"
        )
    if release_config["git"]["tag_template"] != "{new_version}":
        raise ValueError("The release workflow requires unprefixed version tags")


def release_version(before: str, after: str, repository: str) -> None:
    """Publish only a version increase, recovering safely from a failed dispatch."""
    for revision in (before, after):
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision == "0" * 40:
            raise ValueError("Release comparison requires two existing commit SHAs")
    previous = tomllib.loads(read_file(before, "pyproject.toml"))["project"]["version"]
    version = tomllib.loads(read_file(after, "pyproject.toml"))["project"]["version"]
    if previous == version:
        print(f"Version remains {version}; no release needed.")
        return
    if version_tuple(version) <= version_tuple(previous):
        raise ValueError(f"Release version must increase: {previous} -> {version}")
    validate_metadata(after, version)

    tag_exists = bool(run("git", "tag", "--list", version))
    if (
        tag_exists
        and run("git", "rev-parse", f"refs/tags/{version}^{{commit}}") != after
    ):
        raise ValueError(f"Tag {version} already points to a different commit")

    pages = json.loads(
        run("gh", "api", "--paginate", "--slurp", f"repos/{repository}/releases")
    )
    if any(
        release["tag_name"] == version and not release["draft"]
        for page in pages
        for release in page
    ):
        print(f"Version {version} is already published.")
        return

    if not tag_exists:
        run(
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "tag",
            "--annotate",
            "--message",
            f"Release {version}",
            version,
            after,
        )
    run("git", "push", "origin", f"refs/tags/{version}:refs/tags/{version}")

    active_runs = json.loads(
        run(
            "gh",
            "run",
            "list",
            "--repo",
            repository,
            "--workflow",
            "release.yml",
            "--branch",
            version,
            "--json",
            "status",
        )
    )
    if any(workflow_run["status"] != "completed" for workflow_run in active_runs):
        print(f"Publishing {version} is already queued or running.")
        return
    # Tag pushes made with GITHUB_TOKEN do not trigger push workflows.
    run("gh", "workflow", "run", "release.yml", "--repo", repository, "--ref", version)
    print(f"Dispatched publication of {version} at {after}.")


def main() -> int:
    """Read the main-branch push range supplied by GitHub Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--repository", required=True)
    arguments = parser.parse_args()
    try:
        release_version(arguments.before, arguments.after, arguments.repository)
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
