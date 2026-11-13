"""Exercise automatic release preparation against real temporary Git repositories."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import release_version


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """Use real local tags and pushes while replacing GitHub network requests."""
    origin = tmp_path / "origin.git"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    run = release_version.run

    def git(*arguments):
        return run("git", *arguments)

    git("init", "--bare", str(origin))
    git("init")
    git("config", "user.name", "Release tests")
    git("config", "user.email", "tests@example.com")
    git("remote", "add", "origin", str(origin))

    def commit(version, overrides=None):
        files = {
            "pyproject.toml": f'[project]\nversion = "{version}"\n',
            "tbump.toml": (
                f'[version]\ncurrent = "{version}"\n'
                '[git]\ntag_template = "{new_version}"\n'
            ),
            "src/gsim/__init__.py": f'__version__ = "{version}"\n',
            "README.md": f"# gsim {version}\n",
            **(overrides or {}),
        }
        for filename, content in files.items():
            path = checkout / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        git("add", ".")
        git("commit", "--allow-empty", "-m", f"Version {version}")
        return git("rev-parse", "HEAD")

    before = commit("0.10.0")
    after = commit("0.11.0")
    state = SimpleNamespace(releases=[[]], runs=[], calls=[], failure=None)

    def run_with_fake_github(*command):
        if command[0] != "gh":
            return run(*command)
        state.calls.append(command)
        if command[1] == state.failure:
            raise subprocess.CalledProcessError(1, command, stderr="GitHub unavailable")
        if command[1] == "api":
            return json.dumps(state.releases)
        if command[1] == "run":
            return json.dumps(state.runs)
        assert command[:4] == ("gh", "workflow", "run", "release.yml")
        return ""

    monkeypatch.setattr(release_version, "run", run_with_fake_github)
    return SimpleNamespace(
        git=git, commit=commit, before=before, after=after, state=state
    )


def publish(repository, *, after=None):
    release_version.release_version(
        repository.before, after or repository.after, "example/project"
    )


def test_tags_the_tested_commit_and_dispatches_publishing(repository):
    """A later checkout must not move the release away from its tested commit."""
    repository.commit("0.11.0", {"unrelated.txt": "A later commit\n"})

    publish(repository)

    assert repository.git("cat-file", "-t", "refs/tags/0.11.0") == "tag"
    remote_tag = repository.git("ls-remote", "origin", "refs/tags/0.11.0^{}").split()[0]
    assert remote_tag == repository.after
    assert repository.state.calls[-1] == (
        "gh",
        "workflow",
        "run",
        "release.yml",
        "--repo",
        "example/project",
        "--ref",
        "0.11.0",
    )


def test_ordinary_merges_do_not_tag_or_contact_github(repository):
    after = repository.commit("0.10.0")

    publish(repository, after=after)

    assert repository.git("tag", "--list") == ""
    assert repository.state.calls == []


@pytest.mark.parametrize("version", ["0.9.9", "0.11.0rc1", "v0.11.0", "0.011.0"])
def test_rejects_decreases_and_unsupported_versions(repository, version):
    after = repository.commit(version)

    with pytest.raises(ValueError, match=r"must increase|stable major\.minor\.patch"):
        publish(repository, after=after)

    assert repository.git("tag", "--list") == ""
    assert repository.state.calls == []


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("tbump.toml", '[version]\ncurrent = "0.10.0"\n'),
        ("src/gsim/__init__.py", '__version__ = "0.10.0"\n'),
        ("src/gsim/__init__.py", "# Missing runtime version\n"),
        ("README.md", "# gsim 0.10.0\n"),
        (
            "tbump.toml",
            '[version]\ncurrent = "0.11.0"\n[git]\ntag_template = "v{new_version}"\n',
        ),
    ],
)
def test_inconsistent_metadata_prevents_tagging(repository, filename, content):
    after = repository.commit("0.11.0", {filename: content})

    with pytest.raises(ValueError, match=r"Version mismatch|unprefixed version tags"):
        publish(repository, after=after)

    assert repository.git("tag", "--list") == ""
    assert repository.state.calls == []


def test_conflicting_tag_is_never_moved(repository):
    repository.git("tag", "0.11.0", repository.before)

    with pytest.raises(ValueError, match="already points to a different commit"):
        publish(repository)

    assert repository.git("rev-parse", "0.11.0") == repository.before
    assert repository.git("ls-remote", "--tags", "origin") == ""
    assert repository.state.calls == []


def test_retry_reuses_tag_after_dispatch_failure(repository):
    repository.state.failure = "workflow"
    with pytest.raises(subprocess.CalledProcessError):
        publish(repository)
    original_tag = repository.git("rev-parse", "refs/tags/0.11.0")

    repository.state.failure = None
    publish(repository)

    assert repository.git("rev-parse", "refs/tags/0.11.0") == original_tag
    assert repository.state.calls[-1][1] == "workflow"


def test_github_api_failure_does_not_create_a_tag(repository):
    repository.state.failure = "api"

    with pytest.raises(subprocess.CalledProcessError):
        publish(repository)

    assert repository.git("tag", "--list") == ""
    assert not any(call[1] == "workflow" for call in repository.state.calls)


def test_published_versions_are_skipped_across_release_pages(repository):
    repository.state.releases = [
        [{"tag_name": "0.12.0", "draft": True}],
        [{"tag_name": "0.11.0", "draft": False}],
    ]

    publish(repository)

    assert repository.git("tag", "--list") == ""
    assert not any(call[1] == "workflow" for call in repository.state.calls)


def test_draft_release_does_not_prevent_publishing(repository):
    repository.state.releases = [[{"tag_name": "0.11.0", "draft": True}]]

    publish(repository)

    assert repository.state.calls[-1][1] == "workflow"


@pytest.mark.parametrize("status", ["queued", "in_progress", "waiting"])
def test_active_publication_is_not_dispatched_again(repository, status):
    repository.state.runs = [{"status": status}]

    publish(repository)

    assert repository.git("rev-parse", "0.11.0^{commit}") == repository.after
    assert not any(call[1] == "workflow" for call in repository.state.calls)


def test_failed_publication_can_be_retried(repository):
    repository.git("tag", "0.11.0", repository.after)
    repository.state.runs = [{"status": "completed"}]

    publish(repository)

    assert repository.state.calls[-1][1] == "workflow"


@pytest.mark.parametrize("revision", ["0" * 40, "main", "--all"])
def test_requires_a_real_push_comparison(repository, revision):
    with pytest.raises(ValueError, match="two existing commit SHAs"):
        release_version.release_version(revision, repository.after, "example/project")

    assert repository.git("tag", "--list") == ""
    assert repository.state.calls == []
