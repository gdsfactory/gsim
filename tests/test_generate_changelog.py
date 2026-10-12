"""Tests for generating release history without hand-written entries."""

from __future__ import annotations

import io
import json
import sys
from urllib.error import URLError

import pytest

from scripts import generate_changelog


def _release(tag, date, body="", **fields):
    """Create release metadata with the fields used by the docs generator."""
    return {
        "tag_name": tag,
        "published_at": date,
        "html_url": f"https://github.com/example/project/releases/tag/{tag}",
        "body": body,
        **fields,
    }


def test_fetches_all_pages_and_uses_optional_authentication(monkeypatch):
    """Older versions are included, with authentication applied to each request."""
    monkeypatch.setattr(generate_changelog, "RELEASES_PER_PAGE", 2)
    pages = [
        [_release("0.3.0", "2026-03-01"), _release("0.2.0", "2026-02-01")],
        [_release("0.1.0", "2026-01-01")],
    ]
    requested_urls = []

    def read_page(request, *, timeout):
        requested_urls.append(request.full_url)
        assert request.get_header("Authorization") == "Bearer test-token"
        assert timeout == 30
        return io.BytesIO(json.dumps(pages[len(requested_urls) - 1]).encode())

    monkeypatch.setattr(generate_changelog, "urlopen", read_page)

    releases = generate_changelog.fetch_releases(
        "example/project",
        token="test-token",  # noqa: S106
    )

    assert [release["tag_name"] for release in releases] == ["0.3.0", "0.2.0", "0.1.0"]
    assert requested_urls == [
        "https://api.github.com/repos/example/project/releases?per_page=2&page=1",
        "https://api.github.com/repos/example/project/releases?per_page=2&page=2",
    ]


@pytest.mark.parametrize(
    "page", [{"message": "unexpected response"}, ["not a release"]]
)
def test_rejects_invalid_release_responses(monkeypatch, page):
    """Malformed responses must not become an incomplete changelog."""
    monkeypatch.setattr(
        generate_changelog,
        "urlopen",
        lambda request, **kwargs: io.BytesIO(json.dumps(page).encode()),
    )

    with pytest.raises(ValueError, match="invalid release list"):
        generate_changelog.fetch_releases("example/project")


def test_orders_published_releases_and_excludes_drafts():
    """Draft versions remain unpublished while published prereleases are identified."""
    releases = [
        _release("0.1.0", "2026-01-01T12:00:00Z", "- Initial release"),
        _release("0.3.0", None, "private draft", draft=True),
        _release("0.2.0rc1", "2026-02-01T12:00:00Z", "- Preview", prerelease=True),
        _release("0.2.0", "2026-02-02T12:00:00Z", "- Stable release"),
    ]

    changelog = generate_changelog.render_changelog(
        releases, repository="example/project"
    )

    assert changelog.index("## 0.2.0\n") < changelog.index("## 0.2.0rc1 (prerelease)")
    assert changelog.index("## 0.2.0rc1 (prerelease)") < changelog.index("## 0.1.0\n")
    assert "0.3.0" not in changelog
    assert "private draft" not in changelog
    assert "Released 2026-02-02." in changelog
    assert "- Stable release" in changelog


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_nests_release_headings_without_modifying_code(fence):
    """Release sections form a usable TOC while code comments remain unchanged."""
    notes = (
        "# What's Changed\n\n## Features\n\n- Preserve the original PR title (#12)\n\n"
        f"{fence}python\n# A code comment\nprint('example')\n{fence}\n\n## Fixes"
    )

    changelog = generate_changelog.render_changelog(
        [_release("0.1.0", "2026-01-01", notes)], repository="example/project"
    )

    assert "### What's Changed" in changelog
    assert "#### Features" in changelog
    assert "#### Fixes" in changelog
    assert f"{fence}python\n# A code comment\nprint('example')\n{fence}" in changelog
    assert "- Preserve the original PR title (#12)" in changelog


def test_does_not_invent_missing_historical_notes():
    """Versions with no published body retain their release links and dates."""
    changelog = generate_changelog.render_changelog(
        [_release("0.1.0", "2026-01-01", None)], repository="example/project"
    )

    assert "No release notes were published for this version." in changelog
    assert "https://github.com/example/project/releases/tag/0.1.0" in changelog


def test_preserves_existing_output_when_github_is_unavailable(monkeypatch, tmp_path):
    """A failed refresh reports the failure and leaves the existing file intact."""
    output_path = tmp_path / "CHANGELOG.md"
    output_path.write_text("existing release history\n")
    monkeypatch.setattr(
        sys, "argv", ["generate_changelog", "--output", str(output_path)]
    )

    def fail_fetch(*_args, **_kwargs):
        raise URLError("GitHub unavailable")

    monkeypatch.setattr(generate_changelog, "fetch_releases", fail_fetch)

    with pytest.raises(SystemExit, match="Cannot generate changelog"):
        generate_changelog.main()

    assert output_path.read_text() == "existing release history\n"


def test_rejects_empty_published_history():
    """Unexpectedly empty history must not replace a working changelog."""
    with pytest.raises(ValueError, match="No published"):
        generate_changelog.render_changelog(
            [_release("0.1.0", None, draft=True)], repository="example/project"
        )
