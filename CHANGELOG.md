# Changelog

See the [documentation changelog](https://gdsfactory.com/gsim/CHANGELOG/) or
[GitHub Releases](https://github.com/gdsfactory/gsim/releases) for release history.

Release Drafter generates release notes from merged pull requests. The documentation changelog contains only published
GitHub releases, refreshed during docs builds and automatically rebuilt after release publication.

The generated `docs/CHANGELOG.md` snapshot is committed so docs builds can use it if GitHub is unavailable. Release
headings and duplicate PR entries are normalized mechanically; the original release notes remain on GitHub.

Do not add hand-written release entries here or edit the generated snapshot. Run `just sync-changelog` to refresh it
locally.
