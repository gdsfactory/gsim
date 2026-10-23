# AGENTS.md

## Project

`gsim` is a Python 3.12 simulation library with multiple solver and cloud backends. Documented APIs, defaults, results,
saved configurations, generated solver inputs, and published notebooks are user-facing contracts.

## Development

- Use `uv run` for Python commands and tests.
- Before finishing a change, check affected public and documented workflows and run focused tests. Mark tests requiring
  cloud services or local solver installations so the default suite remains runnable.
- Keep optional backends importable without their extras; raise a clear error when a missing backend is used.
- Add regression tests for consequential fixes. Assert public results or generated artifacts, including relevant
  boundary and failure cases.

## Public repository content

- Treat code, documentation, notebooks, test data, issues, PR descriptions, and review comments as public.
- Do not add links to private repositories or other access-controlled company resources. Use a public reference or
  describe the relevant behavior directly.
- Do not publish confidential or proprietary code, data, customer details, credentials, or internal plans.
- Use public product names and clear descriptions. Avoid internal project names, codenames, acronyms, and company slang
  that external readers would not know.
- If a review finds a possible exposure, identify its location without repeating the sensitive content in the review
  comment.

## Public API changes

- An intentional break must document the old and new behavior, reason for the change, affected users, migration steps,
  and target release.
- Introduce the replacement before removing the old API. Keep the old path working with a `DeprecationWarning`, and test
  both paths during the transition. Warn in a released version before changing a default.
- Normally retain a deprecated public API for at least two feature releases and roughly 12 months after the first
  released warning. Remove it only in a release designated for incompatible changes: a feature release before 1.0, or a
  major release after 1.0. Do not make intentional breaks in patch releases.
- Announce deprecations and removals in release notes and update documentation and examples. Retain a safe, inexpensive
  compatibility alias longer when it imposes little maintenance cost.
- If removal is not planned, describe the API as discouraged or soft deprecated without issuing a removal warning.
  Serious correctness or safety problems may justify a shorter transition when the change and migration are documented.

## Code Review Rules

Report actionable defects introduced by the PR. Identify the triggering input or workflow and the resulting incorrect
behavior. Leave formatting and routine lint findings to CI. Apply the public repository content rules without quoting
sensitive material in a review.

- **Preserve public behavior.** Check existing calls, defaults, saved settings, generated solver inputs, and cache keys.
  For an intentional break, check that the Public API changes policy is followed; do not flag the break solely because
  it is incompatible.
- **Check simulation correctness.** Verify material properties, geometry, conductor roles, mesh and port surfaces,
  units, port and mode order, and reference impedances. For numerical changes, use an independent analytic, library, or
  solver reference; a self-consistent round trip is insufficient.
- **Trace complete workflows and failures.** Follow affected documented APIs and notebooks through setup, submission,
  output loading, and reported results. Flag ignored options, missing inputs, incomplete jobs reported as successful,
  lost diagnostics, and cleanup that can affect files outside the job's own directory. Require a focused regression
  check for a concrete break.
