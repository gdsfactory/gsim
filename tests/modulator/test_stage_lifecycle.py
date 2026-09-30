"""Lazy caching, cascade invalidation and verbose reporting of Stages."""

from __future__ import annotations

import pytest

from gsim.modulator.stage import Stage, StageNotRunError


class CountingStage(Stage):
    """Minimal Stage recording how often it actually solved."""

    stage_name = "counting"

    value: int = 0

    def _solve(self):
        self._solves = getattr(self, "_solves", 0) + 1
        return f"solved {self.value}"


def wired_stages():
    """Two stages, the second downstream of the first."""
    upstream = CountingStage()
    downstream = CountingStage()
    upstream.wire(downstream=[downstream])
    return upstream, downstream


class TestNotRun:
    def test_a_stage_that_has_not_run_reports_so(self):
        stage = CountingStage()
        assert stage.has_run is False

    def test_reading_the_result_before_running_is_an_actionable_error(self):
        stage = CountingStage()
        with pytest.raises(StageNotRunError, match="run"):
            _ = stage.result

    def test_the_error_is_a_runtime_error(self):
        assert issubclass(StageNotRunError, RuntimeError)


class TestCaching:
    def test_running_twice_solves_once(self):
        stage = CountingStage()
        first = stage.run()
        assert stage.run() is first
        assert stage._solves == 1

    def test_the_cached_result_is_what_result_returns(self):
        stage = CountingStage()
        assert stage.run() is stage.result
        assert stage.has_run is True

    def test_force_re_solves(self):
        stage = CountingStage()
        stage.run()
        stage.run(force=True)
        assert stage._solves == 2

    def test_elapsed_time_is_recorded(self):
        stage = CountingStage()
        stage.run()
        elapsed = stage.elapsed_s
        assert elapsed is not None
        assert elapsed >= 0.0


class TestInvalidation:
    def test_configuring_a_stage_clears_its_result(self):
        stage = CountingStage()
        stage.run()
        stage(value=3)
        assert stage.has_run is False
        assert stage.run() == "solved 3"

    def test_configuring_a_stage_clears_every_downstream_result(self):
        upstream, downstream = wired_stages()
        upstream.run()
        downstream.run()

        upstream(value=1)

        assert upstream.has_run is False
        assert downstream.has_run is False

    def test_a_downstream_change_leaves_upstream_alone(self):
        upstream, downstream = wired_stages()
        upstream.run()
        downstream.run()

        downstream(value=1)

        assert upstream.has_run is True
        assert downstream.has_run is False

    def test_setting_a_field_directly_invalidates_too(self):
        stage = CountingStage()
        stage.run()
        stage.value = 7
        assert stage.has_run is False

    def test_configuring_returns_the_stage_for_chaining(self):
        stage = CountingStage()
        assert stage(value=2) is stage

    def test_unknown_setting_is_rejected(self):
        stage = CountingStage()
        with pytest.raises(ValueError, match="nope"):
            stage(nope=1)


class TestVerbose:
    def test_silent_by_default(self, capsys):
        CountingStage().run()
        assert capsys.readouterr().out == ""

    def test_verbose_prints_entry_and_exit_with_elapsed_time(self, capsys):
        stage = CountingStage()
        stage.wire(is_verbose=lambda: True)
        stage.run()
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 2
        assert "counting" in out[0]
        assert "counting" in out[1]
        assert "s" in out[1]


class TestForcedRerun:
    def test_forcing_a_re_solve_drops_what_read_the_old_result(self):
        upstream, downstream = wired_stages()
        upstream.run()
        downstream.run()

        upstream.run(force=True)

        assert downstream.has_run is False


class TestSeeding:
    """A result reaches a Stage without a solve through one public operation."""

    def test_a_seeded_result_reads_as_solved(self):
        stage = CountingStage()
        stage.seed("canned")
        assert stage.has_run is True
        assert stage.result == "canned"
        assert stage.run() == "canned"
        assert not hasattr(stage, "_solves")

    def test_seeding_records_no_elapsed_time(self):
        stage = CountingStage()
        stage.run()
        stage.seed("canned")
        assert stage.elapsed_s is None

    def test_seeding_drops_every_downstream_result(self):
        upstream, downstream = wired_stages()
        upstream.run()
        downstream.run()

        upstream.seed("canned")

        assert upstream.has_run is True
        assert downstream.has_run is False

    def test_re_configuring_drops_a_seeded_result_too(self):
        stage = CountingStage()
        stage.seed("canned")
        stage(value=2)
        assert stage.has_run is False
        assert stage.run() == "solved 2"
