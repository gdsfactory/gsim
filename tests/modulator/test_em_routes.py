"""Route selection on the two EM Stages, without any solver runtime.

Both EM Stages answer the same question through either Backend, and the
choice is a Stage setting. What is hermetic about that choice — the
default, the values accepted, the registry a route is found in, what
a strip count means on each Stage, the Staircase the optical Stage
builds when it is routed to Palace, and the error a user selecting a
Route they cannot run gets — is under test here, together with what the
Palace Route asks of a run and what it does with the answer. The
Palace-side halves of that — sizing and reading the impedance paths,
salvaging a crashed run's table, reporting a dead binary — are tested
against ``gsim.palace`` in ``tests/palace``. The Routes actually agreeing
on a number is the runtime-gated ``test_palace_route_runtime.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from pydantic import ValidationError

from gsim.modulator import OpticalStage, RFStage
from gsim.modulator.route import DEFAULT_PALACE_STRIPS
from gsim.palace.results import PalaceTextResults

from .conftest import SLAB


class TestRouteRegistry:
    def test_each_name_resolves_to_its_adapter(self):
        from gsim.modulator.femwell_route import FemwellRoute
        from gsim.modulator.palace_route import PalaceRoute
        from gsim.modulator.route import route_for

        assert isinstance(route_for("femwell"), FemwellRoute)
        assert isinstance(route_for("palace"), PalaceRoute)

    def test_every_run_gets_a_fresh_adapter(self):
        from gsim.modulator.route import route_for

        assert route_for("palace") is not route_for("palace")

    def test_an_unregistered_name_is_reported(self):
        from gsim.modulator.route import route_for

        with pytest.raises(ValueError, match="comsol"):
            route_for("comsol")  # type: ignore[arg-type]

    def test_a_registered_fake_is_what_the_stage_gets(self, biased, fake_route):
        biased.rf(route="palace")
        assert isinstance(biased.rf.resolved_route(), fake_route)

    def test_the_adapters_say_what_they_can_express(self):
        from gsim.modulator.femwell_route import FemwellRoute
        from gsim.modulator.palace_route import PalaceRoute

        assert FemwellRoute.continuous_materials is True
        assert PalaceRoute.continuous_materials is False
        assert FemwellRoute.conductor_model == "volume"
        assert PalaceRoute.conductor_model == "pec"


class TestRouteSelection:
    def test_both_em_stages_default_to_femwell(self):
        assert OpticalStage().route == "femwell"
        assert RFStage().route == "femwell"

    @pytest.mark.parametrize("stage", [OpticalStage, RFStage])
    def test_palace_is_selectable(self, stage):
        assert stage()(route="palace").route == "palace"

    @pytest.mark.parametrize("stage", [OpticalStage, RFStage])
    def test_an_unknown_route_is_rejected(self, stage):
        with pytest.raises(ValidationError, match="route"):
            stage()(route="comsol")

    def test_changing_the_route_invalidates_the_stage(self, biased):
        biased.rf.seed(object())
        biased.rf(route="palace")
        assert biased.rf.has_run is False


class TestOpticalStripCount:
    def test_the_continuous_profile_is_the_default(self):
        stage = OpticalStage()
        assert stage.n_strips is None
        assert stage.effective_n_strips() is None

    def test_the_palace_route_falls_back_to_a_strip_count(self):
        stage = OpticalStage()(route="palace")
        assert stage.effective_n_strips() == DEFAULT_PALACE_STRIPS

    def test_a_configured_count_wins_on_either_route(self):
        assert OpticalStage()(n_strips=7).effective_n_strips() == 7
        assert OpticalStage()(route="palace", n_strips=7).effective_n_strips() == 7

    def test_the_continuous_stage_builds_no_staircase(self, biased):
        point = biased.carriers.run().points[0]
        with pytest.raises(ValueError, match="n_strips"):
            biased.optical.staircase(point)


class TestOpticalStaircase:
    @pytest.fixture
    def staircase(self, biased):
        biased.optical(route="palace", n_strips=4)
        return biased.optical.staircase(biased.carriers.run().points[-1])

    def test_it_tiles_the_doped_slab_with_the_asked_for_strips(self, biased, staircase):
        span = biased.layout.doped_span
        assert len(staircase.strip_names) == 4
        edges = staircase.strips.edges_um
        assert edges[0] == pytest.approx(span[0])
        assert edges[-1] == pytest.approx(span[1])

    def test_it_invents_no_flanking_electrodes(self, staircase):
        """The drawn metal arrives as a surrounding region, not as a flank."""
        assert staircase.electrode_names == ()
        assert "cathode_metal" in {region.name for region in staircase.surroundings}

    def test_it_redraws_the_device_around_the_strips(self, biased, staircase):
        """The staircase is the drawn guide with its doped silicon binned."""
        drawn = {rect.layer_name for rect in biased.section}
        redrawn = {region.name.rsplit("_", 1)[0] for region in staircase.surroundings}
        # The doped regions are what the strips replace; everything else
        # the plane crosses is redrawn beside them.
        assert "slab90" in redrawn
        assert not drawn & {"n_rib", "p_rib"} & redrawn

    def test_its_strips_carry_the_carrier_perturbed_permittivity(
        self, biased, staircase
    ):
        eps = staircase.strips.permittivity
        assert eps.size == 4
        # Free carriers lower the index and add loss (exp(+i omega t)).
        assert np.all(eps.real < biased.optical.unperturbed_index() ** 2)
        assert np.all(eps.imag <= 0.0)

    def test_it_resolves_to_a_meshable_optical_stack(self, staircase):
        stack = staircase.stack()
        assert set(staircase.strip_names) <= set(stack.layers)

    def test_the_strip_span_is_overridable(self, biased):
        biased.optical(route="palace", n_strips=2, strip_span=SLAB)
        staircase = biased.optical.staircase(biased.carriers.run().points[-1])
        edges = staircase.strips.edges_um
        assert edges[0] == pytest.approx(SLAB[0])
        assert edges[-1] == pytest.approx(SLAB[1])


class TestMissingRuntime:
    @pytest.fixture
    def no_palace(self, monkeypatch):
        monkeypatch.setattr(
            "gsim.palace.runtime.resolve_palace_binary", lambda **_kw: None
        )

    @pytest.mark.usefixtures("no_palace")
    @pytest.mark.parametrize("stage_name", ["optical", "rf"])
    def test_palace_without_the_binary_is_actionable(self, biased, stage_name):
        getattr(biased, stage_name)(route="palace")
        with pytest.raises(RuntimeError) as excinfo:
            getattr(biased, stage_name).run()
        message = str(excinfo.value)
        assert "PALACE_BIN" in message
        assert f"study.{stage_name}(route='femwell')" in message

    @pytest.mark.usefixtures("no_palace")
    @pytest.mark.parametrize("stage_name", ["optical", "rf"])
    def test_the_route_is_checked_before_anything_is_meshed(
        self, study, monkeypatch, stage_name
    ):
        """A user whose Route cannot run pays for no mesh and no charge solve."""

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        getattr(study, stage_name)(route="palace")
        with pytest.raises(RuntimeError, match="PALACE_BIN"):
            getattr(study, stage_name).run()

    @pytest.mark.parametrize("stage_name", ["optical", "rf"])
    def test_femwell_still_names_its_extra(self, biased, monkeypatch, stage_name):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            getattr(biased, stage_name).run()


def mode_table(n_modes: int) -> PalaceTextResults:
    """A run's results carrying a complete ``mode-kn.csv`` of *n_modes* Modes."""
    rows = [
        {
            "m": str(m),
            "Re{kn} (1/m)": f"{4.2e7 + m:.6e}",
            "Im{kn} (1/m)": "-1.0e2",
            "Re{n_eff}": f"{2.0 + 0.1 * m:.6e}",
            "Im{n_eff}": "-1.0e-5",
        }
        for m in range(1, n_modes + 1)
    ]
    return PalaceTextResults(
        files={}, csv_tables={"mode-kn.csv": rows}, json_data={}, text_data={}
    )


class FakeSim:
    """A boundary-mode simulation whose run is scripted.

    ``run_local`` hands back whatever *leaves* says. The simulation owns
    the crash salvage and the abort report now, so what a fake stands in
    for here is only the run itself.
    """

    def __init__(self, output_dir, *, leaves=None):
        self.output_dir = output_dir
        self._leaves = leaves
        self.solved = []
        self.ran = []

    def set_boundary_mode(self, **kwargs):
        self.solved.append(kwargs)

    def write_config(self, **_kwargs):
        pass

    def run_local(self, **kwargs):
        self.ran.append(kwargs)
        return self._leaves

    def read_results(self):
        return self._leaves


class TestSolvingOneFrequency:
    """What the Route asks of a run, and what it does with the answer."""

    def _solve(self, sim, num_modes: int = 4):
        from gsim.modulator.palace_route import solve_palace_modes

        return solve_palace_modes(
            sim, freq_hz=10e9, num_modes=num_modes, target=2.0, save=1, binary="palace"
        )

    def test_a_clean_run_hands_back_its_modes_and_results(self, tmp_path):
        sim = FakeSim(tmp_path, leaves=mode_table(2))
        solve = self._solve(sim, num_modes=2)
        assert [mode.n_eff.real for mode in solve.modes] == pytest.approx([2.1, 2.2])
        assert solve.results is sim.read_results()
        assert sim.solved[-1]["save"] == 1

    def test_the_way_back_to_femwell_is_handed_down_to_the_run(self, tmp_path):
        """The simulation writes the runtime report; only this token is ours."""
        sim = FakeSim(tmp_path, leaves=mode_table(2))
        self._solve(sim, num_modes=2)
        assert "route='femwell'" in sim.ran[-1]["remedy"]

    def test_a_run_that_answered_nothing_is_reported(self, tmp_path):
        sim = FakeSim(tmp_path, leaves=mode_table(0))
        with pytest.raises(RuntimeError, match="no mode table"):
            self._solve(sim, num_modes=2)


class TestPalaceRoutePreparesTheLine:
    """The Palace route sizes the paths from the electrodes, not by hand."""

    def _meshed(self, biased, **settings):
        biased.rf(route="palace", frequencies_hz=[10e9], n_strips=3, **settings)
        staircase = biased.rf.staircase()
        sim = biased.rf.simulation(staircase)
        sim.mesh(**biased.rf.mesh)
        return sim, staircase

    def test_the_paths_are_declared_on_the_meshed_simulation(self, biased):
        from gsim.modulator.palace_route import PalaceRoute

        sim, staircase = self._meshed(biased)
        signal, return_ = biased.rf.line_conductors(staircase)
        route = PalaceRoute()

        route.prepare_line(sim, signal=signal, return_=return_, stage_name="rf")

        assert route.impedance_index == 1
        (path,) = sim.mode_paths
        (h_lo, h_hi), (v_lo, v_hi) = signal.extent
        # The voltage path leaves the signal electrode's inner face at
        # its mid-height; the loop surrounds that electrode.
        assert path.voltage_path[0][1] == pytest.approx(0.5 * (v_lo + v_hi))
        loop_h = [p[0] for p in path.current_path]
        assert min(loop_h) < h_lo
        assert max(loop_h) > h_hi

    def test_a_signal_the_window_clips_leaves_the_reading_to_the_fields(self, biased):
        """A Window clipping an electrode is no reason to refuse the solve."""
        from gsim.modulator.palace_route import PalaceRoute

        sim, staircase = self._meshed(biased, conductor_model="volume")
        signal, return_ = biased.rf.line_conductors(staircase)
        # Pretend the mesh stops short of the signal electrode's outer face.
        biased.rf(window=(signal.extent[0][0] + 0.5, signal.extent[0][1] + 10.0))
        sim = biased.rf.simulation(staircase)
        sim.mesh(**biased.rf.mesh)
        route = PalaceRoute()

        with pytest.warns(UserWarning, match="read off the saved fields instead"):
            route.prepare_line(sim, signal=signal, return_=return_, stage_name="rf")

        assert route.impedance_index is None
        assert sim.mode_paths == []

    def test_a_line_without_a_single_return_is_read_off_the_fields(self, biased):
        from gsim.modulator.palace_route import PalaceRoute

        sim, staircase = self._meshed(biased)
        signal, _ = biased.rf.line_conductors(staircase)
        route = PalaceRoute()

        with pytest.warns(UserWarning, match="no single return electrode"):
            route.prepare_line(sim, signal=signal, return_=None, stage_name="rf")

        assert route.impedance_index is None


class TestFemwellRouteReadsTheWallOffTheSimulation:
    """The metallic-boundary flag has one channel: the simulation."""

    @pytest.mark.parametrize("wall", [True, False])
    def test_the_solve_uses_the_simulation_flag(self, monkeypatch, wall):
        from types import SimpleNamespace

        from gsim.modulator.femwell_route import FemwellRoute

        seen = {}

        def fake_solve_modes(_mesh_path, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr("gsim.femwell.adapter.solve_modes", fake_solve_modes)
        sim = SimpleNamespace(mesh_path="mesh.msh", metallic_boundaries=wall)

        FemwellRoute().solve(
            sim,
            freq_hz=1e9,
            num_modes=1,
            target=None,
            order=1,
            verbose=False,
            stage_name="rf",
            epsilon=np.ones(1),
        )

        assert seen["metallic_boundaries"] is wall
