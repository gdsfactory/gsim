"""Contacts, Interfaces and the charge Window come from the device description."""

from __future__ import annotations

import pytest

from tests.modulator.conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT

DOPED_SPAN = (
    CENTER_Y - HALF_WIDTH - PAD_WIDTH,
    CENTER_Y + HALF_WIDTH + PAD_WIDTH,
)


class TestContacts:
    def test_one_contact_per_electrode(self, study):
        contacts = {c.name: c for c in study.layout.contacts}
        assert set(contacts) == {"anode", "cathode"}

    def test_each_contact_binds_its_electrode_to_the_pad_below_it(self, study):
        contacts = {c.name: c for c in study.layout.contacts}
        assert contacts["anode"].region == "p_pad"
        assert contacts["anode"].electrode == "anode_metal"
        assert contacts["cathode"].region == "n_pad"
        assert contacts["cathode"].electrode == "cathode_metal"


class TestInterfaces:
    def test_every_adjacent_doped_pair_is_an_interface(self, study):
        pairs = {frozenset(i.regions) for i in study.layout.interfaces}
        assert pairs == {
            frozenset({"n_pad", "n_rib"}),
            frozenset({"n_rib", "p_rib"}),
            frozenset({"p_rib", "p_pad"}),
        }

    def test_the_pn_interface_is_the_junction(self, study):
        junction = study.layout.junction
        assert set(junction.regions) == {"n_rib", "p_rib"}
        assert junction.name == "junction"
        assert junction in study.layout.interfaces

    def test_interface_names_are_unique(self, study):
        names = [i.name for i in study.layout.interfaces]
        assert len(names) == len(set(names))


class TestJunctionSpan:
    def test_the_span_covers_both_regions_the_junction_separates(self, study):
        span = study.layout.junction_span

        assert span.h == pytest.approx((CENTER_Y - HALF_WIDTH, CENTER_Y + HALF_WIDTH))
        assert span.z == pytest.approx((0.0, RIB_HEIGHT))

    def test_the_span_is_narrower_than_the_charge_window(self, study):
        span = study.layout.junction_span

        assert span.h[0] > study.layout.window[0]
        assert span.h[1] < study.layout.window[1]


class TestChargeWindow:
    def test_window_spans_the_doped_slab(self, study):
        low, high = study.layout.window
        assert low < DOPED_SPAN[0]
        assert high > DOPED_SPAN[1]
        assert low == pytest.approx(DOPED_SPAN[0] - 0.5)
        assert high == pytest.approx(DOPED_SPAN[1] + 0.5)

    def test_margin_is_configurable_on_the_device(self, phase_shifter, tmp_path):
        from gsim.modulator import Device, Study

        component, stack = phase_shifter
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
                window_margin_um=1.5,
            ),
            output_dir=tmp_path,
        )
        assert study.layout.window[0] == pytest.approx(DOPED_SPAN[0] - 1.5)

    def test_explicit_window_overrides_the_derivation(self, study):
        study.charge(window=(-21.0, -19.0))
        assert study.charge.window == (-21.0, -19.0)
        assert study.layout.window != (-21.0, -19.0)


class TestRegionSpans:
    def test_spans_are_read_off_the_cross_section(self, study):
        spans = study.layout.region_spans
        assert spans["p_rib"].h == pytest.approx((CENTER_Y, CENTER_Y + HALF_WIDTH))
        assert spans["p_rib"].z == pytest.approx((0.0, RIB_HEIGHT))
        assert spans["anode_metal"].z == pytest.approx((RIB_HEIGHT, RIB_HEIGHT + 0.5))


class TestDerivationErrors:
    def test_unknown_region_names_the_available_ones(self, phase_shifter, tmp_path):
        from gsim.modulator import Device, Study

        component, stack = phase_shifter
        study = Study(
            component=component,
            stack=stack,
            device=Device(p_regions=["nope"], n_regions=["n_rib"]),
            output_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="p_rib"):
            _ = study.layout

    def test_a_device_without_a_pn_pair_is_reported(self, phase_shifter, tmp_path):
        from gsim.modulator import Device, Study

        component, stack = phase_shifter
        study = Study(
            component=component,
            stack=stack,
            device=Device(p_regions=["p_pad"], n_regions=["n_pad"]),
            output_dir=tmp_path,
        )
        with pytest.raises(ValueError, match=r"[Jj]unction"):
            _ = study.layout

    def test_the_junction_can_be_named_explicitly(self, phase_shifter, tmp_path):
        from gsim.modulator import Device, Study

        component, stack = phase_shifter
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
                junction=("p_rib", "n_rib"),
            ),
            output_dir=tmp_path,
        )
        assert set(study.layout.junction.regions) == {"p_rib", "n_rib"}


class TestDerivationCache:
    def test_the_layout_is_cached(self, study):
        assert study.layout is study.layout

    def test_changing_the_device_drops_the_derived_layout(self, study):
        from gsim.modulator import Device

        first = study.layout
        study.device = Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
            window_margin_um=2.0,
        )
        assert study.layout is not first
        assert study.layout.window[0] == pytest.approx(DOPED_SPAN[0] - 2.0)

    def test_changing_the_device_clears_stage_results(self, study, monkeypatch):
        from gsim.modulator import Device
        from gsim.tcad.results import BiasSweepResult

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda self, biases, contact=None, verbose=False: BiasSweepResult(
                contact="cathode", points=[]
            ),
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        study.charge.run()
        assert study.charge.has_run is True

        study.device = Device(
            p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]
        )

        assert study.charge.has_run is False

    def test_changing_the_component_drops_the_derived_layout(
        self, study, phase_shifter
    ):
        component, _stack = phase_shifter
        first = study.layout
        study.component = component
        assert study.layout is not first

    def test_changing_the_stack_drops_the_derived_layout(self, study, phase_shifter):
        _component, stack = phase_shifter
        first = study.layout
        study.stack = stack
        assert study.layout is not first

    def test_moving_the_plane_is_validated(self, study):
        with pytest.raises(ValueError, match="plane"):
            study.plane = "w=3"

    def test_a_plane_naming_no_coordinate_is_rejected(self, study):
        """The shared parse refuses what the Study's own one accepted."""
        with pytest.raises(ValueError, match="plane"):
            study.plane = "x="
