"""``configure_wave_port`` holds the same mode checks as ``WavePortConfig``.

A wave port reaches the mesher through two doors: the ``WavePortConfig`` model
(``add_wave_port`` / ``add_multimode_wave_port``) and the metadata that
``configure_wave_port`` writes on a gdsfactory port. Only the model used to
check the mode list and the pilot paths, so the metadata could carry
``mode=4, modes=(1, 2, 3)`` (a fourth, active entry) or a pilot path for a
mode that is not followed (silently dropped). Both doors now run one check.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack import LayerStack
from gsim.palace.models.ports import WavePortConfig
from gsim.palace.ports import configure_wave_port, extract_ports
from gsim.palace.ports.modes import check_wave_port_modes

PILOT = [[0.0, 0.0, 1.0], [0.0, 6.0, 1.0]]
NAN = float("nan")

BAD_DECLARATIONS = [
    pytest.param(4, (1, 2, 3), None, "is not in modes", id="mode-not-listed"),
    pytest.param(1, (1, 1), None, "twice", id="mode-twice"),
    pytest.param(1, (0, 1), None, "1-based", id="mode-below-one"),
    pytest.param(1, (1, 2), {3: PILOT}, "not one of", id="pilot-unlisted-mode"),
    pytest.param(1, (), {2: PILOT}, "not one of", id="pilot-for-another-mode"),
    pytest.param(1, (1, 2), {1: [[0.0, 0.0, 0.0]]}, "2\\+ points", id="one-point"),
    pytest.param(
        1, (1, 2), {1: [[0.0, 0.0], [1.0, 1.0]]}, "x, y, z", id="two-coordinates"
    ),
    pytest.param(
        1,
        (1, 2),
        {1: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
        "repeats the point",
        id="repeated-point",
    ),
    pytest.param(
        1, (1, 2), {1: [[0.0, 0.0, 0.0], [NAN, 0.0, 0.0]]}, "finite", id="nan-point"
    ),
]


def _component(*names: str) -> gf.Component:
    gf.gpdk.PDK.activate()
    c = gf.Component()
    for index, name in enumerate(names):
        c.add_port(
            name=name,
            center=(100 * index, 0),
            width=10,
            orientation=0,
            port_type="electrical",
            layer=gf.gpdk.LAYER.M1,
        )
    return c


class TestMetadataRoute:
    def test_an_active_mode_outside_the_list_is_refused(self) -> None:
        """``mode=4, modes=(1, 2, 3)`` used to give a fourth, active entry."""
        c = _component("o1")
        with pytest.raises(ValueError, match=r"mode 4 is not in modes \[1, 2, 3\]"):
            configure_wave_port(c.ports["o1"], layer="metal1", mode=4, modes=(1, 2, 3))

    def test_a_pilot_path_for_an_unlisted_mode_is_refused(self) -> None:
        """It used to be dropped without a word."""
        c = _component("o1")
        with pytest.raises(ValueError, match=r"mode 3, which is not one of \[1, 2\]"):
            configure_wave_port(
                c.ports["o1"], layer="metal1", modes=(1, 2), pilot_paths={3: PILOT}
            )

    @pytest.mark.parametrize(("mode", "modes", "paths", "message"), BAD_DECLARATIONS)
    def test_every_bad_declaration_is_refused_and_leaves_the_port_alone(
        self, mode, modes, paths, message
    ) -> None:
        c = _component("o1", "o2")
        with pytest.raises(ValueError, match=message):
            configure_wave_port(
                c.ports, layer="metal1", mode=mode, modes=modes, pilot_paths=paths
            )
        for port in c.ports:
            assert port.info.get("palace_type") is None
            assert port.info.get("modes") is None

    @pytest.mark.parametrize(("mode", "modes", "paths", "message"), BAD_DECLARATIONS)
    def test_the_model_refuses_the_same_declarations(
        self, mode, modes, paths, message
    ) -> None:
        with pytest.raises(ValueError, match=message):
            WavePortConfig(
                name="o1",
                layer="metal1",
                mode=mode,
                modes=modes,
                active_mode=mode if modes else None,
                pilot_paths=paths,
            )

    def test_a_good_declaration_still_travels_to_the_port(self) -> None:
        c = _component("o1")
        configure_wave_port(
            c.ports["o1"],
            layer="metal1",
            mode=2,
            modes=[1, 2, 3],
            excite_modes="all",
            pilot_paths={1: PILOT, 2: PILOT, 3: PILOT},
        )
        info = c.ports["o1"].info
        assert info["modes"] == [1, 2, 3]
        assert [item["mode"] for item in info["pilot_paths"]] == [1, 2, 3]
        (port,) = extract_ports(c, LayerStack())
        assert (port.mode, port.modes) == (2, (1, 2, 3))
        assert port.pilot_paths == {1: PILOT, 2: PILOT, 3: PILOT}

    def test_an_ordinary_port_may_pin_the_sign_of_its_own_mode(self) -> None:
        c = _component("o1")
        configure_wave_port(
            c.ports["o1"], layer="metal1", mode=2, pilot_paths={2: PILOT}
        )
        (port,) = extract_ports(c, LayerStack())
        assert port.modes == ()
        assert port.pilot_paths == {2: PILOT}

    def test_a_port_without_modes_is_unchanged(self) -> None:
        c = _component("o1")
        configure_wave_port(c.ports["o1"], layer="metal1")
        info = c.ports["o1"].info
        assert info["modes"] == []
        assert info["pilot_paths"] == []

    def test_one_mode_in_the_list_is_fine(self) -> None:
        c = _component("o1")
        configure_wave_port(c.ports["o1"], layer="metal1", mode=3, modes=(3,))
        assert c.ports["o1"].info["modes"] == [3]

    def test_pilot_paths_are_copied_per_port(self) -> None:
        c = _component("o1", "o2")
        configure_wave_port(
            c.ports, layer="metal1", modes=(1, 2), pilot_paths={1: PILOT}
        )
        first = c.ports["o1"].info["pilot_paths"][0]["path"]
        second = c.ports["o2"].info["pilot_paths"][0]["path"]
        assert first == second == PILOT
        assert first is not second
        assert first is not PILOT


class TestSharedCheck:
    def test_the_check_accepts_what_the_model_accepts(self) -> None:
        check_wave_port_modes(
            mode=2, modes=(1, 2, 3), active_mode=2, pilot_paths={1: PILOT}
        )
        check_wave_port_modes(mode=1, modes=(), active_mode=None, pilot_paths=None)

    def test_the_active_mode_is_named_as_the_caller_calls_it(self) -> None:
        with pytest.raises(ValueError, match="my_mode 9 is not in modes"):
            check_wave_port_modes(
                mode=9,
                modes=(1, 2),
                active_mode=9,
                pilot_paths=None,
                active_name="my_mode",
            )

    def test_a_disagreeing_active_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must agree"):
            check_wave_port_modes(mode=1, modes=(1, 2), active_mode=2, pilot_paths=None)
