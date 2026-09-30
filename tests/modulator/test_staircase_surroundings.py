"""What a Staircase carries around its Strips.

The Strips speak for the Carrier map and for nothing else. Everything
else the drawn Cross-section puts inside the meshed Window — the undoped
slab, the metal landing on the pads, an implant the charge solve never
covered — has to reach the Staircase too, or the Staircase is a different
waveguide from the one that was drawn. These are the pure geometry and
bookkeeping of that: the cut against the Strip footprint, the extent a
Carrier map covers, and what the two produce together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.modulator.staircase import (
    CrossSectionOrientationError,
    SurroundingRegion,
    carrier_map_extent,
    surroundings_from_section,
)


@dataclass(frozen=True)
class Rect:
    """A stand-in for one rectangle of a drawn cross-section."""

    layer_name: str
    material: str
    y0: float
    y1: float
    zmin: float
    zmax: float


def stack_with(*layers: Layer) -> LayerStack:
    """A stack carrying just the layers a case needs."""
    return LayerStack(
        layers={layer.name: layer for layer in layers},
        materials={"si": {"permittivity": 12.0}},
    )


def dielectric(name: str, material: str = "si") -> Layer:
    return Layer(
        name=name,
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.22,
        thickness=0.22,
        material=material,
        layer_type="dielectric",
    )


class TestCuttingAgainstTheStrips:
    def test_a_region_the_strips_cover_entirely_disappears(self):
        section = [Rect("core", "si", -0.2, 0.2, 0.0, 0.22)]

        regions = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
        )

        assert regions == ()

    def test_a_region_the_strips_do_not_touch_survives_whole(self):
        section = [Rect("clad_si", "si", 2.0, 3.0, 0.0, 0.09)]

        (region,) = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
        )

        assert region.name == "clad_si"
        assert region.h == (2.0, 3.0)
        assert region.z == (0.0, 0.09)

    def test_a_slab_running_under_the_strips_keeps_its_two_wings(self):
        """The drawn guide's slab is what the strips sit in the middle of."""
        section = [Rect("slab", "si", -5.0, 5.0, 0.0, 0.09)]

        regions = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
        )

        assert [region.h for region in regions] == [(-5.0, -0.6), (0.6, 5.0)]
        assert all(region.z == (0.0, 0.09) for region in regions)
        assert {region.name for region in regions} == {"slab_0", "slab_1"}

    def test_metal_standing_above_the_strips_keeps_its_upper_part(self):
        section = [Rect("pad_metal", "aluminum", -0.6, -0.3, 0.0, 0.7)]

        regions = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
        )

        assert [(region.h, region.z) for region in regions] == [
            ((-0.6, -0.3), (0.22, 0.7))
        ]

    def test_the_cut_pieces_tile_exactly_what_was_left(self):
        """No overlap and no gap: area in equals area out plus the cut."""
        section = [Rect("slab", "si", -5.0, 5.0, -0.5, 0.5)]
        strip_span, strip_z = (-0.6, 0.6), (0.0, 0.22)

        regions = surroundings_from_section(
            section, strip_span=strip_span, strip_z=strip_z
        )

        area = sum(
            (region.h[1] - region.h[0]) * (region.z[1] - region.z[0])
            for region in regions
        )
        cut = (strip_span[1] - strip_span[0]) * (strip_z[1] - strip_z[0])
        assert area == pytest.approx(10.0 * 1.0 - cut)

    def test_a_degenerate_rectangle_is_dropped(self):
        section = [Rect("sliver", "si", 1.0, 1.0, 0.0, 0.22)]

        assert (
            surroundings_from_section(
                section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
            )
            == ()
        )


class TestTheSectionIsXNormal:
    """A y-normal Cross-section is refused by name, not by ``AttributeError``."""

    @dataclass(frozen=True)
    class XZRect:
        """A rectangle of a y-normal Cross-section, as ``Rect2D`` carries it."""

        layer_name: str
        material: str
        x0: float
        x1: float
        zmin: float
        zmax: float

    def test_a_y_normal_section_is_reported(self):
        section = [self.XZRect("slab", "si", -5.0, 5.0, -0.5, 0.5)]

        with pytest.raises(CrossSectionOrientationError, match="x-normal"):
            surroundings_from_section(
                section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
            )


class TestWhatTheRegionsCarry:
    def test_a_conductor_stays_a_conductor(self):
        """ADR 0003: metal is meshed as an outline, not as a domain."""
        section = [Rect("metal", "aluminum", 2.0, 3.0, 0.0, 0.5)]
        stack = LayerStack(
            layers={
                "metal": Layer(
                    name="metal",
                    gds_layer=(41, 0),
                    zmin=0.0,
                    zmax=0.5,
                    thickness=0.5,
                    material="aluminum",
                    layer_type="conductor",
                )
            }
        )

        (region,) = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22), stack=stack
        )

        assert region.layer_type == "conductor"

    def test_a_region_off_the_stack_is_meshed_as_a_dielectric(self):
        section = [Rect("mystery", "si", 2.0, 3.0, 0.0, 0.5)]

        (region,) = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22)
        )

        assert region.layer_type == "dielectric"

    def test_a_material_the_database_does_not_know_travels_with_it(self):
        """A doped material is the stack's, not the database's."""
        section = [Rect("n_pad", "n_pad", 2.0, 3.0, 0.0, 0.22)]
        stack = LayerStack(
            layers={"n_pad": dielectric("n_pad", material="n_pad")},
            materials={"n_pad": {"permittivity": 11.9}},
        )

        (region,) = surroundings_from_section(
            section, strip_span=(-0.6, 0.6), strip_z=(0.0, 0.22), stack=stack
        )

        assert region.material == "n_pad"
        assert region.properties == {"permittivity": 11.9}


class TestCarrierMapExtent:
    @staticmethod
    def _map(h, v):
        class _Carriers:
            x_um = np.asarray(h, dtype=float)
            y_um = np.asarray(v, dtype=float)

        return _Carriers()

    def test_it_is_the_extent_of_the_sampled_nodes(self):
        carriers = self._map([-1.0, 0.0, 2.0], [0.0, 0.1, 0.2])

        assert carrier_map_extent(carriers) == (-1.0, 2.0)

    def test_a_band_measures_only_the_nodes_inside_it(self):
        carriers = self._map([-1.0, 0.0, 2.0], [0.0, 0.1, 0.9])

        assert carrier_map_extent(carriers, (0.0, 0.5)) == (-1.0, 0.0)

    def test_a_band_holding_no_node_is_reported(self):
        carriers = self._map([-1.0, 0.0], [0.0, 0.1])

        with pytest.raises(ValueError, match="No carrier samples"):
            carrier_map_extent(carriers, (5.0, 6.0))


class TestTheRegionsAreDrawable:
    def test_a_non_ascending_extent_is_reported(self):
        import gdsfactory as gf

        from gsim.modulator.staircase import _surrounding_layers

        bad = SurroundingRegion(name="x", h=(1.0, 1.0), z=(0.0, 0.2), material="si")
        with pytest.raises(ValueError, match="in-plane extent"):
            _surrounding_layers(
                gf.Component(), (bad,), length=10.0, base_layer=(310, 0)
            )

    def test_two_regions_of_one_name_are_reported(self):
        import gdsfactory as gf

        from gsim.modulator.staircase import _surrounding_layers

        one = SurroundingRegion(name="x", h=(0.0, 1.0), z=(0.0, 0.2), material="si")
        with pytest.raises(ValueError, match="both named"):
            _surrounding_layers(
                gf.Component(), (one, one), length=10.0, base_layer=(310, 0)
            )
