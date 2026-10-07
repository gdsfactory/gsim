"""One call from a Carrier map to a meshable Staircase cross-section."""

from __future__ import annotations

import sys
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.modulator.staircase import (
    DEFAULT_STRIP_LAYER,
    ElectrodeSpec,
    OpticalStripMaterial,
    RFStripMaterial,
    StaircaseDrawing,
    StripSegment,
    build_staircase_cross_section,
)
from tests._helpers import fake_coupling

#: The optical Stage's strip input, at the wavelength the fixtures solve at.
OPTICAL = OpticalStripMaterial(wavelength_um=1.55, index=3.4757)

CENTER = -20.0
HALF_WIDTH = 0.3
RIB_HEIGHT = 0.22
JUNCTION = (CENTER - HALF_WIDTH, CENTER + HALF_WIDTH)
SLAB_HEIGHT = 0.09

#: A rib beside a thinner slab: the low half of the extent at the rib's
#: height in three Strips, the high half at the slab's in two.
RIB_AND_SLAB = (
    StripSegment(span=(JUNCTION[0], CENTER), z=(0.0, RIB_HEIGHT), n_strips=3),
    StripSegment(span=(CENTER, JUNCTION[1]), z=(0.0, SLAB_HEIGHT), n_strips=2),
)


def analytic_carriers(h):
    """Smooth, strictly varying electron/hole profiles across the junction."""
    u = (np.asarray(h) - CENTER) / HALF_WIDTH
    return 1e18 * np.exp(-((u - 1.0) ** 2)), 1e18 * np.exp(-((u + 1.0) ** 2))


def carrier_map(n_samples=161):
    """Carrier map sampled across the rib, in the mesh frame (x=y_layout)."""
    h = np.linspace(JUNCTION[0], JUNCTION[1], n_samples)
    z = np.linspace(0.0, RIB_HEIGHT, 5)
    hh, zz = (a.ravel() for a in np.meshgrid(h, z))
    electrons, holes = analytic_carriers(hh)
    return SimpleNamespace(x_um=hh, y_um=zz, electrons_cm3=electrons, holes_cm3=holes)


def material_of(stack, region):
    """Material properties of a region, however the stack stores them."""
    from gsim.common.stack.materials import MaterialProperties

    props = stack.materials[stack.layers[region].material]
    return (
        props
        if isinstance(props, MaterialProperties)
        else MaterialProperties.model_validate(props)
    )


def build(carriers=None, **kwargs):
    """Build a staircase with the shared device description."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    params = dict(
        response=fake_coupling,
        material=RFStripMaterial(),
    )
    if "segments" not in kwargs:
        params.update(n_strips=5, junction=JUNCTION, zmin=0.0, zmax=RIB_HEIGHT)
    params.update(kwargs)
    return build_staircase_cross_section(
        carriers if carriers is not None else carrier_map(), **params
    )


class TestOneCall:
    def test_produces_a_meshable_cross_section(self):
        staircase = build()
        stack = staircase.stack()

        assert staircase.strip_names == [f"strip_{i}" for i in range(5)]
        for name in staircase.strip_names:
            assert name in stack.layers
            assert stack.layers[name].material in stack.materials
            layer = stack.layers[name]
            assert (layer.zmin, layer.zmax) == (0.0, RIB_HEIGHT)
        drawn = {tuple(layer) for layer in staircase.component.layers}
        assert all(
            tuple(stack.layers[name].gds_layer) in drawn
            for name in staircase.strip_names
        )

    def test_strips_tile_the_junction_extent(self):
        edges = build().strips.edges_um
        assert edges[0] == pytest.approx(JUNCTION[0])
        assert edges[-1] == pytest.approx(JUNCTION[1])
        assert np.all(np.diff(edges) > 0)

    def test_rejects_a_junction_extent_the_carriers_do_not_cover(self):
        with pytest.raises(ValueError):
            build(junction=(CENTER + 5.0, CENTER + 6.0))


class TestBothMaterialResponses:
    def test_strips_carry_the_coupling_evaluated_on_their_averages(self):
        staircase = build()
        strips = staircase.strips
        assert strips.count == 5
        expected = fake_coupling(strips.electrons_cm3, strips.holes_cm3)
        np.testing.assert_allclose(
            strips.conductivity_s_per_m, expected.conductivity_s_per_m
        )
        np.testing.assert_allclose(strips.index_shift, expected.index_shift)
        np.testing.assert_allclose(strips.absorption_cm, expected.absorption_cm)

    def test_the_coupling_is_the_one_handed_in(self):
        """A different coupling, a different Staircase — same drawing."""

        def twice(n_cm3, p_cm3):
            base = fake_coupling(n_cm3, p_cm3)
            base.conductivity_s_per_m = 2.0 * base.conductivity_s_per_m
            return base

        one = build()
        two = build(response=twice)
        np.testing.assert_allclose(
            two.strips.conductivity_s_per_m, 2.0 * one.strips.conductivity_s_per_m
        )
        np.testing.assert_allclose(two.strips.edges_um, one.strips.edges_um)

    def test_the_rf_stack_carries_conductivity(self):
        staircase = build()
        stack = staircase.stack()
        sigmas = [
            material_of(stack, name).conductivity for name in staircase.strip_names
        ]
        assert sigmas == pytest.approx(list(staircase.strips.conductivity_s_per_m))
        # The lattice permittivity is the RF input's, on every Strip.
        assert np.all(staircase.strips.permittivity == RFStripMaterial().permittivity)

    def test_the_optical_stack_carries_the_perturbed_permittivity(self):
        staircase = build(electrodes=None, material=OPTICAL)
        stack = staircase.stack()
        for i, name in enumerate(staircase.strip_names):
            props = material_of(stack, name)
            expected = complex(staircase.strips.permittivity[i])
            assert props.permittivity == pytest.approx(expected.real)
            assert props.loss_tangent > 0
            # Free carriers lower the index below the unperturbed one.
            assert expected.real < OPTICAL.index**2

    def test_the_extinction_is_built_at_the_solve_wavelength(self):
        """kappa = alpha lambda / 4 pi, and lambda is the solve's.

        The dispersion model's own wavelength is where its coefficients
        were fitted, which says what ``alpha`` is; it does not say what
        wavelength the Stage is solving at. Reading the extinction off
        the fit wavelength inflates the loss of every strip whenever the
        two differ.
        """
        fitted = build(electrodes=None, material=OPTICAL).strips
        solved = build(
            electrodes=None, material=OpticalStripMaterial(wavelength_um=1.31)
        ).strips

        # alpha is the model's answer and does not move with the solve.
        assert solved.absorption_cm == pytest.approx(fitted.absorption_cm)
        assert solved.index_shift == pytest.approx(fitted.index_shift)
        for at_fit, at_solve in zip(
            fitted.permittivity, solved.permittivity, strict=True
        ):
            assert at_solve.imag == pytest.approx(at_fit.imag * 1.31 / 1.55)

    def test_the_stack_is_built_once(self):
        staircase = build(electrodes=None)
        assert staircase.stack() is staircase.stack()

    def test_the_material_says_which_stage_the_staircase_is_for(self):
        assert isinstance(build().material, RFStripMaterial)
        assert build(material=OPTICAL).material is OPTICAL


class TestUnloaded:
    """The same Staircase with its carriers switched off."""

    def test_the_strips_carry_no_response(self):
        bare = build().unloaded()
        strips = bare.strips
        assert np.all(strips.electrons_cm3 == 0.0)
        assert np.all(strips.holes_cm3 == 0.0)
        assert np.all(strips.conductivity_s_per_m == 0.0)
        assert np.all(strips.index_shift == 0.0)
        assert np.all(strips.permittivity.imag == 0.0)

    def test_the_drawing_is_the_loaded_staircases(self):
        loaded = build()
        bare = loaded.unloaded()
        assert bare.component is loaded.component
        assert bare.strip_names == loaded.strip_names
        assert bare.electrode_spans == loaded.electrode_spans
        assert bare.layers == loaded.layers
        np.testing.assert_allclose(bare.strips.edges_um, loaded.strips.edges_um)

    def test_the_bare_stack_carries_no_conductivity(self):
        loaded = build()
        bare = loaded.unloaded()
        for name in bare.strip_names:
            assert material_of(bare.stack(), name).conductivity == 0.0
            assert material_of(loaded.stack(), name).conductivity > 0.0


class TestElectrodes:
    def test_electrodes_flank_the_junction_by_default(self):
        staircase = build()
        stack = staircase.stack()

        assert len(staircase.electrode_names) == 2
        for name in staircase.electrode_names:
            assert name in stack.layers
            assert material_of(stack, name).conductivity > 1e6

    def test_electrode_geometry_follows_the_device_description(self):
        spec = ElectrodeSpec(width_um=1.5, gap_um=0.4, thickness_um=0.6)
        staircase = build(electrodes=spec)
        stack = staircase.stack()

        low, high = staircase.electrode_spans
        assert high[0] == pytest.approx(JUNCTION[1] + 0.4)
        assert high[1] == pytest.approx(JUNCTION[1] + 0.4 + 1.5)
        assert low[1] == pytest.approx(JUNCTION[0] - 0.4)
        assert low[0] == pytest.approx(JUNCTION[0] - 0.4 - 1.5)
        electrode = stack.layers[staircase.electrode_names[0]]
        assert electrode.thickness == pytest.approx(0.6)

    def test_electrodes_can_be_left_out(self):
        staircase = build(electrodes=None)
        assert staircase.electrode_names == ()
        assert staircase.electrode_spans == ()


class TestConductorModel:
    """How the electrode metal reaches the mesh (ADR 0003)."""

    def test_a_volume_electrode_is_a_region_of_lossy_metal(self):
        staircase = build()
        stack = staircase.stack()

        assert staircase.conductor_model == "volume"
        for name in staircase.electrode_names:
            assert stack.layers[name].layer_type == "dielectric"
            assert material_of(stack, name).conductivity > 1e6

    def test_a_pec_electrode_is_a_conductor_layer_without_conductivity(self):
        """A conductor layer is what the native-2D mesher meshes as an
        outline, and no conductivity is what makes that outline perfect
        rather than a surface impedance."""
        staircase = build(electrodes=ElectrodeSpec(conductor_model="pec"))
        stack = staircase.stack()

        assert staircase.conductor_model == "pec"
        for name in staircase.electrode_names:
            assert stack.layers[name].layer_type == "conductor"
            assert not material_of(stack, name).conductivity

    def test_a_pec_electrode_needs_no_optical_permittivity(self):
        """A perfect conductor carries no permittivity to be asked for."""
        staircase = build(
            electrodes=ElectrodeSpec(conductor_model="pec"), material=OPTICAL
        )
        stack = staircase.stack()
        assert stack.layers[staircase.electrode_names[0]].layer_type == "conductor"

    def test_the_model_does_not_move_the_drawn_metal(self):
        """Only how the metal is expressed changes, not where it is."""
        volume = build()
        pec = build(electrodes=ElectrodeSpec(conductor_model="pec"))
        assert pec.electrode_spans == volume.electrode_spans
        assert pec.electrode_extent(pec.electrode_names[0]) == volume.electrode_extent(
            volume.electrode_names[0]
        )


class TestElectrodeExtent:
    def test_it_reports_the_rectangle_the_electrode_occupies(self):
        spec = ElectrodeSpec(width_um=1.5, gap_um=0.4, thickness_um=0.6)
        staircase = build(electrodes=spec)

        h_span, v_span = staircase.electrode_extent("electrode_high")
        assert h_span == pytest.approx((JUNCTION[1] + 0.4, JUNCTION[1] + 0.4 + 1.5))
        assert v_span == pytest.approx((0.0, 0.6))

    def test_an_electrode_the_staircase_never_drew_is_reported(self):
        staircase = build()
        with pytest.raises(ValueError, match="no electrode named 'ground'"):
            staircase.electrode_extent("ground")

    def test_a_staircase_without_electrodes_has_no_extent(self):
        staircase = build(electrodes=None)
        assert staircase.conductor_model is None
        with pytest.raises(ValueError, match="no electrode named"):
            staircase.electrode_extent("electrode_low")


class TestOpticalElectrodes:
    def test_an_rf_electrode_is_refused_by_an_optical_stack(self):
        staircase = build(material=OPTICAL)
        with pytest.raises(ValueError, match="optical_permittivity"):
            staircase.stack()

    def test_the_optical_metal_permittivity_is_used_when_given(self):
        # Aluminium near 1.55 um: n = 1.44, k = 16.0.
        eps = complex((1.44 - 16.0j) ** 2)
        staircase = build(
            electrodes=ElectrodeSpec(optical_permittivity=eps), material=OPTICAL
        )
        stack = staircase.stack()

        props = material_of(stack, staircase.electrode_names[0])
        assert props.permittivity == pytest.approx(eps.real)
        assert props.loss_tangent == pytest.approx(-eps.imag / eps.real)

    def test_the_rf_stack_is_unaffected(self):
        staircase = build()
        stack = staircase.stack()
        assert material_of(stack, staircase.electrode_names[0]).conductivity > 1e6


class TestStripCount:
    def test_one_strip_reproduces_the_uniform_model(self):
        staircase = build(n_strips=1)
        strips = staircase.strips

        assert staircase.strip_names == ["strip_0"]
        assert len(strips.edges_um) == 2
        h = np.linspace(*JUNCTION, 20001)
        electrons, _holes = analytic_carriers(h)
        assert strips.electrons_cm3[0] == pytest.approx(
            np.trapezoid(electrons, h) / (JUNCTION[1] - JUNCTION[0]), rel=1e-3
        )

    def test_more_strips_converge_toward_the_continuous_profile(self):
        errors = []
        for n_strips in (1, 4, 16):
            strips = build(n_strips=n_strips).strips
            edges = np.asarray(strips.edges_um)
            centres = 0.5 * (edges[1:] + edges[:-1])
            exact, _holes = analytic_carriers(centres)
            errors.append(
                float(np.mean(np.abs(strips.electrons_cm3 - exact)) / np.max(exact))
            )
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.01


class TestDrawing:
    """The drawing defaults reach the drawn Regions."""

    def test_the_defaults_are_the_ones_every_stage_uses(self):
        drawing = StaircaseDrawing()
        assert drawing.base_layer == DEFAULT_STRIP_LAYER
        assert drawing.name_prefix == "strip_"
        assert drawing.axis == "x"
        assert drawing.substrate_thickness == 2.0
        assert drawing.component is None

    def test_a_drawing_record_names_and_places_the_strips(self):
        import gdsfactory as gf

        comp = gf.Component()
        staircase = build(
            n_strips=2,
            drawing=StaircaseDrawing(
                base_layer=(77, 3), name_prefix="bin_", component=comp
            ),
        )
        assert staircase.component is comp
        assert staircase.strip_names == ["bin_0", "bin_1"]
        assert staircase.layers["bin_1"].gds_layer == (77, 4)


class TestDrawnGeometry:
    """What the Strips are drawn as, which is what a solver reads back.

    Both of these are the difference between the two EM Routes seeing the
    same problem and seeing two different ones: Palace resolves the drawn
    layers against the stack and honours whatever conductor it finds
    there, while femwell reads only the meshed regions.
    """

    def test_strips_are_not_drawn_on_a_generic_pdk_layer(self):
        """A Strip on a PDK metal or via layer resolves as that conductor."""
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
        from gdsfactory.gpdk.layer_map import LAYER

        pdk_layers = set()
        for name in dir(LAYER):
            if name.startswith("_"):
                continue
            layer = getattr(LAYER, name)
            try:
                pdk_layers.add((int(layer.layer), int(layer.datatype)))
            except (AttributeError, TypeError, ValueError):
                continue

        staircase = build(n_strips=8)
        drawn = {
            (spec.gds_layer[0], spec.gds_layer[1])
            for name, spec in staircase.layers.items()
            if name in staircase.strip_names
        }
        assert DEFAULT_STRIP_LAYER in drawn
        assert not (drawn & pdk_layers)

    @pytest.mark.parametrize("n_strips", [3, 5, 8, 16])
    def test_adjacent_strips_share_their_edge_exactly(self, n_strips):
        """No strip count may snap a sliver of background between Strips.

        Strip edges land off the GDS grid for some counts — eight strips
        across a 0.6 um junction put every centre on a half-nanometre —
        so a Strip drawn from a width and a centre can be rounded a
        nanometre away from its neighbour. Drawn from its two edges, the
        shared edge is one coordinate that rounds once.
        """
        staircase = build(n_strips=n_strips)
        component = staircase.component
        dbu = component.kcl.dbu

        spans = []
        for name in staircase.strip_names:
            spec = staircase.layers[name]
            raw = component.get_polygons(layers=(tuple(spec.gds_layer),), merge=False)
            points = [
                (point.y * dbu)
                for value in raw.values()
                for polygon in (value if isinstance(value, list) else [value])
                for point in polygon.each_point_hull()
            ]
            spans.append((min(points), max(points)))

        assert len(spans) == n_strips
        spans.sort()
        for (_low, high), (next_low, _next_high) in pairwise(spans):
            assert high == next_low


class TestSegments:
    """Strips that follow the drawn device: a run per Region, at its height."""

    def test_each_segment_is_tiled_by_its_own_strips(self):
        edges = build(segments=RIB_AND_SLAB).strips.edges_um

        expected = np.concatenate(
            (
                np.linspace(JUNCTION[0], CENTER, 4),
                np.linspace(CENTER, JUNCTION[1], 3)[1:],
            )
        )
        np.testing.assert_allclose(edges, expected)

    def test_each_strip_is_drawn_at_its_segments_height(self):
        staircase = build(segments=RIB_AND_SLAB)
        stack = staircase.stack()

        heights = [
            (stack.layers[name].zmin, stack.layers[name].zmax)
            for name in staircase.strip_names
        ]
        assert heights == [(0.0, RIB_HEIGHT)] * 3 + [(0.0, SLAB_HEIGHT)] * 2
        np.testing.assert_allclose(
            staircase.strips.zmax_um, [RIB_HEIGHT] * 3 + [SLAB_HEIGHT] * 2
        )

    def test_a_strip_averages_the_carriers_over_its_own_height(self):
        """Electrons only in the bottom 90 nm: the slab Strips hold them
        undiluted, a rib-height Strip over the same ground would not."""
        h = np.linspace(JUNCTION[0], JUNCTION[1], 61)
        z = np.linspace(0.0, RIB_HEIGHT, 23)
        hh, zz = (a.ravel() for a in np.meshgrid(h, z))
        low = np.where(zz <= SLAB_HEIGHT, 1e18, 0.0)
        carriers = SimpleNamespace(
            x_um=hh, y_um=zz, electrons_cm3=low, holes_cm3=np.zeros_like(low)
        )

        strips = build(carriers, segments=RIB_AND_SLAB).strips

        np.testing.assert_allclose(strips.electrons_cm3[3:], 1e18)
        assert np.all(strips.electrons_cm3[:3] < 0.5e18)

    def test_overlapping_segments_are_refused(self):
        clash = (
            StripSegment(span=(JUNCTION[0], CENTER + 0.1), z=(0.0, 0.22), n_strips=2),
            StripSegment(span=(CENTER, JUNCTION[1]), z=(0.0, 0.09), n_strips=2),
        )
        with pytest.raises(ValueError, match="overlap"):
            build(segments=clash)

    def test_segments_replace_the_single_extent(self):
        with pytest.raises(ValueError, match="segments"):
            build(segments=RIB_AND_SLAB, n_strips=5)

    def test_the_unloaded_staircase_keeps_the_heights(self):
        loaded = build(segments=RIB_AND_SLAB)
        np.testing.assert_allclose(
            loaded.unloaded().strips.zmax_um, loaded.strips.zmax_um
        )


def test_builds_without_any_solver_runtime(monkeypatch):
    for name in ("devsim", "femwell", "skfem", "gmsh"):
        monkeypatch.setitem(sys.modules, name, None)
    staircase = build(n_strips=3)
    assert staircase.stack().layers
