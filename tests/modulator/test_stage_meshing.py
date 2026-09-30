"""What the meshing Stages share: their mesh defaults and their Staircase.

Three Stages mesh a Cross-section and two of them build a Staircase from
one Carrier map. Nothing here solves anything: what is under test is that
the shared defaults are shared — and still overridable per Stage — and
that both EM Stages tile the same Strips over the same extent.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.modulator.meshing import STAGE_AIRBOX, STAGE_MESH


class TestSharedMeshingDefaults:
    def test_every_meshing_stage_takes_the_shared_airbox(self, study):
        for stage in (study.charge, study.optical, study.rf):
            assert stage.airbox == STAGE_AIRBOX

    def test_the_em_stages_take_the_shared_mesh_settings(self, study):
        assert study.optical.mesh == STAGE_MESH
        assert study.rf.mesh == STAGE_MESH

    def test_the_charge_stage_differs_only_where_its_physics_does(self, study):
        differing = {
            key for key in STAGE_MESH if study.charge.mesh[key] != STAGE_MESH[key]
        }
        assert differing == {"refined_mesh_size"}
        assert study.charge.mesh["refined_mesh_size"] < STAGE_MESH["refined_mesh_size"]

    def test_a_stage_still_overrides_what_it_wants(self, study):
        study.optical(mesh={**STAGE_MESH, "max_mesh_size": 1.0})

        assert study.optical.mesh["max_mesh_size"] == 1.0
        assert study.rf.mesh["max_mesh_size"] == STAGE_MESH["max_mesh_size"]

    def test_the_defaults_are_copies_no_stage_can_mutate(self, study, tmp_path):
        from gsim.modulator import Study

        study.charge.airbox["material"] = "air"
        study.charge.mesh["preset"] = "fine"

        other = Study(
            component=study.component,
            stack=study.stack,
            device=study.device,
            output_dir=tmp_path / "other",
        )
        assert STAGE_AIRBOX["material"] == "sio2"
        assert STAGE_MESH["preset"] == "coarse"
        assert other.charge.airbox["material"] == "sio2"
        assert other.charge.mesh["preset"] == "coarse"


class TestSharedStaircaseBuilder:
    def test_both_em_stages_tile_the_same_strips(self, biased):
        span = (biased.layout.junction_span.h[0], biased.layout.junction_span.h[1])
        biased.optical(route="palace", n_strips=4, strip_span=span)
        biased.rf(n_strips=4, strip_span=span)

        optical = biased.optical.staircase(biased.carriers.run().points[-1])
        rf = biased.rf.staircase()

        np.testing.assert_allclose(
            np.asarray(optical.strips.edges_um, dtype=float),
            np.asarray(rf.strips.edges_um, dtype=float),
        )
        assert optical.strip_names == rf.strip_names

    def test_both_em_stages_read_the_same_substrate_setting(self, biased):
        assert biased.optical.substrate_thickness_um == pytest.approx(
            biased.rf.substrate_thickness_um
        )
