"""Ideal, lossless air approximation for unassigned simulation space."""

from pdk_schema import Index, Provenance, ScalarValue

from gsim.common.materials._helpers import material_card

AIR = material_card(
    name="Air",
    temperature_ref=None,
    permittivity=Index(
        validity=None,
        variation=None,
        conductivity=None,
        n=ScalarValue(value=1, unit=""),
        k=ScalarValue(value=0, unit=""),
    ),
    provenance=Provenance(
        source="gdsfactory",
        label="Ideal air approximation",
        maturity="toy_model",
        citations=[],
        comment="Lossless, nondispersive n=1 approximation to air.",
        url=None,
        data_url=None,
        info={},
    ),
)

__all__ = ["AIR"]
