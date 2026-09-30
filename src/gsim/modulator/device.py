"""What a modulator Study needs to know about the device it is solving.

The description is written in the device's own words — which Regions are
p, which are n, where the Junction sits — and everything the solvers need
(Contacts, Interfaces, each Stage's Window) is derived from it against the
drawn Cross-section rather than declared one by one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["Device"]


class Device(BaseModel):
    """The device description a Study is built from.

    Attributes:
        p_regions: Names of the p-doped Regions, as named in the stack.
        n_regions: Names of the n-doped Regions.
        junction: The Junction, as the pair of Regions its metallurgical
            boundary separates. Derived from the single adjacent p/n pair
            when left unset.
        electrodes: Names of the metal Regions carrying the terminals.
            Defaults to every conductor layer of the stack that lands on a
            doped Region.
        p_doping_cm3: Acceptor concentration of the p Regions (cm^-3).
        n_doping_cm3: Donor concentration of the n Regions (cm^-3).
        doping: Explicit doping profiles replacing the step profiles the
            concentrations above imply.
        contact_names: Terminal name per electrode Region, overriding the
            derived anode/cathode naming.
        window_margin_um: Margin added around the doped slab when deriving
            the charge Window (um).
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    p_regions: list[str] = Field(min_length=1)
    n_regions: list[str] = Field(min_length=1)
    junction: tuple[str, str] | None = None
    electrodes: list[str] | None = None
    p_doping_cm3: float = Field(default=1e18, gt=0.0)
    n_doping_cm3: float = Field(default=1e18, gt=0.0)
    doping: list[Any] | None = None
    contact_names: dict[str, str] = Field(default_factory=dict)
    window_margin_um: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def _regions_are_disjoint(self) -> Device:
        """A Region is p or n, never both."""
        overlap = sorted(set(self.p_regions) & set(self.n_regions))
        if overlap:
            raise ValueError(
                f"Regions {overlap} are declared both p-doped and n-doped."
            )
        return self

    @property
    def doped_regions(self) -> list[str]:
        """Every doped Region, p side first."""
        return [*self.p_regions, *self.n_regions]

    def dopant_type(self, region: str) -> Literal["acceptor", "donor"]:
        """Dopant type of a doped Region.

        Args:
            region: Region name.

        Returns:
            ``"acceptor"`` for a p Region, ``"donor"`` for an n Region.
        """
        if region in self.p_regions:
            return "acceptor"
        if region in self.n_regions:
            return "donor"
        raise ValueError(f"Region '{region}' is not part of the device description.")

    def concentration_cm3(self, region: str) -> float:
        """Doping concentration of a doped Region (cm^-3).

        Args:
            region: Region name.

        Returns:
            The acceptor or donor concentration, by side.
        """
        return (
            self.p_doping_cm3
            if self.dopant_type(region) == "acceptor"
            else self.n_doping_cm3
        )
