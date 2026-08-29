"""Palik silicon-dioxide material cards."""

from pdk_schema import Lorentz, LorentzTerm, Pole, PoleResidue

from gsim.common.materials._helpers import material_card, wavelength_validity

# The source fit contains negligible non-Lorentz residue components. Dropping
# them makes this equivalent two-pole fit compatible with Meep while changing
# the C-band permittivity by less than 1e-11.
SIO2_PALIK_LOSSLESS = material_card(
    name="SiO2-Palik-Lossless",
    temperature_ref=293.0,
    permittivity=Lorentz(
        validity=wavelength_validity(0.15, 5.0),
        variation=None,
        eps_inf=1.5385442336875639,
        terms=(
            LorentzTerm(
                delta_eps=0.5686340834157791,
                omega_0=1.595196740783775e16,
                gamma=23008278.748555347,
            ),
            LorentzTerm(
                delta_eps=1.1574659369080176,
                omega_0=172280738540723.53,
                gamma=498780.7130088306,
            ),
        ),
    ),
)


SIO2_PALIK = material_card(
    name="SiO2-Palik",
    temperature_ref=293.0,
    permittivity=PoleResidue(
        validity=wavelength_validity(4.0, 250.0),
        variation=None,
        eps_inf=2.1560362571240765,
        poles=(
            Pole(
                a=(-3781744691507.2856, -207719670863343.84),
                c=(-18676276825273.156, -6355596169134.299),
            ),
            Pole(
                a=(-9306968330309.3, -199739685682949.9),
                c=(26685644798963.88, 81265966041216.78),
            ),
            Pole(
                a=(-11649519584911.078, -161489841654821.16),
                c=(-13040029201085.318, 2679209910871.1226),
            ),
            Pole(
                a=(-3052239610863.719, -88355407251640.77),
                c=(-24299959225698.41, 3850586684365.262),
            ),
            Pole(
                a=(-7182184304431.551, -84819227587180.16),
                c=(29330620453153.605, 39789511603200.61),
            ),
        ),
    ),
)


__all__ = ["SIO2_PALIK", "SIO2_PALIK_LOSSLESS"]
