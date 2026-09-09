"""Generate the optical material index plot used by the documentation."""

from pathlib import Path

import plotly.graph_objects as go
from pdk_schema import Index, TabulatedValue

from gsim.common.materials import GSIM_MATERIAL_CARDS, resolve_material_snapshot

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPOSITORY_ROOT / "docs" / "assets" / "plots" / "material-index-vs-wavelength.html"
)
SCALAR_MODEL_RANGES_UM = {
    "Si-Salzberg": (1.357, 11.04),
    "Si-Li-293K": (1.2, 14.0),
    "SiN-Luke": (0.310, 5.504),
    "SiO2-Malitson": (0.21, 6.7),
}
LINE_STYLES = ("solid", "dash", "dashdot", "dot")
ANISOTROPIC_MODEL_COLORS = {
    "LiNbO3-Zelmon": "#FFA15A",
    "LiNbO3-MgO5-Gayer": "#19D3F3",
}
NUMBER_OF_SAMPLES = 300


def sample_wavelengths(minimum_um: float, maximum_um: float) -> list[float]:
    """Return evenly spaced wavelengths across a model's valid range."""
    wavelength_span = maximum_um - minimum_um
    return [
        minimum_um + wavelength_span * sample_index / (NUMBER_OF_SAMPLES - 1)
        for sample_index in range(NUMBER_OF_SAMPLES)
    ]


def wavelength_slice_at_reference_temperature(
    value: TabulatedValue,
    temperature_ref_kelvin: float | None,
) -> tuple[list[float], list[float]]:
    """Read a wavelength curve from a 1-D table or a 2-D reference slice."""
    table = value.data
    wavelengths_um = table.coords["wavelength"].values
    if tuple(table.dims) == ("wavelength",):
        return wavelengths_um, table.values
    if tuple(table.dims) != ("wavelength", "temperature"):
        raise TypeError(f"Unsupported optical table dimensions {table.dims!r}")
    if temperature_ref_kelvin is None:
        raise TypeError("A temperature-dependent optical table needs a reference")

    temperature_coordinate = table.coords["temperature"]
    if temperature_coordinate.unit != "K":
        raise TypeError("Temperature-dependent optical tables must use kelvin")
    temperature_index = min(
        range(len(temperature_coordinate.values)),
        key=lambda index: abs(
            temperature_coordinate.values[index] - temperature_ref_kelvin
        ),
    )
    if (
        abs(temperature_coordinate.values[temperature_index] - temperature_ref_kelvin)
        > 1e-9
    ):
        raise ValueError("The card reference temperature is not present in its table")

    temperature_count = len(temperature_coordinate.values)
    expected_value_count = len(wavelengths_um) * temperature_count
    if len(table.values) != expected_value_count:
        raise ValueError("Temperature-dependent optical table has an invalid shape")
    refractive_indices = [
        table.values[wavelength_index * temperature_count + temperature_index]
        for wavelength_index in range(len(wavelengths_um))
    ]
    return wavelengths_um, refractive_indices


def add_anisotropic_traces(figure: go.Figure, material_name: str, color: str) -> None:
    """Add the ordinary and extraordinary tensor components to the plot."""
    card = GSIM_MATERIAL_CARDS[material_name]
    if card.optical is None or not isinstance(card.optical.permittivity, Index):
        raise TypeError(f"{material_name} is not an optical Index card")
    model = card.optical.permittivity
    if not isinstance(model.n, list) or len(model.n) != 3:
        raise TypeError(f"{material_name} does not have a diagonal index tensor")

    for axis_index, axis_label, line_style in (
        (0, "n<sub>o</sub>", "solid"),
        (2, "n<sub>e</sub>", "dash"),
    ):
        axis_value = model.n[axis_index]
        if not isinstance(axis_value, TabulatedValue):
            raise TypeError(f"{material_name} {axis_label} is not tabulated")
        wavelengths_um, refractive_indices = wavelength_slice_at_reference_temperature(
            axis_value, card.optical.temperature_ref
        )
        temperature_c = (
            None
            if card.optical.temperature_ref is None
            else card.optical.temperature_ref - 273.15
        )
        temperature_hover = (
            "" if temperature_c is None else f"<br>T={temperature_c:g} °C"
        )
        figure.add_trace(
            go.Scatter(
                x=wavelengths_um,
                y=refractive_indices,
                mode="lines",
                name=f"{material_name} {axis_label}",
                legendgroup=material_name,
                line={"color": color, "dash": line_style, "width": 2},
                hovertemplate=(
                    "wavelength=%{x:.3f} µm<br>n=%{y:.5f}"
                    + temperature_hover
                    + "<extra>%{fullData.name}</extra>"
                ),
            )
        )


def generate_plot(output_path: Path = OUTPUT_PATH) -> None:
    """Write the interactive refractive-index comparison plot as HTML."""
    figure = go.Figure()

    for (material_name, valid_range_um), line_style in zip(
        SCALAR_MODEL_RANGES_UM.items(), LINE_STYLES, strict=True
    ):
        wavelengths_um = sample_wavelengths(*valid_range_um)
        refractive_indices = [
            resolve_material_snapshot(
                material_name,
                wavelength_um,
                project_material_cards={},
            ).refractive_index
            for wavelength_um in wavelengths_um
        ]
        figure.add_trace(
            go.Scatter(
                x=wavelengths_um,
                y=refractive_indices,
                mode="lines",
                name=material_name,
                line={"dash": line_style, "width": 2},
                hovertemplate=(
                    "wavelength=%{x:.3f} µm<br>n=%{y:.5f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    for material_name, color in ANISOTROPIC_MODEL_COLORS.items():
        add_anisotropic_traces(figure, material_name, color)

    figure.update_layout(
        template="plotly_white",
        height=480,
        margin={"l": 60, "r": 235, "t": 20, "b": 55},
        hovermode="closest",
        clickmode="event+select",
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
        },
        xaxis={
            "title": "Wavelength (µm)",
            "range": [
                min(valid_range[0] for valid_range in SCALAR_MODEL_RANGES_UM.values()),
                max(valid_range[1] for valid_range in SCALAR_MODEL_RANGES_UM.values()),
            ],
        },
        yaxis={"title": "Refractive index"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
        div_id="material-index-plot",
    )
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n")


if __name__ == "__main__":
    generate_plot()
