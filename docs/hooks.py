"""MkDocs hooks for gsim documentation."""

from pathlib import Path
from typing import Any

DOCS_DIR = Path(__file__).parent
API_MD = DOCS_DIR / "api.md"

# Define the API structure to document
API_SECTIONS = {
    "Types": [
        "gsim.palace.PortType",
        "gsim.palace.MeshPreset",
        "gsim.palace.GroundPlane",
    ],
    "Classes": [
        "gsim.palace.Layer",
        "gsim.palace.LayerStack",
        "gsim.palace.StackLayer",
        "gsim.palace.MaterialProperties",
        "gsim.palace.ValidationResult",
        "gsim.palace.PalacePort",
        "gsim.palace.PortGeometry",
        "gsim.palace.MeshConfig",
        "gsim.palace.MeshResult",
    ],
    "Functions": [
        "gsim.palace.get_stack",
        "gsim.palace.load_stack_yaml",
        "gsim.palace.extract_from_pdk",
        "gsim.palace.extract_layer_stack",
        "gsim.palace.parse_layer_stack",
        "gsim.palace.get_material_properties",
        "gsim.palace.material_is_conductor",
        "gsim.palace.material_is_dielectric",
        "gsim.palace.plot_stack",
        "gsim.palace.print_stack",
        "gsim.palace.print_stack_table",
        "gsim.palace.configure_inplane_port",
        "gsim.palace.configure_via_port",
        "gsim.palace.configure_cpw_port",
        "gsim.palace.extract_ports",
        "gsim.palace.generate_mesh",
    ],
    "Constants": [
        "gsim.palace.MATERIALS_DB",
    ],
}


def generate_api_md() -> str:
    """Generate the API documentation markdown content."""
    lines = ["# API\n"]

    for section, items in API_SECTIONS.items():
        lines.append(f"## {section}\n")
        for item in items:
            lines.append(f"::: {item}\n")

    return "\n".join(lines)


def on_startup(command: str, dirty: bool, **kwargs: Any) -> None:
    """Called once when the MkDocs build starts."""


def on_shutdown(**kwargs: Any) -> None:
    """Called once when the MkDocs build ends."""


def on_config(config: Any, **kwargs: Any) -> Any:
    """Called after config file is loaded but before validation."""
    return config


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Called before the build starts. Generate api.md here."""
    api_content = generate_api_md()
    API_MD.write_text(api_content)


def on_files(files: Any, config: Any, **kwargs: Any) -> Any:
    """Called after files are gathered but before processing."""
    return files


def on_nav(nav: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after navigation is built."""
    return nav


def on_env(env: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after Jinja2 environment is created."""
    return env


def on_post_build(config: Any, **kwargs: Any) -> None:
    """Called after the build is complete."""


def on_pre_page(page: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called before a page is processed."""
    return page


def on_page_markdown(
    markdown: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Process markdown content before it's converted to HTML."""
    return markdown


def on_page_content(
    html: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Called after markdown is converted to HTML."""
    return html


def on_post_page(output: str, page: Any, config: Any, **kwargs: Any) -> str:
    """Called after page is fully processed."""
    return output
