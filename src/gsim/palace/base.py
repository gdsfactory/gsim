"""Base mixin for Palace simulation classes.

Provides common methods shared across all simulation types:
DrivenSim, EigenmodeSim, ElectrostaticSim.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.common import LayerStack
    from gsim.palace.models import (
        MeshConfig,
        SimulationResult,
        ValidationResult,
    )

logger = logging.getLogger(__name__)


class PalaceSimMixin:
    """Mixin providing common methods for all Palace simulation classes.

    Subclasses must define these attributes (typically via Pydantic fields):
        - geometry: Geometry | None
        - stack: LayerStack | None
        - materials: dict[str, MaterialConfig]
        - numerical: NumericalConfig
    """

    geometry: Any = None
    stack: Any = None
    materials: dict[str, Any]
    numerical: Any
    _output_dir: Path | None
    _stack_kwargs: dict[str, Any]
    _last_mesh_result: Any
    _last_ports: list
    _last_terminals: list
    _configured_ports: bool
    _configured_terminals: bool
    _job_id: str | None

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    def set_output_dir(self, path: str | Path) -> None:
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    # -------------------------------------------------------------------------
    # Geometry methods
    # -------------------------------------------------------------------------

    def set_geometry(self, component: Component) -> None:
        from gsim.common import Geometry

        self.geometry = Geometry(component=component)

    @property
    def component(self) -> Component | None:
        return self.geometry.component if self.geometry else None

    @property
    def _component(self) -> Component | None:
        return self.component

    # -------------------------------------------------------------------------
    # Stack methods
    # -------------------------------------------------------------------------

    def set_stack(
        self,
        stack: LayerStack | None = None,
        *,
        yaml_path: str | Path | None = None,
        air_above: float = 200.0,
        substrate_thickness: float = 2.0,
        include_substrate: bool = False,
        **kwargs,
    ) -> None:
        if stack is not None:
            self.stack = stack
            self._stack_kwargs = {"_prebuilt": True}
            return

        self._stack_kwargs = {
            "yaml_path": yaml_path,
            "air_above": air_above,
            "substrate_thickness": substrate_thickness,
            "include_substrate": include_substrate,
            **kwargs,
        }
        self.stack = None

    # -------------------------------------------------------------------------
    # Material methods
    # -------------------------------------------------------------------------

    def set_material(
        self,
        name: str,
        *,
        material_type: Literal["conductor", "dielectric", "semiconductor"]
        | None = None,
        conductivity: float | None = None,
        permittivity: float | None = None,
        loss_tangent: float | None = None,
    ) -> None:
        from gsim.palace.models import MaterialConfig

        resolved_type = material_type
        if resolved_type is None:
            if conductivity is not None and conductivity > 1e4:
                resolved_type = "conductor"
            elif permittivity is not None:
                resolved_type = "dielectric"
            else:
                resolved_type = "dielectric"

        self.materials[name] = MaterialConfig(
            type=resolved_type,
            conductivity=conductivity,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
        )

    def set_numerical(
        self,
        *,
        order: int = 2,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = "Default",
        preconditioner: Literal["Default", "AMS", "BoomerAMG"] = "Default",
        device: Literal["CPU", "GPU"] = "CPU",
        num_processors: int | None = None,
    ) -> None:
        from gsim.palace.models import NumericalConfig

        self.numerical = NumericalConfig(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            num_processors=num_processors,
        )

    # -------------------------------------------------------------------------
    # Port / Terminal methods
    # -------------------------------------------------------------------------

    def add_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        from_layer: str | None = None,
        to_layer: str | None = None,
        length: float | None = None,
        impedance: float = 50.0,
        resistance: float | None = None,
        inductance: float | None = None,
        capacitance: float | None = None,
        excited: bool = True,
        geometry: Literal["inplane", "via"] = "inplane",
    ) -> None:
        from gsim.palace.models import PortConfig

        if not hasattr(self, "ports"):
            self.ports = []
        self.ports = [p for p in self.ports if p.name != name]
        self.ports.append(
            PortConfig(
                name=name,
                layer=layer,
                from_layer=from_layer,
                to_layer=to_layer,
                length=length,
                impedance=impedance,
                resistance=resistance,
                inductance=inductance,
                capacitance=capacitance,
                excited=excited,
                geometry=geometry,
            )
        )

    def add_cpw_port(
        self,
        name: str,
        *,
        layer: str,
        s_width: float,
        gap_width: float,
        length: float,
        impedance: float = 50.0,
        excited: bool = True,
    ) -> None:
        from gsim.palace.models import CPWPortConfig

        if not hasattr(self, "cpw_ports"):
            self.cpw_ports = []
        self.cpw_ports = [p for p in self.cpw_ports if p.name != name]
        self.cpw_ports.append(
            CPWPortConfig(
                name=name,
                layer=layer,
                s_width=s_width,
                gap_width=gap_width,
                length=length,
                impedance=impedance,
                excited=excited,
            )
        )

    def add_terminal(
        self,
        name: str,
        *,
        layer: str,
    ) -> None:
        from gsim.palace.models import TerminalConfig

        if not hasattr(self, "terminals"):
            self.terminals = []
        self.terminals = [t for t in self.terminals if t.name != name]
        self.terminals.append(
            TerminalConfig(
                name=name,
                layer=layer,
            )
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_stack(self) -> LayerStack:
        if self.stack is not None and self._stack_kwargs.get("_prebuilt"):
            for name, props in self.materials.items():
                self.stack.materials[name] = props.to_dict()
            return self.stack

        from gsim.common.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        legacy_stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)
        self._stack_kwargs["yaml_path"] = yaml_path

        for name, props in self.materials.items():
            legacy_stack.materials[name] = props.to_dict()

        self.stack = legacy_stack
        return legacy_stack

    def _build_mesh_config(
        self,
        preset: Literal["coarse", "default", "fine"] | None,
        refined_mesh_size: float | None,
        max_mesh_size: float | None,
        margin: float | None,
        air_above: float | None,
        fmax: float | None,
        planar_conductors: bool | None,
        show_gui: bool,
    ) -> MeshConfig:
        from gsim.palace.models import MeshConfig

        if preset == "coarse":
            mesh_config = MeshConfig.coarse()
        elif preset == "fine":
            mesh_config = MeshConfig.fine()
        else:
            mesh_config = MeshConfig.default()

        if planar_conductors is None:
            existing_config = getattr(self, "mesh_config", None)
            if existing_config is not None:
                mesh_config.planar_conductors = existing_config.planar_conductors
        else:
            mesh_config.planar_conductors = planar_conductors

        overrides = []
        if preset is not None:
            if refined_mesh_size is not None:
                overrides.append(f"refined_mesh_size={refined_mesh_size}")
            if max_mesh_size is not None:
                overrides.append(f"max_mesh_size={max_mesh_size}")
            if margin is not None:
                overrides.append(f"margin={margin}")
            if air_above is not None:
                overrides.append(f"air_above={air_above}")
            if fmax is not None:
                overrides.append(f"fmax={fmax}")
            if planar_conductors is not None:
                overrides.append(f"planar_conductors={planar_conductors}")

            if overrides:
                warnings.warn(
                    f"Preset '{preset}' values overridden by: {', '.join(overrides)}",
                    stacklevel=4,
                )

        if refined_mesh_size is not None:
            mesh_config.refined_mesh_size = refined_mesh_size
        if max_mesh_size is not None:
            mesh_config.max_mesh_size = max_mesh_size
        if margin is not None:
            mesh_config.margin = margin
        if air_above is not None:
            mesh_config.air_above = air_above
        if fmax is not None:
            mesh_config.fmax = fmax
        mesh_config.show_gui = show_gui

        return mesh_config

    def _configure_ports_on_component(self, stack: LayerStack) -> None:
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
        )

        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        ports = getattr(self, "ports", [])
        cpw_ports = getattr(self, "cpw_ports", [])

        for port_config in ports:
            if port_config.name is None:
                continue

            gf_port = None
            for p in component.ports:
                if p.name == port_config.name:
                    gf_port = p
                    break

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available ports: {[p.name for p in component.ports]}"
                )

            if port_config.geometry == "inplane" and port_config.layer is not None:
                configure_inplane_port(
                    gf_port,
                    layer=port_config.layer,
                    length=port_config.length or gf_port.width,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                )
            elif port_config.geometry == "via" and (
                port_config.from_layer is not None and port_config.to_layer is not None
            ):
                configure_via_port(
                    gf_port,
                    from_layer=port_config.from_layer,
                    to_layer=port_config.to_layer,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                )

            if port_config.resistance is not None:
                gf_port.info["resistance"] = port_config.resistance
            if port_config.inductance is not None:
                gf_port.info["inductance"] = port_config.inductance
            if port_config.capacitance is not None:
                gf_port.info["capacitance"] = port_config.capacitance

        for cpw_config in cpw_ports:
            gf_port = None
            for p in component.ports:
                if p.name == cpw_config.name:
                    gf_port = p
                    break
            if gf_port is None:
                raise ValueError(
                    f"CPW port '{cpw_config.name}' not found on component. "
                    f"Available: {[p.name for p in component.ports]}"
                )

            configure_cpw_port(
                gf_port,
                layer=cpw_config.layer,
                s_width=cpw_config.s_width,
                gap_width=cpw_config.gap_width,
                length=cpw_config.length,
                impedance=cpw_config.impedance,
                excited=cpw_config.excited,
            )

        self._configured_ports = True

    def _generate_mesh_internal(
        self,
        output_dir: Path,
        mesh_config: MeshConfig,
        ports: list,
        model_name: str,
        verbose: bool,
        write_config: bool = True,
    ) -> SimulationResult:
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh
        from gsim.palace.models import SimulationResult

        component = self.geometry.component if self.geometry else None

        driven_config = getattr(self, "driven", None)
        eigenmode_config = getattr(self, "eigenmode", None)
        electrostatic_config = getattr(self, "electrostatic", None)
        terminals = getattr(self, "terminals", [])

        effective_fmax = mesh_config.fmax
        if driven_config is not None and mesh_config.fmax == 100e9:
            effective_fmax = driven_config.fmax

        legacy_mesh_config = LegacyMeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=effective_fmax,
            show_gui=mesh_config.show_gui,
            preview_only=mesh_config.preview_only,
            planar_conductors=mesh_config.planar_conductors,
        )

        stack = self._resolve_stack()

        if verbose:
            logger.info("Generating mesh in %s", output_dir)

        mesh_result = generate_mesh(
            component=component,
            stack=stack,
            ports=ports,
            output_dir=output_dir,
            config=legacy_mesh_config,
            model_name=model_name,
            driven_config=driven_config,
            eigenmode_config=eigenmode_config,
            electrostatic_config=electrostatic_config,
            terminals=terminals,
            write_config=write_config,
        )

        self._last_mesh_result = mesh_result
        self._last_ports = ports
        self._last_terminals = terminals

        return SimulationResult(
            mesh_path=mesh_result.mesh_path,
            output_dir=output_dir,
            config_path=mesh_result.config_path,
            port_info=mesh_result.port_info,
            mesh_stats=mesh_result.mesh_stats,
        )

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        from gsim.palace.ports import extract_ports

        component = self.geometry.component if self.geometry else None

        has_ports = bool(getattr(self, "ports", [])) or bool(
            getattr(self, "cpw_ports", [])
        )
        if has_ports:
            self._configure_ports_on_component(stack)
            return extract_ports(component, stack)
        return []

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> ValidationResult:
        from gsim.palace.models import ValidationResult

        errors = []
        warnings_list = []

        if self.geometry is None:
            errors.append("No component set. Call set_geometry(component) first.")

        if self.stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        ports = getattr(self, "ports", [])
        cpw_ports = getattr(self, "cpw_ports", [])
        terminals = getattr(self, "terminals", [])

        if hasattr(self, "driven"):
            has_ports = bool(ports) or bool(cpw_ports)
            if not has_ports:
                warnings_list.append(
                    "No ports configured. Call add_port() or add_cpw_port()."
                )
            if self.driven.excitation_port is not None:
                all_port_names = [p.name for p in ports] + [
                    cpw.name for cpw in cpw_ports
                ]
                if self.driven.excitation_port not in all_port_names:
                    errors.append(
                        f"Excitation port '{self.driven.excitation_port}' not found."
                    )

        if hasattr(self, "eigenmode"):
            has_ports = bool(ports) or bool(cpw_ports)
            if not has_ports:
                warnings_list.append(
                    "No ports configured. Eigenmode finds all modes without port loading."
                )

        if hasattr(self, "electrostatic"):
            if len(terminals) < 2:
                errors.append("Electrostatic simulation requires at least 2 terminals.")

        for port in ports:
            if port.geometry == "inplane" and port.layer is None:
                errors.append(f"Port '{port.name}': inplane ports require 'layer'")
            if port.geometry == "via" and (
                port.from_layer is None or port.to_layer is None
            ):
                errors.append(
                    f"Port '{port.name}': via ports require 'from_layer' and 'to_layer'"
                )

        for cpw in cpw_ports:
            if not cpw.layer:
                errors.append(f"CPW port '{cpw.name}': 'layer' is required")

        for term in terminals:
            if not term.layer:
                errors.append(f"Terminal '{term.name}': 'layer' is required")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Preview & Mesh
    # -------------------------------------------------------------------------

    def preview(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        air_above: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = True,
    ) -> None:
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh

        component = self.geometry.component if self.geometry else None

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
        )

        stack = self._resolve_stack()
        ports = self._get_ports_for_preview(stack)

        legacy_mesh_config = LegacyMeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=mesh_config.fmax,
            show_gui=show_gui,
            preview_only=True,
            planar_conductors=mesh_config.planar_conductors,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=ports,
                output_dir=tmpdir,
                config=legacy_mesh_config,
            )

    def mesh(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        air_above: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = False,
        model_name: str = "palace",
        verbose: bool = True,
    ) -> SimulationResult:
        from gsim.palace.ports import extract_ports

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        component = self.geometry.component if self.geometry else None

        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
        )

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        output_dir = self._output_dir
        stack = self._resolve_stack()

        palace_ports = []
        has_ports = bool(getattr(self, "ports", [])) or bool(
            getattr(self, "cpw_ports", [])
        )
        if has_ports:
            self._configure_ports_on_component(stack)
            palace_ports = extract_ports(component, stack)

        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
            model_name=model_name,
            verbose=verbose,
            write_config=False,
        )

    def write_config(self) -> Path:
        from gsim.palace.mesh.generator import write_config as gen_write_config

        if getattr(self, "_last_mesh_result", None) is None:
            raise ValueError("No mesh result. Call mesh() first.")

        if not self._last_mesh_result.groups:
            raise ValueError(
                "Mesh result has no groups data. "
                "Was mesh() called with write_config=True already?"
            )

        stack = self._resolve_stack()
        config_path = gen_write_config(
            mesh_result=self._last_mesh_result,
            stack=stack,
            ports=getattr(self, "_last_ports", []),
            driven_config=getattr(self, "driven", None),
            eigenmode_config=getattr(self, "eigenmode", None),
            electrostatic_config=getattr(self, "electrostatic", None),
            terminals=getattr(self, "_last_terminals", []),
        )

        return config_path

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        from gsim.common.stack import print_stack_table

        if self.stack is None:
            self._resolve_stack()
        if self.stack is not None:
            print_stack_table(self.stack)

    def plot_stack(self) -> None:
        from gsim.common.stack import plot_stack

        if self.stack is None:
            self._resolve_stack()
        if self.stack is not None:
            plot_stack(self.stack)

    def plot_mesh(
        self,
        output: str | Path | None = None,
        show_groups: list[str] | None = None,
        interactive: bool = True,
    ) -> None:
        from gsim.viz import plot_mesh as _plot_mesh

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_path = self._output_dir / "palace.msh"
        if not mesh_path.exists():
            raise ValueError(f"Mesh file not found: {mesh_path}. Call mesh() first.")

        if output is None and not interactive:
            output = self._output_dir / "mesh.png"

        _plot_mesh(
            msh_path=mesh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
        )

    # -------------------------------------------------------------------------
    # Cloud / Execution
    # -------------------------------------------------------------------------

    def _prepare_upload_dir(self) -> Path:
        import shutil

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        self.write_config()

        tmp = Path(tempfile.mkdtemp(prefix="palace_"))
        for item in self._output_dir.iterdir():
            dest = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return tmp

    def upload(self, *, verbose: bool = True) -> str:
        from gsim import gcloud

        tmp = self._prepare_upload_dir()
        try:
            self._job_id = gcloud.upload(tmp, "palace", verbose=verbose)
        except Exception:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self._job_id

    def start(self, *, verbose: bool = True) -> None:
        from gsim import gcloud

        if getattr(self, "_job_id", None) is None:
            raise ValueError("Call upload() first")
        gcloud.start(self._job_id, verbose=verbose)

    def get_status(self) -> str:
        from gsim import gcloud

        if getattr(self, "_job_id", None) is None:
            raise ValueError("No job submitted yet")
        return gcloud.get_status(self._job_id)

    def wait_for_results(
        self,
        *,
        verbose: bool = True,
        parent_dir: str | Path | None = None,
    ) -> Any:
        from gsim import gcloud

        if getattr(self, "_job_id", None) is None:
            raise ValueError("No job submitted yet")
        return gcloud.wait_for_results(
            self._job_id, verbose=verbose, parent_dir=parent_dir
        )

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: bool = True,
        wait: bool = True,
    ) -> dict[str, Path] | str:
        self.upload(verbose=False)
        self.start(verbose=verbose)
        if not wait:
            return self._job_id
        return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)

    def run_local(
        self,
        *,
        palace_sif_path: str | Path | None = None,
        num_processes: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Path]:
        import os
        import subprocess

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        output_dir = Path(self._output_dir)
        config_path = output_dir / "config.json"
        mesh_path = output_dir / "palace.msh"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. Call write_config() first."
            )

        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {mesh_path}. Call mesh() first."
            )

        if palace_sif_path is None:
            palace_sif_path = os.environ.get("PALACE_SIF")
            if palace_sif_path is None:
                raise ValueError(
                    "Palace SIF path not specified. Either set PALACE_SIF "
                    "environment variable or pass palace_sif_path parameter."
                )
            if verbose:
                logger.info("Using PALACE_SIF from environment: %s", palace_sif_path)

        sif_path = Path(palace_sif_path).expanduser().resolve()

        if not sif_path.exists():
            raise FileNotFoundError(
                f"Palace SIF file not found: {sif_path}. "
                "Install Palace via Apptainer or provide correct path."
            )

        if num_processes is None:
            try:
                import psutil

                num_processes = psutil.cpu_count(logical=True) or 1
            except ImportError:
                num_processes = os.cpu_count() or 1

        cmd = [
            "apptainer",
            "run",
            str(sif_path),
            "-nt",
            str(num_processes),
            "config.json",
        ]

        if verbose:
            logger.info("Running Palace simulation in %s", output_dir)
            logger.info("Command: %s", " ".join(cmd))
            logger.info("Processes: %d", num_processes)

        try:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            if verbose and result.stdout:
                logger.info(result.stdout)
            if verbose and result.stderr:
                logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            error_msg = f"Palace simulation failed with return code {e.returncode}"
            if e.stdout:
                error_msg += f"\n\nStdout:\n{e.stdout}"
            if e.stderr:
                error_msg += f"\n\nStderr:\n{e.stderr}"
            raise RuntimeError(error_msg) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "Apptainer not found. Install Apptainer to run local simulations."
            ) from e

        if verbose:
            logger.info("Simulation completed successfully")

        postpro_dir = output_dir / "output/palace/"

        if verbose:
            logger.info("Results saved to %s", postpro_dir)

        return {
            file.name: file
            for file in postpro_dir.iterdir()
            if file.is_file() and not file.name.startswith(".")
        }
