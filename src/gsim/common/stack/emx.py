"""Import EMX process files (``.proc``) as a gsim :class:`LayerStack`.

Some foundries deliver their RF back-end stack only as an EMX ``.proc`` file.
This module reads the nominal stack out of such a file so it can be used
directly::

    from gsim.common.stack.emx import load_emx_proc

    stack = load_emx_proc("my_process.proc")
    sim.set_stack(stack)

What is imported
----------------

* ``define`` lines with plain arithmetic, and the first (reference) entry of
  table-valued defines.
* The vertical stack, read from the bottom to the top: finite dielectric
  layers, one lossy substrate layer (the layer that carries a resistivity in
  ohm-cm), and embedded conductors with optional ``offset`` lines.
* Conductors given as a sheet resistance (converted to a conductivity with
  the layer thickness) or directly in S/m.
* The GDS layer map (``define NAME = l<layer>t<datatype>``).
* Vias (``via FROM TO { ... S/m ... } NAME``) with their conductivity and the
  two conductors they connect.

What is ignored (a warning is emitted for each feature found in the file):
geometry bias, fill and slotting, via merge operations, and temperature
dependence.  Only nominal values are used.

Conventions of the resulting stack
----------------------------------

* Length unit is um.  ``z = 0`` is the top of the substrate layer, so the
  substrate occupies ``[-thickness, 0]``.  Without a substrate layer, ``z = 0``
  is the bottom of the first layer.
* Every finite dielectric layer (and the substrate) becomes a full-domain
  region in ``LayerStack.dielectrics``.  The semi-infinite layer on top is
  dropped: use ``set_airbox()`` of the simulation to add the air around the
  chip.
* Conductors and vias become ``Layer`` objects (``layer_type`` ``"conductor"``
  and ``"via"``).  A via spans the gap between the top of the lower conductor
  and the bottom of the upper conductor it connects.
* Material names carry the prefix ``emx_`` so that they never collide with the
  built-in material database of gsim, which would silently replace the
  imported values.
* ``LayerStack.simulation["emx"]`` holds metadata: the via connectivity, all
  GDS streams, the ignored features and the warnings.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from gsim.common.stack.extractor import Layer, LayerStack

PORTABLE_SCHEMA = "portable-em-stackup-v1"
"""Schema tag of the portable stackup JSON handled by this module."""

MATERIAL_PREFIX = "emx_"
"""Prefix of all material names created by the importer."""

PathLike = str | Path

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LAYER_TOKEN = re.compile(r"\bl(\d+)t(\d+)\b", re.IGNORECASE)

# Features that exist in EMX files but are not part of the nominal import.
_IGNORED_FEATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("geometry bias", re.compile(r"\bbias\w*", re.IGNORECASE)),
    ("fill and slotting", re.compile(r"\bfill\b|\bslot\w*", re.IGNORECASE)),
    ("via merge operations", re.compile(r"\bmerge\w*", re.IGNORECASE)),
    (
        "temperature dependence",
        re.compile(r"\bdtemp\b|\btemp_reference\b", re.IGNORECASE),
    ),
)

_IGNORED_POLICY = [
    "geometry bias",
    "fill and slotting",
    "via merge operations",
    "temperature dependence",
]


class EmxImportWarning(UserWarning):
    """Warning about a feature of an EMX file that was ignored or simplified."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _decimal(value: Any, field: str) -> Decimal:
    """Convert *value* to a Decimal, naming *field* in the error message."""
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"Invalid number for {field}: {value!r}") from exc


def _number(value: Decimal) -> int | float:
    """Return an int for integral values, else a float."""
    return int(value) if value == value.to_integral_value() else float(value)


def _material_suffix(epsilon: Decimal) -> str:
    """Return a short name fragment for a permittivity (``4.20`` -> ``4p2``)."""
    text = format(epsilon, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def _safe_arithmetic(expression: str, variables: dict[str, Decimal]) -> Decimal:
    """Evaluate ``+ - * /`` expressions over numbers and known scalar defines."""
    tree = ast.parse(expression.strip(), mode="eval")

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Decimal(str(node.value))
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError(f"Unsupported nominal expression: {expression!r}")

    return evaluate(tree)


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment and surrounding whitespace."""
    return line.split("#", 1)[0].strip()


def _collect_via_statements(lines: list[str]) -> list[str]:
    """Join (possibly multi-line) ``via ... { ... } NAME`` declarations."""
    statements: list[str] = []
    collecting: list[str] = []
    for raw in lines:
        line = _strip_comment(raw)
        if collecting:
            collecting.append(line)
            if "}" in line:
                statements.append(" ".join(collecting))
                collecting = []
        elif re.match(r"^via\b", line, re.IGNORECASE):
            if "}" in line:
                statements.append(line)
            else:
                collecting = [line]
    if collecting:
        raise ValueError("Unterminated multiline via declaration")
    return statements


@dataclass(frozen=True)
class _ParsedConductor:
    """A conductor declaration after name, material and thickness are resolved."""

    name: str
    material: str
    thickness_um: Decimal


# ---------------------------------------------------------------------------
# .proc -> portable document
# ---------------------------------------------------------------------------


class _EmxProcParser:
    """Parser that turns the text of a ``.proc`` file into a portable document."""

    def __init__(
        self,
        source: PathLike,
        *,
        substrate_thickness_um: Decimal | None = None,
    ) -> None:
        """Read the process file and keep its lines for parsing."""
        self.source = Path(source).expanduser().resolve()
        text = self.source.read_text(encoding="utf-8", errors="replace")
        self.raw_lines = text.splitlines()
        self.lines = [_strip_comment(line) for line in self.raw_lines]
        self.clean_text = "\n".join(self.lines)
        self.header_text = text
        self.substrate_override = substrate_thickness_um
        self.scalar_defines: dict[str, Decimal] = {"dtemp": Decimal(0)}
        self.define_expressions: dict[str, str] = {}
        self.reference_values: dict[str, Decimal] = {}
        self.materials: dict[str, dict[str, Any]] = {}
        self.layer_stack: list[dict[str, Any]] = []
        self.conductors: dict[str, _ParsedConductor] = {}
        self.warnings: list[str] = []
        self.detected_features: list[str] = []
        self._override_used = False

    # -- top level ---------------------------------------------------------

    def parse(self) -> dict[str, Any]:
        """Parse the file and return the portable stackup document."""
        self._parse_defines()
        self._parse_stack()
        if not any(
            entry["type"] in {"dielectric", "semiconductor", "conductor"}
            for entry in self.layer_stack
        ):
            raise ValueError(
                f"No layer or conductor declarations found in {self.source.name}; "
                "is this an EMX .proc file?"
            )
        if self.substrate_override is not None and not self._override_used:
            self.warnings.append(
                "substrate_thickness_um was given but the file has no substrate "
                "layer (a layer with a resistivity in ohm-cm); it is ignored."
            )
        gds_map = self._parse_gds_map()
        vias = self._parse_vias(gds_map)
        self._detect_ignored_features()
        self._check_stack_direction()

        temperature = re.search(r"temp_reference\s+([\d.]+)", self.clean_text)
        background = re.search(
            r"background_dielectric_constant\s+([\d.]+)", self.clean_text
        )
        process = re.search(
            r"^#\s*process\s+([^\s{]+)\s*\{", self.header_text, re.MULTILINE
        )
        document: dict[str, Any] = {
            "schema": PORTABLE_SCHEMA,
            "name": process.group(1) if process else self.source.stem,
            "source": self.source.name,
            "generator": {
                "script": "gsim.common.stack.emx",
                "policy": "nominal geometry/material extraction",
                "ignored": list(_IGNORED_POLICY),
                "detected_in_source": list(self.detected_features),
                "sheet_resistance_policy": (
                    "Use the first/reference entry of "
                    "width/spacing-dependent sheet-resistance tables."
                ),
                "warnings": self.warnings,
            },
            "units": {
                "length": "um",
                "conductivity": "S/m",
                "sheet_resistance": "ohm/sq",
                "substrate_resistivity": "ohm-cm",
                "temperature": "degC",
            },
            "reference_temperature_C": (
                float(temperature.group(1)) if temperature else 25.0
            ),
            "background_relative_permittivity": (
                float(background.group(1)) if background else 1.0
            ),
            "stack_direction": "bottom_to_top",
            "vertical_semantics": {
                "dielectric_layers": (
                    "Each finite dielectric/semiconductor layer "
                    "advances the vertical interface by its thickness."
                ),
                "conductors": (
                    "An embedded conductor does not advance the "
                    "dielectric interface and starts at the preceding "
                    "finite layer bottom unless offset."
                ),
                "offsets": (
                    "An offset positions the following conductor "
                    "from the preceding finite layer bottom."
                ),
            },
            "gds_import_defaults": {
                "include_optional_fill": False,
                "apply_geometry_bias": False,
                "merge_overlapping_vias": False,
            },
            "materials": self.materials,
            "layer_stack": self.layer_stack,
            "gds_layer_map": gds_map,
            "vias": vias,
        }
        if self.substrate_override is not None and self._override_used:
            document["substrate_override"] = {
                "requested_thickness_um": _number(self.substrate_override),
                "coordinate_policy": (
                    "The substrate top stays at z=0; the backside moves."
                ),
            }
        return document

    # -- defines -----------------------------------------------------------

    def _parse_defines(self) -> None:
        """Collect the scalar ``define NAME = value`` statements."""
        for line in self.lines:
            match = re.match(r"^define\s+(\w+)\s*=\s*(.+)$", line, re.IGNORECASE)
            if not match:
                continue
            name, expression = match.group(1), match.group(2).strip()
            self.define_expressions[name.lower()] = expression
            if "table" in expression.lower():
                first_value = re.search(rf"=>\s*({_NUMBER})", expression)
                if first_value:
                    self.reference_values[name.lower()] = Decimal(first_value.group(1))
                continue
            try:
                value = _safe_arithmetic(expression, self.scalar_defines)
            except (SyntaxError, ValueError, ArithmeticError):
                continue
            self.scalar_defines[name] = value
            self.scalar_defines[name.lower()] = value
            self.reference_values[name.lower()] = value

    # -- vertical stack ----------------------------------------------------

    def _parse_stack(self) -> None:
        """Read the layer, conductor and offset statements from bottom to top."""
        pending_offset: Decimal | None = None
        for line in self.lines:
            if not line:
                continue
            offset_match = re.match(r"^offset\s+(.+)$", line, re.IGNORECASE)
            if offset_match:
                pending_offset = _safe_arithmetic(
                    offset_match.group(1), self.scalar_defines
                )
                continue
            if re.match(r"^layer\b", line, re.IGNORECASE):
                self._parse_layer(line)
                continue
            if re.match(r"^conductor\b", line, re.IGNORECASE):
                conductor = self._parse_conductor(line)
                if pending_offset is not None:
                    self.layer_stack.append(
                        {
                            "name": f"{conductor.name}_offset",
                            "type": "positioning_offset",
                            "offset_um": _number(pending_offset),
                            "applies_to": conductor.name,
                        }
                    )
                    pending_offset = None
                self.layer_stack.append(
                    {
                        "name": conductor.name,
                        "type": "conductor",
                        "thickness_um": _number(conductor.thickness_um),
                        "material": conductor.material,
                        "gds_map": conductor.name,
                    }
                )
                self.conductors[conductor.name] = conductor
        if pending_offset is not None:
            self.warnings.append(
                "An 'offset' line is not followed by a conductor and is ignored."
            )

    def _parse_layer(self, line: str) -> None:
        """Parse one dielectric or substrate ``layer`` statement."""
        match = re.match(r"^layer\s+(\S+)\s+(\S+)\s*(.*)$", line, re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot parse layer declaration: {line}")
        thickness_token, epsilon_token, remainder = match.groups()
        epsilon = _safe_arithmetic(epsilon_token, self.scalar_defines)
        name_match = re.search(r"\bname\s+(\w+)", remainder, re.IGNORECASE)
        is_substrate = "ohm-cm" in remainder.lower()
        is_infinite = thickness_token.lower() == "infinity"

        if is_infinite and not is_substrate:
            name = name_match.group(1) if name_match else "air"
            if epsilon != 1:
                self.warnings.append(
                    f"Semi-infinite layer '{name}' has relative permittivity "
                    f"{epsilon}; it is replaced by the gsim airbox (permittivity 1)."
                )
            self.materials.setdefault(
                "air",
                {
                    "type": "dielectric",
                    "relative_permittivity": float(epsilon),
                    "loss_tangent": None,
                },
            )
            self.layer_stack.append(
                {
                    "name": name,
                    "type": "dielectric",
                    "thickness_um": None,
                    "material": "air",
                    "semi_infinite": True,
                }
            )
            return

        if is_infinite:
            # A semi-infinite substrate cannot be meshed: a finite thickness
            # must be chosen by the caller.
            if self.substrate_override is None:
                raise ValueError(
                    "The substrate layer has infinite thickness; pass "
                    "substrate_thickness_um to choose a finite value."
                )
            declared_thickness = self.substrate_override
        else:
            declared_thickness = _safe_arithmetic(thickness_token, self.scalar_defines)
        if declared_thickness < 0:
            raise ValueError(f"Negative layer thickness in: {line}")

        if is_substrate:
            if self.substrate_override is not None:
                self._override_used = True
            thickness = (
                self.substrate_override
                if self.substrate_override is not None
                else declared_thickness
            )
            name = name_match.group(1) if name_match else "substrate"
            self._add_substrate(name, thickness, declared_thickness, epsilon, remainder)
            return

        name = (
            name_match.group(1) if name_match else f"dielectric_{len(self.layer_stack)}"
        )
        material = (
            "air"
            if epsilon == 1 and "air" in name.lower()
            else f"dielectric_er_{_material_suffix(epsilon)}"
        )
        self.materials.setdefault(
            material,
            {
                "type": "dielectric",
                "relative_permittivity": float(epsilon),
                "loss_tangent": None,
            },
        )
        self.layer_stack.append(
            {
                "name": name,
                "type": "dielectric",
                "thickness_um": _number(declared_thickness),
                "material": material,
            }
        )

    def _add_substrate(
        self,
        name: str,
        thickness: Decimal,
        declared_thickness: Decimal,
        epsilon: Decimal,
        remainder: str,
    ) -> None:
        """Record the substrate layer and its conductivity from the resistivity."""
        # The layer name must not take part in the resistivity product.
        cleaned = re.sub(r"\bname\s+\w+", " ", remainder, flags=re.IGNORECASE)
        before_unit = cleaned.lower().split("ohm-cm", 1)[0].split()
        resistivity: Decimal | None = None
        for token in before_unit:
            try:
                value = _safe_arithmetic(token, self.scalar_defines)
            except (SyntaxError, ValueError, ArithmeticError):
                continue
            resistivity = value if resistivity is None else resistivity * value
        if resistivity is None or resistivity <= 0:
            raise ValueError(
                f"Substrate layer '{name}' needs a positive resistivity in ohm-cm."
            )
        conductivity = Decimal(100) / resistivity
        self.materials["substrate"] = {
            "type": "semiconductor",
            "relative_permittivity": float(epsilon),
            "resistivity_ohm_cm": _number(resistivity),
            "conductivity_S_per_m": float(conductivity),
            "loss_tangent": None,
        }
        entry: dict[str, Any] = {
            "name": name,
            "type": "semiconductor",
            "thickness_um": _number(thickness),
            "material": "substrate",
        }
        if thickness != declared_thickness:
            entry["source_thickness_um"] = _number(declared_thickness)
            entry["thickness_override"] = True
        self.layer_stack.append(entry)

    def _parse_conductor(self, line: str) -> _ParsedConductor:
        """Parse one ``conductor`` statement."""
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"Cannot parse conductor declaration: {line}")
        thickness = _safe_arithmetic(tokens[1], self.scalar_defines)
        if thickness < 0:
            raise ValueError(f"Negative conductor thickness in: {line}")
        name = tokens[-1].upper()
        if name in self.conductors:
            raise ValueError(f"Duplicate conductor name: {name}")
        property_text = " ".join(tokens[2:-1])
        material = f"{name}_metal"
        material_data: dict[str, Any] = {"type": "conductor"}
        si_match = re.search(r"(.+?)\s+S/m\b", property_text, re.IGNORECASE)
        if si_match:
            expression = si_match.group(1)
            nominal_expression = expression.split("*", 1)[0].strip()
            conductivity = _safe_arithmetic(nominal_expression, self.scalar_defines)
            if "*" in expression:
                self.warnings.append(
                    f"Conductor {name}: only the nominal part '{nominal_expression}' "
                    f"of the conductivity expression '{expression.strip()}' is used."
                )
            material_data["conductivity_S_per_m"] = float(conductivity)
            material_data["source_model"] = property_text
        else:
            identifier = re.match(r"([A-Za-z_]\w*)", property_text)
            if identifier and identifier.group(1).lower() in self.reference_values:
                key = identifier.group(1).lower()
                sheet_resistance = self.reference_values[key]
                policy = f"reference value of {identifier.group(1)}"
                if "table" in self.define_expressions.get(key, "").lower():
                    self.warnings.append(
                        f"Conductor {name}: sheet resistance '{identifier.group(1)}' "
                        "is a table; its first (reference) entry is used."
                    )
            else:
                numeric = re.match(_NUMBER, property_text)
                if not numeric:
                    raise ValueError(f"Cannot resolve sheet resistance: {line}")
                sheet_resistance = Decimal(numeric.group(0))
                policy = "constant nominal sheet resistance at dtemp=0"
            if sheet_resistance <= 0:
                raise ValueError(f"Sheet resistance must be positive: {line}")
            if thickness == 0:
                raise ValueError(
                    f"Conductor {name} has a sheet resistance but zero thickness; "
                    "it cannot be converted to a conductivity."
                )
            conductivity = Decimal(1) / (sheet_resistance * thickness * Decimal("1e-6"))
            material_data.update(
                {
                    "conductivity_S_per_m": float(conductivity),
                    "sheet_resistance_ohm_per_sq": float(sheet_resistance),
                    "reference_thickness_um": _number(thickness),
                    "source_model": property_text,
                    "nominalization": policy,
                }
            )
        self.materials[material] = material_data
        return _ParsedConductor(name, material, thickness)

    # -- GDS map -----------------------------------------------------------

    def _via_declarations(self) -> list[tuple[str, str, str, str, str]]:
        """Return ``(from, to, body, raw_name, suffix)`` for every via."""
        result = []
        for statement in _collect_via_statements(self.raw_lines):
            match = re.match(
                r"^via\s+(\w+)\s+(\w+)\s*\{(.*?)\}\s*(\w+)(.*)$",
                statement,
                re.IGNORECASE,
            )
            if not match:
                raise ValueError(f"Cannot parse via declaration: {statement}")
            first, second, body, raw_name, suffix = match.groups()
            result.append(
                (first.upper(), second.upper(), body, raw_name.upper(), suffix)
            )
        return result

    @staticmethod
    def _unique_via_names(
        declarations: list[tuple[str, str, str, str, str]],
    ) -> list[str]:
        """Give vias that reuse a name a unique ``NAME_FROM_TO`` name."""
        seen: set[str] = set()
        names = []
        for first, second, _body, raw_name, _suffix in declarations:
            name = raw_name if raw_name not in seen else f"{raw_name}_{first}_{second}"
            seen.add(raw_name)
            seen.add(name)
            names.append(name)
        return names

    def _parse_gds_map(self) -> dict[str, dict[str, Any]]:
        """Map conductor and via names to their GDS layer streams."""
        declarations = self._via_declarations()
        relevant = set(self.conductors)
        names = self._unique_via_names(declarations)
        for name, decl in zip(names, declarations, strict=True):
            relevant.add(name)
            relevant.add(decl[3])
        # Keep every directly defined GDS stream, even if no conductor or via
        # refers to it.
        relevant.update(
            name.upper()
            for name, expression in self.define_expressions.items()
            if _LAYER_TOKEN.search(expression)
        )
        output: dict[str, dict[str, Any]] = {}

        def resolve(name: str, trail: tuple[str, ...] = ()) -> dict[str, Any] | None:
            key = name.lower()
            if key in trail:
                return None
            expression = self.define_expressions.get(key)
            if expression is None:
                return None
            optional = {
                (int(layer), int(datatype))
                for layer, datatype in re.findall(
                    r"fill\(if\(includefill,\s*l(\d+)t(\d+)", expression, re.IGNORECASE
                )
            }
            all_streams = [
                (int(a), int(b)) for a, b in _LAYER_TOKEN.findall(expression)
            ]
            include = [stream for stream in all_streams if stream not in optional]
            if include:
                result: dict[str, Any] = {
                    "include": [list(item) for item in dict.fromkeys(include)]
                }
                if optional:
                    result["optional_fill"] = [list(item) for item in sorted(optional)]
                return result
            alias = re.fullmatch(r"\s*(\w+)\s*", expression)
            if alias:
                return resolve(alias.group(1), (*trail, key))
            if any(operator in expression for operator in ("*", "-")):
                return {"derived_from": expression.upper()}
            return None

        for name in sorted(relevant):
            mapping = resolve(name)
            if mapping is not None:
                output[name] = mapping
        return output

    # -- vias --------------------------------------------------------------

    def _parse_vias(self, gds_map: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the via entries and the conductors each via connects."""
        vias: list[dict[str, Any]] = []
        declarations = self._via_declarations()
        names = self._unique_via_names(declarations)
        for name, decl in zip(names, declarations, strict=True):
            first, second, body, raw_name, suffix = decl
            conductivity_match = re.search(rf"({_NUMBER})\s*S/m", body, re.IGNORECASE)
            if not conductivity_match:
                raise ValueError(f"Via {name} has no nominal S/m value")
            conductivity = Decimal(conductivity_match.group(1))
            material = f"via_{name}"
            self.materials[material] = {
                "type": "conductor",
                "conductivity_S_per_m": float(conductivity),
                "source_model": body.strip(),
            }
            # A via that reuses a name may have its own define; otherwise it
            # shares the stream of the plain name.
            map_name = next((c for c in (name, raw_name) if c in gds_map), None)
            vias.append(
                {
                    "name": name,
                    "from": first,
                    "to": second,
                    "gds_map": map_name,
                    "material": material,
                    "source_rule": f"{body}{suffix}".strip(),
                }
            )
        return vias

    # -- diagnostics -------------------------------------------------------

    def _detect_ignored_features(self) -> None:
        """Add one warning for each ignored feature found in the file."""
        for label, pattern in _IGNORED_FEATURES:
            for number, line in enumerate(self.lines, start=1):
                if line and pattern.search(line):
                    self.detected_features.append(label)
                    self.warnings.append(
                        f"The file uses {label} (line {number}); only nominal "
                        "values are imported and this feature is ignored."
                    )
                    break

    def _check_stack_direction(self) -> None:
        """Warn if the substrate is not the first layer of the stack."""
        kinds = [
            entry["type"]
            for entry in self.layer_stack
            if entry["type"] in {"dielectric", "semiconductor"}
            and not entry.get("semi_infinite")
        ]
        if "semiconductor" in kinds and kinds.index("semiconductor") > 0:
            self.warnings.append(
                "The substrate layer is not the first layer of the file. The "
                "importer reads layers from the bottom to the top; check that "
                "the file is not listed top to bottom."
            )


def parse_emx_proc(
    path: PathLike,
    *,
    substrate_thickness_um: Decimal | str | float | None = None,
) -> dict[str, Any]:
    """Convert an EMX ``.proc`` file into the portable stackup document.

    The document is a plain ``dict`` that can be written with ``json.dump``
    and read back with :func:`load_portable_stackup_json`.

    Args:
        path: Path of the ``.proc`` file.
        substrate_thickness_um: Optional thickness (um) that replaces the
            declared thickness of the substrate layer. The substrate top stays
            at ``z = 0``.

    Returns:
        The portable stackup document (schema ``portable-em-stackup-v1``).

    Raises:
        ValueError: If the file cannot be parsed or the override is not a
            positive number.
    """
    override = (
        None
        if substrate_thickness_um is None
        else _decimal(substrate_thickness_um, "substrate_thickness_um")
    )
    if override is not None and override <= 0:
        raise ValueError("substrate_thickness_um must be positive")
    return _EmxProcParser(path, substrate_thickness_um=override).parse()


# ---------------------------------------------------------------------------
# portable document -> LayerStack
# ---------------------------------------------------------------------------


def _dielectric_material(props: dict[str, Any]) -> dict[str, Any]:
    """Return a gsim material dict for a dielectric or substrate entry."""
    material: dict[str, Any] = {
        "type": props.get("type", "dielectric"),
        "permittivity": float(props["relative_permittivity"]),
    }
    if props.get("loss_tangent") is not None:
        material["loss_tangent"] = float(props["loss_tangent"])
    if props.get("conductivity_S_per_m") is not None:
        material["conductivity"] = float(props["conductivity_S_per_m"])
    if props.get("resistivity_ohm_cm") is not None:
        material["resistivity_ohm_cm"] = float(props["resistivity_ohm_cm"])
    return material


def _conductor_material(props: dict[str, Any]) -> dict[str, Any]:
    """Return a gsim material dict for a conductor or via entry."""
    material: dict[str, Any] = {
        "type": "conductor",
        "conductivity": float(props["conductivity_S_per_m"]),
    }
    if props.get("sheet_resistance_ohm_per_sq") is not None:
        material["sheet_resistance_ohm_per_sq"] = float(
            props["sheet_resistance_ohm_per_sq"]
        )
    return material


def _first_stream(
    gds_map: dict[str, Any], key: str | None, label: str, notes: list[str]
) -> tuple[int, int] | None:
    """Pick the GDS stream of *label* from the layer map, or None."""
    mapping = gds_map.get(key) if key is not None else None
    if not mapping or not mapping.get("include"):
        notes.append(
            f"{label} has no direct GDS stream in the layer map "
            "(missing or derived define); it is left out of the stack."
        )
        return None
    streams = [(int(a), int(b)) for a, b in mapping["include"]]
    if len(streams) > 1:
        notes.append(
            f"{label} is drawn on several GDS streams {streams}; "
            f"only {streams[0]} is used."
        )
    return streams[0]


def _build_layer_stack(document: dict[str, Any]) -> tuple[LayerStack, list[str]]:
    """Build the gsim stack from a portable document.

    Returns the stack and the warnings raised while building it.
    """
    if document.get("schema") != PORTABLE_SCHEMA:
        raise ValueError(
            f"Unsupported stackup schema {document.get('schema')!r}; "
            f"expected {PORTABLE_SCHEMA!r}."
        )
    entries = document["layer_stack"]
    doc_materials = document.get("materials", {})
    gds_map = document.get("gds_layer_map", {})
    notes: list[str] = []

    offsets = {
        entry["applies_to"]: _decimal(entry["offset_um"], "offset_um")
        for entry in entries
        if entry["type"] == "positioning_offset"
    }

    # Pass 1: raw z positions, bottom to top, in Decimal.
    z_top = Decimal(0)
    previous_bottom: Decimal | None = None
    regions: list[dict[str, Any]] = []
    conductors: dict[str, dict[str, Any]] = {}
    semi_infinite: list[dict[str, Any]] = []
    for entry in entries:
        kind = entry["type"]
        if kind == "positioning_offset":
            continue
        if kind in {"dielectric", "semiconductor"}:
            if entry.get("semi_infinite") or entry.get("thickness_um") is None:
                props = doc_materials[entry["material"]]
                semi_infinite.append(
                    {
                        "name": entry["name"],
                        "relative_permittivity": props["relative_permittivity"],
                        "above_z_raw": z_top,
                    }
                )
                continue
            thickness = _decimal(entry["thickness_um"], "thickness_um")
            regions.append(
                {
                    "name": entry["name"],
                    "kind": kind,
                    "material": entry["material"],
                    "zmin": z_top,
                    "zmax": z_top + thickness,
                }
            )
            previous_bottom = z_top
            z_top += thickness
        elif kind == "conductor":
            thickness = _decimal(entry["thickness_um"], "thickness_um")
            base = previous_bottom if previous_bottom is not None else z_top
            zmin = base + offsets.get(entry["name"], Decimal(0))
            conductors[entry["name"]] = {
                "entry": entry,
                "zmin": zmin,
                "zmax": zmin + thickness,
            }
        else:
            raise ValueError(f"Unknown layer_stack entry type {kind!r}")

    # z = 0 is the top of the (first) substrate layer.
    substrate = next((r for r in regions if r["kind"] == "semiconductor"), None)
    shift = -substrate["zmax"] if substrate is not None else Decimal(0)

    stack = LayerStack(pdk_name=str(document.get("name", "emx")))

    # Dielectric regions (including the substrate) -> background regions.
    used_materials: dict[str, dict[str, Any]] = {}
    for region in regions:
        key = f"{MATERIAL_PREFIX}{region['material']}"
        used_materials[key] = _dielectric_material(doc_materials[region["material"]])
        is_substrate = region is substrate
        stack.dielectrics.append(
            {
                "name": "substrate" if is_substrate else region["name"],
                "zmin": float(region["zmin"] + shift),
                "zmax": float(region["zmax"] + shift),
                "material": key,
            }
        )

    # Conductors -> conductor layers.
    for name, item in conductors.items():
        entry = item["entry"]
        stream = _first_stream(
            gds_map, entry.get("gds_map"), f"Conductor {name}", notes
        )
        if stream is None:
            continue
        key = f"{MATERIAL_PREFIX}{entry['material']}"
        used_materials[key] = _conductor_material(doc_materials[entry["material"]])
        stack.layers[name] = Layer(
            name=name,
            gds_layer=stream,
            zmin=float(item["zmin"] + shift),
            zmax=float(item["zmax"] + shift),
            thickness=float(item["zmax"] - item["zmin"]),
            material=key,
            layer_type="conductor",
        )

    # Vias -> via layers between the two conductors they connect.
    connectivity: dict[str, list[str]] = {}
    for via in document.get("vias", []):
        name = via["name"]
        ends = [str(via["from"]).upper(), str(via["to"]).upper()]
        connectivity[name] = ends
        missing = [end for end in ends if end not in conductors]
        if missing:
            notes.append(
                f"Via {name} connects {ends[0]} and {ends[1]}, but {missing} "
                "is not a conductor of the stack; the via is left out."
            )
            continue
        lower, upper = sorted(
            (conductors[end] for end in ends),
            key=lambda c: (c["zmin"], c["zmax"]),
        )
        gap = upper["zmin"] - lower["zmax"]
        if gap <= 0:
            notes.append(
                f"Via {name} connects layers that touch or overlap in z "
                f"(gap {gap} um); the via is left out."
            )
            continue
        stream = _first_stream(gds_map, via.get("gds_map"), f"Via {name}", notes)
        if stream is None:
            continue
        key = f"{MATERIAL_PREFIX}{via['material']}"
        used_materials[key] = _conductor_material(doc_materials[via["material"]])
        stack.layers[name] = Layer(
            name=name,
            gds_layer=stream,
            zmin=float(lower["zmax"] + shift),
            zmax=float(upper["zmin"] + shift),
            thickness=float(gap),
            material=key,
            layer_type="via",
        )

    stack.materials = used_materials

    # gsim matches polygons to stack layers by GDS layer number only.
    by_number: dict[int, list[str]] = {}
    for layer in stack.layers.values():
        by_number.setdefault(layer.gds_layer[0], []).append(layer.name)
    for number, names in sorted(by_number.items()):
        if len(names) > 1:
            notes.append(
                f"Layers {names} share GDS layer number {number}; gsim matches "
                "polygons by layer number only, so all of them see the same shapes."
            )

    if substrate is None:
        notes.append(
            "The file has no substrate layer; z = 0 is the bottom of the first layer."
        )
    notes.extend(
        f"Semi-infinite layer '{item['name']}' (permittivity "
        f"{item['relative_permittivity']}) is dropped; add the surrounding "
        "medium with set_airbox()."
        for item in semi_infinite
        if item["relative_permittivity"] != 1
    )

    parse_warnings = list(document.get("generator", {}).get("warnings", []))
    all_warnings = parse_warnings + [n for n in notes if n not in parse_warnings]
    stack.simulation = {
        "emx": {
            "process": str(document.get("name", "")),
            "source": str(document.get("source", "")),
            "reference_plane": "substrate top at z=0"
            if substrate is not None
            else "bottom of the first layer at z=0",
            "substrate_thickness_um": (
                float(substrate["zmax"] - substrate["zmin"])
                if substrate is not None
                else None
            ),
            "top_of_stack_um": float(z_top + shift),
            "via_connectivity": connectivity,
            "gds_streams": {
                name: [list(map(int, s)) for s in mapping.get("include", [])]
                for name, mapping in gds_map.items()
                if mapping.get("include")
            },
            "ignored": list(document.get("generator", {}).get("ignored", [])),
            "detected_in_source": list(
                document.get("generator", {}).get("detected_in_source", [])
            ),
            "warnings": all_warnings,
        }
    }
    return stack, all_warnings


def _stack_from_document(document: dict[str, Any]) -> LayerStack:
    """Build the stack from *document* and emit one warning per issue."""
    stack, messages = _build_layer_stack(document)
    for message in messages:
        warnings.warn(message, EmxImportWarning, stacklevel=3)
    return stack


def load_emx_proc(
    path: PathLike,
    *,
    substrate_thickness_um: Decimal | str | float | None = None,
) -> LayerStack:
    """Load an EMX ``.proc`` process file as a gsim :class:`LayerStack`.

    Example:
        >>> stack = load_emx_proc("my_process.proc")
        >>> sim.set_stack(stack)

    Args:
        path: Path of the ``.proc`` file.
        substrate_thickness_um: Optional thickness (um) that replaces the
            declared substrate thickness. The substrate top stays at
            ``z = 0``, so only its bottom moves.

    Returns:
        A stack with the substrate and dielectric layers in
        ``LayerStack.dielectrics``, conductor and via ``Layer`` objects, and
        metadata in ``LayerStack.simulation["emx"]``.

    Warns:
        EmxImportWarning: For every feature of the file that is ignored or
            simplified (see the module documentation).

    Raises:
        ValueError: If the file cannot be parsed.
    """
    document = parse_emx_proc(path, substrate_thickness_um=substrate_thickness_um)
    return _stack_from_document(document)


def load_portable_stackup_json(path: PathLike) -> LayerStack:
    """Load a portable stackup JSON (``portable-em-stackup-v1``) as a stack.

    The JSON is the output of :func:`parse_emx_proc` or of the equivalent
    command line converter. It gives the same stack as :func:`load_emx_proc`
    on the original ``.proc`` file.

    Args:
        path: Path of the JSON file.

    Returns:
        The imported layer stack.

    Raises:
        ValueError: If the schema tag is not ``portable-em-stackup-v1``.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return _stack_from_document(document)


__all__ = [
    "MATERIAL_PREFIX",
    "PORTABLE_SCHEMA",
    "EmxImportWarning",
    "load_emx_proc",
    "load_portable_stackup_json",
    "parse_emx_proc",
]
