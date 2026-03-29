"""Template mapping helpers for Cubism import packages."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


class TemplateMapper:
    """Map generated semantic layers onto a fixed Cubism template."""

    def __init__(self, templates_dir: str | Path | Iterable[str | Path]) -> None:
        if isinstance(templates_dir, (str, Path)):
            self.templates_dirs = [Path(templates_dir)]
        else:
            self.templates_dirs = [Path(path) for path in templates_dir]

    def _candidate_paths(self, template_id: str) -> list[Path]:
        return [directory / f"{template_id}.json" for directory in self.templates_dirs]

    def available_templates(self) -> list[str]:
        names: set[str] = set()
        for directory in self.templates_dirs:
            if not directory.exists():
                continue
            names.update(path.stem for path in directory.glob("*.json"))
        return sorted(names)

    def load_template(self, template_id: str) -> JsonDict:
        for path in self._candidate_paths(template_id):
            if not path.exists():
                continue
            with open(path, encoding="utf-8-sig") as handle:
                return cast(JsonDict, json.load(handle))
        raise FileNotFoundError(f"Unknown template_id: {template_id}")

    def map_layers(self, layers: list[JsonDict], template_id: str) -> JsonDict:
        template = self.load_template(template_id)
        mappings = dict(template.get("mappings", {}))
        by_name = {str(layer.get("name", "")): layer for layer in layers}

        mapped_layers = []
        for layer_name, target in mappings.items():
            layer = by_name.get(layer_name)
            if layer is None:
                continue
            mapped_layers.append(
                {
                    "layer_name": layer_name,
                    "target": target,
                    "group": str(layer.get("group", "Ungrouped")),
                    "path": str(layer.get("path", "")),
                    "bounds": layer.get("bounds", {}),
                    "z_order": int(layer.get("z_order", 0)),
                }
            )

        missing_required = [
            part_name
            for part_name in template.get("required_parts", [])
            if part_name not in by_name
        ]
        coverage = 0.0
        if mappings:
            coverage = round(len(mapped_layers) / len(mappings), 3)

        return {
            "status": "success" if not missing_required else "partial",
            "template_id": template_id,
            "description": template.get("description", ""),
            "required_parts": template.get("required_parts", []),
            "optional_parts": template.get("optional_parts", []),
            "mapped_layers": mapped_layers,
            "missing_required": missing_required,
            "coverage": coverage,
        }
