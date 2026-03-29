"""Build Cubism-friendly PSD import packages from generated layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from psd_tools import PSDImage

JsonDict = dict[str, Any]


class CubismPSDBuilder:
    """Create a PSD and companion manifests for Cubism import."""

    def _canvas_size(self, layers: list[JsonDict]) -> tuple[int, int]:
        width = 1
        height = 1
        for layer in layers:
            bounds = dict(layer.get("bounds", {}))
            width = max(width, int(bounds.get("x", 0)) + int(bounds.get("width", 1)))
            height = max(height, int(bounds.get("y", 0)) + int(bounds.get("height", 1)))
        return width, height

    def _group_name(self, target: str, fallback: str) -> str:
        if "/" in target:
            return target.split("/", 1)[0]
        return fallback or "Ungrouped"

    async def build(
        self,
        layers: list[JsonDict],
        mapping: JsonDict,
        output_dir: str,
        model_name: str,
    ) -> JsonDict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        canvas_size = self._canvas_size(layers)
        psd = PSDImage.new("RGBA", canvas_size, color=0)
        groups: dict[str, Any] = {}

        ordered_layers = sorted(layers, key=lambda item: int(item.get("z_order", 0)))
        mapped_by_name = {
            str(entry.get("layer_name", "")): entry for entry in mapping.get("mapped_layers", [])
        }

        preview = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        exported_layers = []
        for layer in ordered_layers:
            layer_name = str(layer.get("name", "layer"))
            layer_path = Path(str(layer.get("path", "")))
            if not layer_path.exists():
                continue
            mapped = mapped_by_name.get(layer_name, {})
            target = str(mapped.get("target", layer.get("group", "Ungrouped")))
            group_name = self._group_name(target, str(layer.get("group", "Ungrouped")))
            if group_name not in groups:
                groups[group_name] = psd.create_group(name=group_name)
                psd.append(groups[group_name])

            with Image.open(layer_path).convert("RGBA") as image:
                pixel_layer = psd.create_pixel_layer(
                    image,
                    name=layer_name,
                    top=int(dict(layer.get("bounds", {})).get("y", 0)),
                    left=int(dict(layer.get("bounds", {})).get("x", 0)),
                )
                groups[group_name].append(pixel_layer)
                preview.alpha_composite(
                    image,
                    (
                        int(dict(layer.get("bounds", {})).get("x", 0)),
                        int(dict(layer.get("bounds", {})).get("y", 0)),
                    ),
                )

            exported_layers.append(
                {
                    "name": layer_name,
                    "target": target,
                    "group": group_name,
                    "path": str(layer_path),
                    "bounds": layer.get("bounds", {}),
                    "z_order": int(layer.get("z_order", 0)),
                }
            )

        psd_path = output_path / f"{model_name}.psd"
        manifest_path = output_path / f"{model_name}_cubism_import.json"
        mapping_path = output_path / f"{model_name}_template_mapping.json"
        preview_path = output_path / f"{model_name}_preview.png"

        psd.save(psd_path)
        preview.save(preview_path, "PNG")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": model_name,
                    "canvas_size": {"width": canvas_size[0], "height": canvas_size[1]},
                    "layers": exported_layers,
                    "template_id": mapping.get("template_id"),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(mapping, handle, indent=2, ensure_ascii=False)

        return {
            "status": "success" if not mapping.get("missing_required") else "partial",
            "psd_path": str(psd_path),
            "preview_path": str(preview_path),
            "manifest_path": str(manifest_path),
            "mapping_path": str(mapping_path),
            "layers_written": len(exported_layers),
            "template_id": mapping.get("template_id"),
            "missing_required": mapping.get("missing_required", []),
            "coverage": mapping.get("coverage", 0.0),
        }
