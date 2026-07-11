"""Validation helpers for Cubism export bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


class CubismExportValidator:
    """Validate exported Cubism runtime bundles."""

    def validate(self, output_dir: str, model_name: str) -> JsonDict:
        output_path = Path(output_dir)
        expected_files = {
            "moc3": output_path / f"{model_name}.moc3",
            "model3": output_path / "model3.json",
            "textures_dir": output_path / "textures",
        }

        missing = [name for name, path in expected_files.items() if not path.exists()]
        warnings: list[str] = []
        errors: list[str] = []
        model3_checks: JsonDict = {
            "artifact_stage": "unknown",
            "direct_viewer_compatible": None,
            "ready_for_cubism_editor": None,
        }

        metadata_path = output_path / "export_metadata.json"
        has_readiness_metadata = metadata_path.exists()
        if has_readiness_metadata:
            try:
                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                model3_checks.update(
                    {
                        "artifact_stage": str(metadata.get("artifact_stage", "unknown")),
                        "direct_viewer_compatible": (
                            metadata.get("direct_viewer_compatible") is True
                        ),
                        "ready_for_cubism_editor": (
                            metadata.get("ready_for_cubism_editor") is True
                        ),
                    }
                )
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                errors.append(f"Invalid export readiness metadata: {exc}")
        else:
            warnings.append(
                "Export readiness metadata is missing; Cubism readiness cannot be confirmed."
            )

        model3_path = expected_files["model3"]
        if model3_path.exists():
            with open(model3_path, encoding="utf-8") as handle:
                model3_data = json.load(handle)
            file_refs = dict(model3_data.get("FileReferences", {}))
            moc_ref = file_refs.get("Moc")
            textures = list(file_refs.get("Textures", []))
            model3_checks.update(
                {
                    "moc_reference": moc_ref,
                    "texture_count": len(textures),
                }
            )
            if moc_ref != f"{model_name}.moc3":
                errors.append("model3.json does not reference the expected moc3 filename.")
            if not textures:
                warnings.append("model3.json does not reference any textures.")
            for texture in textures:
                if not (output_path / texture).exists():
                    errors.append(f"Missing referenced texture: {texture}")

        structure_valid = not missing and not errors
        ready_for_cubism_editor = model3_checks["ready_for_cubism_editor"] is True
        model3_checks["structure_valid"] = structure_valid
        if not structure_valid:
            status = "error"
        elif not has_readiness_metadata or ready_for_cubism_editor:
            status = "success"
        else:
            status = "partial"
            warnings.append(
                "Export bundle structure is valid, but the artifact is not ready for Cubism Editor."
            )
        return {
            "status": status,
            "missing": missing,
            "warnings": warnings,
            "errors": errors,
            "output_dir": str(output_path),
            "model_name": model_name,
            "checks": model3_checks,
        }
