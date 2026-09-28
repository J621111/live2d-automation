"""Validate declared Cubism bundle files without claiming native model compatibility."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

JsonDict = dict[str, Any]


class CubismExportValidator:
    """Check intermediate or editor-named exports and their local resource references."""

    def validate(self, output_dir: str, model_name: str) -> JsonDict:
        """Report structure separately from readiness supplied by the exporting pipeline."""
        output_path = Path(output_dir)
        native_model3_path = output_path / f"{model_name}.model3.json"
        model3_path = (
            native_model3_path if native_model3_path.is_file() else output_path / "model3.json"
        )
        expected_files = {
            "moc3": output_path / f"{model_name}.moc3",
            "model3": model3_path,
        }
        missing = [name for name, path in expected_files.items() if not path.is_file()]
        warnings: list[str] = []
        errors: list[str] = []
        checks: JsonDict = {
            "artifact_stage": "unknown",
            "direct_viewer_compatible": None,
            "ready_for_cubism_editor": None,
            "model3_file": model3_path.name,
        }
        if not self._safe_reference(output_path, f"{model_name}.moc3"):
            errors.append("Model filename points outside the export directory.")
        if model3_path.is_symlink() and not model3_path.resolve().is_relative_to(
            output_path.resolve()
        ):
            errors.append("Model descriptor points outside the export directory.")

        metadata_path = output_path / "export_metadata.json"
        has_metadata = metadata_path.exists()
        if has_metadata:
            try:
                if not self._safe_reference(output_path, "export_metadata.json"):
                    raise ValueError("Metadata points outside the export directory")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                checks.update(
                    {
                        "artifact_stage": str(metadata.get("artifact_stage", "unknown")),
                        "direct_viewer_compatible": metadata.get("direct_viewer_compatible")
                        is True,
                        "ready_for_cubism_editor": metadata.get("ready_for_cubism_editor") is True,
                    }
                )
            except (OSError, ValueError, AttributeError):
                errors.append("Invalid export readiness metadata.")
        else:
            warnings.append(
                "Export readiness metadata is missing; Cubism readiness cannot be confirmed."
            )

        def reference(label: str, value: Any) -> None:
            if not isinstance(value, str) or not self._safe_reference(output_path, value):
                errors.append(
                    f"{label} reference points outside the export directory or is invalid."
                )
            elif not (output_path / value.replace("\\", "/")).is_file():
                errors.append(f"Missing referenced {label.lower()}: {value}")

        if model3_path.is_file() and not errors:
            try:
                model = json.loads(model3_path.read_text(encoding="utf-8"))
                file_refs = model["FileReferences"]
                if not isinstance(file_refs, dict):
                    raise ValueError
                moc_ref = file_refs.get("Moc")
                textures = file_refs.get("Textures")
                if not isinstance(textures, list):
                    raise ValueError
                checks.update(moc_reference=moc_ref, texture_count=len(textures))
                if moc_ref != f"{model_name}.moc3":
                    errors.append("model3.json does not reference the expected moc3 filename.")
                reference("Moc", moc_ref)
                if not textures:
                    errors.append("model3.json does not reference any textures.")
                for texture in textures:
                    reference("Texture", texture)
                for key in ("DisplayInfo", "Physics", "Pose", "UserData"):
                    if key in file_refs:
                        reference(key, file_refs[key])
                expressions = file_refs.get("Expressions", [])
                motions = file_refs.get("Motions", {})
                if not isinstance(expressions, list) or not isinstance(motions, dict):
                    raise ValueError
                for expression in expressions:
                    reference("Expression", expression["File"])
                for group in motions.values():
                    if not isinstance(group, list):
                        raise ValueError
                    for motion in group:
                        reference("Motion", motion["File"])
                        if "Sound" in motion:
                            reference("Sound", motion["Sound"])
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                errors.append("Invalid model3.json resource structure.")

        valid = not missing and not errors
        checks["structure_valid"] = valid
        if not valid:
            status = "error"
        elif not has_metadata or checks["ready_for_cubism_editor"] is True:
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
            "checks": checks,
        }

    @staticmethod
    def _safe_reference(output_path: Path, reference: str) -> bool:
        """Reject absolute paths, traversal and symlinks that escape the bundle."""
        normalized = reference.replace("\\", "/")
        candidate = Path(normalized)
        return (
            bool(normalized)
            and "\x00" not in normalized
            and not candidate.is_absolute()
            and not PureWindowsPath(reference).drive
            and ".." not in candidate.parts
            and (output_path / candidate).resolve().is_relative_to(output_path.resolve())
        )
