"""Cubism Editor environment discovery and automation-plan helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


class CubismBridge:
    """Prepare deterministic automation plans for Cubism Editor workflows."""

    def __init__(self) -> None:
        env_path = os.getenv("CUBISM_EDITOR_PATH")
        self.default_executable_candidates = [
            Path(env_path).expanduser() if env_path else None,
            Path(r"C:\Program Files\Live2D\Cubism5\Cubism Editor 5\CubismEditor5.exe"),
            Path(r"C:\Program Files\Live2D\Cubism4\Cubism Editor 4\CubismEditor4.exe"),
        ]

    def discover_editor(self, explicit_path: str | None = None) -> JsonDict:
        candidates: list[Path] = []
        if explicit_path:
            candidates.append(Path(explicit_path).expanduser())
        candidates.extend(path for path in self.default_executable_candidates if path is not None)

        checked: list[str] = []
        for candidate in candidates:
            if not str(candidate):
                continue
            checked.append(str(candidate))
            if candidate.exists() and candidate.is_file():
                return {
                    "status": "available",
                    "editor_path": str(candidate.resolve()),
                    "checked_paths": checked,
                }

        return {
            "status": "missing",
            "editor_path": None,
            "checked_paths": checked,
            "message": "Cubism Editor executable was not found in the configured paths.",
        }

    def build_plan(
        self,
        *,
        psd_path: str,
        output_dir: str,
        template_id: str,
        model_name: str,
        editor_info: JsonDict,
    ) -> JsonDict:
        plan_steps = [
            {
                "step": 1,
                "action": "launch_editor",
                "description": "Launch or attach to Cubism Editor.",
                "editor_path": editor_info.get("editor_path"),
            },
            {
                "step": 2,
                "action": "import_psd",
                "description": "Open the generated PSD import package.",
                "psd_path": psd_path,
            },
            {
                "step": 3,
                "action": "apply_template",
                "description": "Apply the selected Cubism template mapping.",
                "template_id": template_id,
            },
            {
                "step": 4,
                "action": "export_embedded_data",
                "description": "Export moc3/model3/texture data from Cubism Editor.",
                "output_dir": output_dir,
                "model_name": model_name,
            },
            {
                "step": 5,
                "action": "validate_export_bundle",
                "description": "Run export validation against the generated files.",
                "output_dir": output_dir,
            },
        ]

        return {
            "status": "ready" if editor_info.get("status") == "available" else "blocked",
            "editor": editor_info,
            "steps": plan_steps,
            "automation_mode": "assisted",
        }

    def write_plan(self, plan: JsonDict, output_dir: str, model_name: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plan_path = output_path / f"{model_name}_cubism_automation_plan.json"
        with open(plan_path, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=False)
        return str(plan_path)
