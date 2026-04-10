from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
_CONTROLLER_TIMEOUT_SECONDS = 10
_SUPPORTED_MODES = {"disabled", "dry_run", "execute"}


class NativeWindowsGUIController:
    """Profile-driven built-in Windows GUI controller for Cubism Editor."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.default_profile = (
            self.project_root / "mcp_server" / "profiles" / "windows_cubism_default.json"
        )

    def resolve(self, profile_hint: str | None = None, mode_hint: str | None = None) -> JsonDict:
        mode = (
            (mode_hint or os.getenv("LIVE2D_NATIVE_GUI_CONTROLLER_MODE") or "disabled")
            .strip()
            .lower()
        )
        if mode not in _SUPPORTED_MODES:
            return {
                "status": "missing",
                "mode": mode,
                "profile_path": None,
                "profile": None,
                "validation_error": (
                    "LIVE2D_NATIVE_GUI_CONTROLLER_MODE must be one of: disabled, dry_run, execute."
                ),
            }
        if mode == "disabled":
            return {
                "status": "disabled",
                "mode": mode,
                "profile_path": None,
                "profile": None,
                "validation_error": None,
            }

        candidate = profile_hint or os.getenv("LIVE2D_NATIVE_GUI_PROFILE")
        profile_path = Path(candidate).expanduser() if candidate else self.default_profile
        if not profile_path.exists() or not profile_path.is_file():
            return {
                "status": "missing",
                "mode": mode,
                "profile_path": str(profile_path),
                "profile": None,
                "validation_error": "The native GUI controller profile could not be found.",
            }

        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        required = {"profile_name", "window_title_contains", "import_shortcut"}
        missing = sorted(required.difference(profile))
        if missing:
            return {
                "status": "missing",
                "mode": mode,
                "profile_path": str(profile_path),
                "profile": None,
                "validation_error": (
                    f"The controller profile is missing required keys: {', '.join(missing)}."
                ),
            }
        return {
            "status": "ready",
            "mode": mode,
            "profile_path": str(profile_path),
            "profile": profile,
            "validation_error": None,
        }

    def execute_launch(self, controller: JsonDict, editor_path: Path, output_dir: Path) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        script_path = output_dir / "native_gui_builtin_launch.ps1"
        artifact_path = output_dir / "native_gui_builtin_launch_result.json"
        script = self._launch_script(editor_path, profile)
        script_path.write_text(script, encoding="utf-8")
        if controller.get("mode") == "dry_run":
            artifact_path.write_text(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "script_path": str(script_path),
                        "editor_path": str(editor_path),
                        "profile": profile,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                "source_action": "launch_editor",
                "status": "recorded",
                "details": "Recorded a built-in controller launch script.",
                "artifact_path": str(artifact_path),
            }

        completed = self._run_powershell(script_path)
        artifact_path.write_text(
            json.dumps(
                {
                    "mode": controller.get("mode"),
                    "script_path": str(script_path),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "source_action": "launch_editor",
            "status": "success" if completed.returncode == 0 else "error",
            "details": "Executed built-in controller launch workflow.",
            "artifact_path": str(artifact_path),
            "returncode": completed.returncode,
        }

    def execute_import(self, controller: JsonDict, psd_path: Path, output_dir: Path) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        script_path = output_dir / "native_gui_builtin_import.ps1"
        artifact_path = output_dir / "native_gui_builtin_import_result.json"
        script = self._import_script(psd_path, profile)
        script_path.write_text(script, encoding="utf-8")
        if controller.get("mode") == "dry_run":
            artifact_path.write_text(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "script_path": str(script_path),
                        "psd_path": str(psd_path),
                        "profile": profile,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                "source_action": "import_psd",
                "status": "recorded",
                "details": "Recorded a built-in controller import script.",
                "artifact_path": str(artifact_path),
            }

        completed = self._run_powershell(script_path)
        artifact_path.write_text(
            json.dumps(
                {
                    "mode": controller.get("mode"),
                    "script_path": str(script_path),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "source_action": "import_psd",
            "status": "success" if completed.returncode == 0 else "error",
            "details": "Executed built-in controller import workflow.",
            "artifact_path": str(artifact_path),
            "returncode": completed.returncode,
        }

    def _run_powershell(self, script_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ],
            capture_output=True,
            text=True,
            timeout=_CONTROLLER_TIMEOUT_SECONDS,
            check=False,
        )

    def _launch_script(self, editor_path: Path, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        delay_ms = int(float(profile.get("launch_wait_seconds", 2.0)) * 1000)
        return (
            '$ErrorActionPreference = "Stop"\n'
            f'Start-Process -FilePath "{editor_path}" | Out-Null\n'
            f"Start-Sleep -Milliseconds {delay_ms}\n"
            "$wshell = New-Object -ComObject WScript.Shell\n"
            f'$null = $wshell.AppActivate("{title}")\n'
        )

    def _import_script(self, psd_path: Path, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        shortcut = str(profile.get("import_shortcut", "^o"))
        activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
        dialog_ms = int(float(profile.get("dialog_wait_seconds", 0.8)) * 1000)
        escaped_psd = str(psd_path).replace("{", "{{").replace("}", "}}")
        return (
            '$ErrorActionPreference = "Stop"\n'
            "$wshell = New-Object -ComObject WScript.Shell\n"
            f'if (-not $wshell.AppActivate("{title}")) {{ throw "Cubism window not found." }}\n'
            f"Start-Sleep -Milliseconds {activation_ms}\n"
            f'$wshell.SendKeys("{shortcut}")\n'
            f"Start-Sleep -Milliseconds {dialog_ms}\n"
            f'$wshell.SendKeys("{escaped_psd}")\n'
            "Start-Sleep -Milliseconds 200\n"
            '$wshell.SendKeys("~")\n'
        )
