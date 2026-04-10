from __future__ import annotations

import json
import os
import subprocess
import time
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
        return self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="launch_editor",
            script_stem="native_gui_builtin_launch",
            script=self._launch_script(editor_path, profile),
            dry_run_payload={
                "editor_path": str(editor_path),
                "profile": profile,
            },
            success_details="Executed built-in controller launch workflow.",
            recorded_details="Recorded a built-in controller launch script.",
        )

    def execute_import(self, controller: JsonDict, psd_path: Path, output_dir: Path) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        return self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="import_psd",
            script_stem="native_gui_builtin_import",
            script=self._import_script(psd_path, profile),
            dry_run_payload={
                "psd_path": str(psd_path),
                "profile": profile,
            },
            success_details="Executed built-in controller import workflow.",
            recorded_details="Recorded a built-in controller import script.",
        )

    def execute_apply_template(
        self,
        controller: JsonDict,
        template_id: str,
        output_dir: Path,
    ) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        return self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="apply_template",
            script_stem="native_gui_builtin_apply_template",
            script=self._apply_template_script(template_id, profile),
            dry_run_payload={
                "template_id": template_id,
                "profile": profile,
            },
            success_details="Executed built-in controller template workflow.",
            recorded_details="Recorded a built-in controller template-application script.",
        )

    def probe_window(self, controller: JsonDict, output_dir: Path) -> JsonDict | None:
        if controller.get("status") != "ready" or controller.get("mode") != "execute":
            return None
        profile = dict(controller.get("profile") or {})
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / "native_gui_builtin_window_probe.ps1"
        artifact_path = output_dir / "native_gui_builtin_window_probe_result.json"
        script_path.write_text(self._window_probe_script(profile), encoding="utf-8")
        try:
            completed = self._run_powershell(script_path)
            payload: JsonDict = {
                "script_path": str(script_path),
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except subprocess.TimeoutExpired as exc:
            payload = {
                "script_path": str(script_path),
                "returncode": None,
                "stdout": "",
                "stderr": f"Timed out after {exc.timeout} seconds.",
                "timed_out": True,
            }
        artifact_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "status": "success" if payload.get("returncode") == 0 else "error",
            "artifact_path": str(artifact_path),
            "script_path": str(script_path),
            "returncode": payload.get("returncode"),
            "timed_out": payload.get("timed_out", False),
        }

    def execute_export(self, controller: JsonDict, output_dir: Path, model_name: str) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        return self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="export_embedded_data",
            script_stem="native_gui_builtin_export",
            script=self._export_script(output_dir, model_name, profile),
            dry_run_payload={
                "output_dir": str(output_dir),
                "model_name": model_name,
                "profile": profile,
            },
            success_details="Executed built-in controller export workflow.",
            recorded_details="Recorded a built-in controller export script.",
        )

    def _execute_action(
        self,
        *,
        controller: JsonDict,
        output_dir: Path,
        source_action: str,
        script_stem: str,
        script: str,
        dry_run_payload: JsonDict,
        success_details: str,
        recorded_details: str,
    ) -> JsonDict:
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / f"{script_stem}.ps1"
        artifact_path = output_dir / f"{script_stem}_result.json"
        profile = dict(controller.get("profile") or {})
        script_path.write_text(script, encoding="utf-8")
        if controller.get("mode") == "dry_run":
            artifact_path.write_text(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "script_path": str(script_path),
                        **dry_run_payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                "source_action": source_action,
                "status": "recorded",
                "details": recorded_details,
                "artifact_path": str(artifact_path),
            }

        retry_attempts = self._retry_attempts(source_action, profile)
        retry_backoff_seconds = self._retry_backoff_seconds(profile)
        attempts: list[JsonDict] = []
        completed: subprocess.CompletedProcess[str] | None = None
        final_timeout: subprocess.TimeoutExpired | None = None
        for attempt_index in range(retry_attempts + 1):
            try:
                completed = self._run_powershell(script_path)
            except subprocess.TimeoutExpired as exc:
                final_timeout = exc
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "returncode": None,
                        "stdout": "",
                        "stderr": f"Timed out after {exc.timeout} seconds.",
                        "timeout": True,
                    }
                )
                if attempt_index < retry_attempts and retry_backoff_seconds > 0:
                    time.sleep(retry_backoff_seconds)
                continue

            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "timeout": False,
                }
            )
            if completed.returncode == 0:
                break
            if attempt_index < retry_attempts and retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds)

        final_returncode = completed.returncode if completed is not None else None
        final_stdout = completed.stdout.strip() if completed is not None else ""
        final_stderr = (
            completed.stderr.strip()
            if completed is not None
            else (
                f"Timed out after {final_timeout.timeout} seconds."
                if final_timeout is not None
                else ""
            )
        )
        timed_out = final_timeout is not None and completed is None
        payload: JsonDict = {
            "mode": controller.get("mode"),
            "script_path": str(script_path),
            "returncode": final_returncode,
            "stdout": final_stdout,
            "stderr": final_stderr,
            "attempt_count": len(attempts),
            "retry_attempts": retry_attempts,
            "attempts": attempts,
            "timed_out": timed_out,
        }
        if final_returncode != 0:
            payload["failure_capture"] = self._capture_failure_context(
                source_action=source_action,
                output_dir=output_dir,
                profile=profile,
            )
        artifact_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "source_action": source_action,
            "status": "success" if final_returncode == 0 else "error",
            "details": success_details,
            "artifact_path": str(artifact_path),
            "returncode": final_returncode,
            "attempt_count": len(attempts),
            "retry_attempts": retry_attempts,
            "timed_out": timed_out,
            **(
                {"failure_capture": payload["failure_capture"]}
                if "failure_capture" in payload
                else {}
            ),
        }

    def _retry_attempts(self, source_action: str, profile: JsonDict) -> int:
        action_overrides = dict(profile.get("action_retry_attempts") or {})
        if source_action in action_overrides:
            return max(0, int(action_overrides[source_action]))
        return max(0, int(profile.get("retry_attempts", 0)))

    def _retry_backoff_seconds(self, profile: JsonDict) -> float:
        return max(0.0, float(profile.get("retry_backoff_seconds", 0.0)))

    def _capture_failure_context(
        self,
        *,
        source_action: str,
        output_dir: Path,
        profile: JsonDict,
    ) -> JsonDict:
        if profile.get("capture_screenshot_on_error", True) is False:
            return {
                "status": "disabled",
                "details": "Failure screenshot capture is disabled in the controller profile.",
            }

        stem = f"native_gui_builtin_{source_action}_failure_capture"
        script_path = output_dir / f"{stem}.ps1"
        screenshot_path = output_dir / f"{stem}.png"
        artifact_path = output_dir / f"{stem}.json"
        script_path.write_text(
            self._capture_script(screenshot_path, profile),
            encoding="utf-8",
        )
        completed = self._run_powershell(script_path)
        payload: JsonDict = {
            "script_path": str(script_path),
            "screenshot_path": str(screenshot_path),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        artifact_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "status": "success" if completed.returncode == 0 else "error",
            "artifact_path": str(artifact_path),
            "script_path": str(script_path),
            "screenshot_path": str(screenshot_path),
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

    def _apply_template_script(self, template_id: str, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        shortcut = str(profile.get("template_shortcut", "^+t"))
        activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
        dialog_ms = int(float(profile.get("template_dialog_wait_seconds", 0.8)) * 1000)
        escaped_template = template_id.replace("{", "{{").replace("}", "}}")
        script = (
            '$ErrorActionPreference = "Stop"\n'
            "$wshell = New-Object -ComObject WScript.Shell\n"
            f'if (-not $wshell.AppActivate("{title}")) {{ throw "Cubism window not found." }}\n'
            f"Start-Sleep -Milliseconds {activation_ms}\n"
            f'$wshell.SendKeys("{shortcut}")\n'
            f"Start-Sleep -Milliseconds {dialog_ms}\n"
            f'$wshell.SendKeys("{escaped_template}")\n'
            "Start-Sleep -Milliseconds 200\n"
            '$wshell.SendKeys("~")\n'
        )
        return script + self._dialog_sequence_script("apply_template", profile)

    def _export_script(self, output_dir: Path, model_name: str, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        shortcut = str(profile.get("export_shortcut", "^+e"))
        activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
        dialog_ms = int(float(profile.get("export_dialog_wait_seconds", 1.0)) * 1000)
        escaped_output = str(output_dir).replace("{", "{{").replace("}", "}}")
        escaped_name = model_name.replace("{", "{{").replace("}", "}}")
        script = (
            '$ErrorActionPreference = "Stop"\n'
            "$wshell = New-Object -ComObject WScript.Shell\n"
            f'if (-not $wshell.AppActivate("{title}")) {{ throw "Cubism window not found." }}\n'
            f"Start-Sleep -Milliseconds {activation_ms}\n"
            f'$wshell.SendKeys("{shortcut}")\n'
            f"Start-Sleep -Milliseconds {dialog_ms}\n"
            f'$wshell.SendKeys("{escaped_output}")\n'
            "Start-Sleep -Milliseconds 200\n"
            '$wshell.SendKeys("~")\n'
            f"Start-Sleep -Milliseconds {dialog_ms}\n"
            f'$wshell.SendKeys("{escaped_name}")\n'
            "Start-Sleep -Milliseconds 200\n"
            '$wshell.SendKeys("~")\n'
        )
        return script + self._dialog_sequence_script("export_embedded_data", profile)

    def _import_script(self, psd_path: Path, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        shortcut = str(profile.get("import_shortcut", "^o"))
        activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
        dialog_ms = int(float(profile.get("dialog_wait_seconds", 0.8)) * 1000)
        escaped_psd = str(psd_path).replace("{", "{{").replace("}", "}}")
        script = (
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
        return script + self._dialog_sequence_script("import_psd", profile)

    def _dialog_sequence_script(self, action: str, profile: JsonDict) -> str:
        sequences = dict(profile.get("known_dialog_sequences") or {}).get(action, [])
        lines: list[str] = []
        for entry in sequences:
            if not isinstance(entry, dict):
                continue
            keys = str(entry.get("keys", "")).strip()
            if not keys:
                continue
            wait_ms = int(float(entry.get("wait_seconds", 0.35)) * 1000)
            escaped_keys = self._escape_send_keys(keys)
            lines.append(f"Start-Sleep -Milliseconds {wait_ms}\n")
            lines.append(f'$wshell.SendKeys("{escaped_keys}")\n')
        return "".join(lines)

    def _escape_send_keys(self, value: str) -> str:
        return value.replace('"', '""')

    def _window_probe_script(self, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        return (
            '$ErrorActionPreference = "Stop"\n'
            "$wshell = New-Object -ComObject WScript.Shell\n"
            f'if ($wshell.AppActivate("{title}")) {{ exit 0 }} else {{ exit 3 }}\n'
        )

    def _capture_script(self, screenshot_path: Path, profile: JsonDict) -> str:
        delay_ms = int(float(profile.get("failure_capture_wait_seconds", 0.2)) * 1000)
        escaped_path = str(screenshot_path).replace("'", "''")
        return (
            '$ErrorActionPreference = "Stop"\n'
            "Add-Type -AssemblyName System.Windows.Forms\n"
            "Add-Type -AssemblyName System.Drawing\n"
            f"Start-Sleep -Milliseconds {delay_ms}\n"
            "$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen\n"
            "$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height\n"
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)\n"
            "$graphics.CopyFromScreen(\n"
            "    $bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size\n"
            ")\n"
            f"$bitmap.Save('{escaped_path}', [System.Drawing.Imaging.ImageFormat]::Png)\n"
            "$graphics.Dispose()\n"
            "$bitmap.Dispose()\n"
        )
