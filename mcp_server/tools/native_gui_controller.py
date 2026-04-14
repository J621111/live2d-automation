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
            payload.update(self._parse_probe_stdout(payload["stdout"]))
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
            **({"diagnostics": payload["diagnostics"]} if "diagnostics" in payload else {}),
            **({"all_titles": payload["all_titles"]} if "all_titles" in payload else {}),
            **(
                {"all_diagnostics": payload["all_diagnostics"]}
                if "all_diagnostics" in payload
                else {}
            ),
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
                if attempt_index < retry_attempts:
                    attempts[-1]["recovery"] = self._run_retry_recovery(
                        source_action,
                        output_dir,
                        profile,
                        attempt_index + 1,
                    )
                    if not self._can_retry_after_recovery(attempts[-1]["recovery"]):
                        break
                    if retry_backoff_seconds > 0:
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
            if attempt_index < retry_attempts:
                attempts[-1]["recovery"] = self._run_retry_recovery(
                    source_action,
                    output_dir,
                    profile,
                    attempt_index + 1,
                )
                if not self._can_retry_after_recovery(attempts[-1]["recovery"]):
                    break
                if retry_backoff_seconds > 0:
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

    def _can_retry_after_recovery(self, recovery: JsonDict | None) -> bool:
        if not recovery or recovery.get("status") != "success":
            return False
        probe = recovery.get("probe")
        if isinstance(probe, dict):
            return probe.get("status") == "success"
        return False

    def _run_retry_recovery(
        self,
        source_action: str,
        output_dir: Path,
        profile: JsonDict,
        attempt_number: int,
    ) -> JsonDict:
        stem = f"native_gui_builtin_{source_action}_retry_recovery_attempt_{attempt_number}"
        script_path = output_dir / f"{stem}.ps1"
        artifact_path = output_dir / f"{stem}.json"
        script_path.write_text(
            self._retry_recovery_script(source_action, profile), encoding="utf-8"
        )
        try:
            completed = self._run_powershell(script_path)
            recovery: JsonDict = {
                "status": "success" if completed.returncode == 0 else "error",
                "script_path": str(script_path),
                "artifact_path": str(artifact_path),
                "dialog_recovery_plan": self._describe_retry_recovery_plan(source_action, profile),
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except subprocess.TimeoutExpired as exc:
            recovery = {
                "status": "error",
                "script_path": str(script_path),
                "artifact_path": str(artifact_path),
                "dialog_recovery_plan": self._describe_retry_recovery_plan(source_action, profile),
                "returncode": None,
                "stdout": "",
                "stderr": f"Timed out after {exc.timeout} seconds.",
                "timed_out": True,
            }
        probe = self.probe_window(
            {"status": "ready", "mode": "execute", "profile": profile}, output_dir
        )
        recovery["probe"] = probe
        artifact_path.write_text(
            json.dumps(recovery, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return recovery

    def _retry_recovery_script(self, source_action: str, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        entries = self._retry_recovery_sequences(source_action, profile)
        lines = [
            '$ErrorActionPreference = "Stop"\n',
            "function Invoke-DialogRecovery {\n",
            "    param(\n",
            "        [string]$TitleFragment,\n",
            "        [string]$Keys,\n",
            "        [int]$WaitMilliseconds\n",
            "    )\n",
            (
                "    if ([string]::IsNullOrWhiteSpace($TitleFragment) -or "
                "[string]::IsNullOrWhiteSpace($Keys)) { return }\n"
            ),
            "    $process = Get-Process | Where-Object {\n",
            '        $_.MainWindowTitle -and $_.MainWindowTitle -like "*$TitleFragment*"\n',
            "    } | Select-Object -First 1\n",
            "    if ($null -eq $process) { return }\n",
            "    $wshell = New-Object -ComObject WScript.Shell\n",
            "    if (-not $wshell.AppActivate($process.Id)) { return }\n",
            "    if ($WaitMilliseconds -gt 0) {\n",
            "        Start-Sleep -Milliseconds $WaitMilliseconds\n",
            "    }\n",
            "    $wshell.SendKeys($Keys)\n",
            "}\n",
            "$wshell = New-Object -ComObject WScript.Shell\n",
            f'if (-not $wshell.AppActivate("{title}")) {{ exit 3 }}\n',
        ]
        for dialog in self._retry_recovery_dialogs(source_action, profile):
            wait_ms = int(float(dialog.get("wait_seconds", 0.2)) * 1000)
            title_fragment = self._escape_powershell_string(
                str(dialog.get("title_contains", "")).strip()
            )
            escaped_keys = self._escape_send_keys(str(dialog.get("keys", "")).strip())
            lines.append(
                "Invoke-DialogRecovery "
                f'-TitleFragment "{title_fragment}" '
                f'-Keys "{escaped_keys}" '
                f"-WaitMilliseconds {wait_ms}\n"
            )
            lines.append(f'if (-not $wshell.AppActivate("{title}")) {{ exit 3 }}\n')
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            keys = str(entry.get("keys", "")).strip()
            if not keys:
                continue
            wait_ms = int(float(entry.get("wait_seconds", 0.2)) * 1000)
            escaped_keys = self._escape_send_keys(keys)
            lines.append(f"Start-Sleep -Milliseconds {wait_ms}\n")
            lines.append(f'$wshell.SendKeys("{escaped_keys}")\n')
        return "".join(lines)

    def _describe_retry_recovery_plan(self, source_action: str, profile: JsonDict) -> JsonDict:
        dialogs_map = dict(profile.get("known_dialog_recovery") or {})
        dialog_source = (
            source_action
            if source_action in dialogs_map and isinstance(dialogs_map.get(source_action), list)
            else ("default" if isinstance(dialogs_map.get("default"), list) else None)
        )
        sequences_map = dict(profile.get("retry_recovery_sequences") or {})
        sequence_source = (
            source_action
            if source_action in sequences_map and isinstance(sequences_map.get(source_action), list)
            else ("default" if isinstance(sequences_map.get("default"), list) else None)
        )
        return {
            "dialog_source": dialog_source,
            "dialogs": self._retry_recovery_dialogs(source_action, profile),
            "sequence_source": sequence_source,
            "sequences": self._retry_recovery_sequences(source_action, profile),
        }

    def _retry_recovery_dialogs(self, source_action: str, profile: JsonDict) -> list[JsonDict]:
        dialogs_map = dict(profile.get("known_dialog_recovery") or {})
        selected = dialogs_map.get(source_action) or dialogs_map.get("default") or []
        return [entry for entry in selected if isinstance(entry, dict)]

    def _retry_recovery_sequences(self, source_action: str, profile: JsonDict) -> list[JsonDict]:
        sequences_map = dict(profile.get("retry_recovery_sequences") or {})
        selected = sequences_map.get(source_action) or sequences_map.get("default") or []
        return [entry for entry in selected if isinstance(entry, dict)]

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

    def _escape_powershell_string(self, value: str) -> str:
        return value.replace('"', '`"')

    def _window_probe_script(self, profile: JsonDict) -> str:
        title = str(profile.get("window_title_contains", "Cubism Editor"))
        diagnostic_fragments = [title]
        diagnostic_fragments.extend(
            str(fragment).strip()
            for fragment in profile.get("window_probe_candidates", [])
            if str(fragment).strip()
        )
        unique_fragments: list[str] = []
        for fragment in diagnostic_fragments:
            if fragment not in unique_fragments:
                unique_fragments.append(fragment)
        fragment_json = json.dumps(unique_fragments, ensure_ascii=False).replace('"', '`"')
        return (
            '$ErrorActionPreference = "Stop"\n'
            f'$fragments = ConvertFrom-Json "{fragment_json}"\n'
            "$windows = Get-Process | Where-Object {\n"
            "    $_.MainWindowTitle\n"
            "} | ForEach-Object {\n"
            "    [PSCustomObject]@{\n"
            "        ProcessId = $_.Id\n"
            "        ProcessName = $_.ProcessName\n"
            "        Title = $_.MainWindowTitle\n"
            "    }\n"
            "}\n"
            "$diagnosticWindows = @($windows | Where-Object {\n"
            "    $title = $_.Title\n"
            '    $fragments | Where-Object { $title -like "*$_*" }\n'
            "})\n"
            "$wshell = New-Object -ComObject WScript.Shell\n"
            f'$activated = $wshell.AppActivate("{title}")\n'
            "$result = [PSCustomObject]@{\n"
            f'    target = "{self._escape_powershell_string(title)}"\n'
            "    matched_titles = @($diagnosticWindows | ForEach-Object { $_.Title })\n"
            "    diagnostics = @($diagnosticWindows)\n"
            "    all_titles = @($windows | ForEach-Object { $_.Title })\n"
            "    all_diagnostics = @($windows)\n"
            "}\n"
            "$result | ConvertTo-Json -Depth 4 -Compress | Write-Output\n"
            "if ($activated) { exit 0 } else { exit 3 }\n"
        )

    def _parse_probe_stdout(self, stdout: str) -> JsonDict:
        if not stdout:
            return {}
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            return {}
        candidate = lines[-1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        extracted: JsonDict = {}
        if "matched_titles" in parsed and isinstance(parsed["matched_titles"], list):
            extracted["matched_titles"] = parsed["matched_titles"]
        if "diagnostics" in parsed and isinstance(parsed["diagnostics"], list):
            extracted["diagnostics"] = parsed["diagnostics"]
        if "all_titles" in parsed and isinstance(parsed["all_titles"], list):
            extracted["all_titles"] = parsed["all_titles"]
        if "all_diagnostics" in parsed and isinstance(parsed["all_diagnostics"], list):
            extracted["all_diagnostics"] = parsed["all_diagnostics"]
        if "target" in parsed:
            extracted["target"] = parsed["target"]
        return extracted

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
