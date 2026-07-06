from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from mcp_server.artifacts import redact_sensitive
from mcp_server.tools import native_gui_runtime, native_gui_scripts

JsonDict = dict[str, Any]
_CONTROLLER_TIMEOUT_SECONDS = 10
_SUPPORTED_MODES = {"disabled", "dry_run", "execute"}


def _artifact_json(payload: JsonDict) -> str:
    return json.dumps(redact_sensitive(payload), indent=2, ensure_ascii=False)


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
            script=native_gui_scripts.launch_script(editor_path, profile),
            dry_run_payload={
                "editor_path": str(editor_path),
                "profile": profile,
            },
            success_details="Executed built-in controller launch workflow.",
            recorded_details="Recorded a built-in controller launch script.",
        )

    def execute_import(
        self,
        controller: JsonDict,
        psd_path: Path,
        output_dir: Path,
        editor_path: Path | None = None,
    ) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        use_launch_argument = bool(profile.get("import_via_launch_argument", True) and editor_path)
        post_success_check = None
        if use_launch_argument:
            assert editor_path is not None
            script = native_gui_scripts.import_via_launch_argument_script(
                editor_path, psd_path, profile
            )

            def _post_success_check() -> JsonDict:
                return self._await_window_title_fragment(
                    output_dir=output_dir,
                    profile=profile,
                    fragment=psd_path.stem,
                    stem="native_gui_builtin_import_psd_title_check",
                )

            post_success_check = _post_success_check
        else:
            script = native_gui_scripts.import_script(psd_path, profile)
        result = self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="import_psd",
            script_stem="native_gui_builtin_import",
            script=script,
            dry_run_payload={
                "psd_path": str(psd_path),
                "editor_path": str(editor_path) if editor_path else None,
                "import_via_launch_argument": use_launch_argument,
                "profile": profile,
            },
            success_details="Executed built-in controller import workflow.",
            recorded_details="Recorded a built-in controller import script.",
            post_success_check=post_success_check,
            capture_on_success=True,
        )
        if (
            use_launch_argument
            and controller.get("mode") == "execute"
            and result.get("status") == "error"
            and profile.get("import_launch_argument_fallback", True)
        ):
            fallback_result = self._execute_action(
                controller=controller,
                output_dir=output_dir,
                source_action="import_psd",
                script_stem="native_gui_builtin_import_fallback",
                script=native_gui_scripts.import_script(psd_path, profile),
                dry_run_payload={
                    "psd_path": str(psd_path),
                    "editor_path": str(editor_path) if editor_path else None,
                    "import_via_launch_argument": False,
                    "profile": profile,
                    "fallback_reason": "launch_argument_import_failed",
                },
                success_details="Executed built-in controller import fallback workflow.",
                recorded_details="Recorded a built-in controller import fallback script.",
                post_success_check=lambda: self._await_window_title_fragment(
                    output_dir=output_dir,
                    profile=profile,
                    fragment=psd_path.stem,
                    stem="native_gui_builtin_import_fallback_title_check",
                ),
                capture_on_success=True,
            )
            return {
                **fallback_result,
                "fallback_from_launch_argument": True,
                "launch_argument_attempt": result,
            }
        return result

    def execute_apply_template(
        self,
        controller: JsonDict,
        template_id: str,
        output_dir: Path,
    ) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        if controller.get(
            "mode"
        ) == "execute" and not native_gui_scripts.has_apply_template_invocation(profile):
            return {
                "source_action": "apply_template",
                "status": "error",
                "details": (
                    "Apply-template automation requires a configured template menu sequence "
                    "or explicit template shortcut in the native GUI profile."
                ),
            }
        return self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="apply_template",
            script_stem="native_gui_builtin_apply_template",
            script=native_gui_scripts.apply_template_script(template_id, profile),
            dry_run_payload={
                "template_id": template_id,
                "profile": profile,
            },
            success_details="Executed built-in controller template workflow.",
            recorded_details="Recorded a built-in controller template-application script.",
            capture_on_success=True,
        )

    def probe_window(self, controller: JsonDict, output_dir: Path) -> JsonDict | None:
        return native_gui_runtime.probe_window(
            controller=controller,
            output_dir=output_dir,
            run_powershell=self._run_powershell,
            artifact_json=_artifact_json,
        )

    def execute_export(self, controller: JsonDict, output_dir: Path, model_name: str) -> JsonDict:
        profile = dict(controller.get("profile") or {})
        if controller.get("mode") == "execute" and not native_gui_scripts.has_export_invocation(
            profile
        ):
            return {
                "source_action": "export_embedded_data",
                "status": "error",
                "details": (
                    "Export automation requires a configured export menu sequence "
                    "or explicit export shortcut in the native GUI profile."
                ),
            }
        return self._execute_action(
            controller=controller,
            output_dir=output_dir,
            source_action="export_embedded_data",
            script_stem="native_gui_builtin_export",
            script=native_gui_scripts.export_script(output_dir, model_name, profile),
            dry_run_payload={
                "output_dir": str(output_dir),
                "model_name": model_name,
                "profile": profile,
            },
            success_details="Executed built-in controller export workflow.",
            recorded_details="Recorded a built-in controller export script.",
            post_success_check=lambda: self._await_export_outputs(output_dir, model_name, profile),
            capture_on_success=True,
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
        post_success_check: Any | None = None,
        capture_on_success: bool = False,
    ) -> JsonDict:
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / f"{script_stem}.ps1"
        artifact_path = output_dir / f"{script_stem}_result.json"
        profile = dict(controller.get("profile") or {})
        script_path.write_text(script, encoding="utf-8")
        if controller.get("mode") == "dry_run":
            artifact_path.write_text(
                _artifact_json(
                    {
                        "mode": "dry_run",
                        "script_path": str(script_path),
                        **dry_run_payload,
                    }
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
        if final_returncode == 0 and callable(post_success_check):
            post_check = post_success_check()
            payload["post_success_check"] = post_check
            if post_check.get("status") != "success":
                if len(attempts) <= retry_attempts:
                    recovery = self._run_retry_recovery(
                        source_action,
                        output_dir,
                        profile,
                        len(attempts),
                    )
                    payload["post_success_recovery"] = recovery
                    if self._can_retry_after_recovery(recovery):
                        post_check = post_success_check()
                        payload["post_success_check"] = post_check
                if post_check.get("status") != "success":
                    final_returncode = 2
                    payload["returncode"] = final_returncode
                    payload["stderr"] = str(
                        post_check.get("details", "Post-success validation failed.")
                    )
                    final_stderr = payload["stderr"]
        if final_returncode == 0 and capture_on_success:
            payload["post_action_probe"] = self._capture_post_action_probe(
                source_action=source_action,
                output_dir=output_dir,
                profile=profile,
            )
            payload["success_capture"] = self._capture_success_context(
                source_action=source_action,
                output_dir=output_dir,
                profile=profile,
            )
        if final_returncode != 0:
            payload["failure_capture"] = self._capture_failure_context(
                source_action=source_action,
                output_dir=output_dir,
                profile=profile,
            )
        artifact_path.write_text(_artifact_json(payload), encoding="utf-8")
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
        artifact_path.write_text(_artifact_json(recovery), encoding="utf-8")
        return recovery

    def _retry_recovery_script(self, source_action: str, profile: JsonDict) -> str:
        fragments = native_gui_scripts.activation_fragments(profile)
        fragment_json = json.dumps(fragments, ensure_ascii=False).replace('"', '`"')
        entries = self._retry_recovery_sequences(source_action, profile)
        lines = [
            '$ErrorActionPreference = "Stop"\n',
            native_gui_scripts.window_activation_helper_script(profile),
            f'$activationFragments = ConvertFrom-Json "{fragment_json}"\n',
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
            "$windowState = Activate-ControllerWindow -Fragments $activationFragments\n",
            "if (-not $windowState.Activated) { exit 3 }\n",
            "$wshell = New-Object -ComObject WScript.Shell\n",
        ]
        for dialog in self._retry_recovery_dialogs(source_action, profile):
            wait_ms = int(float(dialog.get("wait_seconds", 0.2)) * 1000)
            title_fragment = native_gui_scripts.escape_powershell_string(
                str(dialog.get("title_contains", "")).strip()
            )
            escaped_keys = native_gui_scripts.escape_send_keys(str(dialog.get("keys", "")).strip())
            lines.append(
                "Invoke-DialogRecovery "
                f'-TitleFragment "{title_fragment}" '
                f'-Keys "{escaped_keys}" '
                f"-WaitMilliseconds {wait_ms}\n"
            )
            lines.append(
                "$windowState = Activate-ControllerWindow -Fragments $activationFragments\n"
            )
            lines.append("if (-not $windowState.Activated) { exit 3 }\n")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            keys = str(entry.get("keys", "")).strip()
            if not keys:
                continue
            wait_ms = int(float(entry.get("wait_seconds", 0.2)) * 1000)
            escaped_keys = native_gui_scripts.escape_send_keys(keys)
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
        return native_gui_runtime.capture_failure_context(
            source_action=source_action,
            output_dir=output_dir,
            profile=profile,
            run_powershell=self._run_powershell,
            artifact_json=_artifact_json,
        )

    def _capture_success_context(
        self,
        *,
        source_action: str,
        output_dir: Path,
        profile: JsonDict,
    ) -> JsonDict:
        return native_gui_runtime.capture_success_context(
            source_action=source_action,
            output_dir=output_dir,
            profile=profile,
            run_powershell=self._run_powershell,
            artifact_json=_artifact_json,
        )

    def _capture_post_action_probe(
        self,
        *,
        source_action: str,
        output_dir: Path,
        profile: JsonDict,
    ) -> JsonDict:
        return native_gui_runtime.capture_post_action_probe(
            source_action=source_action,
            output_dir=output_dir,
            profile=profile,
            run_powershell=self._run_powershell,
            artifact_json=_artifact_json,
        )

    def _await_export_outputs(
        self,
        output_dir: Path,
        model_name: str,
        profile: JsonDict,
    ) -> JsonDict:
        return native_gui_runtime.await_export_outputs(
            output_dir=output_dir,
            model_name=model_name,
            profile=profile,
        )

    def _await_window_title_fragment(
        self,
        *,
        output_dir: Path,
        profile: JsonDict,
        fragment: str,
        stem: str,
    ) -> JsonDict:
        return native_gui_runtime.await_window_title_fragment(
            output_dir=output_dir,
            profile=profile,
            fragment=fragment,
            stem=stem,
            run_powershell=self._run_powershell,
            artifact_json=_artifact_json,
        )

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
            encoding="utf-8",
            errors="replace",
            timeout=_CONTROLLER_TIMEOUT_SECONDS,
            check=False,
        )
