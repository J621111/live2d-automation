"""Execution-backend helpers for Cubism automation plans."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
_ALLOWED_BACKENDS = {"native_gui", "opencli"}
_OPENCLI_WRAPPERS = {"npx", "pnpm", "pnpx", "bunx", "uvx"}
_OPENCLI_SUFFIXES = (".exe", ".cmd", ".bat")
_PREFLIGHT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    automation_mode: str
    requirements: list[str]
    capabilities: list[str]
    env_vars: list[str]


class CubismAutomationManager:
    """Resolve and prepare Cubism automation backends."""

    def __init__(self) -> None:
        self._descriptors: dict[str, BackendDescriptor] = {
            "native_gui": BackendDescriptor(
                name="native_gui",
                automation_mode="assisted",
                requirements=["cubism_editor"],
                capabilities=[
                    "window_launch",
                    "menu_navigation",
                    "keyboard_shortcuts",
                    "dialog_handling",
                ],
                env_vars=[],
            ),
            "opencli": BackendDescriptor(
                name="opencli",
                automation_mode="connector_assisted",
                requirements=["cubism_editor", "opencli_command", "opencli_runtime"],
                capabilities=[
                    "app_connector_bridge",
                    "step_dispatch",
                    "audit_ready_plan",
                    "browser_extension_handshake",
                ],
                env_vars=["OPENCLI_COMMAND"],
            ),
        }

    def available_backends(self) -> list[str]:
        return sorted(self._descriptors)

    def resolve_backend(self, backend_name: str | None = None) -> BackendDescriptor:
        candidate = backend_name or os.getenv("LIVE2D_CUBISM_AUTOMATION_BACKEND") or "native_gui"
        normalized = candidate.strip().lower()
        if normalized not in _ALLOWED_BACKENDS:
            allowed = ", ".join(sorted(_ALLOWED_BACKENDS))
            raise ValueError(
                f"Unsupported automation backend '{candidate}'. Allowed values: {allowed}."
            )
        return self._descriptors[normalized]

    def _parse_command(self, command_hint: str) -> list[str]:
        return [token for token in shlex.split(command_hint, posix=False) if token]

    def _normalized_name(self, token: str) -> str:
        normalized = Path(token.strip('"')).name.lower()
        for suffix in _OPENCLI_SUFFIXES:
            if normalized.endswith(suffix):
                return normalized[: -len(suffix)]
        return normalized

    def _is_opencli_launcher_token(self, token: str) -> bool:
        return self._normalized_name(token) == "opencli"

    def _is_opencli_package_token(self, token: str) -> bool:
        normalized = token.strip('"').lower()
        return normalized == "opencli"

    def _is_wrapper_token(self, token: str) -> bool:
        return self._normalized_name(token) in _OPENCLI_WRAPPERS

    def _resolve_launcher(self, token: str) -> str | None:
        executable = token.strip('"')
        resolved = shutil.which(executable)
        if resolved is None:
            candidate_path = Path(executable).expanduser()
            if candidate_path.exists() and candidate_path.is_file():
                resolved = str(candidate_path.resolve())
        return resolved

    def _opencli_preflight_commands(self, invocation_prefix: list[str]) -> list[JsonDict]:
        return [
            {
                "name": "doctor",
                "argv": [*invocation_prefix, "doctor"],
                "description": "Check whether the opencli runtime and browser bridge are healthy.",
            },
            {
                "name": "list",
                "argv": [*invocation_prefix, "list"],
                "description": "List available apps/connectors exposed through opencli.",
            },
        ]

    def _run_preflight_commands(self, commands: list[JsonDict]) -> list[JsonDict]:
        results: list[JsonDict] = []
        for command in commands:
            argv = [str(item) for item in command.get("argv", [])]
            if not argv:
                results.append(
                    {
                        "name": command.get("name"),
                        "status": "skipped",
                        "returncode": None,
                        "stdout": "",
                        "stderr": "Empty argv.",
                    }
                )
                continue
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=_PREFLIGHT_TIMEOUT_SECONDS,
                    check=False,
                )
                results.append(
                    {
                        "name": command.get("name"),
                        "status": "success" if completed.returncode == 0 else "error",
                        "returncode": completed.returncode,
                        "stdout": completed.stdout.strip(),
                        "stderr": completed.stderr.strip(),
                    }
                )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "name": command.get("name"),
                        "status": "timeout",
                        "returncode": None,
                        "stdout": "",
                        "stderr": "Command timed out during preflight.",
                    }
                )
            except OSError as exc:
                results.append(
                    {
                        "name": command.get("name"),
                        "status": "error",
                        "returncode": None,
                        "stdout": "",
                        "stderr": str(exc),
                    }
                )
        return results

    def _resolve_opencli_command(self, command_hint: str | None) -> JsonDict:
        if not command_hint:
            return {
                "status": "missing",
                "command_hint": None,
                "argv": [],
                "resolved_executable": None,
                "invocation_prefix": [],
                "preflight_commands": [],
                "preflight_results": [],
                "validation_error": "OPENCLI_COMMAND is not configured.",
            }

        argv = self._parse_command(command_hint)
        if not argv:
            return {
                "status": "missing",
                "command_hint": command_hint,
                "argv": [],
                "resolved_executable": None,
                "invocation_prefix": [],
                "preflight_commands": [],
                "preflight_results": [],
                "validation_error": "OPENCLI_COMMAND could not be parsed.",
            }

        launcher = argv[0]
        resolved_launcher = self._resolve_launcher(launcher)
        invocation_prefix = [launcher]
        validation_error: str | None = None

        if self._is_opencli_launcher_token(launcher):
            pass
        elif self._is_wrapper_token(launcher) and len(argv) >= 2:
            package_token = argv[1]
            if not self._is_opencli_package_token(package_token):
                validation_error = (
                    "OPENCLI_COMMAND wrapper must target the exact opencli package "
                    "as the next token."
                )
            else:
                invocation_prefix = argv[:2]
        else:
            validation_error = (
                "OPENCLI_COMMAND must launch the jackwener/opencli CLI directly "
                "or through a supported wrapper."
            )

        status = "ready"
        if resolved_launcher is None:
            status = "missing"
            if validation_error is None:
                validation_error = "The configured opencli launcher could not be resolved."
        elif validation_error is not None:
            status = "missing"

        preflight_commands = []
        preflight_results = []
        if status == "ready":
            preflight_commands = self._opencli_preflight_commands(invocation_prefix)
            preflight_results = self._run_preflight_commands(preflight_commands)
            failing = [
                result for result in preflight_results if result.get("status") not in {"success"}
            ]
            if failing:
                status = "missing"
                if validation_error is None:
                    failed_names = ", ".join(str(result.get("name")) for result in failing)
                    validation_error = f"opencli preflight commands failed: {failed_names}."

        return {
            "status": status,
            "command_hint": command_hint,
            "argv": argv,
            "resolved_executable": resolved_launcher,
            "invocation_prefix": invocation_prefix,
            "preflight_commands": preflight_commands,
            "preflight_results": preflight_results,
            "validation_error": validation_error,
        }

    def build_dispatch_bundle(
        self,
        backend_name: str,
        *,
        plan: JsonDict,
        execution: JsonDict,
        template_id: str,
        model_name: str,
        psd_path: str,
        output_dir: str,
    ) -> JsonDict:
        descriptor = self.resolve_backend(backend_name)
        dispatch_steps: list[JsonDict] = []
        for step in plan.get("steps", []):
            action = str(step.get("action", ""))
            dispatch_steps.append(
                {
                    "step": int(step.get("step", len(dispatch_steps) + 1)),
                    "source_action": action,
                    "dispatch_kind": (
                        "connector_intent" if descriptor.name == "opencli" else "desktop_intent"
                    ),
                    "target": "Cubism Editor",
                    "intent": self._dispatch_intent_for(action, descriptor.name),
                }
            )

        return {
            "status": execution.get("status", "blocked"),
            "backend": descriptor.name,
            "automation_mode": descriptor.automation_mode,
            "ready_to_execute": execution.get("status") == "ready",
            "template_id": template_id,
            "model_name": model_name,
            "psd_path": psd_path,
            "output_dir": output_dir,
            "integration_target": execution.get("integration_target"),
            "preflight": {
                "commands": execution.get("preflight_commands", []),
                "results": execution.get("preflight_results", []),
            },
            "dispatch_steps": dispatch_steps,
            "warnings": execution.get("warnings", []),
        }

    def write_dispatch_bundle(self, bundle: JsonDict, output_dir: str, model_name: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        bundle_path = output_path / f"{model_name}_cubism_dispatch_bundle.json"
        bundle_path.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(bundle_path)

    def _dispatch_intent_for(self, action: str, backend_name: str) -> str:
        if backend_name == "opencli":
            intents = {
                "launch_editor": "Use opencli to bring Cubism Editor to the foreground.",
                "import_psd": "Use opencli to import the prepared PSD into Cubism Editor.",
                "apply_template": "Use opencli to apply the mapped Cubism template workflow.",
                "export_embedded_data": (
                    "Use opencli to export moc3/model3/textures from Cubism Editor."
                ),
                "validate_export_bundle": (
                    "Return control to MCP validation after export completes."
                ),
            }
        else:
            intents = {
                "launch_editor": (
                    "Launch or focus Cubism Editor through the desktop automation backend."
                ),
                "import_psd": "Open the prepared PSD from the desktop automation backend.",
                "apply_template": "Apply the mapped Cubism template using desktop automation.",
                "export_embedded_data": (
                    "Trigger Cubism embedded-data export through desktop automation."
                ),
                "validate_export_bundle": (
                    "Return control to MCP validation after export completes."
                ),
            }
        return intents.get(action, f"Handle '{action}' through the selected automation backend.")

    def prepare_execution(
        self,
        backend_name: str | None,
        *,
        editor_info: JsonDict,
        plan: JsonDict,
    ) -> JsonDict:
        descriptor = self.resolve_backend(backend_name)
        missing_requirements: list[str] = []
        warnings: list[str] = []
        if editor_info.get("status") != "available":
            missing_requirements.append("cubism_editor")

        command_info: JsonDict | None = None
        if descriptor.name == "opencli":
            command_info = self._resolve_opencli_command(os.getenv("OPENCLI_COMMAND"))
            if command_info.get("resolved_executable") is None:
                missing_requirements.append("opencli_command")
            if command_info.get("status") != "ready":
                missing_requirements.append("opencli_runtime")
            validation_error = command_info.get("validation_error")
            if validation_error:
                warnings.append(str(validation_error))

        status = "ready" if not missing_requirements else "blocked"
        return {
            "status": status,
            "backend": descriptor.name,
            "automation_mode": descriptor.automation_mode,
            "requirements": descriptor.requirements,
            "missing_requirements": sorted(set(missing_requirements)),
            "capabilities": descriptor.capabilities,
            "env_vars": descriptor.env_vars,
            "warnings": warnings,
            "integration_target": "jackwener/opencli" if descriptor.name == "opencli" else None,
            "command_hint": command_info.get("command_hint") if command_info else None,
            "resolved_executable": (
                command_info.get("resolved_executable") if command_info else None
            ),
            "argv": command_info.get("argv", []) if command_info else [],
            "invocation_prefix": (
                command_info.get("invocation_prefix", []) if command_info else []
            ),
            "preflight_commands": (
                command_info.get("preflight_commands", []) if command_info else []
            ),
            "preflight_results": (
                command_info.get("preflight_results", []) if command_info else []
            ),
            "plan_actions": [step.get("action") for step in plan.get("steps", [])],
        }
