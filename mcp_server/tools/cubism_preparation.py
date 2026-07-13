"""Backend resolution and execution preparation for Cubism automation."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mcp_server.cubism_contracts import (
    AutomationMode,
    AutomationPlan,
    BackendName,
    EditorInfo,
    ExecutionPreparation,
    PreflightCommand,
    PreflightResult,
    PreparationStatus,
)
from mcp_server.tools.native_gui_controller import NativeWindowsGUIController

JsonDict = dict[str, Any]
_ALLOWED_BACKENDS = {"native_gui", "opencli"}
_OPENCLI_WRAPPERS = {"npx", "pnpm", "pnpx", "bunx", "uvx"}
_OPENCLI_SUFFIXES = (".exe", ".cmd", ".bat")
_PREFLIGHT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class BackendDescriptor:
    name: BackendName
    automation_mode: AutomationMode
    execution_supported: bool
    requirements: list[str]
    capabilities: list[str]
    env_vars: list[str]


class CubismPreparationService:
    """Resolve Cubism backends and prepare their runtime dependencies."""

    def __init__(self, native_controller: NativeWindowsGUIController) -> None:
        self._native_controller = native_controller
        self._descriptors: dict[BackendName, BackendDescriptor] = {
            "native_gui": BackendDescriptor(
                name="native_gui",
                automation_mode="assisted",
                execution_supported=True,
                requirements=["cubism_editor"],
                capabilities=[
                    "window_launch",
                    "menu_navigation",
                    "keyboard_shortcuts",
                    "dialog_handling",
                    "builtin_controller",
                    "adapter_hook",
                ],
                env_vars=[
                    "LIVE2D_NATIVE_GUI_CONTROLLER_MODE",
                    "LIVE2D_NATIVE_GUI_PROFILE",
                    "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
                ],
            ),
            "opencli": BackendDescriptor(
                name="opencli",
                automation_mode="connector_assisted",
                execution_supported=False,
                requirements=[
                    "cubism_editor",
                    "opencli_command",
                    "opencli_runtime",
                    "dispatch_execution",
                ],
                capabilities=[
                    "app_connector_bridge",
                    "step_dispatch_plan",
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
        return self._descriptors[cast(BackendName, normalized)]

    def prepare_execution(
        self,
        backend_name: str | None,
        *,
        editor_info: EditorInfo,
        plan: AutomationPlan,
    ) -> ExecutionPreparation:
        descriptor = self.resolve_backend(backend_name)
        missing_requirements: list[str] = []
        warnings: list[str] = []
        if editor_info.get("status") != "available":
            missing_requirements.append("cubism_editor")
        if not descriptor.execution_supported:
            missing_requirements.append("dispatch_execution")
            warnings.append(
                f"Backend '{descriptor.name}' supports planning only; "
                "dispatch execution is unavailable."
            )

        command_info: JsonDict | None = None
        native_controller = self._resolve_native_controller()
        native_adapter: JsonDict | None = None
        if descriptor.name == "opencli":
            command_info = self._resolve_opencli_command(os.getenv("OPENCLI_COMMAND"))
            if command_info.get("resolved_executable") is None:
                missing_requirements.append("opencli_command")
            if command_info.get("status") != "ready":
                missing_requirements.append("opencli_runtime")
            validation_error = command_info.get("validation_error")
            if validation_error:
                warnings.append(str(validation_error))
        else:
            controller_error = native_controller.get("validation_error")
            if controller_error:
                warnings.append(str(controller_error))
            native_adapter = self._resolve_native_adapter_command(
                os.getenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND")
            )
            validation_error = native_adapter.get("validation_error") if native_adapter else None
            if validation_error:
                warnings.append(str(validation_error))
            if native_adapter and native_adapter.get("status") != "ready":
                warnings.append(
                    "Native GUI adapter is unavailable, so import/export "
                    "steps stay in record-only mode."
                )

        status: PreparationStatus = "ready" if not missing_requirements else "blocked"
        preparation: ExecutionPreparation = {
            "status": status,
            "backend": descriptor.name,
            "automation_mode": descriptor.automation_mode,
            "execution_supported": descriptor.execution_supported,
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
            "native_controller": native_controller,
            "native_adapter": native_adapter,
            "plan_actions": [step.get("action") for step in plan.get("steps", [])],
        }
        return preparation

    def _parse_command(self, command_hint: str) -> list[str]:
        return [token.strip("\"'") for token in shlex.split(command_hint, posix=False) if token]

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

    def _opencli_preflight_commands(self, invocation_prefix: list[str]) -> list[PreflightCommand]:
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

    def _run_preflight_commands(self, commands: list[PreflightCommand]) -> list[PreflightResult]:
        results: list[PreflightResult] = []
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

    def _resolve_native_controller(self) -> JsonDict:
        return dict(self._native_controller.resolve())

    def _resolve_native_adapter_command(self, command_hint: str | None) -> JsonDict:
        if not command_hint:
            return {
                "status": "missing",
                "command_hint": None,
                "argv": [],
                "resolved_executable": None,
                "validation_error": None,
            }

        argv = self._parse_command(command_hint)
        if not argv:
            return {
                "status": "missing",
                "command_hint": command_hint,
                "argv": [],
                "resolved_executable": None,
                "validation_error": "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND could not be parsed.",
            }

        resolved_launcher = self._resolve_launcher(argv[0])
        if resolved_launcher is None:
            return {
                "status": "missing",
                "command_hint": command_hint,
                "argv": argv,
                "resolved_executable": None,
                "validation_error": (
                    "The configured native GUI adapter launcher could not be resolved."
                ),
            }

        return {
            "status": "ready",
            "command_hint": command_hint,
            "argv": argv,
            "resolved_executable": resolved_launcher,
            "validation_error": None,
        }

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

        preflight_commands: list[PreflightCommand] = []
        preflight_results: list[PreflightResult] = []
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
