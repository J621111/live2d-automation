"""Execution-backend helpers for Cubism automation plans."""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
_ALLOWED_BACKENDS = {"native_gui", "opencli"}
_OPENCLI_WRAPPERS = {"npx", "pnpm", "pnpx", "bunx", "uvx"}


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
                requirements=["cubism_editor", "opencli_command"],
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

    def _is_opencli_token(self, token: str) -> bool:
        normalized = Path(token.strip('"')).name.lower()
        stem = normalized.removesuffix(".exe")
        return stem == "opencli" or stem.startswith("opencli-")

    def _is_wrapper_token(self, token: str) -> bool:
        normalized = Path(token.strip('"')).name.lower()
        stem = normalized.removesuffix(".exe")
        return stem in _OPENCLI_WRAPPERS

    def _resolve_launcher(self, token: str) -> str | None:
        executable = token.strip('"')
        resolved = shutil.which(executable)
        if resolved is None:
            candidate_path = Path(executable).expanduser()
            if candidate_path.exists() and candidate_path.is_file():
                resolved = str(candidate_path.resolve())
        return resolved

    def _opencli_preflight_commands(
        self,
        invocation_prefix: list[str],
    ) -> list[JsonDict]:
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

    def _resolve_opencli_command(self, command_hint: str | None) -> JsonDict:
        if not command_hint:
            return {
                "status": "missing",
                "command_hint": None,
                "argv": [],
                "resolved_executable": None,
                "invocation_prefix": [],
                "preflight_commands": [],
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
                "validation_error": "OPENCLI_COMMAND could not be parsed.",
            }

        launcher = argv[0]
        resolved_launcher = self._resolve_launcher(launcher)
        invocation_prefix = [launcher]
        validation_error: str | None = None

        if self._is_opencli_token(launcher):
            pass
        elif self._is_wrapper_token(launcher) and len(argv) >= 2:
            package_token = argv[1]
            if not self._is_opencli_token(package_token):
                validation_error = (
                    "OPENCLI_COMMAND wrapper must target the opencli package " "as the next token."
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

        preflight_commands = (
            self._opencli_preflight_commands(invocation_prefix) if status == "ready" else []
        )
        return {
            "status": status,
            "command_hint": command_hint,
            "argv": argv,
            "resolved_executable": resolved_launcher,
            "invocation_prefix": invocation_prefix,
            "preflight_commands": preflight_commands,
            "validation_error": validation_error,
        }

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
            if command_info.get("status") != "ready":
                missing_requirements.append("opencli_command")
            validation_error = command_info.get("validation_error")
            if validation_error:
                warnings.append(str(validation_error))

        status = "ready" if not missing_requirements else "blocked"
        return {
            "status": status,
            "backend": descriptor.name,
            "automation_mode": descriptor.automation_mode,
            "requirements": descriptor.requirements,
            "missing_requirements": missing_requirements,
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
            "plan_actions": [step.get("action") for step in plan.get("steps", [])],
        }
