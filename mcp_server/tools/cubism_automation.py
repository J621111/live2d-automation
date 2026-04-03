"""Execution-backend helpers for Cubism automation plans."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

JsonDict = dict[str, Any]
_ALLOWED_BACKENDS = {"native_gui", "opencli"}


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

    def prepare_execution(
        self,
        backend_name: str | None,
        *,
        editor_info: JsonDict,
        plan: JsonDict,
    ) -> JsonDict:
        descriptor = self.resolve_backend(backend_name)
        missing_requirements: list[str] = []
        if editor_info.get("status") != "available":
            missing_requirements.append("cubism_editor")
        if descriptor.name == "opencli" and not os.getenv("OPENCLI_COMMAND"):
            missing_requirements.append("opencli_command")

        status = "ready" if not missing_requirements else "blocked"
        return {
            "status": status,
            "backend": descriptor.name,
            "automation_mode": descriptor.automation_mode,
            "requirements": descriptor.requirements,
            "missing_requirements": missing_requirements,
            "capabilities": descriptor.capabilities,
            "env_vars": descriptor.env_vars,
            "command_hint": os.getenv("OPENCLI_COMMAND") if descriptor.name == "opencli" else None,
            "plan_actions": [step.get("action") for step in plan.get("steps", [])],
        }
