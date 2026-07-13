"""Typed contracts shared by Cubism planning and execution boundaries."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

BackendName: TypeAlias = Literal["native_gui", "opencli"]
AutomationMode: TypeAlias = Literal["assisted", "connector_assisted"]
PreparationStatus: TypeAlias = Literal["ready", "blocked"]
DispatchStatus: TypeAlias = Literal["success", "partial", "blocked", "error"]
StepStatus: TypeAlias = Literal["success", "error", "recorded", "pending", "skipped"]
PreflightStatus: TypeAlias = Literal["success", "error", "timeout", "skipped"]

# Native GUI adapters are external extension points whose profile-specific fields are dynamic.
ExternalPayload: TypeAlias = dict[str, Any]


class AutomationPlanStep(TypedDict, total=False):
    step: int
    action: str
    description: str


class AutomationPlan(TypedDict, total=False):
    status: str
    backend: BackendName
    automation_mode: str
    steps: list[AutomationPlanStep]


class EditorInfo(TypedDict, total=False):
    status: str
    editor_path: str | None
    version: str | None
    source: str


class PreflightCommand(TypedDict, total=False):
    name: str
    argv: list[str]
    description: str


class PreflightResult(TypedDict):
    name: str | None
    status: PreflightStatus
    returncode: int | None
    stdout: str
    stderr: str


class ExecutionPreparation(TypedDict, total=False):
    status: PreparationStatus
    backend: BackendName
    automation_mode: AutomationMode
    execution_supported: bool
    requirements: list[str]
    missing_requirements: list[str]
    capabilities: list[str]
    env_vars: list[str]
    warnings: list[str]
    integration_target: str | None
    command_hint: str | None
    resolved_executable: str | None
    argv: list[str]
    invocation_prefix: list[str]
    preflight_commands: list[PreflightCommand]
    preflight_results: list[PreflightResult]
    native_controller: ExternalPayload
    native_adapter: ExternalPayload | None
    plan_actions: list[str | None]


class DispatchStep(TypedDict, total=False):
    step: int
    source_action: str
    dispatch_kind: Literal["connector_intent", "desktop_intent"]
    target: str
    intent: str


class DispatchPreflight(TypedDict):
    commands: list[PreflightCommand]
    results: list[PreflightResult]


class DispatchBundle(TypedDict, total=False):
    status: str
    backend: BackendName
    automation_mode: AutomationMode
    ready_to_execute: bool
    execution_supported: bool
    template_id: str
    model_name: str
    psd_path: str
    output_dir: str
    editor: EditorInfo
    integration_target: str | None
    native_controller: ExternalPayload | None
    native_adapter: ExternalPayload | None
    preflight: DispatchPreflight
    dispatch_steps: list[DispatchStep]
    warnings: list[str]
    bundle_path: str


class ExecutionStep(TypedDict, total=False):
    step: int | None
    source_action: str
    status: StepStatus
    details: str
    artifact_path: str


class ResumeInfo(TypedDict):
    requested: bool
    skipped_actions: list[str | None]
    previous_successes: list[str]
    cumulative_successes: list[str]
    window_probe: ExternalPayload | None


class DispatchExecution(TypedDict, total=False):
    status: DispatchStatus
    backend: BackendName
    executed_steps: list[ExecutionStep]
    artifacts: list[str]
    resume: ResumeInfo
    message: str
    execution_path: str
