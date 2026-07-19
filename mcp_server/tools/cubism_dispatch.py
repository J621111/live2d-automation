"""Dispatch bundle construction and execution for Cubism automation."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, overload

from mcp_server.artifacts import ArtifactStore, redact_command
from mcp_server.cubism_contracts import (
    AutomationPlan,
    DispatchBundle,
    DispatchExecution,
    DispatchStatus,
    DispatchStep,
    EditorInfo,
    ExecutionPreparation,
    ExecutionStep,
)
from mcp_server.tools.cubism_preparation import BackendDescriptor, CubismPreparationService
from mcp_server.tools.export_validator import CubismExportValidator
from mcp_server.tools.native_gui_controller import NativeWindowsGUIController

JsonDict = dict[str, Any]
_EXECUTION_TIMEOUT_SECONDS = 5
_NATIVE_ADAPTER_UNSUPPORTED_EXIT_CODE = 64
_SCRIPT_SUFFIXES = {"", ".cmd", ".bat", ".sh", ".ps1"}


class CubismDispatchExecutor:
    """Build and execute dispatch bundles for prepared Cubism backends."""

    def __init__(
        self,
        native_controller: NativeWindowsGUIController,
        preparation_service: CubismPreparationService,
    ) -> None:
        self._native_controller = native_controller
        self._preparation_service = preparation_service

    def resolve_backend(self, backend_name: str | None = None) -> BackendDescriptor:
        return self._preparation_service.resolve_backend(backend_name)

    @overload
    def build_dispatch_bundle(
        self,
        backend_name: str,
        *,
        plan: AutomationPlan,
        execution: ExecutionPreparation,
        template_id: str,
        model_name: str,
        psd_path: str,
        output_dir: str,
        editor_info: EditorInfo,
    ) -> DispatchBundle: ...

    @overload
    def build_dispatch_bundle(
        self,
        backend_name: str,
        *,
        plan: Mapping[str, object],
        execution: Mapping[str, object],
        template_id: str,
        model_name: str,
        psd_path: str,
        output_dir: str,
        editor_info: Mapping[str, object],
    ) -> Any: ...

    def build_dispatch_bundle(
        self,
        backend_name: str,
        *,
        plan: object,
        execution: object,
        template_id: str,
        model_name: str,
        psd_path: str,
        output_dir: str,
        editor_info: object,
    ) -> object:
        typed_plan = cast(AutomationPlan, plan)
        typed_execution = cast(ExecutionPreparation, execution)
        typed_editor_info = cast(EditorInfo, editor_info)
        descriptor = self.resolve_backend(backend_name)
        dispatch_steps: list[DispatchStep] = []
        for step in typed_plan.get("steps", []):
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

        bundle: DispatchBundle = {
            "status": typed_execution.get("status", "blocked"),
            "backend": descriptor.name,
            "automation_mode": descriptor.automation_mode,
            "ready_to_execute": (
                descriptor.execution_supported and typed_execution.get("status") == "ready"
            ),
            "execution_supported": descriptor.execution_supported,
            "template_id": template_id,
            "model_name": model_name,
            "psd_path": psd_path,
            "output_dir": output_dir,
            "editor": typed_editor_info,
            "integration_target": typed_execution.get("integration_target"),
            "native_controller": typed_execution.get("native_controller"),
            "native_adapter": typed_execution.get("native_adapter"),
            "preflight": {
                "commands": typed_execution.get("preflight_commands", []),
                "results": typed_execution.get("preflight_results", []),
            },
            "dispatch_steps": dispatch_steps,
            "warnings": typed_execution.get("warnings", []),
        }
        return bundle

    @overload
    def execute_dispatch_bundle(
        self,
        bundle: DispatchBundle,
        *,
        previous_execution: DispatchExecution | None = None,
        resume: bool = False,
    ) -> DispatchExecution: ...

    @overload
    def execute_dispatch_bundle(
        self,
        bundle: Mapping[str, object],
        *,
        previous_execution: Mapping[str, object] | None = None,
        resume: bool = False,
    ) -> Any: ...

    def execute_dispatch_bundle(
        self,
        bundle: object,
        *,
        previous_execution: object | None = None,
        resume: bool = False,
    ) -> object:
        typed_bundle = cast(DispatchBundle, bundle)
        typed_previous_execution = cast(DispatchExecution | None, previous_execution)
        backend = str(typed_bundle.get("backend", "native_gui"))
        descriptor = self.resolve_backend(backend)
        if not descriptor.execution_supported:
            unsupported_execution: DispatchExecution = {
                "status": "blocked",
                "backend": descriptor.name,
                "executed_steps": [],
                "artifacts": [],
                "message": (
                    f"Backend '{descriptor.name}' supports planning only; "
                    "dispatch execution is unavailable."
                ),
            }
            return unsupported_execution
        if not bool(typed_bundle.get("ready_to_execute", False)):
            unready_execution: DispatchExecution = {
                "status": "blocked",
                "backend": descriptor.name,
                "executed_steps": [],
                "artifacts": [],
                "message": "Dispatch bundle is not ready_to_execute.",
            }
            return unready_execution
        artifact_store = ArtifactStore(str(typed_bundle.get("output_dir", ".")))
        output_dir = artifact_store.output_dir
        editor_info = dict(typed_bundle.get("editor", {}))
        editor_path = (
            Path(str(editor_info.get("editor_path", "")))
            if editor_info.get("editor_path")
            else None
        )
        psd_path = (
            Path(str(typed_bundle.get("psd_path", ""))) if typed_bundle.get("psd_path") else None
        )
        native_controller = dict(typed_bundle.get("native_controller") or {})
        native_adapter = dict(typed_bundle.get("native_adapter") or {})

        executed_steps: list[ExecutionStep] = []
        artifacts: list[str] = []
        step_statuses: dict[str, str] = {}
        previous_successes = self._resume_successes(typed_previous_execution) if resume else set()
        resume_probe = (
            self._probe_native_resume_window(native_controller, output_dir)
            if resume and previous_successes
            else None
        )
        if resume_probe and resume_probe.get("artifact_path"):
            artifacts.append(str(resume_probe["artifact_path"]))
        halted_after_error = False
        for step in typed_bundle.get("dispatch_steps", []):
            result: ExecutionStep
            action = str(step.get("source_action", ""))
            can_skip = action in previous_successes and self._can_skip_resumed_action(
                action,
                native_controller,
                resume_probe,
            )
            if can_skip:
                result = {
                    "step": step.get("step"),
                    "source_action": action,
                    "status": "skipped",
                    "details": (
                        "Skipped because this step already succeeded in the previous execution."
                    ),
                }
                step_statuses[action] = "success"
            elif halted_after_error:
                result = {
                    "step": step.get("step"),
                    "source_action": action,
                    "status": "pending",
                    "details": "Skipped because a previous automation step failed.",
                }
            elif action == "launch_editor" and self._should_defer_native_launch_to_import(
                native_controller
            ):
                result = {
                    "step": step.get("step"),
                    "source_action": action,
                    "status": "success",
                    "details": "Launch is deferred to import_psd via direct document open.",
                }
            elif action == "launch_editor":
                result = cast(
                    ExecutionStep,
                    self._execute_native_launch(
                        editor_path,
                        output_dir,
                        native_controller,
                        native_adapter,
                        typed_bundle,
                    ),
                )
            elif action == "import_psd":
                result = cast(
                    ExecutionStep,
                    self._execute_native_import(
                        psd_path,
                        output_dir,
                        native_controller,
                        native_adapter,
                        typed_bundle,
                    ),
                )
            elif action == "apply_template":
                result = cast(
                    ExecutionStep,
                    self._execute_native_apply_template(
                        output_dir, native_controller, native_adapter, typed_bundle
                    ),
                )
            elif action == "export_embedded_data":
                result = cast(
                    ExecutionStep,
                    self._execute_native_export(
                        output_dir, native_controller, native_adapter, typed_bundle
                    ),
                )
            elif action == "validate_export_bundle":
                if step_statuses.get("export_embedded_data") != "success":
                    result = {
                        "step": step.get("step"),
                        "source_action": action,
                        "status": "pending",
                        "details": "Validation is deferred until export_embedded_data succeeds.",
                    }
                else:
                    result = cast(
                        ExecutionStep,
                        self._execute_local_validation(output_dir, typed_bundle),
                    )
            else:
                result = {
                    "step": step.get("step"),
                    "source_action": action,
                    "status": "pending",
                    "details": "Execution PoC does not handle this step yet.",
                }
                step_statuses[action] = str(result.get("status", "pending"))
            if not can_skip:
                step_statuses[action] = str(result.get("status", "pending"))
            executed_steps.append(result)
            artifact_path = result.get("artifact_path")
            if artifact_path:
                artifacts.append(str(artifact_path))
            if result.get("status") == "error":
                halted_after_error = True

        failed = [step for step in executed_steps if step.get("status") == "error"]
        recorded = [step for step in executed_steps if step.get("status") == "recorded"]
        pending = [step for step in executed_steps if step.get("status") == "pending"]
        skipped = [step for step in executed_steps if step.get("status") == "skipped"]
        current_execution: DispatchExecution = {"executed_steps": executed_steps}
        current_successes = self._successful_actions(current_execution)
        cumulative_successes = sorted(previous_successes.union(current_successes))
        if failed:
            status: DispatchStatus = "error"
        elif recorded or pending:
            status = "partial"
        else:
            status = "success"
        dispatch_execution: DispatchExecution = {
            "status": status,
            "backend": descriptor.name,
            "executed_steps": executed_steps,
            "artifacts": artifacts,
            "resume": {
                "requested": resume,
                "skipped_actions": [step.get("source_action") for step in skipped],
                "previous_successes": sorted(previous_successes),
                "cumulative_successes": cumulative_successes,
                "window_probe": resume_probe,
            },
            "message": (
                "Native GUI execution PoC completed."
                if status == "success"
                else (
                    "Native GUI execution PoC produced partial execution records."
                    if status == "partial"
                    else "Native GUI execution PoC hit errors."
                )
            ),
        }
        return dispatch_execution

    def _probe_native_resume_window(
        self,
        native_controller: JsonDict,
        output_dir: Path,
    ) -> JsonDict | None:
        if native_controller.get("status") != "ready":
            return None
        result = self._native_controller.probe_window(native_controller, output_dir)
        return dict(result) if result is not None else None

    def _can_skip_resumed_action(
        self,
        action: str,
        native_controller: JsonDict,
        resume_probe: JsonDict | None,
    ) -> bool:
        if action not in {"launch_editor", "import_psd"}:
            return True
        if native_controller.get("status") != "ready" or native_controller.get("mode") != "execute":
            return False
        return bool(resume_probe and resume_probe.get("status") == "success")

    def _should_defer_native_launch_to_import(self, native_controller: JsonDict) -> bool:
        if native_controller.get("status") != "ready":
            return False
        if native_controller.get("mode") != "execute":
            return False
        profile = dict(native_controller.get("profile") or {})
        return bool(profile.get("import_via_launch_argument", True))

    def _resume_successes(self, execution: DispatchExecution | None) -> set[str]:
        if not execution:
            return set()
        resume = dict(execution.get("resume") or {})
        cumulative = resume.get("cumulative_successes")
        if isinstance(cumulative, list):
            return {str(action) for action in cumulative if str(action)}
        previous = resume.get("previous_successes")
        skipped = resume.get("skipped_actions")
        actions: set[str] = self._successful_actions(execution)
        if isinstance(previous, list):
            actions.update(str(action) for action in previous if str(action))
        if isinstance(skipped, list):
            actions.update(str(action) for action in skipped if str(action))
        return actions

    def _successful_actions(self, execution: DispatchExecution | None) -> set[str]:
        if not execution:
            return set()
        actions: set[str] = set()
        for step in execution.get("executed_steps", []):
            if str(step.get("status")) == "success":
                actions.add(str(step.get("source_action", "")))
        return actions

    def write_dispatch_execution(
        self, execution: JsonDict, output_dir: str, model_name: str, suffix: str = ""
    ) -> str:
        artifact_store = ArtifactStore(output_dir)
        execution_path = artifact_store.write_json(
            f"{model_name}_cubism_dispatch_execution{suffix}.json", execution
        )
        return str(execution_path)

    def _execute_native_launch(
        self,
        editor_path: Path | None,
        output_dir: Path,
        native_controller: JsonDict,
        native_adapter: JsonDict,
        bundle: DispatchBundle,
    ) -> JsonDict:
        if editor_path is None or not editor_path.exists():
            return {
                "source_action": "launch_editor",
                "status": "error",
                "details": "Editor path is missing.",
            }

        controller_result = self._execute_native_controller_launch(
            native_controller,
            editor_path,
            output_dir,
        )
        if controller_result is not None:
            return controller_result

        adapter_result = self._execute_native_adapter_action(
            native_adapter,
            action="launch_editor",
            output_dir=output_dir,
            bundle=bundle,
            editor_path=editor_path,
            psd_path=None,
        )
        if adapter_result is not None:
            return adapter_result

        script_like = editor_path.suffix.lower() in _SCRIPT_SUFFIXES
        if not script_like and os.getenv("LIVE2D_NATIVE_GUI_ALLOW_BINARY_EXEC") != "1":
            artifact_path = ArtifactStore(output_dir).write_json(
                "native_gui_launch_request.json",
                {
                    "editor_path": str(editor_path),
                    "mode": "record_only",
                    "reason": "Binary launch is disabled for the execution PoC.",
                },
            )
            return {
                "source_action": "launch_editor",
                "status": "recorded",
                "details": "Recorded a launch request without executing the binary.",
                "artifact_path": str(artifact_path),
            }

        command = self._native_launch_command(editor_path)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_EXECUTION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "source_action": "launch_editor",
                "status": "error",
                "details": "Editor launch timed out.",
            }
        except OSError as exc:
            return {
                "source_action": "launch_editor",
                "status": "error",
                "details": str(exc),
            }

        artifact_path = ArtifactStore(output_dir).write_json(
            "native_gui_launch_result.json",
            {
                "command": redact_command(command),
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            },
        )
        return {
            "source_action": "launch_editor",
            "status": "success" if completed.returncode == 0 else "error",
            "details": "Executed launch command for the native GUI backend.",
            "returncode": completed.returncode,
            "artifact_path": str(artifact_path),
        }

    def _native_launch_command(self, editor_path: Path) -> list[str]:
        suffix = editor_path.suffix.lower()
        if suffix == ".ps1":
            return [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(editor_path),
            ]
        return [str(editor_path)]

    def _execute_native_import(
        self,
        psd_path: Path | None,
        output_dir: Path,
        native_controller: JsonDict,
        native_adapter: JsonDict,
        bundle: DispatchBundle,
    ) -> JsonDict:
        if psd_path is None or not psd_path.exists():
            return {
                "source_action": "import_psd",
                "status": "error",
                "details": "PSD path is missing.",
            }

        editor_info = dict(bundle.get("editor", {}))
        editor_path = (
            Path(str(editor_info.get("editor_path", "")))
            if editor_info.get("editor_path")
            else None
        )
        controller_result = self._execute_native_controller_import(
            native_controller,
            psd_path,
            output_dir,
            editor_path,
        )
        if controller_result is not None:
            return controller_result

        adapter_result = self._execute_native_adapter_action(
            native_adapter,
            action="import_psd",
            output_dir=output_dir,
            bundle=bundle,
            editor_path=editor_path,
            psd_path=psd_path,
        )
        if adapter_result is not None:
            return adapter_result

        artifact_path = ArtifactStore(output_dir).write_json(
            "native_gui_import_request.json",
            {
                "psd_path": str(psd_path),
                "target": "Cubism Editor",
                "model_name": bundle.get("model_name"),
                "template_id": bundle.get("template_id"),
                "dispatch_kind": "desktop_intent",
            },
        )
        return {
            "source_action": "import_psd",
            "status": "recorded",
            "details": "Recorded a native GUI import request for the prepared PSD.",
            "artifact_path": str(artifact_path),
        }

    def _execute_native_controller_launch(
        self,
        native_controller: JsonDict,
        editor_path: Path | None,
        output_dir: Path,
    ) -> JsonDict | None:
        if native_controller.get("status") != "ready":
            return None
        if editor_path is None or not editor_path.exists():
            return {
                "source_action": "launch_editor",
                "status": "error",
                "details": "Editor path is missing.",
            }
        return dict(
            self._native_controller.execute_launch(native_controller, editor_path, output_dir)
        )

    def _execute_native_controller_apply_template(
        self,
        native_controller: JsonDict,
        template_id: str,
        output_dir: Path,
    ) -> JsonDict | None:
        if native_controller.get("status") != "ready":
            return None
        return dict(
            self._native_controller.execute_apply_template(
                native_controller,
                template_id,
                output_dir,
            )
        )

    def _execute_native_controller_export(
        self,
        native_controller: JsonDict,
        output_dir: Path,
        model_name: str,
    ) -> JsonDict | None:
        if native_controller.get("status") != "ready":
            return None
        return dict(
            self._native_controller.execute_export(
                native_controller,
                output_dir,
                model_name,
            )
        )

    def _execute_native_controller_import(
        self,
        native_controller: JsonDict,
        psd_path: Path | None,
        output_dir: Path,
        editor_path: Path | None,
    ) -> JsonDict | None:
        if native_controller.get("status") != "ready":
            return None
        if psd_path is None or not psd_path.exists():
            return {
                "source_action": "import_psd",
                "status": "error",
                "details": "PSD path is missing.",
            }
        return dict(
            self._native_controller.execute_import(
                native_controller,
                psd_path,
                output_dir,
                editor_path=editor_path,
            )
        )

    def _execute_native_adapter_action(
        self,
        native_adapter: JsonDict,
        *,
        action: str,
        output_dir: Path,
        bundle: DispatchBundle,
        editor_path: Path | None,
        psd_path: Path | None,
        fallback_on_nonzero: bool = False,
    ) -> JsonDict | None:
        if native_adapter.get("status") != "ready":
            return None

        argv = [str(item) for item in native_adapter.get("argv", [])]
        if not argv:
            return None

        command = [
            *argv,
            action,
            "--output-dir",
            str(output_dir),
            "--model-name",
            str(bundle.get("model_name", "ATRI")),
            "--template-id",
            str(bundle.get("template_id", "")),
        ]
        if editor_path is not None:
            command.extend(["--editor-path", str(editor_path)])
        if psd_path is not None:
            command.extend(["--psd-path", str(psd_path)])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_EXECUTION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "source_action": action,
                "status": "error",
                "details": f"Native GUI adapter timed out while handling {action}.",
            }
        except OSError as exc:
            return {
                "source_action": action,
                "status": "error",
                "details": str(exc),
            }

        if completed.returncode == _NATIVE_ADAPTER_UNSUPPORTED_EXIT_CODE and fallback_on_nonzero:
            return None

        artifact_path = ArtifactStore(output_dir).write_json(
            f"native_gui_{action}_adapter_result.json",
            {
                "command": redact_command(command),
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "adapter": native_adapter,
            },
        )
        return {
            "source_action": action,
            "status": "success" if completed.returncode == 0 else "error",
            "details": f"Executed native GUI adapter for {action}.",
            "returncode": completed.returncode,
            "artifact_path": str(artifact_path),
        }

    def _execute_native_apply_template(
        self,
        output_dir: Path,
        native_controller: JsonDict,
        native_adapter: JsonDict,
        bundle: DispatchBundle,
    ) -> JsonDict:
        controller_result = self._execute_native_controller_apply_template(
            native_controller,
            str(bundle.get("template_id", "")),
            output_dir,
        )
        if controller_result is not None:
            return controller_result

        adapter_result = self._execute_native_adapter_action(
            native_adapter,
            action="apply_template",
            output_dir=output_dir,
            bundle=bundle,
            editor_path=None,
            psd_path=None,
            fallback_on_nonzero=True,
        )
        if adapter_result is not None:
            return adapter_result

        artifact_path = ArtifactStore(output_dir).write_json(
            "native_gui_apply_template_request.json",
            {
                "template_id": bundle.get("template_id"),
                "model_name": bundle.get("model_name"),
                "dispatch_kind": "desktop_intent",
            },
        )
        return {
            "source_action": "apply_template",
            "status": "recorded",
            "details": "Recorded a native GUI template-application request.",
            "artifact_path": str(artifact_path),
        }

    def _execute_native_export(
        self,
        output_dir: Path,
        native_controller: JsonDict,
        native_adapter: JsonDict,
        bundle: DispatchBundle,
    ) -> JsonDict:
        controller_result = self._execute_native_controller_export(
            native_controller,
            output_dir,
            str(bundle.get("model_name", "ATRI")),
        )
        if controller_result is not None:
            return controller_result

        adapter_result = self._execute_native_adapter_action(
            native_adapter,
            action="export_embedded_data",
            output_dir=output_dir,
            bundle=bundle,
            editor_path=None,
            psd_path=None,
            fallback_on_nonzero=True,
        )
        if adapter_result is not None:
            return adapter_result

        artifact_path = ArtifactStore(output_dir).write_json(
            "native_gui_export_request.json",
            {
                "output_dir": str(output_dir),
                "model_name": bundle.get("model_name"),
                "dispatch_kind": "desktop_intent",
            },
        )
        return {
            "source_action": "export_embedded_data",
            "status": "recorded",
            "details": "Recorded a native GUI export request.",
            "artifact_path": str(artifact_path),
        }

    def _execute_local_validation(self, output_dir: Path, bundle: DispatchBundle) -> JsonDict:
        model_name = str(bundle.get("model_name", "ATRI"))
        validator = CubismExportValidator()
        result = validator.validate(str(output_dir), model_name)
        artifact_path = ArtifactStore(output_dir).write_json(
            "native_gui_validate_export_result.json", result
        )
        return {
            "source_action": "validate_export_bundle",
            "status": "success" if result.get("status") == "success" else "error",
            "details": (
                "Validated the exported Cubism bundle."
                if result.get("status") == "success"
                else "Cubism export validation failed."
            ),
            "artifact_path": str(artifact_path),
            "validation": result,
        }

    def write_dispatch_bundle(
        self, bundle: Mapping[str, object], output_dir: str, model_name: str
    ) -> str:
        artifact_store = ArtifactStore(output_dir)
        bundle_path = artifact_store.write_json(
            f"{model_name}_cubism_dispatch_bundle.json",
            cast(JsonDict, bundle),
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
