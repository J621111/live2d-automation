"""Compatibility facade for Cubism automation services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast, overload

from mcp_server.cubism_contracts import (
    AutomationPlan,
    DispatchBundle,
    DispatchExecution,
    EditorInfo,
    ExecutionPreparation,
)
from mcp_server.tools import cubism_calibration
from mcp_server.tools.cubism_dispatch import CubismDispatchExecutor
from mcp_server.tools.cubism_preparation import BackendDescriptor, CubismPreparationService
from mcp_server.tools.native_gui_controller import NativeWindowsGUIController

JsonDict = dict[str, Any]


class CubismAutomationManager:
    """Preserve the public automation API while delegating service responsibilities."""

    def __init__(self) -> None:
        native_controller = NativeWindowsGUIController()
        self._preparation_service = CubismPreparationService(native_controller)
        self._dispatch_executor = CubismDispatchExecutor(
            native_controller,
            self._preparation_service,
        )

    def available_backends(self) -> list[str]:
        return self._preparation_service.available_backends()

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
        return self._dispatch_executor.build_dispatch_bundle(
            backend_name,
            plan=typed_plan,
            execution=typed_execution,
            template_id=template_id,
            model_name=model_name,
            psd_path=psd_path,
            output_dir=output_dir,
            editor_info=typed_editor_info,
        )

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
        return self._dispatch_executor.execute_dispatch_bundle(
            typed_bundle,
            previous_execution=typed_previous_execution,
            resume=resume,
        )

    def write_dispatch_execution(
        self, execution: JsonDict, output_dir: str, model_name: str, suffix: str = ""
    ) -> str:
        return self._dispatch_executor.write_dispatch_execution(
            execution,
            output_dir,
            model_name,
            suffix,
        )

    def build_profile_calibration_report(
        self,
        bundle: JsonDict,
        execution: JsonDict,
    ) -> JsonDict:
        return cubism_calibration.build_profile_calibration_report(bundle, execution)

    def write_profile_calibration_report(
        self, report: JsonDict, output_dir: str, model_name: str, suffix: str = ""
    ) -> str:
        return cubism_calibration.write_profile_calibration_report(
            report,
            output_dir,
            model_name,
            suffix=suffix,
        )

    def write_dispatch_bundle(
        self, bundle: Mapping[str, object], output_dir: str, model_name: str
    ) -> str:
        return self._dispatch_executor.write_dispatch_bundle(bundle, output_dir, model_name)

    @overload
    def prepare_execution(
        self,
        backend_name: str | None,
        *,
        editor_info: EditorInfo,
        plan: AutomationPlan,
        run_preflight: bool = False,
    ) -> ExecutionPreparation: ...

    @overload
    def prepare_execution(
        self,
        backend_name: str | None,
        *,
        editor_info: Mapping[str, object],
        plan: Mapping[str, object],
        run_preflight: bool = False,
    ) -> Any: ...

    def prepare_execution(
        self,
        backend_name: str | None,
        *,
        editor_info: object,
        plan: object,
        run_preflight: bool = False,
    ) -> object:
        typed_editor_info = cast(EditorInfo, editor_info)
        typed_plan = cast(AutomationPlan, plan)
        return self._preparation_service.prepare_execution(
            backend_name,
            editor_info=typed_editor_info,
            plan=typed_plan,
            run_preflight=run_preflight,
        )
