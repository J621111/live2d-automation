from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp_server.cubism_contracts import (
    DispatchBundle,
    DispatchExecution,
    ExecutionPreparation,
)
from mcp_server.tools.cubism_automation import CubismAutomationManager

if TYPE_CHECKING:

    def _assert_manager_contracts(
        manager: CubismAutomationManager,
        preparation: ExecutionPreparation,
        bundle: DispatchBundle,
        execution: DispatchExecution,
    ) -> None:
        typed_preparation: ExecutionPreparation = manager.prepare_execution(
            "native_gui",
            editor_info={"status": "available"},
            plan={"steps": []},
        )
        typed_bundle: DispatchBundle = manager.build_dispatch_bundle(
            "native_gui",
            plan={"steps": []},
            execution=preparation,
            template_id="standard_bust_up",
            model_name="ATRI",
            psd_path="/tmp/ATRI.psd",
            output_dir="/tmp/output",
            editor_info={"status": "available"},
        )
        typed_execution: DispatchExecution = manager.execute_dispatch_bundle(
            bundle,
            previous_execution=execution,
        )
        assert typed_preparation and typed_bundle and typed_execution


def test_dispatch_contracts_remain_json_compatible() -> None:
    bundle: DispatchBundle = {
        "status": "ready",
        "backend": "native_gui",
        "automation_mode": "assisted",
        "ready_to_execute": True,
        "execution_supported": True,
        "template_id": "standard_bust_up",
        "model_name": "ATRI",
        "psd_path": "/tmp/ATRI.psd",
        "output_dir": "/tmp/output",
        "editor": {"status": "available", "editor_path": "/opt/CubismEditor"},
        "integration_target": None,
        "native_controller": {"status": "ready"},
        "native_adapter": None,
        "preflight": {"commands": [], "results": []},
        "dispatch_steps": [],
        "warnings": [],
    }
    execution: DispatchExecution = {
        "status": "success",
        "backend": "native_gui",
        "executed_steps": [],
        "artifacts": [],
        "resume": {
            "requested": False,
            "skipped_actions": [],
            "previous_successes": [],
            "cumulative_successes": [],
            "window_probe": None,
        },
        "message": "completed",
    }

    assert json.loads(json.dumps(bundle))["backend"] == "native_gui"
    assert json.loads(json.dumps(execution))["status"] == "success"
