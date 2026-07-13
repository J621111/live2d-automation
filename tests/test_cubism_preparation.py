from __future__ import annotations

import pytest

from mcp_server.tools.cubism_preparation import CubismPreparationService
from mcp_server.tools.native_gui_controller import NativeWindowsGUIController


def test_preparation_service_resolves_catalog_and_native_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIVE2D_CUBISM_AUTOMATION_BACKEND", raising=False)
    monkeypatch.delenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND", raising=False)
    monkeypatch.setenv("LIVE2D_NATIVE_GUI_CONTROLLER_MODE", "disabled")
    service = CubismPreparationService(NativeWindowsGUIController())

    descriptor = service.resolve_backend()
    preparation = service.prepare_execution(
        descriptor.name,
        editor_info={"status": "available", "editor_path": "CubismEditor5.exe"},
        plan={"steps": [{"step": 1, "action": "launch_editor"}]},
    )

    assert service.available_backends() == ["native_gui", "opencli"]
    assert descriptor.name == "native_gui"
    assert preparation["status"] == "ready"
    assert preparation["backend"] == "native_gui"
    assert preparation["automation_mode"] == "assisted"
    assert preparation["missing_requirements"] == []
    assert preparation["native_controller"]["status"] == "disabled"
    assert preparation["plan_actions"] == ["launch_editor"]
