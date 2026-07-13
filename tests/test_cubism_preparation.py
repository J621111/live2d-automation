from __future__ import annotations

import subprocess
from pathlib import Path

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


def test_opencli_preflight_reports_wrapper_command_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "uvx"
    launcher.write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENCLI_COMMAND", f'"{launcher}" opencli run cubism')
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0 if argv[-1] == "doctor" else 3,
            stdout="healthy" if argv[-1] == "doctor" else "",
            stderr="connector unavailable" if argv[-1] == "list" else "",
        )

    monkeypatch.setattr("mcp_server.tools.cubism_preparation.subprocess.run", fake_run)
    service = CubismPreparationService(NativeWindowsGUIController())

    preparation = service.prepare_execution(
        "opencli",
        editor_info={"status": "available", "editor_path": "CubismEditor5.exe"},
        plan={"steps": []},
        run_preflight=True,
    )

    assert calls == [
        [str(launcher), "opencli", "doctor"],
        [str(launcher), "opencli", "list"],
    ]
    assert preparation["invocation_prefix"] == [str(launcher), "opencli"]
    assert [result["status"] for result in preparation["preflight_results"]] == [
        "success",
        "error",
    ]
    assert preparation["missing_requirements"] == [
        "dispatch_execution",
        "opencli_runtime",
    ]
    assert any("preflight commands failed" in warning for warning in preparation["warnings"])


def test_opencli_preflight_rejects_non_opencli_wrapper_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "uvx"
    launcher.write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENCLI_COMMAND", f'"{launcher}" not-opencli run cubism')

    def unexpected_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("Invalid wrapper targets must not run preflight commands.")

    monkeypatch.setattr("mcp_server.tools.cubism_preparation.subprocess.run", unexpected_run)
    service = CubismPreparationService(NativeWindowsGUIController())

    preparation = service.prepare_execution(
        "opencli",
        editor_info={"status": "available", "editor_path": "CubismEditor5.exe"},
        plan={"steps": []},
        run_preflight=True,
    )

    assert preparation["preflight_commands"] == []
    assert preparation["preflight_results"] == []
    assert preparation["missing_requirements"] == [
        "dispatch_execution",
        "opencli_runtime",
    ]
    assert any(
        "wrapper must target the exact opencli package" in warning
        for warning in preparation["warnings"]
    )
