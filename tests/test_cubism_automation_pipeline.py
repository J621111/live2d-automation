from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mcp_server.secure_server_impl import (
    analyze_photo,
    build_cubism_psd,
    close_session,
    execute_cubism_dispatch,
    export_model,
    generate_layers,
    prepare_cubism_automation,
    resume_cubism_dispatch,
    validate_cubism_export,
)
from mcp_server.tools.cubism_automation import CubismAutomationManager
from mcp_server.tools.native_gui_controller import NativeWindowsGUIController


def _create_sample_character_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (768, 1024), (245, 247, 252, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((220, 340, 548, 980), radius=48, fill=(72, 114, 196, 255))
    draw.ellipse((244, 84, 524, 380), fill=(255, 224, 197, 255))
    draw.ellipse((296, 178, 356, 238), fill=(255, 255, 255, 255))
    draw.ellipse((412, 178, 472, 238), fill=(255, 255, 255, 255))
    draw.ellipse((318, 198, 340, 220), fill=(48, 67, 110, 255))
    draw.ellipse((434, 198, 456, 220), fill=(48, 67, 110, 255))
    draw.arc((332, 248, 436, 314), start=15, end=165, fill=(180, 82, 102, 255), width=5)
    draw.rectangle((250, 118, 518, 164), fill=(34, 45, 92, 255))
    image.save(path, format="PNG")
    return path


def _write_python_adapter(path: Path) -> Path:
    script = """import json
import os
import subprocess
import sys
from pathlib import Path

action = sys.argv[1]
args = sys.argv[2:]
parsed = {}
index = 0
while index < len(args):
    key = args[index]
    if key.startswith("--") and index + 1 < len(args):
        parsed[key[2:]] = args[index + 1]
        index += 2
    else:
        index += 1

if action == "export_embedded_data":
    output_dir = Path(parsed["output-dir"])
    model_name = parsed["model-name"]
    textures_dir = output_dir / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_name}.moc3").write_bytes(b"moc")
    (textures_dir / "texture_00.png").write_bytes(b"png")
    (output_dir / "model3.json").write_text(
        json.dumps({
            "FileReferences": {
                "Moc": f"{model_name}.moc3",
                "Textures": ["textures/texture_00.png"],
            }
        }),
        encoding="utf-8",
    )

supported_actions = {"launch_editor", "import_psd", "apply_template", "export_embedded_data"}
sys.exit(0 if action in supported_actions else 1)
"""
    path.write_text(script, encoding="utf-8")
    return path


def _write_command_script(path: Path, windows_lines: list[str], posix_lines: list[str]) -> Path:
    if os.name == "nt":
        script = "@echo off\n" + "\n".join(windows_lines) + "\n"
        target = path.with_suffix(".cmd")
        target.write_text(script, encoding="utf-8")
        return target

    script = "#!/usr/bin/env sh\nset -eu\n" + "\n".join(posix_lines) + "\n"
    target = path.with_suffix("")
    target.write_text(script, encoding="utf-8")
    target.chmod(0o755)
    return target


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    return _create_sample_character_image(tmp_path / "input_image" / "character.png")


@pytest.mark.asyncio
async def test_prepare_cubism_automation_writes_assisted_plan(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        layers_result = await generate_layers(session_id, str(tmp_path / "semantic_layers"))
        assert layers_result["status"] == "success"

        psd_result = await build_cubism_psd(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3Plan",
        )
        assert psd_result["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3Plan",
        )

        assert prepare_result["status"] == "blocked"
        assert prepare_result["editor"]["status"] == "missing"
        assert prepare_result["automation_backend"] == "native_gui"
        assert prepare_result["automation_mode"] == "assisted"
        assert prepare_result["missing_requirements"] == ["cubism_editor"]
        assert len(prepare_result["steps"]) == 5
        assert prepare_result["steps"][1]["action"] == "import_psd"

        plan_path = Path(prepare_result["plan_path"])
        dispatch_bundle_path = Path(prepare_result["dispatch_bundle_path"])
        assert plan_path.exists()
        assert dispatch_bundle_path.exists()
        assert "Stage3Plan" in plan_path.name
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        dispatch_data = json.loads(dispatch_bundle_path.read_text(encoding="utf-8"))
        assert plan_data["automation_backend"] == "native_gui"
        assert plan_data["execution"]["missing_requirements"] == ["cubism_editor"]
        assert dispatch_data["backend"] == "native_gui"
        assert dispatch_data["dispatch_steps"][0]["dispatch_kind"] == "desktop_intent"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_accepts_explicit_editor_path(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3Ready",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3Ready",
            editor_path=str(fake_editor),
        )

        assert prepare_result["status"] == "ready"
        assert prepare_result["editor"]["status"] == "available"
        assert prepare_result["automation_backend"] == "native_gui"
        assert prepare_result["missing_requirements"] == []
        assert Path(prepare_result["editor"]["editor_path"]).name == "CubismEditor5.exe"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_runs_native_gui_poc(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    fake_editor = _write_command_script(
        tmp_path / "CubismEditor5",
        ["echo launched-native-gui", "exit /b 0"],
        ["echo launched-native-gui", "exit 0"],
    )

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage4NativeExec",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage4NativeExec",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "partial"
        assert Path(execute_result["execution_path"]).exists()
        assert Path(execute_result["calibration_report_path"]).exists()
        assert len(execute_result["executed_steps"]) >= 2
        launch_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "launch_editor"
        )
        import_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "import_psd"
        )
        assert launch_step["status"] == "success"
        assert import_step["status"] == "recorded"
        assert Path(launch_step["artifact_path"]).exists()
        assert Path(import_step["artifact_path"]).exists()

        execution_data = json.loads(
            Path(execute_result["execution_path"]).read_text(encoding="utf-8")
        )
        calibration_data = json.loads(
            Path(execute_result["calibration_report_path"]).read_text(encoding="utf-8")
        )
        assert execution_data["backend"] == "native_gui"
        assert execution_data["status"] == "partial"
        assert execution_data["executed_steps"][0]["source_action"] == "launch_editor"
        assert calibration_data["backend"] == "native_gui"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_uses_native_adapter_for_launch_and_import(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    fake_adapter = _write_command_script(
        tmp_path / "native_gui_adapter",
        [
            'if /I "%1"=="launch_editor" exit /b 0',
            'if /I "%1"=="import_psd" exit /b 0',
            "exit /b 64",
        ],
        [
            'if [ "$1" = "launch_editor" ]; then exit 0; fi',
            'if [ "$1" = "import_psd" ]; then exit 0; fi',
            "exit 64",
        ],
    )
    monkeypatch.setenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND", str(fake_adapter))

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage4AdapterExec",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage4AdapterExec",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "partial"
        launch_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "launch_editor"
        )
        import_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "import_psd"
        )
        assert launch_step["status"] == "success"
        assert import_step["status"] == "success"
        assert (
            Path(launch_step["artifact_path"]).name
            == "native_gui_launch_editor_adapter_result.json"
        )
        assert (
            Path(import_step["artifact_path"]).name == "native_gui_import_psd_adapter_result.json"
        )
        assert Path(launch_step["artifact_path"]).exists()
        assert Path(import_step["artifact_path"]).exists()

        execution_data = json.loads(
            Path(execute_result["execution_path"]).read_text(encoding="utf-8")
        )
        assert execution_data["status"] == "partial"
        assert execution_data["executed_steps"][0]["status"] == "success"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_treats_adapter_failures_as_errors(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    failing_adapter = _write_command_script(
        tmp_path / "native_gui_failing_adapter",
        [
            'if /I "%1"=="launch_editor" exit /b 0',
            'if /I "%1"=="import_psd" exit /b 0',
            'if /I "%1"=="apply_template" exit /b 2',
            'if /I "%1"=="export_embedded_data" exit /b 2',
            "exit /b 2",
        ],
        [
            'if [ "$1" = "launch_editor" ]; then exit 0; fi',
            'if [ "$1" = "import_psd" ]; then exit 0; fi',
            'if [ "$1" = "apply_template" ]; then exit 2; fi',
            'if [ "$1" = "export_embedded_data" ]; then exit 2; fi',
            "exit 2",
        ],
    )
    monkeypatch.setenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND", str(failing_adapter))

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage4AdapterFailure",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage4AdapterFailure",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "error"
        apply_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "apply_template"
        )
        assert apply_step["status"] == "error"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_with_bundled_demo_adapter(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    demo_adapter = Path(__file__).resolve().parents[1] / "scripts" / "native_gui_adapter_demo.py"
    monkeypatch.setenv(
        "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
        f'"{sys.executable}" "{demo_adapter}" --mode full',
    )

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        package_dir = tmp_path / "cubism_package"
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(package_dir),
                template_id="standard_bust_up",
                model_name="Stage4BundledDemo",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(package_dir),
            template_id="standard_bust_up",
            model_name="Stage4BundledDemo",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "success"
        assert (package_dir / "Stage4BundledDemo.moc3").exists()
        assert (package_dir / "model3.json").exists()
        assert (package_dir / "textures" / "texture_00.png").exists()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_completes_adapter_backed_flow(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    adapter_path = _write_python_adapter(tmp_path / "native_gui_adapter.py")
    monkeypatch.setenv(
        "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
        f'"{sys.executable}" "{adapter_path}"',
    )

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        package_dir = tmp_path / "cubism_package"
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(package_dir),
                template_id="standard_bust_up",
                model_name="Stage4AdapterFull",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(package_dir),
            template_id="standard_bust_up",
            model_name="Stage4AdapterFull",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "success"
        assert "completed" in execute_result["message"].lower()
        step_statuses = {
            step["source_action"]: step["status"] for step in execute_result["executed_steps"]
        }
        assert step_statuses == {
            "launch_editor": "success",
            "import_psd": "success",
            "apply_template": "success",
            "export_embedded_data": "success",
            "validate_export_bundle": "success",
        }
        assert (package_dir / "Stage4AdapterFull.moc3").exists()
        assert (package_dir / "model3.json").exists()
        assert (package_dir / "textures" / "texture_00.png").exists()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_resolves_builtin_controller(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    monkeypatch.setenv("LIVE2D_NATIVE_GUI_CONTROLLER_MODE", "dry_run")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage5BuiltinPrepare",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage5BuiltinPrepare",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"
        execution = json.loads(Path(prepare_result["plan_path"]).read_text(encoding="utf-8"))
        assert execution["execution"]["native_controller"]["status"] == "ready"
        assert execution["execution"]["native_controller"]["mode"] == "dry_run"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_uses_builtin_controller_dry_run(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    monkeypatch.setenv("LIVE2D_NATIVE_GUI_CONTROLLER_MODE", "dry_run")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        package_dir = tmp_path / "cubism_package"
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(package_dir),
                template_id="standard_bust_up",
                model_name="Stage5BuiltinDryRun",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(package_dir),
            template_id="standard_bust_up",
            model_name="Stage5BuiltinDryRun",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "partial"
        step_statuses = {
            step["source_action"]: step["status"] for step in execute_result["executed_steps"]
        }
        assert step_statuses["launch_editor"] == "recorded"
        assert step_statuses["import_psd"] == "recorded"
        assert step_statuses["apply_template"] == "recorded"
        assert step_statuses["export_embedded_data"] == "recorded"
        assert (package_dir / "native_gui_builtin_launch.ps1").exists()
        assert (package_dir / "native_gui_builtin_import.ps1").exists()
        assert (package_dir / "native_gui_builtin_apply_template.ps1").exists()
        assert (package_dir / "native_gui_builtin_export.ps1").exists()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_records_binary_launch_request(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage4BinaryRecord",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage4BinaryRecord",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "partial"
        assert "partial execution records" in execute_result["message"].lower()
        launch_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "launch_editor"
        )
        import_step = next(
            step
            for step in execute_result["executed_steps"]
            if step["source_action"] == "import_psd"
        )
        assert launch_step["status"] == "recorded"
        assert import_step["status"] == "recorded"
        assert Path(launch_step["artifact_path"]).name == "native_gui_launch_request.json"
        assert Path(launch_step["artifact_path"]).exists()

        execution_data = json.loads(
            Path(execute_result["execution_path"]).read_text(encoding="utf-8")
        )
        assert execution_data["status"] == "partial"
        launch_requests = [
            step
            for step in execution_data["executed_steps"]
            if step["source_action"] == "launch_editor"
        ]
        assert launch_requests[0]["status"] == "recorded"
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_blocks_not_ready_bundle(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage4BlockedExec",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage4BlockedExec",
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "blocked"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "blocked"
        assert "not ready_to_execute" in execute_result["message"].lower()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_execute_cubism_dispatch_blocks_opencli_backend(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    fake_opencli = _write_command_script(
        tmp_path / "opencli",
        [
            'if /I "%1"=="doctor" exit /b 0',
            'if /I "%1"=="list" exit /b 0',
            "exit /b 1",
        ],
        [
            'if [ "$1" = "doctor" ]; then exit 0; fi',
            'if [ "$1" = "list" ]; then exit 0; fi',
            "exit 1",
        ],
    )
    monkeypatch.setenv("OPENCLI_COMMAND", str(fake_opencli))

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage4OpenCLIExec",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage4OpenCLIExec",
            editor_path=str(fake_editor),
            automation_backend="opencli",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "blocked"
        assert "only the native_gui backend" in execute_result["message"].lower()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_supports_direct_opencli_backend(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    fake_opencli = _write_command_script(
        tmp_path / "opencli",
        [
            'if /I "%1"=="doctor" exit /b 0',
            'if /I "%1"=="list" exit /b 0',
            "exit /b 1",
        ],
        [
            'if [ "$1" = "doctor" ]; then exit 0; fi',
            'if [ "$1" = "list" ]; then exit 0; fi',
            "exit 1",
        ],
    )
    monkeypatch.setenv("OPENCLI_COMMAND", f"{fake_opencli} run cubism")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3OpenCLI",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3OpenCLI",
            editor_path=str(fake_editor),
            automation_backend="opencli",
        )

        assert prepare_result["status"] == "ready"
        assert prepare_result["automation_backend"] == "opencli"
        assert prepare_result["automation_mode"] == "connector_assisted"
        assert prepare_result["backend_capabilities"]
        assert prepare_result["missing_requirements"] == []

        plan_data = json.loads(Path(prepare_result["plan_path"]).read_text(encoding="utf-8"))
        dispatch_data = json.loads(
            Path(prepare_result["dispatch_bundle_path"]).read_text(encoding="utf-8")
        )
        assert plan_data["automation_backend"] == "opencli"
        assert plan_data["execution"]["integration_target"] == "jackwener/opencli"
        assert plan_data["execution"]["command_hint"] == f"{fake_opencli} run cubism"
        assert plan_data["execution"]["resolved_executable"] == str(fake_opencli.resolve())
        assert plan_data["execution"]["argv"][0] == str(fake_opencli)
        assert plan_data["execution"]["invocation_prefix"] == [str(fake_opencli)]
        assert plan_data["execution"]["preflight_commands"][0]["argv"] == [
            str(fake_opencli),
            "doctor",
        ]
        assert [result["status"] for result in plan_data["execution"]["preflight_results"]] == [
            "success",
            "success",
        ]
        assert plan_data["execution"]["automation_mode"] == "connector_assisted"
        assert dispatch_data["backend"] == "opencli"
        assert dispatch_data["ready_to_execute"] is True
        assert dispatch_data["dispatch_steps"][0]["dispatch_kind"] == "connector_intent"
        assert dispatch_data["dispatch_steps"][1]["intent"].lower().startswith("use opencli")
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_supports_wrapped_opencli_backend(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    fake_uvx = _write_command_script(
        tmp_path / "uvx",
        [
            'if /I not "%1"=="opencli" exit /b 2',
            'if /I "%2"=="doctor" exit /b 0',
            'if /I "%2"=="list" exit /b 0',
            "exit /b 1",
        ],
        [
            'if [ "$1" != "opencli" ]; then exit 2; fi',
            'if [ "$2" = "doctor" ]; then exit 0; fi',
            'if [ "$2" = "list" ]; then exit 0; fi',
            "exit 1",
        ],
    )
    monkeypatch.setenv("OPENCLI_COMMAND", f"{fake_uvx} opencli run cubism")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3WrappedOpenCLI",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3WrappedOpenCLI",
            editor_path=str(fake_editor),
            automation_backend="opencli",
        )

        assert prepare_result["status"] == "ready"
        plan_data = json.loads(Path(prepare_result["plan_path"]).read_text(encoding="utf-8"))
        assert Path(prepare_result["dispatch_bundle_path"]).exists()
        assert plan_data["execution"]["resolved_executable"] == str(fake_uvx.resolve())
        assert plan_data["execution"]["invocation_prefix"] == [str(fake_uvx), "opencli"]
        assert plan_data["execution"]["preflight_commands"][1]["argv"] == [
            str(fake_uvx),
            "opencli",
            "list",
        ]
        assert [result["status"] for result in plan_data["execution"]["preflight_results"]] == [
            "success",
            "success",
        ]
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_blocks_unresolvable_opencli_command(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    monkeypatch.setenv("OPENCLI_COMMAND", "missing-opencli-command run cubism")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3OpenCLIBlocked",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3OpenCLIBlocked",
            editor_path=str(fake_editor),
            automation_backend="opencli",
        )

        assert prepare_result["status"] == "blocked"
        assert prepare_result["automation_backend"] == "opencli"
        assert prepare_result["missing_requirements"] == ["opencli_command", "opencli_runtime"]

        plan_data = json.loads(Path(prepare_result["plan_path"]).read_text(encoding="utf-8"))
        assert plan_data["execution"]["resolved_executable"] is None
        assert plan_data["execution"]["argv"][0] == "missing-opencli-command"
        assert plan_data["execution"]["preflight_commands"] == []
        assert plan_data["execution"]["warnings"]
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_blocks_non_opencli_wrapper(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    fake_uvx = _write_command_script(
        tmp_path / "uvx",
        [
            'if /I "%1"=="doctor" exit /b 0',
            'if /I "%1"=="list" exit /b 0',
            "exit /b 1",
        ],
        [
            'if [ "$1" = "doctor" ]; then exit 0; fi',
            'if [ "$1" = "list" ]; then exit 0; fi',
            "exit 1",
        ],
    )
    monkeypatch.setenv("OPENCLI_COMMAND", f"{fake_uvx} not-opencli run cubism")

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3OpenCLIMismatch",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3OpenCLIMismatch",
            editor_path=str(fake_editor),
            automation_backend="opencli",
        )

        assert prepare_result["status"] == "blocked"
        plan_data = json.loads(Path(prepare_result["plan_path"]).read_text(encoding="utf-8"))
        assert plan_data["execution"]["resolved_executable"] == str(fake_uvx.resolve())
        assert plan_data["execution"]["preflight_commands"] == []
        assert any(
            "wrapper must target the exact opencli package" in warning.lower()
            for warning in plan_data["execution"]["warnings"]
        )
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_blocks_opencli_preflight_failures(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    fake_opencli = _write_command_script(
        tmp_path / "opencli",
        [
            'if /I "%1"=="doctor" exit /b 0',
            'if /I "%1"=="list" exit /b 3',
            "exit /b 1",
        ],
        [
            'if [ "$1" = "doctor" ]; then exit 0; fi',
            'if [ "$1" = "list" ]; then exit 3; fi',
            "exit 1",
        ],
    )
    monkeypatch.setenv("OPENCLI_COMMAND", str(fake_opencli))

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3OpenCLIPreflightFail",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3OpenCLIPreflightFail",
            editor_path=str(fake_editor),
            automation_backend="opencli",
        )

        assert prepare_result["status"] == "blocked"
        assert prepare_result["missing_requirements"] == ["opencli_runtime"]

        plan_data = json.loads(Path(prepare_result["plan_path"]).read_text(encoding="utf-8"))
        results = {item["name"]: item for item in plan_data["execution"]["preflight_results"]}
        assert results["doctor"]["status"] == "success"
        assert results["list"]["status"] == "error"
        assert any(
            "preflight commands failed" in warning.lower()
            for warning in plan_data["execution"]["warnings"]
        )
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_rejects_unknown_backend(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(tmp_path / "cubism_package"),
                template_id="standard_bust_up",
                model_name="Stage3BadBackend",
            )
        )["status"] == "success"

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3BadBackend",
            automation_backend="unknown_backend",
        )

        assert prepare_result["status"] == "error"
        assert prepare_result["error_code"] == "invalid_input"
        assert "unsupported automation backend" in prepare_result["message"].lower()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_prepare_cubism_automation_rejects_missing_psd_package(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        psd_result = await build_cubism_psd(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3MissingPSD",
        )
        assert psd_result["status"] == "success"

        Path(psd_result["psd_path"]).unlink()
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage3MissingPSD",
        )

        assert prepare_result["status"] == "error"
        assert prepare_result["error_code"] == "invalid_input"
        assert "missing" in prepare_result["message"].lower()
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_validate_cubism_export_accepts_mock_export_bundle(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        layers_result = await generate_layers(session_id, str(tmp_path / "semantic_layers"))
        assert layers_result["status"] == "success"

        export_result = await export_model(
            session_id,
            str(tmp_path / "mock_export"),
            model_name="Stage3Export",
        )
        assert export_result["status"] == "success"

        validate_result = await validate_cubism_export(
            str(tmp_path / "mock_export"),
            model_name="Stage3Export",
        )
        assert validate_result["status"] == "success"
        assert validate_result["missing"] == []
        assert validate_result["errors"] == []
        assert validate_result["checks"]["moc_reference"] == "Stage3Export.moc3"
        assert validate_result["checks"]["texture_count"] >= 1
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_validate_cubism_export_rejects_broken_model_references(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await export_model(
                session_id,
                str(tmp_path / "mock_export"),
                model_name="Stage3Broken",
            )
        )["status"] == "success"

        model3_path = tmp_path / "mock_export" / "model3.json"
        model3_data = json.loads(model3_path.read_text(encoding="utf-8"))
        model3_data["FileReferences"]["Moc"] = "wrong_name.moc3"
        if model3_data["FileReferences"]["Textures"]:
            broken_texture = model3_data["FileReferences"]["Textures"][0]
            (tmp_path / "mock_export" / broken_texture).unlink()
        model3_path.write_text(json.dumps(model3_data, indent=2), encoding="utf-8")

        validate_result = await validate_cubism_export(
            str(tmp_path / "mock_export"),
            model_name="Stage3Broken",
        )
        assert validate_result["status"] == "error"
        assert validate_result["errors"]
        assert any("expected moc3" in error.lower() for error in validate_result["errors"])
        assert any(
            "missing referenced texture" in error.lower() for error in validate_result["errors"]
        )
    finally:
        await close_session(session_id)


def test_native_gui_controller_captures_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_failure"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "export_shortcut": "^+e",
            "activation_wait_seconds": 0.0,
            "export_dialog_wait_seconds": 0.0,
            "capture_screenshot_on_error": True,
            "failure_capture_wait_seconds": 0.0,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name.endswith("_failure_capture.ps1"):
            screenshot_path = (
                output_dir / "native_gui_builtin_export_embedded_data_failure_capture.png"
            )
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"png")
            return subprocess.CompletedProcess([str(script_path)], 0, "captured", "")
        return subprocess.CompletedProcess([str(script_path)], 1, "", "export failed")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_export(controller_state, output_dir, "FailureCase")

    assert result["status"] == "error"
    failure_capture = result["failure_capture"]
    assert failure_capture["status"] == "success"
    assert Path(failure_capture["artifact_path"]).exists()
    assert Path(failure_capture["script_path"]).exists()
    assert Path(failure_capture["screenshot_path"]).exists()

    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["returncode"] == 1
    assert payload["failure_capture"]["status"] == "success"


def test_native_gui_controller_can_disable_failure_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_failure_disabled"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "template_shortcut": "^+t",
            "activation_wait_seconds": 0.0,
            "template_dialog_wait_seconds": 0.0,
            "capture_screenshot_on_error": False,
        },
    }

    monkeypatch.setattr(
        controller,
        "_run_powershell",
        lambda script_path: subprocess.CompletedProcess([str(script_path)], 2, "", "apply failed"),
    )

    result = controller.execute_apply_template(controller_state, "standard_bust_up", output_dir)

    assert result["status"] == "error"
    failure_capture = result["failure_capture"]
    assert failure_capture["status"] == "disabled"
    assert "artifact_path" not in failure_capture
    assert not list(output_dir.glob("*failure_capture*"))


def test_native_gui_controller_embeds_known_dialog_sequences(tmp_path: Path) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_dialog_sequences"
    controller_state = {
        "status": "ready",
        "mode": "dry_run",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "template_shortcut": "^+t",
            "export_shortcut": "^+e",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "template_dialog_wait_seconds": 0.0,
            "export_dialog_wait_seconds": 0.0,
            "known_dialog_sequences": {
                "import_psd": [{"keys": "{ENTER}", "wait_seconds": 0.1}],
                "apply_template": [{"keys": "%y", "wait_seconds": 0.2}],
                "export_embedded_data": [{"keys": "{TAB}{ENTER}", "wait_seconds": 0.3}],
            },
        },
    }

    controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)
    controller.execute_apply_template(controller_state, "standard_bust_up", output_dir)
    controller.execute_export(controller_state, output_dir, "DialogCase")

    import_script = (output_dir / "native_gui_builtin_import.ps1").read_text(encoding="utf-8")
    template_script = (output_dir / "native_gui_builtin_apply_template.ps1").read_text(
        encoding="utf-8"
    )
    export_script = (output_dir / "native_gui_builtin_export.ps1").read_text(encoding="utf-8")

    assert "Start-Sleep -Milliseconds 100" in import_script
    assert '$wshell.SendKeys("{ENTER}")' in import_script
    assert "Start-Sleep -Milliseconds 200" in template_script
    assert '$wshell.SendKeys("%y")' in template_script
    assert "Start-Sleep -Milliseconds 300" in export_script
    assert '$wshell.SendKeys("{TAB}{ENTER}")' in export_script


def test_native_gui_controller_retries_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_retry"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
        },
    }
    attempts = {"count": 0}

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return subprocess.CompletedProcess([str(script_path)], 1, "", "first failure")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    assert result["status"] == "success"
    assert result["attempt_count"] == 2
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["attempt_count"] == 2
    assert payload["attempts"][0]["returncode"] == 1
    assert payload["attempts"][1]["returncode"] == 0


@pytest.mark.asyncio
async def test_resume_cubism_dispatch_skips_previous_successes(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    partial_adapter = _write_command_script(
        tmp_path / "native_gui_partial_resume_adapter",
        [
            'if /I "%1"=="launch_editor" exit /b 0',
            'if /I "%1"=="import_psd" exit /b 0',
            'if /I "%1"=="apply_template" exit /b 64',
            'if /I "%1"=="export_embedded_data" exit /b 64',
            'if /I "%1"=="validate_export_bundle" exit /b 64',
            "exit /b 64",
        ],
        [
            'if [ "$1" = "launch_editor" ]; then exit 0; fi',
            'if [ "$1" = "import_psd" ]; then exit 0; fi',
            'if [ "$1" = "apply_template" ]; then exit 64; fi',
            'if [ "$1" = "export_embedded_data" ]; then exit 64; fi',
            'if [ "$1" = "validate_export_bundle" ]; then exit 64; fi',
            "exit 64",
        ],
    )
    full_adapter = _write_python_adapter(tmp_path / "native_gui_resume_adapter.py")
    monkeypatch.setenv(
        "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
        f'"{partial_adapter}"',
    )

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        package_dir = tmp_path / "cubism_package"
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(package_dir),
                template_id="standard_bust_up",
                model_name="Stage6Resume",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(package_dir),
            template_id="standard_bust_up",
            model_name="Stage6Resume",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "partial"
        first_step_statuses = {
            step["source_action"]: step["status"] for step in execute_result["executed_steps"]
        }
        assert first_step_statuses["launch_editor"] == "success"
        assert first_step_statuses["import_psd"] == "success"
        assert first_step_statuses["apply_template"] == "recorded"

        monkeypatch.setenv(
            "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
            f'"{sys.executable}" "{full_adapter}"',
        )
        resume_result = await resume_cubism_dispatch(session_id)
        assert resume_result["status"] == "success"
        resume_statuses = {
            step["source_action"]: step["status"] for step in resume_result["executed_steps"]
        }
        assert resume_statuses["launch_editor"] == "success"
        assert resume_statuses["import_psd"] == "success"
        assert resume_statuses["apply_template"] == "success"
        assert resume_statuses["export_embedded_data"] == "success"
        assert resume_statuses["validate_export_bundle"] == "success"
        assert resume_result["resume"]["requested"] is True
        assert resume_result["resume"]["skipped_actions"] == []
        assert Path(resume_result["execution_path"]).name.endswith("_resume.json")
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_resume_cubism_dispatch_preserves_cumulative_successes(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    partial_adapter = _write_command_script(
        tmp_path / "native_gui_partial_resume_cumulative_adapter",
        [
            'if /I "%1"=="launch_editor" exit /b 0',
            'if /I "%1"=="import_psd" exit /b 0',
            'if /I "%1"=="apply_template" exit /b 64',
            'if /I "%1"=="export_embedded_data" exit /b 64',
            'if /I "%1"=="validate_export_bundle" exit /b 64',
            "exit /b 64",
        ],
        [
            'if [ "$1" = "launch_editor" ]; then exit 0; fi',
            'if [ "$1" = "import_psd" ]; then exit 0; fi',
            'if [ "$1" = "apply_template" ]; then exit 64; fi',
            'if [ "$1" = "export_embedded_data" ]; then exit 64; fi',
            'if [ "$1" = "validate_export_bundle" ]; then exit 64; fi',
            "exit 64",
        ],
    )
    full_adapter = _write_python_adapter(tmp_path / "native_gui_resume_cumulative_adapter.py")
    monkeypatch.setenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND", f'"{partial_adapter}"')

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        package_dir = tmp_path / "cubism_package"
        assert (await generate_layers(session_id, str(tmp_path / "semantic_layers")))[
            "status"
        ] == "success"
        assert (
            await build_cubism_psd(
                session_id,
                str(package_dir),
                template_id="standard_bust_up",
                model_name="Stage6ResumeAgain",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(package_dir),
            template_id="standard_bust_up",
            model_name="Stage6ResumeAgain",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        first_result = await execute_cubism_dispatch(session_id)
        assert first_result["status"] == "partial"

        monkeypatch.setenv(
            "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
            f'"{sys.executable}" "{full_adapter}"',
        )
        second_result = await resume_cubism_dispatch(session_id)
        assert second_result["status"] == "success"
        assert sorted(second_result["resume"]["cumulative_successes"]) == [
            "apply_template",
            "export_embedded_data",
            "import_psd",
            "launch_editor",
            "validate_export_bundle",
        ]

        monkeypatch.setenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND", f'"{partial_adapter}"')
        third_result = await resume_cubism_dispatch(session_id)
        assert third_result["status"] == "success"
        third_statuses = {
            step["source_action"]: step["status"] for step in third_result["executed_steps"]
        }
        assert third_statuses["launch_editor"] == "success"
        assert third_statuses["import_psd"] == "success"
        assert third_statuses["apply_template"] == "skipped"
        assert third_statuses["export_embedded_data"] == "skipped"
        assert third_statuses["validate_export_bundle"] == "skipped"
        assert sorted(third_result["resume"]["skipped_actions"]) == [
            "apply_template",
            "export_embedded_data",
            "validate_export_bundle",
        ]
    finally:
        await close_session(session_id)


def test_native_gui_controller_retries_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_timeout_retry"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
        },
    }
    attempts = {"count": 0}

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise subprocess.TimeoutExpired([str(script_path)], 10)
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    assert result["status"] == "success"
    assert result["timed_out"] is False
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["attempt_count"] == 2
    assert payload["attempts"][0]["timeout"] is True
    assert payload["attempts"][1]["returncode"] == 0


def test_native_gui_controller_captures_failure_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_timeout_failure"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "export_shortcut": "^+e",
            "activation_wait_seconds": 0.0,
            "export_dialog_wait_seconds": 0.0,
            "retry_attempts": 0,
            "capture_screenshot_on_error": True,
            "failure_capture_wait_seconds": 0.0,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name.endswith("_failure_capture.ps1"):
            screenshot_path = (
                output_dir / "native_gui_builtin_export_embedded_data_failure_capture.png"
            )
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"png")
            return subprocess.CompletedProcess([str(script_path)], 0, "captured", "")
        raise subprocess.TimeoutExpired([str(script_path)], 10)

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_export(controller_state, output_dir, "TimeoutCase")

    assert result["status"] == "error"
    assert result["timed_out"] is True
    failure_capture = result["failure_capture"]
    assert failure_capture["status"] == "success"
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["timed_out"] is True
    assert payload["attempts"][0]["timeout"] is True


def test_native_gui_controller_marks_export_missing_outputs_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_export_missing_outputs"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "export_dialog_wait_seconds": 0.0,
            "export_output_timeout_seconds": 0.0,
            "export_output_poll_seconds": 0.05,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name == "native_gui_builtin_export.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "", "")
        if script_path.name == "native_gui_builtin_export_embedded_data_failure_capture.ps1":
            screenshot_path = (
                output_dir / "native_gui_builtin_export_embedded_data_failure_capture.png"
            )
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"png")
            return subprocess.CompletedProcess([str(script_path)], 0, "captured", "")
        return subprocess.CompletedProcess([str(script_path)], 0, "", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_export(controller_state, output_dir, "MissingExport")

    assert result["status"] == "error"
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["returncode"] == 2
    assert payload["post_success_check"]["status"] == "error"
    assert payload["post_success_check"]["missing"] == ["moc3", "model3", "textures_dir"]
    assert payload["failure_capture"]["status"] == "success"


def test_native_gui_controller_records_success_probe_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    psd_path = tmp_path / "input.psd"
    psd_path.write_bytes(b"psd")
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "failure_capture_wait_seconds": 0.0,
            "success_capture_wait_seconds": 0.0,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name == "native_gui_builtin_import.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "", "")
        if script_path.name == "native_gui_builtin_import_psd_post_probe.ps1":
            stdout = json.dumps(
                {
                    "target": "Cubism Editor",
                    "matched_titles": ["Live2D Cubism Editor 5.3.01    [ FREE version ]  -"],
                    "all_titles": ["Live2D Cubism Editor 5.3.01    [ FREE version ]  -"],
                }
            )
            return subprocess.CompletedProcess([str(script_path)], 0, stdout, "")
        if script_path.name == "native_gui_builtin_import_psd_success_capture.ps1":
            screenshot_path = output_dir / "native_gui_builtin_import_psd_success_capture.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"png")
            return subprocess.CompletedProcess([str(script_path)], 0, "captured", "")
        return subprocess.CompletedProcess([str(script_path)], 0, "", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, psd_path, output_dir)

    assert result["status"] == "success"
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["post_action_probe"]["status"] == "success"
    assert payload["post_action_probe"]["matched_titles"] == [
        "Live2D Cubism Editor 5.3.01    [ FREE version ]  -"
    ]
    assert payload["success_capture"]["status"] == "success"
    assert Path(payload["success_capture"]["artifact_path"]).exists()
    assert Path(payload["success_capture"]["screenshot_path"]).exists()


def test_native_gui_controller_import_can_use_launch_argument_and_title_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    psd_path = tmp_path / "ATRI.psd"
    psd_path.write_bytes(b"psd")
    editor_path = tmp_path / "CubismEditor5.exe"
    editor_path.write_text("stub", encoding="utf-8")
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "import_via_launch_argument": True,
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "document_window_timeout_seconds": 0.1,
            "document_window_poll_seconds": 0.05,
            "failure_capture_wait_seconds": 0.0,
            "success_capture_wait_seconds": 0.0,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name == "native_gui_builtin_import.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "", "")
        if script_path.name == "native_gui_builtin_import_psd_title_check.ps1":
            stdout = json.dumps(
                {
                    "target": "Cubism Editor",
                    "matched_titles": ["Live2D Cubism Editor 5.3.01    [ FREE version ]  - ATRI"],
                    "all_titles": ["Live2D Cubism Editor 5.3.01    [ FREE version ]  - ATRI"],
                    "all_diagnostics": [
                        {
                            "ProcessName": "java",
                            "Title": "Live2D Cubism Editor 5.3.01    [ FREE version ]  - ATRI",
                        }
                    ],
                }
            )
            return subprocess.CompletedProcess([str(script_path)], 0, stdout, "")
        if script_path.name == "native_gui_builtin_import_psd_post_probe.ps1":
            stdout = json.dumps(
                {
                    "target": "Cubism Editor",
                    "matched_titles": ["Live2D Cubism Editor 5.3.01    [ FREE version ]  - ATRI"],
                    "all_titles": ["Live2D Cubism Editor 5.3.01    [ FREE version ]  - ATRI"],
                }
            )
            return subprocess.CompletedProcess([str(script_path)], 0, stdout, "")
        if script_path.name == "native_gui_builtin_import_psd_success_capture.ps1":
            screenshot_path = output_dir / "native_gui_builtin_import_psd_success_capture.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"png")
            return subprocess.CompletedProcess([str(script_path)], 0, "captured", "")
        return subprocess.CompletedProcess([str(script_path)], 0, "", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(
        controller_state,
        psd_path,
        output_dir,
        editor_path=editor_path,
    )

    assert result["status"] == "success"
    script = (output_dir / "native_gui_builtin_import.ps1").read_text(encoding="utf-8")
    assert f'Start-Process -FilePath "{editor_path}"' in script
    assert str(psd_path) in script
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["post_success_check"]["status"] == "success"
    assert payload["post_success_check"]["matched_titles"] == [
        "Live2D Cubism Editor 5.3.01    [ FREE version ]  - ATRI"
    ]


def test_native_gui_controller_import_title_check_ignores_non_cubism_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    psd_path = tmp_path / "ATRI.psd"
    psd_path.write_bytes(b"psd")
    editor_path = tmp_path / "CubismEditor5.exe"
    editor_path.write_text("stub", encoding="utf-8")
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "import_via_launch_argument": True,
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "document_window_timeout_seconds": 0.1,
            "document_window_poll_seconds": 0.05,
            "failure_capture_wait_seconds": 0.0,
            "success_capture_wait_seconds": 0.0,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name == "native_gui_builtin_import.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "", "")
        if script_path.name == "native_gui_builtin_import_psd_title_check.ps1":
            stdout = json.dumps(
                {
                    "target": "Cubism Editor",
                    "all_titles": ["atri dev"],
                    "all_diagnostics": [{"ProcessName": "QQ", "Title": "atri dev"}],
                }
            )
            return subprocess.CompletedProcess([str(script_path)], 0, stdout, "")
        if script_path.name == "native_gui_builtin_import_psd_failure_capture.ps1":
            screenshot_path = output_dir / "native_gui_builtin_import_psd_failure_capture.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"png")
            return subprocess.CompletedProcess([str(script_path)], 0, "captured", "")
        return subprocess.CompletedProcess([str(script_path)], 0, "", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(
        controller_state,
        psd_path,
        output_dir,
        editor_path=editor_path,
    )

    assert result["status"] == "error"
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["post_success_check"]["status"] == "error"
    assert payload["failure_capture"]["status"] == "success"


def test_resume_requires_live_window_probe_before_skipping_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = CubismAutomationManager()
    bundle = {
        "backend": "native_gui",
        "ready_to_execute": True,
        "output_dir": str(tmp_path),
        "model_name": "ProbeCase",
        "template_id": "standard_bust_up",
        "editor": {"editor_path": str(tmp_path / "CubismEditor5.exe")},
        "native_controller": {"status": "ready", "mode": "execute", "profile": {}},
        "native_adapter": {"status": "disabled"},
        "dispatch_steps": [
            {"step": 1, "source_action": "launch_editor"},
            {"step": 2, "source_action": "import_psd"},
            {"step": 3, "source_action": "apply_template"},
            {"step": 4, "source_action": "export_embedded_data"},
            {"step": 5, "source_action": "validate_export_bundle"},
        ],
    }
    previous_execution = {
        "executed_steps": [
            {"source_action": "launch_editor", "status": "success"},
            {"source_action": "import_psd", "status": "success"},
        ],
        "resume": {"cumulative_successes": ["launch_editor", "import_psd"]},
    }

    monkeypatch.setattr(
        manager,
        "_probe_native_resume_window",
        lambda native_controller, output_dir: {
            "status": "error",
            "artifact_path": str(tmp_path / "probe.json"),
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_launch",
        lambda editor_path, output_dir, native_controller, native_adapter, bundle: {
            "source_action": "launch_editor",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_import",
        lambda psd_path, output_dir, native_controller, native_adapter, bundle: {
            "source_action": "import_psd",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_apply_template",
        lambda output_dir, native_controller, native_adapter, bundle: {
            "source_action": "apply_template",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_export",
        lambda output_dir, native_controller, native_adapter, bundle: {
            "source_action": "export_embedded_data",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_local_validation",
        lambda output_dir, bundle: {"source_action": "validate_export_bundle", "status": "success"},
    )

    execution = manager.execute_dispatch_bundle(
        bundle,
        previous_execution=previous_execution,
        resume=True,
    )

    statuses = {step["source_action"]: step["status"] for step in execution["executed_steps"]}
    assert statuses["launch_editor"] == "success"
    assert statuses["import_psd"] == "success"
    assert execution["resume"]["window_probe"]["status"] == "error"


def test_execute_dispatch_defers_launch_when_import_uses_direct_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = CubismAutomationManager()
    output_dir = tmp_path / "dispatch"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "backend": "native_gui",
        "ready_to_execute": True,
        "output_dir": str(output_dir),
        "model_name": "DirectOpen",
        "template_id": "standard_bust_up",
        "editor": {"editor_path": str(tmp_path / "CubismEditor5.exe")},
        "psd_path": str(tmp_path / "DirectOpen.psd"),
        "native_controller": {
            "status": "ready",
            "mode": "execute",
            "profile": {"import_via_launch_argument": True},
        },
        "native_adapter": {"status": "disabled"},
        "dispatch_steps": [
            {"step": 1, "source_action": "launch_editor"},
            {"step": 2, "source_action": "import_psd"},
        ],
    }

    monkeypatch.setattr(
        manager,
        "_execute_native_import",
        lambda *args, **kwargs: {
            "source_action": "import_psd",
            "status": "success",
            "details": "imported",
        },
    )

    execution = manager.execute_dispatch_bundle(bundle)

    assert execution["executed_steps"][0]["source_action"] == "launch_editor"
    assert execution["executed_steps"][0]["status"] == "success"
    assert "deferred to import_psd" in execution["executed_steps"][0]["details"]
    assert execution["executed_steps"][1]["status"] == "success"


def test_execute_dispatch_halts_after_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = CubismAutomationManager()
    output_dir = tmp_path / "dispatch"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "backend": "native_gui",
        "ready_to_execute": True,
        "output_dir": str(output_dir),
        "model_name": "StopOnError",
        "template_id": "standard_bust_up",
        "editor": {"editor_path": str(tmp_path / "CubismEditor5.exe")},
        "psd_path": str(tmp_path / "StopOnError.psd"),
        "native_controller": {
            "status": "ready",
            "mode": "execute",
            "profile": {"import_via_launch_argument": True},
        },
        "native_adapter": {"status": "disabled"},
        "dispatch_steps": [
            {"step": 1, "source_action": "launch_editor"},
            {"step": 2, "source_action": "import_psd"},
            {"step": 3, "source_action": "apply_template"},
            {"step": 4, "source_action": "export_embedded_data"},
        ],
    }

    monkeypatch.setattr(
        manager,
        "_execute_native_import",
        lambda *args, **kwargs: {
            "source_action": "import_psd",
            "status": "error",
            "details": "import failed",
        },
    )

    execution = manager.execute_dispatch_bundle(bundle)

    assert execution["executed_steps"][1]["status"] == "error"
    assert execution["executed_steps"][2]["status"] == "pending"
    assert execution["executed_steps"][3]["status"] == "pending"
    assert "previous automation step failed" in execution["executed_steps"][2]["details"]


def test_resume_does_not_skip_prerequisites_when_controller_cannot_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = CubismAutomationManager()
    bundle = {
        "backend": "native_gui",
        "ready_to_execute": True,
        "output_dir": str(tmp_path),
        "model_name": "ProbeUnavailableCase",
        "template_id": "standard_bust_up",
        "editor": {"editor_path": str(tmp_path / "CubismEditor5.exe")},
        "native_controller": {"status": "missing", "mode": "disabled", "profile": {}},
        "native_adapter": {"status": "disabled"},
        "dispatch_steps": [
            {"step": 1, "source_action": "launch_editor"},
            {"step": 2, "source_action": "import_psd"},
            {"step": 3, "source_action": "apply_template"},
            {"step": 4, "source_action": "export_embedded_data"},
            {"step": 5, "source_action": "validate_export_bundle"},
        ],
    }
    previous_execution = {
        "executed_steps": [
            {"source_action": "launch_editor", "status": "success"},
            {"source_action": "import_psd", "status": "success"},
        ],
        "resume": {"cumulative_successes": ["launch_editor", "import_psd"]},
    }

    monkeypatch.setattr(
        manager,
        "_execute_native_launch",
        lambda editor_path, output_dir, native_controller, native_adapter, bundle: {
            "source_action": "launch_editor",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_import",
        lambda psd_path, output_dir, native_controller, native_adapter, bundle: {
            "source_action": "import_psd",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_apply_template",
        lambda output_dir, native_controller, native_adapter, bundle: {
            "source_action": "apply_template",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_native_export",
        lambda output_dir, native_controller, native_adapter, bundle: {
            "source_action": "export_embedded_data",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        manager,
        "_execute_local_validation",
        lambda output_dir, bundle: {"source_action": "validate_export_bundle", "status": "success"},
    )

    execution = manager.execute_dispatch_bundle(
        bundle,
        previous_execution=previous_execution,
        resume=True,
    )

    statuses = {step["source_action"]: step["status"] for step in execution["executed_steps"]}
    assert statuses["launch_editor"] == "success"
    assert statuses["import_psd"] == "success"
    assert execution["resume"]["window_probe"] is None


def test_native_gui_controller_runs_recovery_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_recovery_retry"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
            "retry_recovery_sequences": {"default": [{"keys": "{ESC}", "wait_seconds": 0.0}]},
        },
    }
    calls: list[str] = []

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        calls.append(script_path.name)
        if (
            script_path.name == "native_gui_builtin_import.ps1"
            and calls.count(script_path.name) == 1
        ):
            return subprocess.CompletedProcess([str(script_path)], 1, "", "first failure")
        if script_path.name == "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "recovered", "")
        if script_path.name == "native_gui_builtin_window_probe.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "probe", "")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    assert result["status"] == "success"
    assert calls == [
        "native_gui_builtin_import.ps1",
        "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1",
        "native_gui_builtin_window_probe.ps1",
        "native_gui_builtin_import.ps1",
        "native_gui_builtin_import_psd_post_probe.ps1",
        "native_gui_builtin_import_psd_success_capture.ps1",
    ]
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    recovery = payload["attempts"][0]["recovery"]
    assert recovery["status"] == "success"
    assert recovery["probe"]["status"] == "success"
    script_text = (
        output_dir / "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1"
    ).read_text(encoding="utf-8")
    assert "function Activate-ControllerWindow" in script_text
    assert "$windowState = Activate-ControllerWindow -Fragments $activationFragments" in script_text
    assert '$wshell.SendKeys("{ESC}")' in script_text


def test_native_gui_controller_stops_retry_when_recovery_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_recovery_probe_fail"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
            "retry_recovery_sequences": {"default": [{"keys": "{ESC}", "wait_seconds": 0.0}]},
            "capture_screenshot_on_error": False,
        },
    }
    calls: list[str] = []

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        calls.append(script_path.name)
        if script_path.name == "native_gui_builtin_import.ps1":
            return subprocess.CompletedProcess([str(script_path)], 1, "", "first failure")
        if script_path.name == "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "recovered", "")
        if script_path.name == "native_gui_builtin_window_probe.ps1":
            return subprocess.CompletedProcess([str(script_path)], 3, "", "window missing")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    assert result["status"] == "error"
    assert calls == [
        "native_gui_builtin_import.ps1",
        "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1",
        "native_gui_builtin_window_probe.ps1",
    ]
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["attempt_count"] == 1
    assert payload["attempts"][0]["recovery"]["probe"]["status"] == "error"


def test_native_gui_controller_stops_retry_after_timeout_when_recovery_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_timeout_probe_fail"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
            "retry_recovery_sequences": {"default": [{"keys": "{ESC}", "wait_seconds": 0.0}]},
            "capture_screenshot_on_error": False,
        },
    }
    calls: list[str] = []

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        calls.append(script_path.name)
        if script_path.name == "native_gui_builtin_import.ps1":
            raise subprocess.TimeoutExpired([str(script_path)], 10)
        if script_path.name == "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "recovered", "")
        if script_path.name == "native_gui_builtin_window_probe.ps1":
            return subprocess.CompletedProcess([str(script_path)], 3, "", "window missing")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    assert result["status"] == "error"
    assert calls == [
        "native_gui_builtin_import.ps1",
        "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1",
        "native_gui_builtin_window_probe.ps1",
    ]
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["attempt_count"] == 1
    assert payload["attempts"][0]["timeout"] is True
    assert payload["attempts"][0]["recovery"]["probe"]["status"] == "error"


def test_native_gui_controller_uses_action_specific_recovery_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_action_specific_recovery"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
            "retry_recovery_sequences": {
                "default": [{"keys": "{ESC}", "wait_seconds": 0.0}],
                "import_psd": [{"keys": "%{F4}", "wait_seconds": 0.0}],
            },
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name == "native_gui_builtin_import.ps1":
            return subprocess.CompletedProcess([str(script_path)], 1, "", "first failure")
        if script_path.name == "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "recovered", "")
        if script_path.name == "native_gui_builtin_window_probe.ps1":
            return subprocess.CompletedProcess([str(script_path)], 3, "", "window missing")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    script_text = (
        output_dir / "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1"
    ).read_text(encoding="utf-8")
    assert '$wshell.SendKeys("%{F4}")' in script_text
    assert '$wshell.SendKeys("{ESC}")' not in script_text


def test_native_gui_controller_recovery_script_targets_known_dialogs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_known_dialog_recovery"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
            "retry_recovery_sequences": {"default": [{"keys": "{ESC}", "wait_seconds": 0.0}]},
            "known_dialog_recovery": {
                "import_psd": [
                    {
                        "title_contains": "Import PSD",
                        "keys": "%y",
                        "wait_seconds": 0.0,
                    }
                ]
            },
            "capture_screenshot_on_error": False,
        },
    }

    calls: list[str] = []

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        calls.append(script_path.name)
        if (
            script_path.name == "native_gui_builtin_import.ps1"
            and calls.count(script_path.name) == 1
        ):
            return subprocess.CompletedProcess([str(script_path)], 1, "", "first failure")
        if script_path.name == "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "recovered", "")
        if script_path.name == "native_gui_builtin_window_probe.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "probe", "")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    result = controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    assert result["status"] == "success"
    script_text = (
        output_dir / "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1"
    ).read_text(encoding="utf-8")
    assert "function Invoke-DialogRecovery" in script_text
    assert 'TitleFragment "Import PSD"' in script_text
    assert "$wshell.AppActivate($process.Id)" in script_text
    assert 'Invoke-DialogRecovery -TitleFragment "Import PSD" -Keys "%y"' in script_text
    assert "function Activate-ControllerWindow" in script_text
    assert "$windowState = Activate-ControllerWindow -Fragments $activationFragments" in script_text
    assert "if (-not $windowState.Activated) { exit 3 }" in script_text
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    plan = payload["attempts"][0]["recovery"]["dialog_recovery_plan"]
    assert plan["dialog_source"] == "import_psd"
    assert plan["sequence_source"] == "default"
    assert plan["dialogs"][0]["title_contains"] == "Import PSD"
    assert plan["sequences"][0]["keys"] == "{ESC}"


def test_native_gui_controller_prefers_action_specific_dialog_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    output_dir = tmp_path / "controller_action_dialog_recovery"
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "import_shortcut": "^o",
            "activation_wait_seconds": 0.0,
            "dialog_wait_seconds": 0.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 0.0,
            "retry_recovery_sequences": {"default": [{"keys": "{ESC}", "wait_seconds": 0.0}]},
            "known_dialog_recovery": {
                "default": [
                    {"title_contains": "Generic Dialog", "keys": "{ENTER}", "wait_seconds": 0.0}
                ],
                "import_psd": [{"title_contains": "Import PSD", "keys": "%y", "wait_seconds": 0.0}],
            },
            "capture_screenshot_on_error": False,
        },
    }

    def fake_run(script_path: Path) -> subprocess.CompletedProcess[str]:
        if script_path.name == "native_gui_builtin_import.ps1":
            return subprocess.CompletedProcess([str(script_path)], 1, "", "first failure")
        if script_path.name == "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1":
            return subprocess.CompletedProcess([str(script_path)], 0, "recovered", "")
        if script_path.name == "native_gui_builtin_window_probe.ps1":
            return subprocess.CompletedProcess([str(script_path)], 3, "", "window missing")
        return subprocess.CompletedProcess([str(script_path)], 0, "ok", "")

    monkeypatch.setattr(controller, "_run_powershell", fake_run)

    controller.execute_import(controller_state, tmp_path / "demo.psd", output_dir)

    script_text = (
        output_dir / "native_gui_builtin_import_psd_retry_recovery_attempt_1.ps1"
    ).read_text(encoding="utf-8")
    assert 'TitleFragment "Import PSD"' in script_text
    assert 'TitleFragment "Generic Dialog"' not in script_text
    payload = json.loads(
        (output_dir / "native_gui_builtin_import_result.json").read_text(encoding="utf-8")
    )
    assert (
        payload["attempts"][0]["recovery"]["dialog_recovery_plan"]["dialog_source"] == "import_psd"
    )


def test_default_windows_profile_contains_seed_dialog_recovery_rules() -> None:
    profile = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "mcp_server"
            / "profiles"
            / "windows_cubism_default.json"
        ).read_text(encoding="utf-8")
    )

    known_dialog_recovery = profile["known_dialog_recovery"]
    assert known_dialog_recovery["import_psd"]
    assert known_dialog_recovery["apply_template"]
    assert known_dialog_recovery["export_embedded_data"]
    assert known_dialog_recovery["import_psd"][0]["title_contains"] == "Open"
    assert "Import PSD" in profile["window_probe_candidates"]


def test_probe_window_captures_window_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = NativeWindowsGUIController()
    controller_state = {
        "status": "ready",
        "mode": "execute",
        "profile": {
            "window_title_contains": "Cubism Editor",
            "window_probe_candidates": ["Import PSD", "Export"],
            "import_shortcut": "^o",
        },
    }
    probe_payload = {
        "target": "Cubism Editor",
        "matched_titles": ["Cubism Editor", "Import PSD"],
        "diagnostics": [
            {
                "ProcessId": 101,
                "ProcessName": "CubismEditor5",
                "Title": "Cubism Editor",
            },
            {
                "ProcessId": 202,
                "ProcessName": "CubismEditor5",
                "Title": "Import PSD",
            },
        ],
        "all_titles": ["Cubism Editor", "Import PSD", "Live2D Cubism 5.3"],
        "all_diagnostics": [
            {
                "ProcessId": 101,
                "ProcessName": "CubismEditor5",
                "Title": "Cubism Editor",
            },
            {
                "ProcessId": 202,
                "ProcessName": "CubismEditor5",
                "Title": "Import PSD",
            },
            {
                "ProcessId": 303,
                "ProcessName": "CubismEditor5",
                "Title": "Live2D Cubism 5.3",
            },
        ],
    }

    monkeypatch.setattr(
        controller,
        "_run_powershell",
        lambda script_path: subprocess.CompletedProcess(
            [str(script_path)], 0, json.dumps(probe_payload), ""
        ),
    )

    result = controller.probe_window(controller_state, tmp_path)

    assert result is not None
    assert result["status"] == "success"
    assert result["diagnostics"][0]["Title"] == "Cubism Editor"
    assert "Live2D Cubism 5.3" in result["all_titles"]
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["matched_titles"] == ["Cubism Editor", "Import PSD"]
    assert "Live2D Cubism 5.3" in payload["all_titles"]
    assert payload["target"] == "Cubism Editor"


def test_profile_calibration_report_collects_probe_and_dialog_suggestions(
    tmp_path: Path,
) -> None:
    manager = CubismAutomationManager()
    output_dir = tmp_path / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_artifact = output_dir / "native_gui_builtin_import_result.json"
    recovery_artifact.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "attempt": 1,
                        "returncode": 1,
                        "recovery": {
                            "dialog_recovery_plan": {
                                "dialog_source": "import_psd",
                                "dialogs": [{"title_contains": "Import PSD", "keys": "%y"}],
                                "sequence_source": "default",
                                "sequences": [{"keys": "{ESC}"}],
                            },
                            "probe": {
                                "matched_titles": ["Cubism Editor", "PSD Import"],
                                "diagnostics": [
                                    {"Title": "Cubism Editor"},
                                    {"Title": "PSD Import"},
                                ],
                                "all_titles": [
                                    "Cubism Editor",
                                    "PSD Import",
                                    "Live2D Cubism 5.3",
                                ],
                            },
                        },
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle = {
        "backend": "native_gui",
        "model_name": "DiagCase",
        "native_controller": {
            "profile": {
                "window_title_contains": "Cubism Editor",
                "window_probe_candidates": ["Cubism Editor", "Open"],
                "known_dialog_recovery": {
                    "import_psd": [{"title_contains": "Import PSD", "keys": "%y"}]
                },
            }
        },
    }
    execution = {
        "resume": {
            "window_probe": {
                "matched_titles": ["Cubism Editor", "PSD Import"],
                "diagnostics": [
                    {"Title": "Cubism Editor"},
                    {"Title": "PSD Import"},
                ],
                "all_titles": [
                    "Cubism Editor",
                    "PSD Import",
                    "Live2D Cubism 5.3",
                ],
            }
        },
        "executed_steps": [
            {
                "source_action": "import_psd",
                "status": "error",
                "artifact_path": str(recovery_artifact),
            }
        ],
    }

    report = manager.build_profile_calibration_report(bundle, execution)

    assert report["status"] == "ready"
    assert "PSD Import" in report["observed_titles"]
    assert "Live2D Cubism 5.3" in report["visible_window_titles"]
    assert "Live2D Cubism 5.3" in report["likely_window_titles"]
    assert "PSD Import" in report["missing_probe_candidates"]
    assert report["action_diagnostics"]["import_psd"]["configured_dialog_titles"] == ["Import PSD"]
    assert any("window_probe_candidates" in suggestion for suggestion in report["suggestions"])


def test_profile_calibration_report_surfaces_likely_titles_when_probe_target_misses(
    tmp_path: Path,
) -> None:
    manager = CubismAutomationManager()
    bundle = {
        "backend": "native_gui",
        "model_name": "DiagMiss",
        "native_controller": {
            "profile": {
                "window_title_contains": "Cubism Editor",
                "window_probe_candidates": ["Cubism Editor"],
                "known_dialog_recovery": {},
            }
        },
    }
    execution = {
        "resume": {
            "window_probe": {
                "matched_titles": [],
                "diagnostics": [],
                "all_titles": [
                    "Live2D Cubism 5.3",
                    "PSD Import",
                    "Visual Studio Code",
                ],
            }
        },
        "executed_steps": [],
    }

    report = manager.build_profile_calibration_report(bundle, execution)

    assert report["observed_titles"] == []
    assert "Live2D Cubism 5.3" in report["visible_window_titles"]
    assert "PSD Import" in report["likely_window_titles"]
    assert "Live2D Cubism 5.3" in report["missing_probe_candidates"]
    assert any("window_probe_candidates" in suggestion for suggestion in report["suggestions"])


def test_profile_calibration_report_surfaces_missing_apply_template_invocation() -> None:
    manager = CubismAutomationManager()
    bundle = {
        "backend": "native_gui",
        "model_name": "TemplateDiag",
        "native_controller": {
            "profile": {
                "window_title_contains": "Cubism Editor",
                "window_probe_candidates": ["Cubism Editor"],
                "known_dialog_recovery": {},
                "template_menu_sequence": [],
            }
        },
    }
    execution = {
        "resume": {"window_probe": None},
        "executed_steps": [
            {
                "source_action": "apply_template",
                "status": "error",
                "details": (
                    "Apply-template automation requires a configured template menu sequence "
                    "or explicit template shortcut in the native GUI profile."
                ),
            }
        ],
    }

    report = manager.build_profile_calibration_report(bundle, execution)

    assert report["action_diagnostics"]["apply_template"]["status"] == "error"
    assert report["action_diagnostics"]["apply_template"]["apply_template_ready"] is False
    assert (
        report["action_diagnostics"]["apply_template"]["template_menu_sequence_configured"] is False
    )
    assert any("template_menu_sequence" in suggestion for suggestion in report["suggestions"])


def test_profile_calibration_report_ignores_unusable_apply_template_sequence() -> None:
    manager = CubismAutomationManager()
    bundle = {
        "backend": "native_gui",
        "model_name": "TemplateDiagInvalid",
        "native_controller": {
            "profile": {
                "window_title_contains": "Cubism Editor",
                "window_probe_candidates": ["Cubism Editor"],
                "known_dialog_recovery": {},
                "template_menu_sequence": [{"wait_seconds": 0.2}],
            }
        },
    }
    execution = {
        "resume": {"window_probe": None},
        "executed_steps": [
            {
                "source_action": "apply_template",
                "status": "error",
                "details": "Apply-template calibration is still missing.",
            }
        ],
    }

    report = manager.build_profile_calibration_report(bundle, execution)

    diagnostics = report["action_diagnostics"]["apply_template"]
    assert diagnostics["template_menu_sequence_entries"] == 1
    assert diagnostics["template_menu_sequence_length"] == 0
    assert diagnostics["template_menu_sequence_configured"] is False
    assert diagnostics["apply_template_ready"] is False
    assert diagnostics["recommended_menu_path"] == [
        "Modeling",
        "Model template",
        "Apply template",
    ]
    assert "Modeling" in diagnostics["documentation_hint"]
    assert any("template_menu_sequence" in suggestion for suggestion in report["suggestions"])
