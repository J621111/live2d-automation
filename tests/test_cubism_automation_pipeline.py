from __future__ import annotations

import json
import os
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
    validate_cubism_export,
)


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
        assert execution_data["backend"] == "native_gui"
        assert execution_data["status"] == "partial"
        assert execution_data["executed_steps"][0]["source_action"] == "launch_editor"
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
        assert (package_dir / "native_gui_builtin_launch.ps1").exists()
        assert (package_dir / "native_gui_builtin_import.ps1").exists()
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
