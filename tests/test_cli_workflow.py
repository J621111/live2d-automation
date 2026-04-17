from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


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


def _create_stub_psd(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"8BPS-stub")
    return path


def test_cli_run_with_demo_adapter_full(tmp_path: Path) -> None:
    image_path = _create_sample_character_image(tmp_path / "input_image" / "character.png")
    output_dir = tmp_path / "cli_full_output"
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_server.cli",
            "run",
            "--image-path",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "CLIFull",
            "--editor-path",
            str(fake_editor),
            "--demo-adapter-mode",
            "full",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    report_path = Path(payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert (output_dir / "CLIFull.moc3").exists()
    assert (output_dir / "model3.json").exists()
    assert (output_dir / "textures" / "texture_00.png").exists()


def test_cli_run_with_builtin_controller_dry_run(tmp_path: Path) -> None:
    image_path = _create_sample_character_image(tmp_path / "input_image" / "character.png")
    output_dir = tmp_path / "cli_builtin_output"
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_server.cli",
            "run",
            "--image-path",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "CLIBuiltin",
            "--editor-path",
            str(fake_editor),
            "--native-gui-controller-mode",
            "dry_run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout.strip())
    report_path = Path(payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert (output_dir / "native_gui_builtin_launch.ps1").exists()
    assert (output_dir / "native_gui_builtin_import.ps1").exists()
    assert (output_dir / "native_gui_builtin_apply_template.ps1").exists()
    assert (output_dir / "native_gui_builtin_export.ps1").exists()


def test_cli_run_with_demo_adapter_partial(tmp_path: Path) -> None:
    image_path = _create_sample_character_image(tmp_path / "input_image" / "character.png")
    output_dir = tmp_path / "cli_partial_output"
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_server.cli",
            "run",
            "--image-path",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "CLIPartial",
            "--editor-path",
            str(fake_editor),
            "--demo-adapter-mode",
            "partial",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout.strip())
    report_path = Path(payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["steps"]["execute_cubism_dispatch"]["status"] == "partial"


def test_cli_calibrate_template_with_demo_adapter_full(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_calibrate_full_output"
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    psd_path = _create_stub_psd(output_dir / "CalibrateFull.psd")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_server.cli",
            "calibrate-template",
            "--output-dir",
            str(output_dir),
            "--model-name",
            "CalibrateFull",
            "--psd-path",
            str(psd_path),
            "--editor-path",
            str(fake_editor),
            "--demo-adapter-mode",
            "full",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    report_path = Path(payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["steps"]["prepare_cubism_automation"]["status"] == "ready"
    assert report["steps"]["execute_cubism_dispatch"]["status"] == "success"
    assert (output_dir / "CalibrateFull_cubism_automation_plan.json").exists()
    assert (output_dir / "CalibrateFull_cubism_dispatch_bundle.json").exists()
    assert (output_dir / "CalibrateFull_cubism_dispatch_execution.json").exists()
    assert (output_dir / "CalibrateFull_cubism_profile_calibration.json").exists()
    assert (output_dir / "CalibrateFull.moc3").exists()


def test_cli_calibrate_template_with_builtin_controller_dry_run_infers_psd_path(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cli_calibrate_builtin_output"
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    _create_stub_psd(output_dir / "CalibrateBuiltin.psd")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_server.cli",
            "calibrate-template",
            "--output-dir",
            str(output_dir),
            "--model-name",
            "CalibrateBuiltin",
            "--editor-path",
            str(fake_editor),
            "--native-gui-controller-mode",
            "dry_run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout.strip())
    report_path = Path(payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["steps"]["prepare_cubism_automation"]["status"] == "ready"
    assert report["steps"]["execute_cubism_dispatch"]["status"] == "partial"
    assert (output_dir / "native_gui_builtin_import.ps1").exists()
    assert (output_dir / "native_gui_builtin_apply_template.ps1").exists()


def test_cli_calibrate_template_ignores_stale_resume_execution(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_calibrate_resume_output"
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    psd_path = _create_stub_psd(output_dir / "CalibrateResume.psd")
    stale_execution_path = output_dir / "CalibrateResume_cubism_dispatch_execution.json"
    stale_execution_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "resume_context": {
                    "command": "calibrate-template",
                    "automation_backend": "native_gui",
                    "editor_path": str(fake_editor),
                    "model_name": "CalibrateResume",
                    "native_gui_controller_mode": "disabled",
                    "native_gui_profile": None,
                    "psd_path": str((output_dir / "Other.psd").resolve()),
                    "psd_size": 12,
                    "psd_mtime_ns": 1,
                    "template_id": "standard_bust_up",
                },
                "executed_steps": [
                    {"source_action": "launch_editor", "status": "success"},
                    {"source_action": "import_psd", "status": "success"},
                    {"source_action": "apply_template", "status": "success"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_server.cli",
            "calibrate-template",
            "--output-dir",
            str(output_dir),
            "--model-name",
            "CalibrateResume",
            "--psd-path",
            str(psd_path),
            "--editor-path",
            str(fake_editor),
            "--demo-adapter-mode",
            "partial",
            "--resume",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout.strip())
    report_path = Path(payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["steps"]["resume_validation"]["status"] == "ignored"
    assert "does not match" in report["steps"]["resume_validation"]["message"]
    executed_steps = report["steps"]["execute_cubism_dispatch"]["executed_steps"]
    launch_step = next(step for step in executed_steps if step["source_action"] == "launch_editor")
    assert launch_step["status"] == "success"
    execution = json.loads(
        (output_dir / "CalibrateResume_cubism_dispatch_execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["resume_context"]["psd_path"] == str(psd_path.resolve())
