from __future__ import annotations

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
    generate_layers,
    prepare_cubism_automation,
)

DEMO_ADAPTER = Path(__file__).resolve().parent.parent / "scripts" / "native_gui_adapter_demo.py"


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


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    return _create_sample_character_image(tmp_path / "input_image" / "character.png")


def test_demo_adapter_cli_modes(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo_output"

    partial = subprocess.run(
        [
            sys.executable,
            str(DEMO_ADAPTER),
            "apply_template",
            "--output-dir",
            str(output_dir),
            "--model-name",
            "DemoPartial",
            "--template-id",
            "standard_bust_up",
            "--mode",
            "partial",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert partial.returncode == 64
    assert (output_dir / "demo_adapter_apply_template.json").exists()

    failure = subprocess.run(
        [
            sys.executable,
            str(DEMO_ADAPTER),
            "export_embedded_data",
            "--output-dir",
            str(output_dir),
            "--model-name",
            "DemoFail",
            "--template-id",
            "standard_bust_up",
            "--mode",
            "fail",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failure.returncode == 2

    full = subprocess.run(
        [
            sys.executable,
            str(DEMO_ADAPTER),
            "export_embedded_data",
            "--output-dir",
            str(output_dir),
            "--model-name",
            "DemoFull",
            "--template-id",
            "standard_bust_up",
            "--mode",
            "full",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert full.returncode == 0
    assert (output_dir / "DemoFull.moc3").exists()
    assert (output_dir / "model3.json").exists()
    assert (output_dir / "textures" / "texture_00.png").exists()


@pytest.mark.asyncio
async def test_demo_adapter_runs_full_dispatch_flow(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    monkeypatch.setenv(
        "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND",
        f'"{sys.executable}" "{DEMO_ADAPTER}" --mode full',
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
                model_name="DemoAdapterFlow",
            )
        )["status"] == "success"
        prepare_result = await prepare_cubism_automation(
            session_id,
            str(package_dir),
            template_id="standard_bust_up",
            model_name="DemoAdapterFlow",
            editor_path=str(fake_editor),
            automation_backend="native_gui",
        )
        assert prepare_result["status"] == "ready"

        execute_result = await execute_cubism_dispatch(session_id)
        assert execute_result["status"] == "success"
        assert (package_dir / "DemoAdapterFlow.moc3").exists()
        assert (package_dir / "model3.json").exists()
        assert (package_dir / "textures" / "texture_00.png").exists()
        assert (package_dir / "demo_adapter_export_embedded_data.json").exists()
    finally:
        await close_session(session_id)
