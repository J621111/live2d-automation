from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mcp_server.secure_server_impl import (
    analyze_photo,
    build_cubism_psd,
    close_session,
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
        assert plan_path.exists()
        assert "Stage3Plan" in plan_path.name
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan_data["automation_backend"] == "native_gui"
        assert plan_data["execution"]["missing_requirements"] == ["cubism_editor"]
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
async def test_prepare_cubism_automation_supports_opencli_backend(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_editor = tmp_path / "CubismEditor5.exe"
    fake_editor.write_bytes(b"stub")
    monkeypatch.setenv("OPENCLI_COMMAND", "opencli run cubism")

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
        assert plan_data["automation_backend"] == "opencli"
        assert plan_data["execution"]["command_hint"] == "opencli run cubism"
        assert plan_data["execution"]["automation_mode"] == "connector_assisted"
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
