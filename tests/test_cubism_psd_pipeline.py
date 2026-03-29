from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from psd_tools import PSDImage

from mcp_server.secure_server_impl import (
    analyze_photo,
    build_cubism_psd,
    close_session,
    generate_layers,
    get_templates,
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
async def test_build_cubism_psd_creates_template_mapped_package(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        layers_result = await generate_layers(session_id, str(tmp_path / "semantic_layers"))
        assert layers_result["status"] == "success"

        build_result = await build_cubism_psd(
            session_id,
            str(tmp_path / "cubism_package"),
            template_id="standard_bust_up",
            model_name="Stage2Model",
        )
        assert build_result["status"] == "success"
        assert build_result["coverage"] > 0.3
        assert not build_result["missing_required"]

        psd_path = Path(build_result["psd_path"])
        manifest_path = Path(build_result["manifest_path"])
        mapping_path = Path(build_result["mapping_path"])
        preview_path = Path(build_result["preview_path"])
        assert psd_path.exists()
        assert manifest_path.exists()
        assert mapping_path.exists()
        assert preview_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapped_names = {entry["name"] for entry in manifest["layers"]}
        assert {"torso", "head", "face_base", "left_eye", "right_eye", "mouth"}.issubset(
            mapped_names
        )

        psd = PSDImage.open(psd_path)
        flattened_names = [layer.name for layer in psd.descendants() if not layer.is_group()]
        assert "left_eye" in flattened_names
        assert "mouth" in flattened_names
    finally:
        await close_session(session_id)


def test_get_templates_lists_stage_two_template() -> None:
    payload = json.loads(get_templates())
    assert "standard_bust_up" in payload["templates"]
