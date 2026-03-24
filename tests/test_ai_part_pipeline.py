from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mcp_server.secure_server_impl import (
    analyze_parts_with_ai,
    analyze_photo,
    close_session,
    generate_layers,
    segment_detected_parts,
)


def _part_by_name(parts: list[dict[str, object]], name: str) -> dict[str, object]:
    for part in parts:
        if part.get("name") == name:
            return part
    raise AssertionError(f"missing part: {name}")


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
async def test_analyze_parts_with_ai_returns_structured_face_parts(
    sample_image_path: Path,
) -> None:
    result = await analyze_parts_with_ai(str(sample_image_path))
    assert result["status"] == "success"
    assert result["part_count"] >= 8
    assert result["detector_used"] == "semantic_refine_v1"

    parts = result["parts"]
    names = {part["name"] for part in parts}
    assert {"left_eye", "right_eye", "mouth", "left_eyebrow", "right_eyebrow"}.issubset(names)

    left_eye = _part_by_name(parts, "left_eye")
    right_eye = _part_by_name(parts, "right_eye")
    assert left_eye["side"] == "left"
    assert right_eye["side"] == "right"
    assert left_eye["bbox"]["x"] < right_eye["bbox"]["x"]
    assert left_eye["attributes"]["refined"] is True
    assert len(left_eye["polygon"]) >= 3

    await close_session(result["session_id"])


@pytest.mark.asyncio
async def test_segment_detected_parts_tightens_eye_crop(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_parts_with_ai(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        left_eye_part = _part_by_name(analyze_result["parts"], "left_eye")
        segment_result = await segment_detected_parts(session_id, str(tmp_path / "ai_segmentation"))
        assert segment_result["status"] == "success"
        assert segment_result["layers_generated"] >= 5

        left_eye = _part_by_name(segment_result["layers"], "left_eye")
        image_path = Path(str(left_eye["path"]))
        mask_path = Path(str(left_eye["mask_path"]))
        assert image_path.exists()
        assert mask_path.exists()

        detected_width = left_eye_part["bbox"]["width"]
        detected_height = left_eye_part["bbox"]["height"]
        with Image.open(image_path).convert("RGBA") as layer:
            alpha = layer.getchannel("A")
            extrema = alpha.getextrema()
            assert extrema is not None
            assert extrema[1] > 0
            assert layer.width < detected_width or layer.height < detected_height
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_generate_layers_prefers_semantic_refine_path(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        result = await generate_layers(session_id, str(tmp_path / "semantic_layers"))
        assert result["status"] == "success"
        assert result["detector_used"] == "semantic_refine_v1"
        names = {layer["name"] for layer in result["layers"]}
        assert {"left_eye", "right_eye", "hair_front", "torso"}.issubset(names)
    finally:
        await close_session(session_id)
