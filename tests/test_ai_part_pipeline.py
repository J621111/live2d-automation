from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mcp_server.secure_server_impl import (
    analyze_parts_with_ai,
    analyze_photo,
    close_session,
    generate_layers,
    segment_detected_parts,
)
from mcp_server.tools.ai_part_detector import AIPartDetector
from mcp_server.tools.part_detection_backends.api_backend import APIPartDetectionBackend

JsonDict = dict[str, Any]


def _part_by_name(parts: list[JsonDict], name: str) -> JsonDict:
    for part in parts:
        if part.get("name") == name:
            return part
    raise AssertionError(f"missing part: {name}")


def _layer_stats(layer_payload: JsonDict) -> dict[str, float]:
    with Image.open(Path(str(layer_payload["path"]))).convert("RGBA") as image:
        rgba = np.array(image)

    alpha = rgba[:, :, 3] > 0
    assert np.count_nonzero(alpha) > 0
    active_pixels = rgba[alpha]
    occupancy = float(np.count_nonzero(alpha)) / float(alpha.size)
    return {
        "occupancy": occupancy,
        "mean_r": float(active_pixels[:, 0].mean()),
        "mean_g": float(active_pixels[:, 1].mean()),
        "mean_b": float(active_pixels[:, 2].mean()),
        "mean_gray": float(active_pixels[:, :3].mean(axis=1).mean()),
    }


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
    draw.ellipse((324, 202, 330, 208), fill=(252, 253, 255, 255))
    draw.ellipse((440, 202, 446, 208), fill=(252, 253, 255, 255))
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

    parts: list[JsonDict] = result["parts"]
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
        analyze_parts: list[JsonDict] = analyze_result["parts"]
        left_eye_part = _part_by_name(analyze_parts, "left_eye")
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


@pytest.mark.asyncio
async def test_ai_part_detector_api_backend_uses_remote_keypoint_overrides(
    sample_image_path: Path,
) -> None:
    def fake_transport(_url: str, payload: JsonDict, _headers: JsonDict) -> JsonDict:
        fallback_names = {part["name"] for part in payload["fallback_parts"]}
        assert {"left_eye", "right_eye", "mouth"}.issubset(fallback_names)
        return {
            "parts": [
                {
                    "name": "left_eye",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 288, "y": 184, "width": 82, "height": 38},
                    "polygon": [
                        {"x": 290, "y": 190},
                        {"x": 366, "y": 188},
                        {"x": 360, "y": 220},
                    ],
                    "confidence": 0.97,
                },
                {
                    "name": "left_eye_white",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 294, "y": 190, "width": 62, "height": 24},
                    "polygon": [
                        {"x": 296, "y": 194},
                        {"x": 354, "y": 194},
                        {"x": 350, "y": 212},
                    ],
                    "confidence": 0.95,
                },
                {
                    "name": "left_iris",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 314, "y": 194, "width": 22, "height": 22},
                    "polygon": [
                        {"x": 316, "y": 196},
                        {"x": 334, "y": 196},
                        {"x": 334, "y": 214},
                    ],
                    "confidence": 0.94,
                },
                {
                    "name": "left_eye_highlight",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 320, "y": 198, "width": 8, "height": 8},
                    "polygon": [
                        {"x": 321, "y": 199},
                        {"x": 327, "y": 199},
                        {"x": 327, "y": 205},
                    ],
                    "confidence": 0.92,
                },
                {
                    "name": "right_eye",
                    "group": "face",
                    "side": "right",
                    "bbox": {"x": 404, "y": 184, "width": 84, "height": 38},
                    "polygon": [
                        {"x": 406, "y": 190},
                        {"x": 484, "y": 188},
                        {"x": 476, "y": 220},
                    ],
                    "confidence": 0.96,
                },
            ]
        }

    detector = AIPartDetector(backend_name="api", api_transport=fake_transport)
    result = await detector.analyze(str(sample_image_path))

    assert result["backend_used"] == "api"
    assert result["detector_used"] == "hybrid_api_v1"
    assert {
        "left_eye",
        "left_eye_white",
        "left_iris",
        "left_eye_highlight",
        "right_eye",
    }.issubset(set(result["api_metadata"]["refined_parts"]))

    left_eye = _part_by_name(result["parts"], "left_eye")
    left_eye_white = _part_by_name(result["parts"], "left_eye_white")
    left_iris = _part_by_name(result["parts"], "left_iris")
    left_eye_highlight = _part_by_name(result["parts"], "left_eye_highlight")
    right_eye = _part_by_name(result["parts"], "right_eye")
    assert left_eye["bbox"]["x"] == 288
    assert right_eye["bbox"]["x"] == 404
    assert left_eye_white["bbox"]["width"] < left_eye["bbox"]["width"]
    assert left_iris["bbox"]["width"] < left_eye_white["bbox"]["width"]
    assert left_eye_highlight["attributes"]["source"] == "api_backend"


@pytest.mark.asyncio
async def test_ai_part_detector_api_backend_falls_back_without_configuration(
    sample_image_path: Path,
) -> None:
    detector = AIPartDetector(backend_name="api")
    result = await detector.analyze(str(sample_image_path))

    assert result["backend_used"] == "api"
    assert result["detector_used"] == "semantic_refine_v1"
    assert "using heuristic fallback" in str(result["fallback_reason"]).lower()


@pytest.mark.asyncio
async def test_mcp_pipeline_can_use_api_backend_via_environment(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_transport(
        self: APIPartDetectionBackend, _url: str, payload: JsonDict, _headers: JsonDict
    ) -> JsonDict:
        fallback_names = {part["name"] for part in payload["fallback_parts"]}
        assert {"left_eye", "right_eye"}.issubset(fallback_names)
        return {
            "parts": [
                {
                    "name": "left_eye",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 280, "y": 182, "width": 88, "height": 42},
                    "polygon": [
                        {"x": 282, "y": 190},
                        {"x": 364, "y": 188},
                        {"x": 356, "y": 222},
                    ],
                    "confidence": 0.98,
                },
                {
                    "name": "right_eye",
                    "group": "face",
                    "side": "right",
                    "bbox": {"x": 404, "y": 182, "width": 88, "height": 42},
                    "polygon": [
                        {"x": 406, "y": 190},
                        {"x": 488, "y": 188},
                        {"x": 480, "y": 222},
                    ],
                    "confidence": 0.98,
                },
            ]
        }

    monkeypatch.setenv("LIVE2D_PART_BACKEND", "api")
    monkeypatch.setenv("LIVE2D_PART_API_URL", "https://example.invalid/parts")
    monkeypatch.setattr(APIPartDetectionBackend, "_default_transport", fake_transport)

    analyze_result = await analyze_parts_with_ai(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        left_eye = _part_by_name(analyze_result["parts"], "left_eye")
        right_eye = _part_by_name(analyze_result["parts"], "right_eye")
        assert analyze_result["backend_used"] == "api"
        assert analyze_result["detector_used"] == "hybrid_api_v1"
        assert left_eye["bbox"]["x"] == 280
        assert right_eye["bbox"]["x"] == 404

        segment_result = await segment_detected_parts(
            session_id, str(tmp_path / "api_segmentation")
        )
        assert segment_result["status"] == "success"
        segmented_left_eye = _part_by_name(segment_result["layers"], "left_eye")
        assert segmented_left_eye["bounds"]["x"] >= 280
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_generate_layers_can_use_api_backend_via_environment(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_transport(
        self: APIPartDetectionBackend, _url: str, _payload: JsonDict, _headers: JsonDict
    ) -> JsonDict:
        return {
            "parts": [
                {
                    "name": "left_eye",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 282, "y": 182, "width": 88, "height": 40},
                    "polygon": [
                        {"x": 284, "y": 190},
                        {"x": 366, "y": 188},
                        {"x": 358, "y": 222},
                    ],
                    "confidence": 0.98,
                },
                {
                    "name": "left_eye_white",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 292, "y": 190, "width": 60, "height": 24},
                    "polygon": [
                        {"x": 294, "y": 194},
                        {"x": 350, "y": 194},
                        {"x": 346, "y": 212},
                    ],
                    "confidence": 0.95,
                },
                {
                    "name": "left_iris",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 312, "y": 194, "width": 22, "height": 22},
                    "polygon": [
                        {"x": 314, "y": 196},
                        {"x": 332, "y": 196},
                        {"x": 332, "y": 214},
                    ],
                    "confidence": 0.94,
                },
                {
                    "name": "left_eye_highlight",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 320, "y": 198, "width": 8, "height": 8},
                    "polygon": [
                        {"x": 321, "y": 199},
                        {"x": 327, "y": 199},
                        {"x": 327, "y": 205},
                    ],
                    "confidence": 0.92,
                },
                {
                    "name": "right_eye",
                    "group": "face",
                    "side": "right",
                    "bbox": {"x": 404, "y": 182, "width": 88, "height": 40},
                    "polygon": [
                        {"x": 406, "y": 190},
                        {"x": 488, "y": 188},
                        {"x": 480, "y": 222},
                    ],
                    "confidence": 0.98,
                },
            ]
        }

    monkeypatch.setenv("LIVE2D_PART_BACKEND", "api")
    monkeypatch.setenv("LIVE2D_PART_API_URL", "https://example.invalid/parts")
    monkeypatch.setattr(APIPartDetectionBackend, "_default_transport", fake_transport)

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        layers_result = await generate_layers(session_id, str(tmp_path / "api_layers"))
        assert layers_result["status"] == "success"
        assert layers_result["backend_used"] == "api"
        assert layers_result["detector_used"] == "hybrid_api_v1"

        left_eye = _part_by_name(layers_result["layers"], "left_eye")
        left_eye_white = _part_by_name(layers_result["layers"], "left_eye_white")
        left_iris = _part_by_name(layers_result["layers"], "left_iris")
        left_eye_highlight = _part_by_name(layers_result["layers"], "left_eye_highlight")
        right_eye = _part_by_name(layers_result["layers"], "right_eye")
        assert left_eye["bounds"]["x"] >= 282
        assert right_eye["bounds"]["x"] >= 404
        assert left_eye_white["bounds"]["width"] < left_eye["bounds"]["width"]
        assert left_iris["bounds"]["width"] <= left_eye_white["bounds"]["width"]
        assert left_eye_highlight["bounds"]["width"] <= left_iris["bounds"]["width"]
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_generate_layers_api_backend_validates_eye_mask_quality(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_transport(
        self: APIPartDetectionBackend, _url: str, _payload: JsonDict, _headers: JsonDict
    ) -> JsonDict:
        return {
            "parts": [
                {
                    "name": "left_eye_white",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 294, "y": 190, "width": 62, "height": 24},
                    "polygon": [
                        {"x": 296, "y": 194},
                        {"x": 354, "y": 194},
                        {"x": 354, "y": 214},
                        {"x": 296, "y": 214},
                    ],
                    "confidence": 0.95,
                },
                {
                    "name": "left_iris",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 316, "y": 196, "width": 20, "height": 20},
                    "polygon": [
                        {"x": 316, "y": 196},
                        {"x": 336, "y": 196},
                        {"x": 336, "y": 216},
                        {"x": 316, "y": 216},
                    ],
                    "confidence": 0.94,
                },
                {
                    "name": "left_eye_highlight",
                    "group": "face",
                    "side": "left",
                    "bbox": {"x": 322, "y": 200, "width": 8, "height": 8},
                    "polygon": [
                        {"x": 322, "y": 200},
                        {"x": 330, "y": 200},
                        {"x": 330, "y": 208},
                        {"x": 322, "y": 208},
                    ],
                    "confidence": 0.92,
                },
            ]
        }

    monkeypatch.setenv("LIVE2D_PART_BACKEND", "api")
    monkeypatch.setenv("LIVE2D_PART_API_URL", "https://example.invalid/parts")
    monkeypatch.setattr(APIPartDetectionBackend, "_default_transport", fake_transport)

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        layers_result = await generate_layers(session_id, str(tmp_path / "quality_layers"))
        assert layers_result["status"] == "success"

        eye_white_stats = _layer_stats(_part_by_name(layers_result["layers"], "left_eye_white"))
        iris_stats = _layer_stats(_part_by_name(layers_result["layers"], "left_iris"))
        highlight_stats = _layer_stats(_part_by_name(layers_result["layers"], "left_eye_highlight"))

        assert 0.25 <= eye_white_stats["occupancy"] <= 0.95
        assert eye_white_stats["mean_gray"] >= 235.0

        assert 0.20 <= iris_stats["occupancy"] <= 0.90
        assert iris_stats["mean_b"] > iris_stats["mean_r"] + 25.0
        assert iris_stats["mean_b"] > iris_stats["mean_g"] + 15.0
        assert iris_stats["mean_gray"] < eye_white_stats["mean_gray"] - 60.0

        assert 0.08 <= highlight_stats["occupancy"] <= 0.85
        assert highlight_stats["mean_gray"] >= 235.0
        assert highlight_stats["mean_gray"] > iris_stats["mean_gray"] + 80.0
    finally:
        await close_session(session_id)
