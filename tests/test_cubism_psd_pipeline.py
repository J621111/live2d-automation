from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from mcp_server.tools.part_detection_backends.api_backend import APIPartDetectionBackend

JsonDict = dict[str, Any]


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


@pytest.mark.asyncio
async def test_build_cubism_psd_includes_finer_eye_parts_from_api_layers(
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
                {
                    "name": "right_eye_white",
                    "group": "face",
                    "side": "right",
                    "bbox": {"x": 408, "y": 190, "width": 60, "height": 24},
                    "polygon": [
                        {"x": 410, "y": 194},
                        {"x": 466, "y": 194},
                        {"x": 462, "y": 212},
                    ],
                    "confidence": 0.95,
                },
                {
                    "name": "right_iris",
                    "group": "face",
                    "side": "right",
                    "bbox": {"x": 432, "y": 194, "width": 22, "height": 22},
                    "polygon": [
                        {"x": 434, "y": 196},
                        {"x": 452, "y": 196},
                        {"x": 452, "y": 214},
                    ],
                    "confidence": 0.94,
                },
                {
                    "name": "right_eye_highlight",
                    "group": "face",
                    "side": "right",
                    "bbox": {"x": 440, "y": 198, "width": 8, "height": 8},
                    "polygon": [
                        {"x": 441, "y": 199},
                        {"x": 447, "y": 199},
                        {"x": 447, "y": 205},
                    ],
                    "confidence": 0.92,
                },
                {
                    "name": "mouth",
                    "group": "face",
                    "side": None,
                    "bbox": {"x": 332, "y": 248, "width": 104, "height": 66},
                    "polygon": [
                        {"x": 334, "y": 250},
                        {"x": 434, "y": 250},
                        {"x": 434, "y": 314},
                    ],
                    "confidence": 0.95,
                },
            ]
        }

    monkeypatch.setenv("LIVE2D_PART_BACKEND", "api")
    monkeypatch.setenv("LIVE2D_PART_API_URL", "https://example.invalid/parts")
    monkeypatch.setenv("LIVE2D_PART_API_ALLOW_UPLOAD", "1")
    monkeypatch.setattr(APIPartDetectionBackend, "_default_transport", fake_transport)

    analyze_result = await analyze_photo(str(sample_image_path))
    assert analyze_result["status"] == "success"
    session_id = analyze_result["session_id"]

    try:
        layers_result = await generate_layers(session_id, str(tmp_path / "api_layers"))
        assert layers_result["status"] == "success"
        assert layers_result["backend_used"] == "api"

        build_result = await build_cubism_psd(
            session_id,
            str(tmp_path / "cubism_package_fine_parts"),
            template_id="standard_bust_up",
            model_name="FineEyeModel",
        )
        assert build_result["status"] == "success"
        assert build_result["coverage"] > 0.4

        manifest = json.loads(Path(build_result["manifest_path"]).read_text(encoding="utf-8"))
        manifest_by_name = {entry["name"]: entry for entry in manifest["layers"]}
        assert manifest_by_name["left_eye_white"]["target"] == "Face/EyeL/White"
        assert manifest_by_name["left_iris"]["target"] == "Face/EyeL/Iris"
        assert manifest_by_name["left_eye_highlight"]["target"] == "Face/EyeL/Highlight"
        assert manifest_by_name["left_eye_white"]["group"] == "Face/EyeL"
        assert manifest_by_name["right_eye_white"]["group"] == "Face/EyeR"

        mapping = json.loads(Path(build_result["mapping_path"]).read_text(encoding="utf-8"))
        mapped_names = {entry["layer_name"] for entry in mapping["mapped_layers"]}
        assert {
            "left_eye_white",
            "left_iris",
            "left_eye_highlight",
            "right_eye_white",
            "right_iris",
            "right_eye_highlight",
        }.issubset(mapped_names)

        psd = PSDImage.open(Path(build_result["psd_path"]))
        flattened_names = {layer.name for layer in psd.descendants() if not layer.is_group()}
        group_names = {layer.name for layer in psd.descendants() if layer.is_group()}
        assert {"left_eye_white", "left_iris", "left_eye_highlight"}.issubset(flattened_names)
        assert {"Face", "EyeL", "EyeR"}.issubset(group_names)
    finally:
        await close_session(session_id)


def test_get_templates_lists_stage_two_template() -> None:
    payload = json.loads(get_templates())
    assert "standard_bust_up" in payload["templates"]
