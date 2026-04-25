import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mcp_server.artifacts import redact_command
from mcp_server.secure_server_impl import (
    InputValidationError,
    _resolve_output_dir,
    analyze_photo,
    close_session,
    configure_physics,
    create_mesh,
    detect_face_features,
    export_model,
    full_pipeline,
    generate_layers,
    generate_motions,
    get_status,
    session_store,
    setup_rigging,
)
from mcp_server.tools.auto_rigger import AutoRigger
from mcp_server.tools.image_processor import ImageProcessor
from mcp_server.tools.moc3_generator import Live2DExporter
from mcp_server.tools.motion_generator import MotionGenerator
from mcp_server.tools.physics_setup import PhysicsSetup

pytestmark = pytest.mark.filterwarnings(
    "ignore:Image size .* exceeds limit .* "
    "decompression bomb DOS attack.:PIL.Image.DecompressionBombWarning"
)


def _case_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_sample_character_image(path: Path) -> Path:
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


def test_redact_command_masks_inline_secret_assignments() -> None:
    command = [
        "adapter.exe",
        "--api-key=super-secret",
        "--token",
        "plain-secret",
        "--safe=value",
    ]

    assert redact_command(command) == [
        "adapter.exe",
        "--api-key=<redacted>",
        "--token",
        "<redacted>",
        "--safe=value",
    ]


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    return _create_sample_character_image(_case_dir(tmp_path, "input_image") / "character.png")


@pytest.mark.asyncio
async def test_status_hides_session_ids(sample_image_path: Path) -> None:
    result = await analyze_photo(str(sample_image_path))
    assert result["status"] == "success"
    session_id = result["session_id"]

    try:
        status = json.loads(get_status())
        assert status["session_ids_hidden"] is True
        assert "sessions" not in status
        assert session_id not in json.dumps(status)
        assert "metrics" in status
    finally:
        await close_session(session_id)


@pytest.mark.asyncio
async def test_large_image_is_rejected(tmp_path: Path) -> None:
    huge_image = _case_dir(tmp_path, "huge_image") / "huge.png"
    Image.new("RGBA", (5000, 5000), (255, 0, 0, 255)).save(huge_image, format="PNG")

    result = await analyze_photo(str(huge_image))

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_input"
    assert "Image" in result["message"]


@pytest.mark.asyncio
async def test_invalid_motion_types_are_rejected() -> None:
    result = await generate_motions("job_invalid12", ["dance"])

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_input"
    assert "Unsupported motion_type" in result["message"]


@pytest.mark.asyncio
async def test_export_contract_uses_moc3(tmp_path: Path) -> None:
    bundle_dir = _case_dir(tmp_path, "bundle")
    texture_path = bundle_dir / "layer_head.png"
    Image.new("RGBA", (32, 32), (0, 255, 255, 255)).save(texture_path, format="PNG")

    state = {
        "layers": [
            {
                "name": "head",
                "texture_path": str(texture_path),
                "bounds": {"x": 0, "y": 0, "width": 32, "height": 32},
                "z_order": 1,
            }
        ],
        "rigging": {"bones": [], "parameters": [], "groups": [], "hit_areas": []},
        "motions": [],
        "physics": {},
        "face_layers": [],
    }

    exporter = Live2DExporter()
    result = await exporter.export("UnitModel", str(bundle_dir), state)

    assert result["status"] == "success"
    assert result["validation"]["contract_valid"] is True
    assert "model3.moc3" in result["files"]
    assert result["ready_for_cubism_editor"] is False

    model3_path = bundle_dir / "model3.json"
    with open(model3_path, encoding="utf-8") as handle:
        model3 = json.load(handle)
    assert model3["FileReferences"]["Moc"] == "UnitModel.moc3"


@pytest.mark.asyncio
async def test_full_pipeline_closes_session(sample_image_path: Path, tmp_path: Path) -> None:
    before = len(session_store)
    result = await full_pipeline(
        image_path=str(sample_image_path),
        output_dir=str(_case_dir(tmp_path, "full_pipeline")),
        model_name="FullModel",
        motion_types=["idle"],
    )
    assert result["status"] == "success"
    assert result["session_id"] not in session_store
    assert len(session_store) == before


@pytest.mark.asyncio
async def test_step_flow_retains_session_until_close_and_matches_export(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    step_output = _case_dir(tmp_path, "step_flow")
    full_output = _case_dir(tmp_path, "full_flow")

    analyze = await analyze_photo(str(sample_image_path))
    assert analyze["status"] == "success"
    session_id = analyze["session_id"]

    try:
        assert session_id in session_store
        assert (await detect_face_features(session_id, str(step_output)))["status"] == "success"
        assert (await generate_layers(session_id, str(step_output)))["status"] == "success"
        assert (await create_mesh(session_id))["status"] == "success"
        assert (await setup_rigging(session_id))["status"] == "success"
        assert (await configure_physics(session_id))["status"] == "success"
        assert (await generate_motions(session_id, ["idle", "move"]))["status"] == "success"
        step_export = await export_model(session_id, str(step_output), "StepModel")
        assert step_export["status"] == "success"
        assert session_id in session_store

        full_result = await full_pipeline(
            image_path=str(sample_image_path),
            output_dir=str(full_output),
            model_name="FullModel2",
            motion_types=["idle", "move"],
        )
        assert full_result["status"] == "success"
        assert set(step_export["model_files"].keys()) == set(full_result["model_files"].keys())
    finally:
        await close_session(session_id)

    assert session_id not in session_store


@pytest.mark.asyncio
async def test_analyze_photo_reports_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_path: Path,
) -> None:
    monkeypatch.setattr(
        ImageProcessor,
        "_load_pose_backend",
        lambda self: (None, "heuristic", "forced fallback for test"),
    )

    result = await analyze_photo(str(sample_image_path))
    assert result["status"] == "success"
    assert result["detector_used"] == "heuristic"
    assert result["fallback_reason"] == "forced fallback for test"

    await close_session(result["session_id"])


@pytest.mark.parametrize("candidate", ["../outside", r"..\outside"])
def test_output_dir_traversal_is_rejected(candidate: str) -> None:
    with pytest.raises(InputValidationError):
        _resolve_output_dir(candidate)


@pytest.mark.asyncio
async def test_physics_and_motion_reference_declared_parameters() -> None:
    segments = {"body_parts": {"head": {"center": (10, 10)}, "torso": {"center": (20, 20)}}}
    meshes: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        "head": {"mesh": {"vertices": []}},
        "body": {"mesh": {"vertices": []}},
    }
    rigging = await AutoRigger().setup(meshes, segments)
    physics = await PhysicsSetup().configure(rigging, segments)
    motions = await MotionGenerator().generate(rigging, ["idle", "move", "emotional"])

    parameter_ids = {param["id"] for param in rigging["parameters"]}
    physics_ids = {
        entry["id"]
        for group in physics["groups"]
        for side in ("input", "output")
        for entry in group[side]
    }
    motion_ids = {curve["Id"] for motion in motions for curve in motion["data"]["Curves"]}

    assert physics_ids.issubset(parameter_ids)
    assert motion_ids.issubset(parameter_ids)


@pytest.mark.asyncio
async def test_motion_keyframes_stay_within_parameter_ranges() -> None:
    rigging = await AutoRigger().setup({}, {"body_parts": {}})
    motions = await MotionGenerator().generate(rigging, ["idle", "move", "emotional"])
    ranges = {param["id"]: (param["min"], param["max"]) for param in rigging["parameters"]}

    for motion in motions:
        for curve in motion["data"]["Curves"]:
            minimum, maximum = ranges[curve["Id"]]
            values = curve["Segments"][1::2]
            assert all(minimum <= value <= maximum for value in values), (
                motion["name"],
                curve["Id"],
                values,
            )
