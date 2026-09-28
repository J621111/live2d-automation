"""Native export references are safe; a valid structure does not prove readiness."""

import json

import pytest

from mcp_server.tools.export_validator import CubismExportValidator


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "Avatar.moc3").write_bytes(b"MOC3\x04\0\0\0")
    (root / "Avatar.2048").mkdir()
    (root / "Avatar.2048/texture_00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    data = {
        "Version": 3,
        "FileReferences": {"Moc": "Avatar.moc3", "Textures": ["Avatar.2048/texture_00.png"]},
    }
    (root / "Avatar.model3.json").write_text(json.dumps(data))
    return root


def update_refs(bundle, **changes):
    path = bundle / "Avatar.model3.json"
    data = json.loads(path.read_text())
    data["FileReferences"].update(changes)
    path.write_text(json.dumps(data))


def test_native_names_and_atlas_folder_are_valid_without_claiming_readiness(bundle):
    result = CubismExportValidator().validate(str(bundle), "Avatar")
    assert result["status"] == "success"
    assert result["checks"]["structure_valid"] is True
    assert result["checks"]["model3_file"] == "Avatar.model3.json"
    assert result["checks"]["ready_for_cubism_editor"] is None
    assert result["checks"]["direct_viewer_compatible"] is None


@pytest.mark.parametrize(
    "refs",
    [
        {"Textures": ["../outside.png"]},
        {"Textures": ["C:\\outside.png"]},
        {"Textures": []},
        {"Textures": ["missing.png"]},
        {"Physics": "../outside.json"},
        {"Pose": "missing.pose3.json"},
        {"UserData": 12},
        {"DisplayInfo": "missing.cdi3.json"},
        {"Expressions": [{"Name": "Smile", "File": "missing.exp3.json"}]},
        {"Motions": {"Idle": [{"File": "missing.motion3.json"}]}},
    ],
)
def test_invalid_declared_resources_are_errors(bundle, refs):
    update_refs(bundle, **refs)
    result = CubismExportValidator().validate(str(bundle), "Avatar")
    assert result["status"] == "error"
    assert result["errors"]


def test_sound_traversal_is_rejected_with_existing_motion_and_sound(bundle):
    (bundle / "motion.json").write_text("{}")
    (bundle.parent / "sound.wav").write_bytes(b"sound")
    update_refs(bundle, Motions={"Idle": [{"File": "motion.json", "Sound": "../sound.wav"}]})
    result = CubismExportValidator().validate(str(bundle), "Avatar")
    assert result["status"] == "error"
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("Sound reference")


def test_symlink_reference_outside_bundle_is_rejected(bundle):
    target = bundle.parent / "outside.png"
    target.write_bytes(b"outside")
    (bundle / "linked.png").symlink_to(target)
    update_refs(bundle, Textures=["linked.png"])
    result = CubismExportValidator().validate(str(bundle), "Avatar")
    assert result["status"] == "error"
    assert any("outside" in error.lower() for error in result["errors"])


@pytest.mark.parametrize("payload", ["[]", "null", '{"FileReferences":[]}'])
def test_malformed_model_data_returns_error(bundle, payload):
    (bundle / "Avatar.model3.json").write_text(payload)
    result = CubismExportValidator().validate(str(bundle), "Avatar")
    assert result["status"] == "error"
    assert result["errors"]


def test_intermediate_metadata_cannot_be_promoted_to_ready(bundle):
    (bundle / "export_metadata.json").write_text(
        json.dumps(
            {
                "artifact_stage": "mock-intermediate",
                "direct_viewer_compatible": False,
                "ready_for_cubism_editor": False,
            }
        )
    )
    result = CubismExportValidator().validate(str(bundle), "Avatar")
    assert result["status"] == "partial"
    assert result["checks"]["ready_for_cubism_editor"] is False
