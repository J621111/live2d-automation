"""Runtime JSON contracts use synthetic authored parameter bounds."""

import importlib
import io

import pytest

PARAMETERS = [
    {"Id": "Angle", "Min": -30, "Max": 30, "Default": 0},
    {"Id": "Sway", "Min": -1, "Max": 1, "Default": 0},
]


@pytest.fixture
def assets():
    try:
        return importlib.import_module("mcp_server.tools.runtime_assets")
    except ModuleNotFoundError:
        pytest.fail("Runtime builders have not been ported")


def test_motion_counts_and_closed_loop(assets):
    data = assets.motion(PARAMETERS, 2, {"Angle": [(0, 0), (1, 12), (2, 0)]}, loop=True)
    assert data["Meta"]["TotalPointCount"] == 3
    assert data["Meta"]["TotalSegmentCount"] == 2
    assert data["Curves"][0]["Segments"] == [0, 0, 0, 1, 12, 0, 2, 0]
    with pytest.raises(ValueError, match="loop"):
        assets.motion(PARAMETERS, 2, {"Angle": [(0, 0), (2, 12)]}, loop=True)


def test_motion_fades_are_available_in_sdk_metadata(assets):
    data = assets.motion(PARAMETERS, 1, {"Angle": [(0, 0), (1, 0)]})
    assert data["Meta"]["FadeInTime"] == 0.5
    assert data["Meta"]["FadeOutTime"] == 0.5


@pytest.mark.parametrize("key,value", [("Missing", 1), ("Angle", 31), ("Angle", float("nan"))])
def test_expression_rejects_unknown_and_outside_authored_range(assets, key, value):
    with pytest.raises(ValueError):
        assets.expression(PARAMETERS, {key: value})


def test_expression_preserves_blink_multiplier(assets):
    parameters = PARAMETERS + [{"Id": "EyeOpen", "Min": 0, "Max": 1, "Default": 1}]
    data = assets.expression(parameters, {"EyeOpen": 0.3, "Angle": 5}, multiply={"EyeOpen"})
    assert data["Parameters"] == [
        {"Id": "EyeOpen", "Value": 0.3, "Blend": "Multiply"},
        {"Id": "Angle", "Value": 5, "Blend": "Overwrite"},
    ]
    with pytest.raises(ValueError):
        assets.expression(parameters, {"Angle": 5}, multiply={"EyeOpen"})


@pytest.mark.parametrize("samples", [[(0, 0), (1, 2), (0.5, 1), (2, 0)], [(1, 0), (2, 0)]])
def test_motion_rejects_invalid_time_span(assets, samples):
    with pytest.raises(ValueError, match="time"):
        assets.motion(PARAMETERS, 2, {"Angle": samples})


def test_physics_bounds_and_no_feedback(assets):
    group = assets.physics_group(PARAMETERS, "Cloth", "Angle", "Sway")
    assert group["Output"][0]["VertexIndex"] == 1
    assert len(group["Vertices"]) == 2
    assert group["Output"][0]["Destination"]["Id"] == "Sway"
    with pytest.raises(ValueError, match="feedback"):
        assets.physics_group(PARAMETERS, "Cloth", "Sway", "Sway")
    with pytest.raises(ValueError):
        assets.physics_group(PARAMETERS, "Cloth", "Angle", "Missing")


def test_texture_urls_change_with_bytes_and_repeated_call_is_idempotent(assets, tmp_path):
    helper = getattr(assets, "version_textures", None)
    assert callable(helper), "Content-addressed texture helper is missing"
    texture = tmp_path / "original.png"
    texture.write_bytes(b"abc")
    references = {"Textures": ["original.png"]}
    helper(tmp_path, references)
    assert references["Textures"] == ["textures/atlas_0_ba7816bf8f01cfea.png"]
    target = tmp_path / references["Textures"][0]
    assert target.read_bytes() == b"abc"
    before = target.stat().st_mtime_ns
    helper(tmp_path, references)
    assert target.stat().st_mtime_ns == before
    assert texture.read_bytes() == b"abc"


@pytest.mark.parametrize("invalid", ["../outside.png", "C:\\outside.png", "missing.png"])
def test_texture_preflight_prevents_partial_writes(assets, tmp_path, invalid):
    helper = getattr(assets, "version_textures", None)
    assert callable(helper), "Content-addressed texture helper is missing"
    (tmp_path / "valid.png").write_bytes(b"abc")
    references = {"Textures": ["valid.png", invalid]}
    with pytest.raises((ValueError, OSError)):
        helper(tmp_path, references)
    assert references == {"Textures": ["valid.png", invalid]}
    assert not (tmp_path / "textures").exists()


def test_texture_destination_escape_and_hash_collision_fail_before_write(assets, tmp_path):
    helper = getattr(assets, "version_textures", None)
    assert callable(helper), "Content-addressed texture helper is missing"
    root = tmp_path / "model"
    root.mkdir()
    (root / "original.png").write_bytes(b"abc")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "textures").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        helper(root, {"Textures": ["original.png"]})
    assert not list(outside.iterdir())
    (root / "textures").unlink()
    (root / "textures").mkdir()
    existing = root / "textures/atlas_0_ba7816bf8f01cfea.png"
    existing.write_bytes(b"conflicting-data")
    with pytest.raises(ValueError):
        helper(root, {"Textures": ["original.png"]})
    assert existing.read_bytes() == b"conflicting-data"


def test_texture_partial_write_can_be_retried_without_cleanup(assets, tmp_path, monkeypatch):
    (tmp_path / "original.png").write_bytes(b"abc")
    references = {"Textures": ["original.png"]}
    original_open = io.open

    class FailingWrite:
        def __init__(self, handle):
            self.handle = handle

        def write(self, data):
            self.handle.write(data[:1])
            self.handle.flush()
            raise OSError("No space left on device")

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    def interrupted_open(file, mode="r", *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        return FailingWrite(handle) if "w" in mode or "x" in mode else handle

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", interrupted_open)
        with pytest.raises(OSError, match="No space"):
            assets.version_textures(tmp_path, references)
    assert references == {"Textures": ["original.png"]}
    assert list((tmp_path / "textures").iterdir()) == []
    assets.version_textures(tmp_path, references)
    assert (tmp_path / references["Textures"][0]).read_bytes() == b"abc"
