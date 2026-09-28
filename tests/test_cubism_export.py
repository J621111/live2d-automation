"""Native export workflows must not report stale files or overwrite silently."""

import importlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
from PIL import Image


def exports():
    assert importlib.util.find_spec(
        "mcp_server.tools.cubism_export"
    ), "Native export is not implemented"
    return importlib.import_module("mcp_server.tools.cubism_export")


class ExportEditor:
    """Model the external editor dialogs and an optional generated output file."""

    document = "Avatar.cmo3"

    def __init__(self, *, creates=True):
        self.calls = []
        self.path = None
        self.creates = creates

    def snapshot(self):
        return {"windows": [{"text": "Cubism - Avatar.cmo3"}]}, []

    def menu(self, name):
        self.calls.append(name)

    def wait_window(self, name):
        """Return protocol-shaped settings or Save controls."""
        return {
            "id": 10,
            "text": name,
            "children": [
                {"id": 1, "kind": "combo", "options": ["For SDK 4.2 / Cubism4.2"]},
                {"id": 2, "kind": "text", "showing": True},
                {"id": 3, "kind": "button", "text": "Export Current Display Content as PSD"},
            ],
        }

    def action(self, payload):
        self.calls.append(payload)
        if payload["action"] == "set_text":
            self.path = Path(payload["value"])

    def click(self, widget):
        self.calls.append(widget)

    def dialog_button(self, window, name):
        self.calls.append((window["text"], name))
        if name == "Save" and self.creates:
            self.path.write_bytes(b"MOC3 synthetic export")


def test_export_uses_document_basename_and_actual_sdk_selection(tmp_path):
    ui = ExportEditor()
    path = exports().export_moc3(ui, tmp_path / "bundle")
    assert path.name == "Avatar.moc3"
    assert path.read_bytes() == b"MOC3 synthetic export"
    assert {"action": "select_combo", "id": 1, "value": "For SDK 4.2 / Cubism4.2"} in ui.calls


def test_existing_bundle_is_rejected_before_editor_actions(tmp_path):
    (tmp_path / "unrelated.txt").write_text("keep")
    ui = ExportEditor()
    with pytest.raises(FileExistsError):
        exports().export_moc3(ui, tmp_path)
    assert ui.calls == []
    assert (tmp_path / "unrelated.txt").read_text() == "keep"


def test_stale_export_is_not_reported_as_success(tmp_path):
    (tmp_path / "Avatar.moc3").write_bytes(b"old")
    with pytest.raises(TimeoutError):
        exports().export_moc3(ExportEditor(creates=False), tmp_path, overwrite=True, timeout=0.01)


def test_unknown_initial_dialog_stops_before_export(tmp_path):
    ui = ExportEditor()
    ui.snapshot = lambda: ({"windows": [{}, {"text": "Unsaved model", "modal": True}]}, [])
    with pytest.raises(RuntimeError, match="dialog"):
        exports().export_moc3(ui, tmp_path / "new")
    assert ui.calls == []


def save_dialog_editor(
    path,
    *,
    prompt="File already exists. Overwrite?",
    prompt_title="Confirm",
    blocked_confirmation=False,
):
    """Exercise the real UI guards with an overwrite modal above its Save parent."""
    from mcp_server.tools.cubism_ui import EditorUI

    main = {"id": 1, "text": "Cubism - Avatar.cmo3", "blocked_by": 2}
    save = {
        "id": 2,
        "text": "Save",
        "blocked_by": None,
        "modal": True,
        "showing": True,
        "class": "javax.swing.JDialog",
        "bounds": [0, 0, 600, 400],
        "children": [
            {"id": 3, "kind": "text", "showing": True},
            {"id": 4, "kind": "button", "text": "Save"},
        ],
    }
    confirm = {
        "id": 5,
        "text": prompt_title,
        "blocked_by": None,
        "modal": True,
        "showing": True,
        "class": "javax.swing.JDialog",
        "children": [
            {"id": 6, "text": prompt},
            {"id": 7, "kind": "button", "text": "Yes"},
        ],
    }
    windows, clicks = [main, save], []
    closing_snapshots = []

    def send(payload):
        if payload["action"] == "snapshot":
            if 7 in clicks:
                closing_snapshots.append(True)
                if len(closing_snapshots) > 1:
                    windows[:] = [main]
                    main["blocked_by"] = None
            return {"status": "success", "windows": windows.copy()}
        if payload["action"] == "set_text":
            assert payload["id"] == 3 and payload["value"] == str(path)
        elif payload["action"] == "click":
            clicks.append(payload["id"])
            if payload["id"] == 4:
                save["blocked_by"] = 5
                windows.append(confirm)
                if blocked_confirmation:
                    windows[:] = [
                        main,
                        confirm,
                        save,
                        {
                            "id": 8,
                            "text": "Confirm",
                            "modal": True,
                            "blocked_by": 5,
                            "children": [
                                {"id": 9, "text": "Overwrite file?"},
                                {"id": 10, "kind": "button", "text": "Yes"},
                            ],
                        },
                    ]
            elif payload["id"] == 7:
                path.write_bytes(b"fresh MOC3")
                save["blocked_by"] = None
                windows[:] = [main, save]
            else:
                pytest.fail("Unexpected export click")
            return {"status": "accepted"}
        return {"status": "success"}

    return EditorUI("Avatar.cmo3", sender=send), clicks, closing_snapshots


@pytest.mark.parametrize("blocked_confirmation", [False, True])
def test_overwrite_confirmation_is_handled_above_save_parent(tmp_path, blocked_confirmation):
    path = tmp_path / "Avatar.moc3"
    path.write_bytes(b"old")
    ui, clicks, closing_snapshots = save_dialog_editor(
        path, blocked_confirmation=blocked_confirmation
    )
    assert exports()._save(ui, path, overwrite=True, timeout=1) == path
    assert path.read_bytes() == b"fresh MOC3"
    assert clicks == [4, 7]
    assert len(closing_snapshots) == 2


def test_nested_overwrite_without_permission_preserves_existing_file(tmp_path):
    path = tmp_path / "Avatar.moc3"
    path.write_bytes(b"old")
    ui, clicks, _ = save_dialog_editor(path)
    with pytest.raises(FileExistsError):
        exports()._save(ui, path, overwrite=False, timeout=1)
    assert path.read_bytes() == b"old"
    assert clicks == [4]


@pytest.mark.parametrize("title", ["Unexpected", "Save"])
def test_export_rejects_unrecognized_dialog_even_with_known_save_parent(tmp_path, title):
    path = tmp_path / "Avatar.moc3"
    path.write_bytes(b"old")
    ui, clicks, _ = save_dialog_editor(path, prompt="Something else?", prompt_title=title)
    with pytest.raises(RuntimeError, match="Unexpected export dialog"):
        exports()._save(ui, path, overwrite=True, timeout=1)
    assert path.read_bytes() == b"old"
    assert clicks == [4]


def bundle(path, *, pixel="red", moc=b"MOC3 test"):
    path.mkdir()
    (path / "Avatar.moc3").write_bytes(moc)
    Image.new("RGBA", (2, 2), pixel).save(path / "texture.png")
    (path / "Avatar.model3.json").write_text(
        json.dumps({"FileReferences": {"Moc": "Avatar.moc3", "Textures": ["texture.png"]}})
    )
    return path / "Avatar.model3.json"


def test_export_comparison_checks_moc_and_decoded_pixels(tmp_path):
    before = bundle(tmp_path / "before")
    after = bundle(tmp_path / "after")
    assert exports().compare_exports(before, after)["identical"]
    Image.new("RGBA", (2, 2), "blue").save(after.parent / "texture.png")
    assert not exports().compare_exports(before, after)["identical"]
    Image.new("RGBA", (2, 2), "red").save(after.parent / "texture.png")
    (after.parent / "Avatar.moc3").write_bytes(b"different")
    assert not exports().compare_exports(before, after)["identical"]


def test_export_comparison_rejects_escape_in_descriptor(tmp_path):
    before = bundle(tmp_path / "before")
    after = bundle(tmp_path / "after")
    before.write_text(
        json.dumps({"FileReferences": {"Moc": "../secret.moc3", "Textures": ["texture.png"]}})
    )
    with pytest.raises(ValueError, match="reference"):
        exports().compare_exports(before, after)


def test_psd_export_keeps_current_pose_and_uses_current_display_option(tmp_path):
    ui = ExportEditor()
    path = exports().export_psd(ui, tmp_path / "pose.psd")
    assert path.exists()
    assert "Reset to default values" not in ui.calls
    assert any(
        isinstance(value, dict) and value.get("text") == "Export Current Display Content as PSD"
        for value in ui.calls
    )


def test_psd_collision_is_rejected_before_editor_actions(tmp_path):
    path = tmp_path / "pose.psd"
    path.write_bytes(b"original")
    ui = ExportEditor()
    with pytest.raises(FileExistsError):
        exports().export_psd(ui, path)
    assert path.read_bytes() == b"original" and not ui.calls


@pytest.fixture
def export_root(tmp_path, monkeypatch):
    """Use an isolated allowed root while keeping the production path validator active."""
    from mcp_server import validation

    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(validation, "OUTPUT_ROOT", root)
    return root


class UnreachableEditor(ExportEditor):
    """Ensure invalid destinations are rejected before any editor interaction."""

    def snapshot(self):
        pytest.fail("Invalid export destination reached the editor")


@pytest.mark.parametrize("kind", ["moc3", "psd"])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("path_kind", ["absolute", "parent_traversal", "symlink_parent"])
def test_export_rejects_paths_outside_root_before_editor(
    export_root, tmp_path, monkeypatch, kind, overwrite, path_kind
):
    from mcp_server.validation import InputValidationError

    outside = tmp_path / "outside"
    name = "Avatar" if kind == "moc3" else "Avatar.psd"
    if path_kind == "absolute":
        destination = outside / name
    elif path_kind == "parent_traversal":
        monkeypatch.chdir(tmp_path)
        destination = Path("output/../outside") / name
    else:
        (export_root / "alias").symlink_to(outside, target_is_directory=True)
        destination = export_root / "alias" / name
    with pytest.raises(InputValidationError):
        getattr(exports(), f"export_{kind}")(UnreachableEditor(), destination, overwrite=overwrite)
    assert not outside.exists()


@pytest.mark.parametrize("kind", ["moc3", "psd"])
def test_export_rejects_target_file_symlink_escape(export_root, tmp_path, kind):
    from mcp_server.validation import InputValidationError

    original = tmp_path / f"outside.{kind}"
    original.write_bytes(b"original")
    destination = export_root / f"Avatar.{kind}"
    destination.symlink_to(original)
    requested = export_root if kind == "moc3" else destination
    with pytest.raises(InputValidationError):
        getattr(exports(), f"export_{kind}")(UnreachableEditor(), requested, overwrite=True)
    assert original.read_bytes() == b"original"


@pytest.mark.parametrize("kind", ["moc3", "psd"])
@pytest.mark.parametrize("parent_checkout", [False, True])
def test_export_keeps_paths_relative_to_calling_directory(
    tmp_path, monkeypatch, kind, parent_checkout
):
    from mcp_server import validation

    monkeypatch.chdir(tmp_path)
    relative_root = Path("live2d-automation/output" if parent_checkout else "output")
    root = tmp_path / relative_root
    monkeypatch.setattr(validation, "OUTPUT_ROOT", root)
    destination = relative_root / ("Avatar" if kind == "moc3" else "Avatar.psd")
    result = getattr(exports(), f"export_{kind}")(ExportEditor(), destination)
    expected = root / "Avatar/Avatar.moc3" if kind == "moc3" else root / "Avatar.psd"
    assert result == expected
    assert result.is_file()


@pytest.mark.parametrize("kind", ["moc3", "psd"])
def test_export_cli_rejects_outside_destination(export_root, tmp_path, monkeypatch, capsys, kind):
    from mcp_server.tools import cubism_ui

    destination = tmp_path / "outside" / ("Avatar" if kind == "moc3" else "Avatar.psd")
    ui = UnreachableEditor()
    monkeypatch.setattr(cubism_ui, "EditorUI", lambda *args, **kwargs: ui)
    monkeypatch.setattr(
        "sys.argv",
        ["live2d-cubism-export", kind, "--document", "Avatar.cmo3", "--output", str(destination)],
    )
    with pytest.raises(SystemExit) as caught:
        exports().main()
    assert caught.value.code == 1
    assert "output directory" in capsys.readouterr().err
    assert not destination.parent.exists()


@pytest.mark.parametrize(
    "escape",
    [
        "atlas_directory",
        "texture_file",
        "descriptor",
        "alias_chain",
        "moc_alias",
        "dangling_atlas",
        pytest.param(
            "space_directory",
            marks=pytest.mark.skipif(
                os.name != "posix", reason="Requires literal trailing spaces in directory names"
            ),
        ),
    ],
)
def test_overwrite_rejects_companion_escape_before_editor(export_root, tmp_path, escape):
    from mcp_server.validation import InputValidationError

    folder = export_root / "bundle"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original = outside / "untouched.bin"
    original.write_bytes(b"original")
    if escape == "atlas_directory":
        (folder / "Avatar.2048").symlink_to(outside, target_is_directory=True)
    elif escape == "texture_file":
        (folder / "Avatar.4096").mkdir()
        (folder / "Avatar.4096/texture_00.png").symlink_to(original)
    elif escape == "descriptor":
        (folder / "Avatar.model3.json").symlink_to(original)
    elif escape == "alias_chain":
        shared = export_root / "shared"
        shared.mkdir()
        (folder / "Avatar.2048").symlink_to(shared, target_is_directory=True)
        (shared / "nested").symlink_to(outside, target_is_directory=True)
    elif escape == "moc_alias":
        effective = export_root / "effective"
        effective.mkdir()
        (effective / "Other.moc3").write_bytes(b"existing MOC")
        (folder / "Avatar.moc3").symlink_to(effective / "Other.moc3")
        (effective / "Other.2048").symlink_to(outside, target_is_directory=True)
    elif escape == "dangling_atlas":
        (folder / "Avatar.2048").symlink_to(outside / "future", target_is_directory=True)
    else:
        nested = folder / "nested "
        nested.mkdir()
        (nested / "texture.png").symlink_to(original)
    with pytest.raises(InputValidationError):
        exports().export_moc3(UnreachableEditor(), folder, overwrite=True)
    assert original.read_bytes() == b"original"


def test_overwrite_accepts_confined_links_and_terminates_directory_cycles(export_root, monkeypatch):
    folder, shared = export_root / "bundle", export_root / "shared"
    folder.mkdir()
    shared.mkdir()
    texture = shared / "texture.png"
    texture.write_bytes(b"existing texture")
    (folder / "Avatar.2048").symlink_to(shared, target_is_directory=True)
    (shared / "back").symlink_to(folder, target_is_directory=True)
    original_iterdir = Path.iterdir
    visits = []

    def bounded_iterdir(path):
        visits.append(path)
        assert len(visits) < 10, "Companion validation did not terminate a directory cycle"
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", bounded_iterdir)
    result = exports().export_moc3(ExportEditor(), folder, overwrite=True)
    assert result == folder / "Avatar.moc3"
    assert result.is_file()
    assert texture.read_bytes() == b"existing texture"
    assert (folder / "Avatar.2048").is_symlink()
