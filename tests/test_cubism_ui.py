"""Named UI operations preserve ownership and wait for native selection state."""

import importlib
import importlib.util

import pytest


def editor(nodes, *, title="Avatar.cmo3", modals=None, delay_name=False, table_rows=None):
    """Simulate the external Swing transport, including delayed Inspector updates."""
    assert importlib.util.find_spec("mcp_server.tools.cubism_ui"), "Named UI is not implemented"
    module = importlib.import_module("mcp_server.tools.cubism_ui")
    actions, snapshots = [], [0]

    def send(payload):
        if payload["action"] == "snapshot":
            snapshots[0] += 1
            if delay_name and actions and snapshots[0] > 5:
                nodes[-1]["children"][1]["text"] = "Collar"
            return {
                "status": "success",
                "windows": [
                    {
                        "id": 1,
                        "text": f"Cubism - {title}",
                        "children": nodes,
                        "blocked_by": modals[-1]["id"] if modals else None,
                    }
                ]
                + [{"blocked_by": None, **w} for w in (modals or [])],
            }
        actions.append(payload)
        if payload["action"] == "find_table_rows":
            table = next(node for node in nodes if node["id"] == payload["id"])
            source = table_rows if table_rows is not None else table["rows"]
            selected = []
            for name in payload["values"]:
                matches = [i for i, row in enumerate(source) if row[payload["column"]] == name]
                if len(matches) != 1:
                    return {"status": "error"}
                selected.append(matches[0])
            return {"status": "success", "rows": selected}
        return {
            "status": "accepted" if payload["action"] in {"click", "mouse", "drag"} else "success"
        }

    return module.EditorUI("Avatar.cmo3", sender=send, timeout=0.08), actions


def button(widget_id, text, **extra):
    return {"id": widget_id, "kind": "button", "text": text, "showing": True, **extra}


def test_file_menu_ignores_bookmark_save():
    ui, actions = editor([button(2, "File", children=[button(3, "Save")]), button(4, "Save")])
    ui.file_menu("Save")
    assert actions == [{"action": "click", "id": 3, "expected_document": "Avatar.cmo3"}]


def test_wrong_document_never_receives_click():
    ui, actions = editor([button(2, "Save")], title="Different.cmo3")
    with pytest.raises(ValueError, match="document"):
        ui.menu("Save")
    assert actions == []


def test_active_modal_prevents_background_action():
    ui, actions = editor(
        [button(2, "Save")],
        modals=[{"id": 3, "text": "Confirm", "modal": True, "children": [button(4, "Cancel")]}],
    )
    with pytest.raises(ValueError, match="modal"):
        ui.menu("Save")
    assert actions == []


def test_duplicate_commands_are_not_silently_selected():
    ui, actions = editor([button(2, "Export"), button(3, "Export")])
    with pytest.raises(RuntimeError, match="Expected one"):
        ui.menu("Export")
    assert actions == []


@pytest.mark.parametrize("dialog_width", [160, 600])
def test_wait_window_ignores_hidden_menu_with_same_title(dialog_width):
    hidden_menu = button(
        2, "Save", showing=False, bounds=[0, 0, 260, 24], **{"class": "javax.swing.JMenuItem"}
    )
    dialog = {
        "id": 3,
        "text": "Save",
        "class": "javax.swing.JDialog",
        "modal": True,
        "showing": True,
        "bounds": [0, 0, dialog_width, 400],
    }
    ui, _ = editor([hidden_menu], modals=[dialog])
    assert ui.wait_window("Save")["id"] == 3


def test_wait_window_times_out_when_only_a_matching_menu_exists():
    ui, _ = editor([button(2, "Save", showing=False, bounds=[0, 0, 260, 24])])
    with pytest.raises(TimeoutError):
        ui.wait_window("Save")


def test_selection_waits_for_visible_inspector_name():
    table = {"id": 2, "class": "CPartsTreeTable", "rows": [["", "", "Collar"]]}
    row = {
        "id": 3,
        "kind": "component",
        "showing": True,
        "children": [
            {"id": 4, "kind": "component", "text": "Name", "showing": True},
            {"id": 5, "kind": "text", "text": "Torso", "showing": True},
        ],
    }
    ui, _ = editor([table, row], delay_name=True)
    ui.select_parts(["Collar"])
    assert row["children"][1]["text"] == "Collar"


def test_selection_times_out_if_inspector_never_updates():
    ui, _ = editor([{"id": 2, "class": "CPartsTreeTable", "rows": [["", "", "Collar"]]}])
    with pytest.raises(TimeoutError, match="appear"):
        ui.select_parts(["Collar"])


def test_parts_selection_finds_names_beyond_snapshot_rows():
    rows = [["", "", f"Part {i}"] for i in range(320)]
    table = {"id": 2, "class": "CPartsTreeTable", "rows": rows[:150]}
    ui, actions = editor([table], table_rows=rows)
    ui.select_parts(["Part 150", "Part 319"])
    assert [a["row"] for a in actions if a["action"] == "scroll_table_row"] == [150, 319]
    assert [a["y"] for a in actions if a["action"] == "mouse"] == [3311, 7029]
    assert all(a["expected_document"] == "Avatar.cmo3" for a in actions)


def test_parts_selection_rejects_duplicate_names_after_snapshot_limit():
    rows = [["", "", f"Part {i}"] for i in range(151)]
    rows[-1][2] = "Part 0"
    table = {"id": 2, "class": "CPartsTreeTable", "rows": rows[:150]}
    ui, actions = editor([table], table_rows=rows)
    with pytest.raises(RuntimeError):
        ui.select_parts(["Part 0"])
    assert not any(a["action"] in {"scroll_table_row", "mouse"} for a in actions)


def test_nonfinite_parameter_does_not_reach_editor():
    ui, actions = editor([])
    with pytest.raises(ValueError, match="finite"):
        ui.parameter_value("Body", float("nan"))
    assert actions == []


def test_parameter_scrolls_then_commits_exact_value():
    row = {
        "id": 2,
        "text": "singleRangeBox",
        "children": [
            {"id": 3, "text": "Body"},
            {"id": 4, "class": "CCustomPaint", "bounds": [0, 2000, 104, 26]},
            {"id": 5, "class": "editor.CSlidableFloat$b"},
        ],
    }
    ui, actions = editor([row])
    ui.parameter_value("Body", 0.25)
    assert actions[0] == {"action": "scroll_into_view", "id": 2, "expected_document": "Avatar.cmo3"}
    assert actions[-1] == {
        "action": "set_number",
        "id": 5,
        "value": 0.25,
        "expected_document": "Avatar.cmo3",
    }


def psd_import_editor(options):
    """Simulate PSD dialogs at the transport boundary while exercising real UI guards."""
    module = importlib.import_module("mcp_server.tools.cubism_ui")
    actions = []
    phase = [0]
    dialogs = [
        {
            "id": 10,
            "text": "Open",
            "children": [{"id": 11, "kind": "text", "showing": True}, button(12, "Open")],
        },
        {
            "id": 20,
            "text": "Model settings",
            "children": [
                {"id": 21, "options": ["Other (Model)", "Avatar (Model)"]},
                button(22, "OK"),
            ],
        },
        {
            "id": 30,
            "text": "Re-import settings",
            "children": [{"id": 31, "options": options}, button(32, "OK")],
        },
    ]
    for dialog in dialogs:
        dialog.update(
            modal=True, blocked_by=None, bounds=[0, 0, 600, 400], **{"class": "javax.swing.JDialog"}
        )

    def send(payload):
        if payload["action"] == "snapshot":
            windows = [
                {
                    "id": 1,
                    "text": "Cubism - Avatar.cmo3",
                    "blocked_by": dialogs[phase[0] - 1]["id"] if 0 < phase[0] < 4 else None,
                    "children": [button(2, "File", children=[button(3, "Open...")])],
                }
            ]
            if 0 < phase[0] < 4:
                windows.append(dialogs[phase[0] - 1])
            return {"status": "success", "windows": windows}
        actions.append(payload)
        if payload["action"] == "click":
            phase[0] = {3: 1, 12: 2, 22: 3, 32: 4}[payload["id"]]
        return {"status": "success"}

    return module.EditorUI("Avatar.cmo3", sender=send, timeout=0.08), actions


def test_psd_replacement_selects_exact_filename_among_multiple_sources(tmp_path):
    path = tmp_path / "outfit.psd"
    path.write_bytes(b"synthetic PSD")
    ui, actions = psd_import_editor(
        [
            "< Add all layers as new ArtMesh >",
            "Replace [ outfit-back.psd ] (2026/09/27 11:22)",
            "Replace [ outfit.psd.bak ] (2026/09/27 11:23)",
            "Replace [ outfit.psd ] (2026/09/28 12:34)",
            "Replace [ face.psd ] (2026/09/28 12:35)",
        ]
    )
    ui.import_psd(path, "Avatar", replace=True)
    assert {
        "action": "select_list",
        "id": 31,
        "index": 3,
        "expected_document": "Avatar.cmo3",
    } in actions
    assert actions[-1] == {"action": "click", "id": 32, "expected_document": "Avatar.cmo3"}


def test_psd_replacement_allows_explicit_source_for_different_input_name(tmp_path):
    path = tmp_path / "updated-outfit.psd"
    path.write_bytes(b"synthetic PSD")
    ui, actions = psd_import_editor(
        [
            "< Add all layers as new ArtMesh >",
            "Replace [outfit.psd] (2026/09/28 12:34)",
            "Replace [ updated-outfit.psd ] (2026/09/28 12:35)",
        ]
    )
    ui.import_psd(path, "Avatar", replace=True, replace_source="outfit.psd")
    assert {
        "action": "select_list",
        "id": 31,
        "index": 1,
        "expected_document": "Avatar.cmo3",
    } in actions


@pytest.mark.parametrize(
    "replacements",
    [
        ["Replace [ other.psd ] (2026/09/28 12:34)"],
        ["Replace [ outfit.psd ] (2026/09/27 12:34)", "Replace [ outfit.psd ] (2026/09/28 12:34)"],
    ],
)
def test_psd_replacement_rejects_missing_or_ambiguous_source_without_confirmation(
    tmp_path, replacements
):
    path = tmp_path / "outfit.psd"
    path.write_bytes(b"synthetic PSD")
    ui, actions = psd_import_editor(["< Add all layers as new ArtMesh >", *replacements])
    with pytest.raises(RuntimeError, match="source"):
        ui.import_psd(path, "Avatar", replace=True)
    assert not any(action["id"] in (31, 32) for action in actions)


def test_psd_add_new_preserves_existing_sources(tmp_path):
    path = tmp_path / "outfit.psd"
    path.write_bytes(b"synthetic PSD")
    ui, actions = psd_import_editor(
        [
            "Replace [ outfit.psd ] (2026/09/28 12:34)",
            "< Add all layers as new ArtMesh >",
        ]
    )
    ui.import_psd(path, "Avatar")
    assert {
        "action": "select_list",
        "id": 31,
        "index": 1,
        "expected_document": "Avatar.cmo3",
    } in actions


@pytest.mark.parametrize(
    "options",
    [
        {"replace_source": "outfit.psd"},
        {"replace": True, "replace_source": ""},
        {"replace": True, "replace_source": "../outfit.psd"},
        {"replace": True, "replace_source": "folder\\outfit.psd"},
    ],
)
def test_invalid_psd_replacement_arguments_stop_before_editor_actions(tmp_path, options):
    path = tmp_path / "outfit.psd"
    path.write_bytes(b"synthetic PSD")
    ui, actions = psd_import_editor([])
    with pytest.raises(ValueError):
        ui.import_psd(path, "Avatar", **options)
    assert actions == []
