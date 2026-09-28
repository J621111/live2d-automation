"""Synthetic AppleScript boundary checks never operate a running editor."""

import subprocess
import sys

import pytest


def runner(windows, calls):
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, windows if len(calls) == 1 else "", "")

    return run


def test_status_parses_configured_editor_without_shell_interpolation(monkeypatch):
    from mcp_server.tools.macos_cubism_controller import MacOSCubismController

    monkeypatch.setattr("sys.platform", "darwin")
    calls = []
    controller = MacOSCubismController(
        runner("Editor - portrait.cmo3\n", calls),
        process_name='Cubism "Preview"',
        bundle_id="com.example.editor",
    )
    result = controller.status()
    assert result["active_document"] == "portrait.cmo3"
    assert calls[0][:2] == ["osascript", "-e"]
    assert r'name is "Cubism \"Preview\""' in calls[0][-1]
    assert 'bundle identifier is "com.example.editor"' in calls[0][-1]


@pytest.mark.parametrize("windows", ["Editor - other.cmo3\n", "Dialog\nEditor - portrait.cmo3\n"])
def test_wrong_document_or_modal_never_sends_a_menu_click(monkeypatch, windows):
    from mcp_server.tools.macos_cubism_controller import MacOSCubismController

    monkeypatch.setattr("sys.platform", "darwin")
    calls = []
    result = MacOSCubismController(runner(windows, calls)).open_menu("export_moc3", "portrait.cmo3")
    assert result["status"] == "error" and len(calls) == 1


def test_menu_script_rechecks_document_inside_the_same_click_transaction(monkeypatch):
    from mcp_server.tools.macos_cubism_controller import MacOSCubismController

    monkeypatch.setattr("sys.platform", "darwin")
    calls = []
    result = MacOSCubismController(runner('Editor - portrait "smile".cmo3\n', calls)).open_menu(
        "export_moc3", 'portrait "smile".cmo3'
    )
    assert result["status"] == "success"
    script = calls[-1][-1]
    assert "count of windows" in script and "name of window 1" in script
    assert 'menu item "Export as moc3 file..."' in script
    assert script.index("name of window 1") < script.index("click menu item")


def test_nonmac_and_unknown_action_have_no_external_side_effect(monkeypatch):
    from mcp_server.tools.macos_cubism_controller import MacOSCubismController

    monkeypatch.setattr("sys.platform", "linux")
    calls = []
    controller = MacOSCubismController(runner("", calls))
    assert controller.status()["status"] == "unsupported"
    assert controller.open_menu("eval", "portrait.cmo3")["status"] == "error"
    assert calls == []


def test_bad_process_control_characters_are_rejected():
    from mcp_server.tools.macos_cubism_controller import MacOSCubismController

    with pytest.raises(ValueError):
        MacOSCubismController(process_name='Editor"\n do shell script "bad')


@pytest.mark.skipif(sys.platform != "darwin", reason="Uses the native AppleScript interpreter")
def test_menu_recheck_rejects_a_document_with_the_same_filename_suffix():
    from mcp_server.tools.macos_cubism_controller import MacOSCubismController

    calls = []
    MacOSCubismController(runner("Editor - Avatar.cmo3\n", calls)).open_menu(
        "export_moc3", "Avatar.cmo3"
    )
    guard = next(line for line in calls[-1][-1].splitlines() if "if (name of window 1)" in line)
    script = 'set observedTitle to "Editor - Draft - Avatar.cmo3"\n' + guard.replace(
        "(name of window 1)", "observedTitle"
    )
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "document changed" in result.stderr
