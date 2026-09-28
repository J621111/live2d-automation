"""Guarded macOS Cubism Editor automation through System Events."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from typing import Any

JsonDict = dict[str, Any]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_MENU_PATHS: dict[str, tuple[str, ...]] = {
    "auto_deformer": ("Modeling", "Deformer", "Auto Generation of Deformer..."),
    "face_deformer": (
        "Modeling",
        "Parameter",
        "Auto Generation of Face Motion",
        "Generate Face Deformer...",
    ),
    "face_motion": (
        "Modeling",
        "Parameter",
        "Auto Generation of Face Motion",
        "Generate Face Motion...",
    ),
    "sway_motion": ("Modeling", "Parameter", "Auto Generation of Sway Motion..."),
    "eye_lip_settings": ("Modeling", "Parameter", "Settings for Eye Blinking and Lip-sync..."),
    "texture_atlas": ("Modeling", "Texture Atlas", "Edit Texture Atlas..."),
    "model_template": ("Modeling", "Model Template", "Apply Template..."),
    "export_moc3": ("File", "Export For Runtime", "Export as moc3 file..."),
}


def _quoted(value: str) -> str:
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError("Editor identifiers must be nonempty and contain no control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class MacOSCubismController:
    """Operate only a named Cubism document through known editor menus."""

    def __init__(
        self,
        run_command: CommandRunner = subprocess.run,
        *,
        process_name: str = "CubismEditor",
        bundle_id: str | None = None,
    ) -> None:
        self._run_command = run_command
        self._process_name = _quoted(process_name)
        configured_bundle = bundle_id or os.environ.get("LIVE2D_CUBISM_BUNDLE_ID")
        self._bundle_id = _quoted(configured_bundle) if configured_bundle else None

    def _script(self, body: str) -> str:
        predicate = f"name is {self._process_name}"
        if self._bundle_id:
            predicate += f" and bundle identifier is {self._bundle_id}"
        return (
            'tell application "System Events"\n'
            f"  set matches to (application processes whose {predicate})\n"
            '  if (count of matches) is not 1 then error "Expected one matching Cubism process"\n'
            "  tell item 1 of matches\n" + body + "\n  end tell\nend tell"
        )

    def status(self) -> JsonDict:
        """Report the matching editor's model and windows without activating it."""
        if sys.platform != "darwin":
            return {"status": "unsupported", "windows": [], "active_document": None}
        result = self._run(
            self._script(
                "    set windowNames to get name of every window\n"
                '    set windowText to ""\n'
                "    repeat with windowName in windowNames\n"
                "      set windowText to windowText & (windowName as text) & linefeed\n"
                "    end repeat\n"
                "    return windowText"
            )
        )
        if result.returncode != 0:
            return {
                "status": "missing",
                "windows": [],
                "active_document": None,
                "details": result.stderr.strip(),
            }
        windows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        document = None
        for title in windows:
            match = re.search(r" - ([^/\\]+\.cmo3)$", title)
            if match:
                document = match.group(1)
                break
        return {
            "status": "ready" if document else "missing",
            "windows": windows,
            "active_document": document,
        }

    def open_menu(self, action: str, expected_document: str) -> JsonDict:
        """Open a whitelisted menu only when the named model has no blocking dialog."""
        path = _MENU_PATHS.get(action)
        if path is None:
            return {"status": "error", "details": f"Unsupported Cubism action: {action}"}
        if (
            not isinstance(expected_document, str)
            or not expected_document.endswith(".cmo3")
            or "/" in expected_document
            or "\\" in expected_document
            or any(ord(char) < 32 for char in expected_document)
        ):
            return {"status": "error", "details": "Expected a Cubism document filename."}
        editor = self.status()
        if editor["status"] != "ready":
            return {"status": "error", "details": "Cubism Editor has no open model."}
        if editor["active_document"] != expected_document:
            return {"status": "error", "details": "The active Cubism document does not match."}
        if len(editor["windows"]) != 1:
            return {"status": "error", "details": "Close the current Cubism dialog first."}

        menu_item = f'menu item "{path[-1]}"'
        for parent in reversed(path[1:-1]):
            menu_item += f' of menu 1 of menu item "{parent}"'
        menu_item += f' of menu 1 of menu bar item "{path[0]}" of menu bar 1'
        script = self._script(
            '    if (count of windows) is not 1 then error "A Cubism dialog is open"\n'
            f'    if (name of window 1) is not {_quoted(editor["windows"][0])} '
            'then error "The active Cubism document changed"\n'
            "    set frontmost to true\n"
            f"    click {menu_item}\n"
        )
        result = self._run(script)
        return {
            "status": "success" if result.returncode == 0 else "error",
            "action": action,
            **({"details": result.stderr.strip()} if result.returncode != 0 else {}),
        }

    def _run(self, script: str) -> subprocess.CompletedProcess[str]:
        try:
            return self._run_command(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(["osascript"], 1, "", str(exc))
