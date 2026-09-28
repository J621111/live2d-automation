"""Native editor exports and byte/pixel comparisons, without shipping a Cubism SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from PIL import Image

from mcp_server import validation

if TYPE_CHECKING:
    from mcp_server.tools.cubism_ui import EditorUI


def _nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    result = [node]
    for child in node.get("children", []):
        result.extend(_nodes(child))
    return result


def _one(nodes: list[dict[str, Any]], predicate: Any, label: str) -> dict[str, Any]:
    found = [node for node in nodes if predicate(node)]
    if len(found) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(found)}")
    return found[0]


def _stamp(path: Path) -> tuple[int, int, int] | None:
    if not path.is_file():
        return None
    info = path.stat()
    return info.st_mtime_ns, info.st_ctime_ns, info.st_size


def _ready(ui: EditorUI, timeout: float) -> None:
    if not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("Export timeout must be finite and between zero and 120 seconds")
    state, _ = ui.snapshot()
    if any(w.get("text") for w in state["windows"][1:]):
        raise RuntimeError("Resolve the open dialog before exporting")


def _save(ui: EditorUI, path: Path, *, overwrite: bool, timeout: float) -> Path:
    previous = _stamp(path)
    save_window = ui.wait_window("Save")
    field = _one(
        _nodes(save_window),
        lambda n: n.get("kind") == "text" and n.get("showing"),
        "export path field",
    )
    ui.action({"action": "set_text", "id": field["id"], "value": str(path)})
    ui.dialog_button(save_window, "Save")
    deadline = time.monotonic() + timeout
    while True:
        state, _ = ui.snapshot()
        dialogs = [w for w in state["windows"][1:] if w.get("text")]
        for window in dialogs:
            if "blocked_by" not in window:
                raise RuntimeError("Missing modal blocking information; reattach the bridge")
            if window["blocked_by"] is not None:
                continue
            if window.get("id") == save_window["id"] and window.get("text") == "Save":
                continue
            nodes = _nodes(window)
            if any("overwrite" in str(n.get("text", "")).lower() for n in nodes):
                if not overwrite:
                    raise FileExistsError("Cubism requested an overwrite; export was not confirmed")
                button = _one(
                    nodes,
                    lambda n: n.get("kind") == "button"
                    and n.get("text") in ("Yes(Y)", "Yes", "OK"),
                    "overwrite confirmation",
                )
                ui.click(button)
                break
            elif window.get("text") != "Progress":
                raise RuntimeError(f"Unexpected export dialog: {window.get('text')}")
        current = _stamp(path)
        if not dialogs and current is not None and current != previous and current[2] > 0:
            return path
        if time.monotonic() >= deadline:
            raise TimeoutError("Cubism did not produce a fresh export before the timeout")
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def _validate_existing_export_paths(*folders: Path) -> None:
    """Check every existing companion path, following confined directory aliases once."""
    pending = list(folders)
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if not validation.is_relative_to(path, validation.OUTPUT_ROOT):
            raise validation.InputValidationError(
                "Export companion path must stay inside the project output directory."
            )
        if path in visited or not path.is_dir():
            continue
        visited.add(path)
        pending.extend(path.iterdir())


def export_moc3(
    ui: EditorUI,
    output_dir: Path,
    *,
    basename: str | None = None,
    sdk_version: str = "4.2",
    overwrite: bool = False,
    timeout: float = 30,
) -> Path:
    """Export the expected open document through the editor's SDK export command.

    The output directory must be empty unless overwrite is explicitly enabled,
    because the editor writes the MOC, descriptor, atlas and companion files.
    Destinations must resolve inside the configured project output root.
    Overwrite also checks existing companion paths before contacting the editor.
    """
    basename = basename or Path(ui.document).stem
    if not basename or basename in (".", "..") or any(c in basename for c in "/\\\x00"):
        raise ValueError("Export basename must be a filename stem")
    folder = validation.resolve_output_dir(Path(output_dir).resolve())
    if folder.exists() and (not folder.is_dir() or (any(folder.iterdir()) and not overwrite)):
        raise FileExistsError("Export directory already contains files; use an empty directory")
    path = validation.resolve_output_dir(folder / f"{basename}.moc3")
    if overwrite:
        _validate_existing_export_paths(folder, path.parent)
    _ready(ui, timeout)
    folder.mkdir(parents=True, exist_ok=True)
    ui.menu("Reset to default values")
    ui.menu("Deselect")
    ui.menu("Export as moc3 file...")
    window = ui.wait_window("Export settings")
    option = f"For SDK {sdk_version} / Cubism{sdk_version}"
    combo = _one(
        _nodes(window),
        lambda n: n.get("kind") == "combo" and option in n.get("options", []),
        f"SDK {sdk_version} selector",
    )
    ui.action({"action": "select_combo", "id": combo["id"], "value": option})
    ui.dialog_button(window, "OK")
    return _save(ui, path, overwrite=overwrite, timeout=timeout)


def export_psd(
    ui: EditorUI, destination: Path, *, overwrite: bool = False, timeout: float = 30
) -> Path:
    """Export the current pose as PSD inside the configured project output root."""
    path = validation.resolve_output_dir(Path(destination).resolve())
    if path.suffix.lower() != ".psd":
        raise ValueError("PSD export destination must end in .psd")
    if path.exists() and (not path.is_file() or not overwrite):
        raise FileExistsError("PSD output exists; choose another destination or allow overwrite")
    _ready(ui, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    ui.menu("Deselect")
    ui.menu("PSD images(beta)...")
    window = ui.wait_window("Export PSD (beta)")
    radio = _one(
        _nodes(window),
        lambda n: n.get("text") == "Export Current Display Content as PSD",
        "current-pose export option",
    )
    ui.click(radio)
    ui.dialog_button(window, "OK")
    ui.dialog_button(ui.wait_window("PSD Image Output Settings"), "OK")
    return _save(ui, path, overwrite=overwrite, timeout=timeout)


def _resource(folder: Path, reference: Any) -> Path:
    if not isinstance(reference, str) or not reference or "\x00" in reference:
        raise ValueError("Invalid export resource reference")
    relative = Path(reference.replace("\\", "/"))
    path = (folder / relative).resolve()
    if (
        relative.is_absolute()
        or PureWindowsPath(reference).drive
        or ".." in relative.parts
        or not path.is_relative_to(folder)
        or not path.is_file()
    ):
        raise ValueError("Export reference is missing or escapes its bundle")
    return path


def _bundle(model_path: Path) -> tuple[bytes, list[tuple[tuple[int, int], bytes]]]:
    path = Path(model_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        refs = data["FileReferences"]
        if not isinstance(refs["Textures"], list) or not refs["Textures"]:
            raise ValueError("Export must reference at least one texture")
        moc = _resource(path.parent, refs["Moc"]).read_bytes()
        textures = []
        for name in refs["Textures"]:
            with Image.open(_resource(path.parent, name)) as image:
                textures.append((image.size, image.convert("RGBA").tobytes()))
        return moc, textures
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid export resource structure") from exc


def compare_exports(before: Path, after: Path) -> dict[str, Any]:
    """Compare MOC bytes and decoded atlas pixels from two native exports.

    This checks export stability after a manual save/reopen. It does not prove
    binary SDK compatibility or semantic equivalence of different MOC files.
    """
    first, first_textures = _bundle(before)
    second, second_textures = _bundle(after)
    return {
        "identical": first == second and first_textures == second_textures,
        "moc_identical": first == second,
        "textures_identical": first_textures == second_textures,
        "before_sha256": hashlib.sha256(first).hexdigest(),
        "after_sha256": hashlib.sha256(second).hexdigest(),
        "texture_count_before": len(first_textures),
        "texture_count_after": len(second_textures),
    }


def main() -> None:
    """Run installed native-export commands with explicit document and destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for name in ("moc3", "psd"):
        command = actions.add_parser(name)
        command.add_argument("--document", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--session-dir", type=Path)
        command.add_argument("--timeout", type=float, default=30)
        command.add_argument("--overwrite", action="store_true")
        if name == "moc3":
            command.add_argument("--basename")
            command.add_argument("--sdk-version", default="4.2")
    compare = actions.add_parser("compare")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            result = compare_exports(args.before, args.after)
            print(json.dumps(result, indent=2))
            if not result["identical"]:
                raise SystemExit(1)
        else:
            from mcp_server.tools.cubism_ui import EditorUI

            ui = EditorUI(args.document, session_dir=args.session_dir)
            options = {"overwrite": args.overwrite, "timeout": args.timeout}
            if args.command == "moc3":
                path = export_moc3(
                    ui, args.output, basename=args.basename, sdk_version=args.sdk_version, **options
                )
            else:
                path = export_psd(ui, args.output, **options)
            print(json.dumps({"exported": str(path)}))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
