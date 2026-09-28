"""Named operations for Cubism's English Swing UI, with per-action document guards."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

from mcp_server.tools.cubism_swing import guarded_action, request

Widget = dict[str, Any]
Sender = Callable[[Widget], Widget]


def flatten(node: Widget) -> Iterator[Widget]:
    """Yield a widget and its descendants in snapshot order."""
    yield node
    for child in node.get("children", []):
        yield from flatten(child)


def one(nodes: list[Widget], predicate: Callable[[Widget], bool], label: str) -> Widget:
    """Require an unambiguous current widget instead of choosing a duplicate label."""
    matches = [node for node in nodes if predicate(node)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(matches)}")
    return matches[0]


class EditorUI:
    """Operate one open document using the Cubism 5.3 English widget layout.

    Widget classes and table geometry are editor-version specific. Each mutation
    rechecks the document and front modal; a layout mismatch fails explicitly.
    """

    def __init__(
        self,
        expected_document: str,
        *,
        session_dir: Path | None = None,
        sender: Sender | None = None,
        timeout: float = 3,
    ) -> None:
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("UI timeout must be finite and between zero and 120 seconds")
        self.document = expected_document
        self.timeout = timeout
        self._sender = sender or partial(request, session_dir=session_dir)

    def action(self, payload: Widget) -> Widget:
        """Send a guarded request; asynchronous pointer events require a later state check."""
        result = guarded_action(self.document, payload, sender=self._sender)
        statuses = {"success"}
        if payload.get("action") in {"click", "mouse", "drag", "close_dialog"}:
            statuses.add("accepted")
        if result.get("status") not in statuses:
            raise RuntimeError(f"Cubism action failed: {payload.get('action')}")
        return result

    def snapshot(self) -> tuple[Widget, list[Widget]]:
        """Read the current document and flatten its widget tree."""
        result = self.action({"action": "snapshot", "depth": 42})
        return result, [node for window in result["windows"] for node in flatten(window)]

    def click(self, widget: Widget) -> Widget:
        """Click a widget that still belongs to the expected document or front modal."""
        return self.action({"action": "click", "id": widget["id"]})

    def wait_widget(self, predicate: Callable[[Widget], bool]) -> Widget:
        """Wait for a state change, bounded by this client's UI timeout."""
        deadline = time.monotonic() + self.timeout
        while True:
            _, nodes = self.snapshot()
            match = next((node for node in nodes if predicate(node)), None)
            if match is not None:
                return match
            if time.monotonic() >= deadline:
                raise TimeoutError("Expected Cubism widget did not appear")
            time.sleep(min(0.04, max(0, deadline - time.monotonic())))

    def wait_window(self, title: str) -> Widget:
        """Wait for a top-level window with the exact title, bounded by the UI timeout."""
        deadline = time.monotonic() + self.timeout
        while True:
            state, _ = self.snapshot()
            windows: list[Widget] = state["windows"]
            match = next((w for w in windows if w.get("text") == title), None)
            if match is not None:
                return match
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Expected Cubism window {title!r} did not appear")
            time.sleep(min(0.04, max(0, deadline - time.monotonic())))

    def dialog_button(self, window: Widget, text: str) -> Widget:
        """Click a uniquely named button inside the specified dialog."""
        return self.click(
            one(
                list(flatten(window)),
                lambda n: n.get("kind") == "button" and n.get("text") == text,
                f"dialog button {text!r}",
            )
        )

    def menu(self, text: str) -> Widget:
        """Invoke one exact menu command, rejecting ambiguous names."""
        _, nodes = self.snapshot()
        return self.click(
            one(
                nodes,
                lambda n: n.get("text") == text and n.get("kind") == "button",
                f"menu command {text!r}",
            )
        )

    def file_menu(self, text: str) -> Widget:
        """Resolve File actions by menu ownership, excluding duplicate bookmarks."""
        _, nodes = self.snapshot()
        parent = one(
            nodes, lambda n: n.get("text") == "File" and n.get("kind") == "button", "File menu"
        )
        return self.click(
            one(
                list(flatten(parent)),
                lambda n: n.get("text") == text and n.get("kind") == "button",
                f"File command {text!r}",
            )
        )

    def filter_parts(self, value: str) -> None:
        """Enable and edit the Parts text filter in the default modeling workspace."""
        _, nodes = self.snapshot()
        toggle = one(
            nodes,
            lambda n: n.get("tooltip") == "Filter by Text" and n.get("showing", False),
            "Parts text filter",
        )
        if not toggle.get("selected"):
            self.click(toggle)
            _, nodes = self.snapshot()
        field = one(
            nodes,
            lambda n: n.get("kind") == "text"
            and n.get("showing", False)
            and n.get("bounds", [999, 999])[0] < 300
            and n.get("bounds", [999, 999])[1] < 230,
            "Parts filter field",
        )
        self.action({"action": "set_text", "id": field["id"], "value": value})
        for tooltip in ("Collapse all", "Expand all"):
            _, nodes = self.snapshot()
            matches = [
                n
                for n in nodes
                if n.get("tooltip") == tooltip
                and n.get("showing")
                and n.get("bounds", [999, 999])[0] < 300
                and n.get("bounds", [999, 999])[1] < 250
            ]
            if len(matches) > 1:
                raise RuntimeError(f"Expected one Parts {tooltip} control")
            if matches:
                self.click(matches[0])

    def select_parts(self, names: list[str]) -> None:
        """Find exact names across the current Parts table and await Inspector selection."""
        if not names or len(set(names)) != len(names):
            raise ValueError("Part names must be nonempty and unique")
        _, nodes = self.snapshot()
        table = one(
            nodes, lambda n: "CPartsTreeTable" in n.get("class", "") and "rows" in n, "Parts table"
        )
        selected = self.action(
            {"action": "find_table_rows", "id": table["id"], "column": 2, "values": names}
        )["rows"]
        for index, row in enumerate(selected):
            self.action({"action": "scroll_table_row", "id": table["id"], "row": row})
            self.action(
                {
                    "action": "mouse",
                    "id": table["id"],
                    "x": 145,
                    "y": row * 22 + 11,
                    "modifiers": 0 if index == 0 else 256,
                }
            )
        if len(names) == 1:
            self.wait_widget(
                lambda n: bool(n.get("showing"))
                and n.get("kind") != "button"
                and any(
                    c.get("text") == "Name" and c.get("kind") != "button" and c.get("showing")
                    for c in n.get("children", [])
                )
                and any(
                    c.get("kind") == "text" and c.get("text") == names[0] and c.get("showing")
                    for c in n.get("children", [])
                )
            )
        else:
            time.sleep(0.08)

    def _parameter_row(self, name: str) -> Widget:
        _, nodes = self.snapshot()
        return one(
            nodes,
            lambda n: n.get("text") == "singleRangeBox"
            and any(c.get("text") == name for c in n.get("children", [])),
            f"parameter {name!r}",
        )

    def parameter(self, name: str, fraction: float) -> Widget:
        """Move a visible parameter slider to a normalized position between zero and one."""
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Parameter fraction must be finite and within zero to one")
        row = self._parameter_row(name)
        slider = one(row["children"], lambda n: "CCustomPaint" in n.get("class", ""), "slider")
        self.action({"action": "scroll_into_view", "id": row["id"]})
        return self.action(
            {
                "action": "mouse",
                "id": slider["id"],
                "x": round(10 + (slider["bounds"][2] - 20) * fraction),
                "y": 13,
            }
        )

    def parameter_value(self, name: str, value: float) -> None:
        """Commit a finite exact value through the parameter's numeric change handler."""
        if not math.isfinite(value):
            raise ValueError("Parameter value must be finite")
        row = self._parameter_row(name)
        slider = one(row["children"], lambda n: "CCustomPaint" in n.get("class", ""), "slider")
        control = one(
            list(flatten(row)),
            lambda n: n.get("class", "").endswith("CSlidableFloat$b"),
            "parameter numeric field",
        )
        self.action({"action": "scroll_into_view", "id": row["id"]})
        self.action({"action": "mouse", "id": slider["id"], "x": slider["bounds"][2] // 2, "y": 13})
        self.action({"action": "set_number", "id": control["id"], "value": value})

    def inspector_number(self, label: str, value: float) -> None:
        """Commit a finite number to an exact visible Inspector field."""
        if not math.isfinite(value):
            raise ValueError("Inspector value must be finite")
        row = self.wait_widget(
            lambda n: bool(n.get("showing"))
            and any(c.get("text") == label for c in n.get("children", []))
        )
        control = one(
            list(flatten(row)),
            lambda n: "CSlidable" in n.get("class", "") and n.get("class", "").endswith("$b"),
            f"Inspector {label!r}",
        )
        self.action({"action": "set_number", "id": control["id"], "value": value})

    def add_keys(self, count: int) -> Widget:
        """Add the requested keyform preset for the current selection."""
        _, nodes = self.snapshot()
        return self.click(
            one(
                nodes,
                lambda n: n.get("tooltip") == f"Add {count} keys" and n.get("showing", False),
                "keyform button",
            )
        )

    def focus_selection(self) -> Widget:
        """Center the canvas on the selected object using the editor's own command."""
        _, nodes = self.snapshot()
        return self.click(
            one(
                nodes,
                lambda n: "Focus on selected element" in str(n.get("tooltip"))
                and n.get("showing", False),
                "focus button",
            )
        )

    def reparent_parts(
        self, filter_text: str, names: list[str], deformer: str, part: str | None = None
    ) -> None:
        """Assign selected objects to one existing named deformer and optional Part folder."""
        self.filter_parts(filter_text)
        self.select_parts(names)
        for marker, target in [("[Root]", deformer), ("Root Part", part)]:
            if target is None:
                continue

            def matches(node: Widget, option: str = marker) -> bool:
                return (
                    bool(node.get("showing"))
                    and node.get("kind") == "combo"
                    and option in node.get("options", [])
                )

            combo = self.wait_widget(matches)
            indices = [
                i for i, name in enumerate(combo["options"]) if name.split("[")[0].strip() == target
            ]
            if len(indices) != 1:
                raise RuntimeError(f"Expected one destination named {target!r}")
            self.action({"action": "select_combo", "id": combo["id"], "index": indices[0]})

    def import_psd(
        self,
        path: Path,
        model_name: str,
        *,
        replace: bool = False,
        replace_source: str | None = None,
    ) -> None:
        """Import a PSD; replace one exact source filename, defaulting to the input name."""
        if replace_source is not None and (
            not replace
            or not isinstance(replace_source, str)
            or not replace_source.strip()
            or any(character in replace_source for character in "/\\\x00\n\r")
        ):
            raise ValueError("replace_source requires replace=True and a source filename")
        source = replace_source if replace_source is not None else Path(path).name
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() != ".psd":
            raise ValueError("Expected an existing PSD file")
        self.file_menu("Open...")
        window = self.wait_window("Open")
        field = one(
            list(flatten(window)),
            lambda n: n.get("kind") == "text" and n.get("showing", False),
            "file path field",
        )
        self.action({"action": "set_text", "id": field["id"], "value": str(path)})
        self.dialog_button(window, "Open")
        window = self.wait_window("Model settings")
        model = f"{model_name} (Model)"
        choices = one(
            list(flatten(window)), lambda n: model in n.get("options", []), "target model"
        )
        self.action(
            {"action": "select_list", "id": choices["id"], "index": choices["options"].index(model)}
        )
        self.dialog_button(window, "OK")
        window = self.wait_window("Re-import settings")
        choices = one(
            list(flatten(window)),
            lambda n: "< Add all layers as new ArtMesh >" in n.get("options", []),
            "PSD import action",
        )
        options = choices["options"]
        if replace:
            indices = [
                i
                for i, value in enumerate(options)
                if value.startswith("Replace [")
                and value.partition("[")[2].rpartition("]")[0].strip() == source
            ]
            if len(indices) != 1:
                raise RuntimeError(
                    f"PSD replacement requires exactly one source named {source!r}; "
                    f"found {len(indices)}"
                )
            index = indices[0]
        else:
            index = options.index("< Add all layers as new ArtMesh >")
        self.action({"action": "select_list", "id": choices["id"], "index": index})
        self.dialog_button(window, "OK")
