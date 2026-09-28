"""Public MCP entrypoints retain editor guards and redact boundary errors."""

import pytest

from mcp_server import secure_server_impl as server


@pytest.mark.asyncio
async def test_parameters_only_exposes_ranges_and_keyforms(monkeypatch):
    tool = getattr(server, "cubism_editor_parameters", None)
    assert callable(tool), "Editor parameter MCP tool is missing"

    class Editor:
        def __init__(self, expected_document):
            assert expected_document == "Avatar.cmo3"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def parameters(self):
            return [
                {
                    "Id": "Angle",
                    "Name": "Angle",
                    "Min": -10,
                    "Default": 0,
                    "Max": 10,
                    "Keyform": [{"Value": 0}],
                    "Token": "do-not-return",
                }
            ]

    monkeypatch.setattr(server, "CubismEditorAPI", Editor)
    result = await tool("Avatar.cmo3")
    assert result == {
        "status": "success",
        "document": "Avatar.cmo3",
        "parameters": [
            {"id": "Angle", "name": "Angle", "min": -10, "default": 0, "max": 10, "keyforms": [0]}
        ],
    }


@pytest.mark.asyncio
async def test_widget_tools_route_through_exact_document_guard(monkeypatch):
    tool = getattr(server, "cubism_editor_widget_action", None)
    assert callable(tool), "Guarded widget MCP tool is missing"
    seen = []

    def guard(document, action):
        seen.append((document, action))
        return {"status": "error", "details": "Unexpected document"}

    monkeypatch.setattr(server, "_guarded_editor_action", guard)
    result = await tool("Avatar.cmo3", {"action": "click", "id": "component-1"})
    assert result["status"] == "error"
    assert seen == [("Avatar.cmo3", {"action": "click", "id": "component-1"})]
    result = await server.cubism_editor_widgets("Avatar.cmo3")
    assert seen[-1] == ("Avatar.cmo3", {"action": "snapshot"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("cubism_editor_status", ()),
        ("cubism_editor_open_menu", ("save", "Avatar.cmo3")),
        ("cubism_editor_widgets", ("Avatar.cmo3",)),
        ("cubism_editor_widget_action", ("Avatar.cmo3", {"action": "snapshot"})),
        ("cubism_editor_parameters", ("Avatar.cmo3",)),
    ],
)
async def test_unavailable_editor_errors_are_bounded_and_redacted(monkeypatch, tool_name, args):
    tool = getattr(server, tool_name, None)
    assert callable(tool), "Editor MCP tool is missing"

    def unavailable(*args, **kwargs):
        raise RuntimeError("connector leaked token=private-secret")

    monkeypatch.setattr(server, "_macos_controller", unavailable)
    monkeypatch.setattr(server, "_guarded_editor_action", unavailable)
    monkeypatch.setattr(server, "CubismEditorAPI", unavailable)
    result = await tool(*args)
    assert result["status"] == "error"
    assert "secret" not in str(result)
    assert len(str(result)) < 400


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [0, 8])
async def test_widget_action_snapshot_honors_depth(monkeypatch, depth):
    from mcp_server.tools import cubism_swing

    requests = []

    def request(payload, **kwargs):
        requests.append(payload)
        return {"status": "success", "windows": [{"text": "Cubism - Avatar.cmo3"}]}

    monkeypatch.setattr(cubism_swing, "request", request)
    result = await server.cubism_editor_widget_action(
        "Avatar.cmo3", {"action": "snapshot", "depth": depth}
    )
    assert result["status"] == "success"
    assert requests == [{"action": "snapshot", "depth": depth}]
