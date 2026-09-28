"""Authorized editor transport and token boundaries, without a live editor."""

import importlib
import json
import os
import stat

import pytest


class EditorSocket:
    """Reply with protocol-shaped messages at the external WebSocket boundary."""

    def __init__(self, *, document="Avatar.cmo3", approved=True, token="test-secret"):
        self.remote_address = ("127.0.0.1", 22033)
        self.document = document
        self.approved = approved
        self.token = token
        self.calls = []
        self.closed = False
        self.override = None

    async def send(self, payload):
        self.calls.append(json.loads(payload))

    async def recv(self):
        request = self.calls[-1]
        method = request["Method"]
        data = {
            "RegisterPlugin": {"Token": self.token},
            "GetIsApproval": {"Result": self.approved},
            "GetCurrentModelUID": {"ModelUID": "model-1"},
            "GetDocuments": {
                "ModelingDocuments": [
                    {
                        "DocumentFilePath": f"/models/{self.document}",
                        "Views": [{"ModelUID": "model-1"}],
                    }
                ]
            },
            "GetParameters": {
                "Parameters": [
                    {
                        "Id": "Angle",
                        "Name": "Angle",
                        "Min": -10,
                        "Max": 10,
                        "Default": 0,
                        "Keyform": [{"Value": -10}, {"Value": 0}, {"Value": 10}],
                    }
                ]
            },
            "GetParameterValues": {"Parameters": [{"Id": "Angle", "Value": 3.5}]},
            "SetParameterValues": {},
            "ClearParameterValues": {},
        }[method]
        response = {
            "Type": "Response",
            "RequestId": request["RequestId"],
            "Method": method,
            "Data": data,
        }
        return json.dumps(self.override(response) if self.override else response)

    async def close(self):
        self.closed = True


@pytest.fixture
def api_class():
    try:
        return importlib.import_module("mcp_server.tools.cubism_editor_api").CubismEditorAPI
    except ModuleNotFoundError:
        pytest.fail("The approved local editor API has not been ported")


@pytest.fixture
def token_path(tmp_path):
    path = tmp_path / "token.json"
    path.write_text('{"Token":"test-secret"}')
    path.chmod(0o600)
    return path


def connect_to(socket):
    async def connect(url, **kwargs):
        assert url == "ws://127.0.0.1:22033"
        assert 0 < kwargs["open_timeout"] <= 15
        return socket

    return connect


@pytest.mark.asyncio
async def test_exact_document_preview_restores_values_and_closes(api_class, token_path):
    socket = EditorSocket()
    async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)) as api:
        assert (await api.parameters())[0]["Keyform"] == [
            {"Value": -10},
            {"Value": 0},
            {"Value": 10},
        ]
        await api.set_preview("Angle", -10)
    updates = [r["Data"]["Parameters"] for r in socket.calls if r["Method"] == "SetParameterValues"]
    assert updates == [[{"Id": "Angle", "Value": -10}], [{"Id": "Angle", "Value": 3.5}]]
    assert socket.calls[-1]["Method"] == "ClearParameterValues"
    assert socket.closed
    assert socket.calls[0]["Data"]["Name"] == "Live2D Automation"


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_second", [False, True])
async def test_preview_restores_only_touched_parameters_at_their_first_change(
    api_class, token_path, preview_second
):
    socket = EditorSocket()
    values = {"Angle": 3.5, "Eye": 1.0}

    def respond(response):
        if response["Method"] == "GetParameters":
            response["Data"]["Parameters"].append({"Id": "Eye", "Min": 0, "Default": 1, "Max": 1})
        elif response["Method"] == "GetParameterValues":
            response["Data"]["Parameters"] = [
                {"Id": key, "Value": value} for key, value in values.items()
            ]
        elif response["Method"] == "SetParameterValues":
            values.update({p["Id"]: p["Value"] for p in socket.calls[-1]["Data"]["Parameters"]})
        return response

    socket.override = respond
    async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)) as api:
        await api.set_preview("Angle", -10)
        values["Eye"] = 0.7
        await api.set_preview("Angle", 5)
        if preview_second:
            await api.set_preview("Eye", 0)
    assert values == {"Angle": 3.5, "Eye": 0.7}
    restored = [
        r["Data"]["Parameters"] for r in socket.calls if r["Method"] == "SetParameterValues"
    ][-1]
    assert restored == (
        [{"Id": "Angle", "Value": 3.5}, {"Id": "Eye", "Value": 0.7}]
        if preview_second
        else [{"Id": "Angle", "Value": 3.5}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("original", [[], [{"Id": "Angle", "Value": 1}] * 2])
async def test_preview_requires_one_original_value_before_changing_parameter(
    api_class, token_path, original
):
    socket = EditorSocket()
    socket.override = lambda response: (
        {**response, "Data": {"Parameters": original}}
        if response["Method"] == "GetParameterValues"
        else response
    )
    async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)) as api:
        with pytest.raises(RuntimeError, match="preview values"):
            await api.set_preview("Angle", 1)
    assert not any(r["Method"] == "SetParameterValues" for r in socket.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings,error",
    [
        ({"document": "Other.cmo3"}, ValueError),
        ({"approved": False}, PermissionError),
    ],
)
async def test_no_parameter_access_before_document_and_approval(
    api_class, token_path, settings, error
):
    socket = EditorSocket(**settings)
    with pytest.raises(error):
        async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)):
            pytest.fail("Unapproved or wrong document was accepted")
    assert not any(r["Method"] == "GetParameters" for r in socket.calls)
    assert socket.closed


@pytest.mark.asyncio
async def test_stale_token_is_preserved_until_explicit_renewal(api_class, token_path):
    socket = EditorSocket(token="renewed-secret", approved=False)
    with pytest.raises(PermissionError) as caught:
        async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)):
            pass
    assert json.loads(token_path.read_text()) == {"Token": "test-secret"}
    if os.name == "posix":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    assert "secret" not in str(caught.value)
    assert not any(r["Method"] == "GetCurrentModelUID" for r in socket.calls)
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [-11, 11, float("nan"), float("inf")])
async def test_preview_rejects_invalid_values_before_write(api_class, token_path, value):
    socket = EditorSocket()
    async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)) as api:
        with pytest.raises(ValueError):
            await api.set_preview("Angle", value)
    assert not any(r["Method"] == "SetParameterValues" for r in socket.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["list", "method", "error", "unmatched", "parameters"])
async def test_malformed_and_unmatched_replies_fail_without_secret(api_class, token_path, kind):
    socket = EditorSocket()

    def override(response):
        if kind == "list":
            return ["test-secret"]
        if kind == "method":
            return {**response, "Method": "Wrong"}
        if kind == "error":
            return {**response, "Type": "Error", "Data": {"ErrorType": "test-secret"}}
        if kind == "unmatched":
            return {**response, "RequestId": "unrelated"}
        if response["Method"] == "GetParameters":
            return {**response, "Data": {"Parameters": [{"Id": "Angle", "Min": "test-secret"}]}}
        return response

    socket.override = override
    with pytest.raises(RuntimeError) as caught:
        async with api_class(
            "Avatar.cmo3", token_path=token_path, connector=connect_to(socket)
        ) as api:
            await api.parameters()
    assert "secret" not in str(caught.value)
    assert socket.closed


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://example.com:22033",
        "ws://127.0.0.1.example.com",
        "file:///tmp/socket",
        "ws://user:password@127.0.0.1:22033",
        "ws://127.0.0.1:22033/#fragment",
    ],
)
def test_remote_or_credentialed_endpoints_are_rejected(api_class, endpoint):
    with pytest.raises(ValueError, match="loopback"):
        api_class("Avatar.cmo3", endpoint=endpoint)


@pytest.mark.asyncio
async def test_missing_optional_dependency_is_bounded(api_class, token_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "websockets.asyncio.client", None)
    with pytest.raises(RuntimeError, match="optional"):
        async with api_class("Avatar.cmo3", token_path=token_path):
            pass


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX token permission bits")
async def test_world_readable_token_is_rejected_without_connection(api_class, token_path):
    token_path.chmod(0o644)

    async def must_not_connect(*args, **kwargs):
        pytest.fail("Unsafe token reached a network connector")

    with pytest.raises(PermissionError):
        async with api_class("Avatar.cmo3", token_path=token_path, connector=must_not_connect):
            pass


@pytest.mark.asyncio
async def test_first_registration_saves_token_without_accessing_models(api_class, tmp_path):
    register = getattr(api_class, "register", None)
    assert callable(register), "Explicit registration is missing"
    socket = EditorSocket(approved=False, token="new-private-secret")
    token_file = tmp_path / "state" / "editor-token.json"
    api = api_class(token_path=token_file, connector=connect_to(socket))
    result = await api.register()
    assert json.loads(token_file.read_text()) == {"Token": "new-private-secret"}
    if os.name == "posix":
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(token_file.parent.stat().st_mode) == 0o700
    assert result["approved"] is False
    assert "secret" not in json.dumps(result)
    assert [call["Method"] for call in socket.calls] == ["RegisterPlugin", "GetIsApproval"]
    assert socket.calls[0]["Data"] == {"Name": "Live2D Automation"}
    assert socket.closed


@pytest.mark.asyncio
async def test_registration_existing_token_requires_explicit_renewal(api_class, token_path):
    assert callable(getattr(api_class, "register", None)), "Explicit registration is missing"

    async def must_not_connect(*args, **kwargs):
        pytest.fail("Existing token should be checked before any connection")

    api = api_class(token_path=token_path, connector=must_not_connect)
    original = token_path.read_bytes()
    with pytest.raises(FileExistsError, match="renew"):
        await api.register()
    assert token_path.read_bytes() == original


@pytest.mark.asyncio
async def test_explicit_renewal_replaces_token_after_valid_protocol(api_class, token_path):
    assert callable(getattr(api_class, "register", None)), "Explicit registration is missing"
    socket = EditorSocket(approved=False, token="renewed-secret")
    result = await api_class(token_path=token_path, connector=connect_to(socket)).register(
        renew=True
    )
    assert json.loads(token_path.read_text()) == {"Token": "renewed-secret"}
    assert socket.calls[0]["Data"] == {"Name": "Live2D Automation"}
    assert result["approved"] is False
    assert "secret" not in str(result)
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["transport", "error", "token", "approval"])
async def test_failed_renewal_preserves_existing_token(api_class, token_path, fault):
    assert callable(getattr(api_class, "register", None)), "Explicit registration is missing"
    socket = EditorSocket(token="renewed-secret")

    async def connect(url, **kwargs):
        if fault == "transport":
            raise OSError("request contains private-secret")
        return socket

    def override(response):
        if fault == "error":
            return {**response, "Type": "Error", "Data": {"ErrorType": "private-secret"}}
        if fault == "token" and response["Method"] == "RegisterPlugin":
            return {**response, "Data": {"Token": None}}
        if fault == "approval" and response["Method"] == "GetIsApproval":
            return {**response, "Data": {"Result": "private-secret"}}
        return response

    socket.override = override
    original = token_path.read_bytes()
    with pytest.raises(RuntimeError) as caught:
        await api_class(token_path=token_path, connector=connect).register(renew=True)
    assert token_path.read_bytes() == original
    assert "secret" not in str(caught.value)
    assert fault == "transport" or socket.closed


def test_registration_cli_uses_shared_default_path_without_disclosing_token(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("websockets.asyncio.client")
    module = importlib.import_module("mcp_server.tools.cubism_editor_api")
    main = getattr(module, "main", None)
    assert callable(main), "Registration CLI is missing"
    socket = EditorSocket(approved=False, token="new-private-secret")
    monkeypatch.setenv("LIVE2D_AUTOMATION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("websockets.asyncio.client.connect", connect_to(socket))
    assert main(["register"]) == 0
    output = capsys.readouterr()
    assert "secret" not in output.out + output.err
    assert "approve" in output.out.lower() and "Cubism" in output.out
    saved = tmp_path / "state" / "editor-token.json"
    assert json.loads(saved.read_text()) == {"Token": "new-private-secret"}
    api = module.CubismEditorAPI("Avatar.cmo3")
    assert api.token_path == saved
    assert main(["register"]) == 1
    output = capsys.readouterr()
    assert "renew" in output.err
    assert "secret" not in output.out + output.err


def test_registration_cli_rejects_remote_endpoint_before_connecting(monkeypatch, tmp_path, capsys):
    pytest.importorskip("websockets.asyncio.client")
    module = importlib.import_module("mcp_server.tools.cubism_editor_api")
    main = getattr(module, "main", None)
    assert callable(main), "Registration CLI is missing"

    async def must_not_connect(*args, **kwargs):
        pytest.fail("Remote endpoint reached connector")

    monkeypatch.setattr("websockets.asyncio.client.connect", must_not_connect)
    path = tmp_path / "token.json"
    assert (
        main(["register", "--endpoint", "ws://example.com:22033", "--token-path", str(path)]) == 1
    )
    assert "loopback" in capsys.readouterr().err
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", [("203.0.113.9", 22033), ("2001:db8::9", 22033), None])
async def test_nonlocal_peer_is_closed_before_any_registration(api_class, token_path, peer):
    socket = EditorSocket()
    socket.remote_address = peer
    original = token_path.read_bytes()
    with pytest.raises(RuntimeError):
        async with api_class("Avatar.cmo3", token_path=token_path, connector=connect_to(socket)):
            pytest.fail("Nonlocal or unknown peer was accepted")
    assert socket.calls == []
    assert socket.closed
    assert token_path.read_bytes() == original


@pytest.mark.asyncio
async def test_redirect_is_rejected_before_connecting_to_a_different_origin(
    api_class, token_path, monkeypatch
):
    pytest.importorskip("websockets.asyncio.client")
    from websockets.asyncio.client import connect
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    attempts = []
    sockets = []

    class HandshakeSocket(EditorSocket):
        def __init__(self, redirected):
            super().__init__()
            self.redirected = redirected
            self.transport = self

        async def handshake(self, *args):
            if not self.redirected:
                raise InvalidStatus(
                    Response(302, "Found", Headers([("Location", "ws://203.0.113.9:22033/")]), b"")
                )

        def abort(self):
            self.closed = True

        def start_keepalive(self):
            pass

    async def create_connection(connection):
        attempts.append(connection.uri)
        socket = HandshakeSocket(redirected=len(attempts) > 1)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(connect, "create_connection", create_connection)
    with pytest.raises(RuntimeError):
        async with api_class("Avatar.cmo3", token_path=token_path):
            pytest.fail("A cross-origin redirect was accepted")
    assert attempts == ["ws://127.0.0.1:22033"]
    assert all(socket.calls == [] for socket in sockets)


@pytest.mark.asyncio
async def test_real_connector_disables_proxy_and_sensitive_frame_logging(
    api_class, token_path, monkeypatch, caplog
):
    import logging

    pytest.importorskip("websockets.asyncio.client")
    from websockets.asyncio.client import connect
    from websockets.frames import OP_TEXT, Frame
    from websockets.uri import parse_uri

    seen = {}
    socket = EditorSocket()

    async def create_connection(connection):
        seen["proxy"] = getattr(connection, "proxy", None)
        seen["host"] = connection.connection_kwargs.get("host")
        seen["port"] = connection.connection_kwargs.get("port")
        protocol = connection.protocol_factory(parse_uri(connection.uri)).protocol

        async def handshake(*args):
            pass

        async def send(payload):
            protocol.send_frame(Frame(OP_TEXT, payload.encode()))
            await EditorSocket.send(socket, payload)

        socket.handshake = handshake
        socket.send = send
        socket.start_keepalive = lambda: None
        return socket

    monkeypatch.setattr(connect, "create_connection", create_connection)
    caplog.set_level(logging.DEBUG, logger="websockets.client")
    async with api_class("Avatar.cmo3", token_path=token_path):
        pass
    assert seen == {"proxy": None, "host": "127.0.0.1", "port": 22033}
    assert "test-secret" not in caplog.text


@pytest.mark.asyncio
async def test_connector_without_proxy_option_accepts_ipv6_loopback(api_class, token_path):
    socket = EditorSocket()
    socket.remote_address = ("::1", 22033, 0, 0)

    async def connect(uri, *, open_timeout, close_timeout, host, port, logger):
        assert uri == "ws://[::1]:22033"
        assert host == "::1" and port == 22033
        return socket

    async with api_class(
        "Avatar.cmo3", endpoint="ws://[::1]:22033", token_path=token_path, connector=connect
    ) as api:
        assert (await api.parameters())[0]["Id"] == "Angle"
    assert socket.closed
