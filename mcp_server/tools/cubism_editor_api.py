"""Bounded, authorized access to one local Cubism Editor document."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import ipaddress
import json
import logging
import math
import os
import stat
import sys
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

Connector = Callable[..., Awaitable[Any]]


class CubismEditorAPI:
    """Register an application or inspect one exact document after editor approval."""

    def __init__(
        self,
        expected_document: str | None = None,
        *,
        endpoint: str = "ws://127.0.0.1:22033",
        timeout: float = 5,
        token_path: Path | None = None,
        connector: Connector | None = None,
    ) -> None:
        if expected_document is not None and (
            not expected_document.endswith(".cmo3")
            or any(character in expected_document for character in "/\\\x00")
            or Path(expected_document).name != expected_document
        ):
            raise ValueError("Expected a Cubism document filename")
        try:
            address = urlsplit(endpoint)
            local = (
                address.hostname == "localhost"
                or ipaddress.ip_address(address.hostname or "").is_loopback
            )
            valid = (
                local
                and address.scheme in {"ws", "wss"}
                and address.username is None
                and address.password is None
                and not address.fragment
                and address.port is not None
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Cubism endpoint must be a loopback WebSocket URL with a port")
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("Cubism timeout must be between zero and 60 seconds")
        if token_path is None:
            from mcp_server.tools.cubism_paths import state_directory

            token_path = state_directory() / "editor-token.json"
        self.expected_document = expected_document
        self.endpoint = endpoint
        self.timeout = timeout
        self.token_path = Path(token_path)
        self.connector = connector
        self._socket: Any = None
        self._model_uid: str | None = None
        self._parameters: list[dict[str, Any]] | None = None
        self._original_values: dict[str, dict[str, Any]] = {}
        self._sequence = 0

    def _read_token(self) -> str:
        try:
            with self.token_path.open("r", encoding="utf-8") as handle:
                info = os.fstat(handle.fileno())
                if self.token_path.is_symlink() or not stat.S_ISREG(info.st_mode):
                    raise PermissionError
                if os.name == "posix" and (
                    stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid()
                ):
                    raise PermissionError
                token = json.loads(handle.read(65536))["Token"]
                if not isinstance(token, str) or not token:
                    raise ValueError
                return token
        except (OSError, KeyError, ValueError, TypeError):
            raise PermissionError(
                "Cubism permission token is unavailable or not private; "
                "register the application and restrict its token file to the current user"
            ) from None

    def _save_token(self, token: str, *, renew: bool) -> None:
        temporary = None
        try:
            self.token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.token_path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"Token": token}, handle)
            temporary.chmod(0o600)
            if renew:
                temporary.replace(self.token_path)
            else:
                os.link(temporary, self.token_path)
        except FileExistsError:
            raise FileExistsError(
                "Token file already exists; use register --renew explicitly"
            ) from None
        except OSError:
            raise PermissionError("Could not privately save the Cubism token") from None
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    async def _connect(self) -> None:
        connector = self.connector
        if connector is None:
            try:
                from websockets.asyncio.client import connect
            except ImportError:
                raise RuntimeError(
                    "Cubism parameter access needs the optional websockets dependency"
                ) from None
            connector = connect
        address = urlsplit(self.endpoint)
        logger = logging.Logger("live2d-automation.cubism-api")
        logger.disabled = True
        options: dict[str, Any] = {
            "open_timeout": self.timeout,
            "close_timeout": self.timeout,
            "host": address.hostname,
            "port": address.port,
            "logger": logger,
        }
        if "proxy" in inspect.signature(connector).parameters:
            options["proxy"] = None
        try:
            self._socket = await asyncio.wait_for(
                connector(self.endpoint, **options),
                timeout=self.timeout,
            )
            peer = self._socket.remote_address
            if (
                not isinstance(peer, tuple)
                or not peer
                or not ipaddress.ip_address(peer[0]).is_loopback
            ):
                raise ValueError("Cubism peer is not a loopback address")
        except Exception:
            await self._close()
            raise RuntimeError("Could not establish a loopback Cubism API connection") from None

    async def register(self, *, renew: bool = False) -> dict[str, Any]:
        """Request a token without model access; editor approval remains a manual step.

        Existing tokens require explicit renewal. Both registration and approval
        replies must validate before a new token is saved; transport failures leave
        the previous token unchanged.
        """
        if self._socket is not None:
            raise RuntimeError("Register before opening an editor document connection")
        if self.token_path.exists() or self.token_path.is_symlink():
            if not renew:
                raise FileExistsError("Token file already exists; use register --renew explicitly")
            if self.token_path.is_symlink() or not self.token_path.is_file():
                raise PermissionError("Token destination must be a regular, non-symlink file")
        await self._connect()
        try:
            registration = await self._request("RegisterPlugin", {"Name": "Live2D Automation"})
            token = registration.get("Token")
            if not isinstance(token, str) or not token:
                raise RuntimeError("Cubism returned an invalid registration response")
            approval = (await self._request("GetIsApproval", {})).get("Result")
            if not isinstance(approval, bool):
                raise RuntimeError("Cubism returned an invalid approval response")
            self._save_token(token, renew=renew)
            return {
                "status": "registered",
                "approved": approval,
                "token_file": str(self.token_path),
            }
        finally:
            await self._close()

    async def __aenter__(self) -> CubismEditorAPI:
        if self.expected_document is None:
            raise ValueError("Expected a Cubism document filename")
        token = self._read_token()
        await self._connect()
        try:
            registration = await self._request(
                "RegisterPlugin", {"Name": "Live2D Automation", "Token": token}
            )
            returned = registration.get("Token")
            if not isinstance(returned, str) or not returned:
                raise RuntimeError("Cubism returned an invalid registration response")
            if returned != token:
                raise PermissionError(
                    "Cubism no longer recognizes the saved token; "
                    "run live2d-cubism-api register --renew, then approve it in the editor"
                )
            approval = await self._request("GetIsApproval", {})
            if approval.get("Result") is not True:
                raise PermissionError("Cubism has not approved this local connection")
            self._model_uid = await self._check_document()
            self._parameters = None
            return self
        except BaseException:
            await self._close()
            raise

    async def __aexit__(self, *_args: object) -> None:
        try:
            await self.clear_preview()
        finally:
            await self._close()

    async def _close(self) -> None:
        socket, self._socket = self._socket, None
        self._model_uid = None
        self._parameters = None
        if socket is not None:
            with suppress(Exception):
                await asyncio.wait_for(socket.close(), timeout=self.timeout)

    async def _check_document(self) -> str:
        current = await self._request("GetCurrentModelUID", {})
        model_uid = current.get("ModelUID")
        if not isinstance(model_uid, str) or not model_uid:
            raise RuntimeError("Cubism has no current model")
        documents = (await self._request("GetDocuments", {})).get("ModelingDocuments")
        if not isinstance(documents, list):
            raise RuntimeError("Cubism returned an invalid document list")
        try:
            matching = [
                document
                for document in documents
                if any(view.get("ModelUID") == model_uid for view in document.get("Views", []))
            ]
            matches = len(matching) == 1 and (
                str(matching[0].get("DocumentFilePath", "")).replace("\\", "/").rsplit("/", 1)[-1]
                == self.expected_document
            )
        except (AttributeError, TypeError):
            raise RuntimeError("Cubism returned an invalid document list") from None
        if not matches:
            raise ValueError("The active Cubism document does not match")
        return model_uid

    async def parameters(self) -> list[dict[str, Any]]:
        """Return finite authored ranges and keyforms after document authentication."""
        if self._model_uid is None:
            raise RuntimeError("Connect to Cubism before reading parameters")
        if self._parameters is None:
            parameters = (await self._request("GetParameters", {"ModelUID": self._model_uid})).get(
                "Parameters"
            )
            try:
                if not isinstance(parameters, list):
                    raise ValueError
                ids = set()
                for parameter in parameters:
                    if not isinstance(parameter["Id"], str) or not parameter["Id"]:
                        raise ValueError
                    if parameter["Id"] in ids:
                        raise ValueError
                    ids.add(parameter["Id"])
                    low, default, high = [float(parameter[k]) for k in ("Min", "Default", "Max")]
                    if not all(math.isfinite(v) for v in (low, default, high)):
                        raise ValueError
                    if not low <= default <= high:
                        raise ValueError
                    for key in parameter.get("Keyform", []):
                        value = float(key["Value"])
                        if not math.isfinite(value) or not low <= value <= high:
                            raise ValueError
            except (KeyError, TypeError, ValueError, AttributeError):
                raise RuntimeError("Cubism returned an invalid parameter list") from None
            self._parameters = parameters
        return self._parameters

    async def set_preview(self, parameter_id: str, value: float) -> None:
        """Set a temporary value within the authored range; restore it on context exit."""
        if not math.isfinite(value):
            raise ValueError("Parameter value must be finite")
        parameter = next((p for p in await self.parameters() if p["Id"] == parameter_id), None)
        if parameter is None:
            raise ValueError("Unknown Cubism parameter")
        if not float(parameter["Min"]) <= value <= float(parameter["Max"]):
            raise ValueError("Parameter value is outside its range")
        if await self._check_document() != self._model_uid:
            raise ValueError("The active Cubism document changed")
        if parameter_id not in self._original_values:
            original = (
                await self._request("GetParameterValues", {"ModelUID": self._model_uid})
            ).get("Parameters")
            if not isinstance(original, list) or any(
                not isinstance(p, dict)
                or not isinstance(p.get("Id"), str)
                or not isinstance(p.get("Value"), (int, float))
                or not math.isfinite(p["Value"])
                for p in original
            ):
                raise RuntimeError("Cubism returned invalid preview values")
            matching = [p for p in original if p["Id"] == parameter_id]
            if len(matching) != 1:
                raise RuntimeError("Cubism returned invalid preview values")
            self._original_values[parameter_id] = matching[0]
        await self._request(
            "SetParameterValues",
            {"ModelUID": self._model_uid, "Parameters": [{"Id": parameter_id, "Value": value}]},
        )

    async def clear_preview(self) -> None:
        """Restore only previewed parameters to their values before the first override."""
        if self._original_values:
            await self._request(
                "SetParameterValues",
                {"ModelUID": self._model_uid, "Parameters": list(self._original_values.values())},
            )
            await self._request("ClearParameterValues", {"ModelUID": self._model_uid})
            self._original_values.clear()

    async def _request(self, method: str, data: dict[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("Cubism socket is not connected")
        self._sequence += 1
        request_id = str(self._sequence)

        async def exchange() -> dict[str, Any]:
            await self._socket.send(
                json.dumps(
                    {
                        "Version": "1.0.1",
                        "RequestId": request_id,
                        "Type": "Request",
                        "Method": method,
                        "Data": data,
                    }
                )
            )
            for _ in range(8):
                response = json.loads(await self._socket.recv())
                if not isinstance(response, dict):
                    raise ValueError
                if response.get("RequestId") != request_id:
                    continue
                payload = response.get("Data")
                if (
                    response.get("Type") != "Response"
                    or response.get("Method") != method
                    or not isinstance(payload, dict)
                ):
                    raise ValueError
                return payload
            raise ValueError

        try:
            return await asyncio.wait_for(exchange(), timeout=self.timeout)
        except Exception:
            raise RuntimeError(
                f"Cubism request failed or returned invalid data: {method}"
            ) from None


def main(argv: list[str] | None = None) -> int:
    """Register the application explicitly, without printing tokens or granting approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="Request a private token for manual approval")
    register.add_argument("--endpoint", default="ws://127.0.0.1:22033")
    register.add_argument("--token-path", type=Path)
    register.add_argument("--timeout", type=float, default=5)
    register.add_argument("--renew", action="store_true", help="Replace an existing token")
    args = parser.parse_args(argv)
    try:
        api = CubismEditorAPI(
            endpoint=args.endpoint, token_path=args.token_path, timeout=args.timeout
        )
        result = asyncio.run(api.register(renew=args.renew))
    except FileExistsError:
        print("Token file already exists; use register --renew explicitly.", file=sys.stderr)
        return 1
    except (PermissionError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result["message"] = (
        "Cubism already approved Live2D Automation."
        if result["approved"]
        else "Approve Live2D Automation in Cubism's External Application Integration settings."
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
