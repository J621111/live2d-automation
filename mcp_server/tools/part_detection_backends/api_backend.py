"""Remote API-assisted backend for semantic part detection."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib import request
from urllib.parse import urlparse

from mcp_server.schemas import BoundingBox, DetectedPart
from mcp_server.tools.part_detection_backends.base import PartDetectionBackend
from mcp_server.tools.part_detection_backends.heuristic_backend import (
    HeuristicPartDetectionBackend,
)

JsonDict = dict[str, Any]
Transport = Callable[[str, JsonDict, dict[str, str]], JsonDict]


class APIPartDetectionBackend(PartDetectionBackend):
    """Hybrid detector that lets a remote AI refine key semantic parts."""

    backend_name = "api"
    detector_name = "hybrid_api_v1"
    key_part_names = {
        "left_eye",
        "right_eye",
        "left_eye_white",
        "right_eye_white",
        "left_iris",
        "right_iris",
        "left_eye_highlight",
        "right_eye_highlight",
        "left_eyebrow",
        "right_eyebrow",
        "mouth",
        "nose",
        "hair_front",
    }
    fine_grained_face_parts = {
        "left_eye_white",
        "right_eye_white",
        "left_iris",
        "right_iris",
        "left_eye_highlight",
        "right_eye_highlight",
    }

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        transport: Transport | None = None,
        fallback_backend: HeuristicPartDetectionBackend | None = None,
        allow_remote_upload: bool | None = None,
    ) -> None:
        configured_url = api_url or os.getenv("LIVE2D_PART_API_URL")
        self.api_url = configured_url or ("inmemory://part-detector" if transport else None)
        self.api_key = api_key or os.getenv("LIVE2D_PART_API_KEY")
        self.model = model or os.getenv("LIVE2D_PART_API_MODEL", "part-locator-v1")
        self.transport = transport or self._default_transport
        self.fallback_backend = fallback_backend or HeuristicPartDetectionBackend()
        self.allow_remote_upload = (
            allow_remote_upload
            if allow_remote_upload is not None
            else self._env_flag("LIVE2D_PART_API_ALLOW_UPLOAD", default=transport is not None)
        )
        self.allowed_hosts = self._allowed_hosts()
        self.timeout_seconds = self._env_int("LIVE2D_PART_API_TIMEOUT_SECONDS", 30)
        self.max_response_bytes = self._env_int("LIVE2D_PART_API_MAX_RESPONSE_BYTES", 512 * 1024)

    async def analyze(self, image_path: str) -> JsonDict:
        fallback_result = await self.fallback_backend.analyze(image_path)
        if not self.api_url:
            fallback_result["backend_used"] = self.backend_name
            fallback_result["fallback_reason"] = self._merge_reasons(
                fallback_result.get("fallback_reason"),
                "LIVE2D_PART_API_URL is not configured; using heuristic fallback.",
            )
            return fallback_result
        if not self._remote_upload_allowed():
            fallback_result["backend_used"] = self.backend_name
            fallback_result["fallback_reason"] = self._merge_reasons(
                fallback_result.get("fallback_reason"),
                "Remote image upload is disabled; set LIVE2D_PART_API_ALLOW_UPLOAD=1 to opt in.",
            )
            return fallback_result
        host_error = self._validate_api_host()
        if host_error:
            fallback_result["backend_used"] = self.backend_name
            fallback_result["fallback_reason"] = self._merge_reasons(
                fallback_result.get("fallback_reason"),
                host_error,
            )
            return fallback_result

        try:
            response = self.transport(
                self.api_url,
                self._build_payload(image_path, fallback_result),
                self._headers(),
            )
        except Exception as exc:
            fallback_result["backend_used"] = self.backend_name
            fallback_result["fallback_reason"] = self._merge_reasons(
                fallback_result.get("fallback_reason"),
                f"API refinement failed: {exc}",
            )
            return fallback_result

        remote_parts = self._parse_remote_parts(response)
        merged = self._merge_parts(fallback_result.get("parts", []), remote_parts)
        normalized = [part.to_dict() for part in merged]
        confidences = [part.confidence for part in merged]
        fallback_result.update(
            {
                "backend_used": self.backend_name,
                "detector_used": self.detector_name,
                "parts": normalized,
                "part_count": len(normalized),
                "confidence_summary": {
                    "count": len(confidences),
                    "average": (
                        round(sum(confidences) / len(confidences), 3) if confidences else 0.0
                    ),
                    "minimum": round(min(confidences), 3) if confidences else 0.0,
                    "maximum": round(max(confidences), 3) if confidences else 0.0,
                },
                "api_metadata": {
                    "model": self.model,
                    "remote_upload_opt_in": self.allow_remote_upload,
                    "refined_parts": [
                        part.name
                        for part in merged
                        if part.attributes.get("source") == "api_backend"
                    ],
                },
            }
        )
        return fallback_result

    def _build_payload(self, image_path: str, fallback_result: JsonDict) -> JsonDict:
        image_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return {
            "model": self.model,
            "image_base64": encoded,
            "instructions": (
                "Locate left_eye, right_eye, left_eye_white, right_eye_white, "
                "left_iris, right_iris, left_eye_highlight, right_eye_highlight, "
                "left_eyebrow, right_eyebrow, mouth, nose, and hair_front. "
                "Return strict JSON with parts[]. Each part needs name, "
                "bbox{x,y,width,height}, optional polygon, confidence, and side."
            ),
            "fallback_parts": [
                part
                for part in fallback_result.get("parts", [])
                if isinstance(part, dict) and part.get("name") in self.key_part_names
            ],
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _default_transport(self, url: str, payload: JsonDict, headers: dict[str, str]) -> JsonDict:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw_response = response.read(self.max_response_bytes + 1)
        if len(raw_response) > self.max_response_bytes:
            raise ValueError("remote response exceeded LIVE2D_PART_API_MAX_RESPONSE_BYTES")
        payload_text = raw_response.decode("utf-8")
        parsed = json.loads(payload_text)
        if not isinstance(parsed, dict):
            raise ValueError("remote response must decode to a JSON object")
        return parsed

    def _env_flag(self, name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _env_int(self, name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    def _allowed_hosts(self) -> set[str]:
        raw = os.getenv("LIVE2D_PART_API_ALLOWED_HOSTS", "")
        return {item.strip().lower() for item in raw.split(",") if item.strip()}

    def _remote_upload_allowed(self) -> bool:
        return bool(
            self.api_url and (self.api_url.startswith("inmemory://") or self.allow_remote_upload)
        )

    def _validate_api_host(self) -> str | None:
        api_url = self.api_url
        if not api_url or api_url.startswith("inmemory://") or not self.allowed_hosts:
            return None
        parsed = urlparse(api_url)
        host = (parsed.hostname or "").lower()
        if host not in self.allowed_hosts:
            allowed = ", ".join(sorted(self.allowed_hosts))
            return (
                f"LIVE2D_PART_API_URL host '{host}' is not in "
                f"LIVE2D_PART_API_ALLOWED_HOSTS ({allowed})."
            )
        if parsed.scheme != "https":
            return "LIVE2D_PART_API_URL must use https when host allowlisting is enabled."
        return None

    def _parse_remote_parts(self, response: JsonDict) -> list[DetectedPart]:
        parts_payload = response.get("parts", [])
        if not isinstance(parts_payload, list):
            raise ValueError("remote response must include a parts list")

        parsed: list[DetectedPart] = []
        for item in parts_payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            bbox = item.get("bbox")
            if not name or not isinstance(bbox, dict):
                continue
            polygon_payload = item.get("polygon", [])
            polygon = [
                {"x": int(point.get("x", 0)), "y": int(point.get("y", 0))}
                for point in polygon_payload
                if isinstance(point, dict)
            ]
            parsed.append(
                DetectedPart(
                    name=name,
                    group=str(
                        item.get(
                            "group",
                            (
                                "face"
                                if name in self.key_part_names
                                or name in self.fine_grained_face_parts
                                else "body"
                            ),
                        )
                    ),
                    side=str(item.get("side")) if item.get("side") in {"left", "right"} else None,
                    bbox=BoundingBox(
                        x=int(bbox.get("x", 0)),
                        y=int(bbox.get("y", 0)),
                        width=max(1, int(bbox.get("width", 1))),
                        height=max(1, int(bbox.get("height", 1))),
                    ),
                    confidence=float(item.get("confidence", 0.8)),
                    detector=self.detector_name,
                    polygon=polygon,
                    attributes={"source": "api_backend", "refined": True},
                )
            )
        return parsed

    def _merge_parts(
        self, fallback_parts_payload: Any, remote_parts: list[DetectedPart]
    ) -> list[DetectedPart]:
        best_by_name: dict[str, DetectedPart] = {}
        for item in fallback_parts_payload if isinstance(fallback_parts_payload, list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            bbox = item.get("bbox")
            if not name or not isinstance(bbox, dict):
                continue
            best_by_name[name] = DetectedPart(
                name=name,
                group=str(item.get("group", "face")),
                side=str(item.get("side")) if item.get("side") in {"left", "right"} else None,
                bbox=BoundingBox(
                    x=int(bbox.get("x", 0)),
                    y=int(bbox.get("y", 0)),
                    width=max(1, int(bbox.get("width", 1))),
                    height=max(1, int(bbox.get("height", 1))),
                ),
                confidence=float(item.get("confidence", 0.0)),
                detector=str(item.get("detector", self.fallback_backend.detector_name)),
                polygon=list(item.get("polygon", [])),
                occluded=bool(item.get("occluded", False)),
                attributes=dict(item.get("attributes", {})),
            )
        for remote_part in remote_parts:
            existing = best_by_name.get(remote_part.name)
            if (
                existing is None
                or remote_part.confidence >= existing.confidence
                or remote_part.name in self.key_part_names
            ):
                best_by_name[remote_part.name] = remote_part
        return sorted(best_by_name.values(), key=lambda item: (item.group, item.name))

    def _merge_reasons(self, existing: Any, extra: str) -> str:
        base = str(existing).strip() if existing else ""
        return extra if not base else f"{base}; {extra}"
