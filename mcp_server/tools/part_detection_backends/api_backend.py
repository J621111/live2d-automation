"""Remote API-assisted backend for semantic part detection."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib import request

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
        "left_eyebrow",
        "right_eyebrow",
        "mouth",
        "nose",
        "hair_front",
    }

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        transport: Transport | None = None,
        fallback_backend: HeuristicPartDetectionBackend | None = None,
    ) -> None:
        configured_url = api_url or os.getenv("LIVE2D_PART_API_URL")
        self.api_url = configured_url or ("inmemory://part-detector" if transport else None)
        self.api_key = api_key or os.getenv("LIVE2D_PART_API_KEY")
        self.model = model or os.getenv("LIVE2D_PART_API_MODEL", "part-locator-v1")
        self.transport = transport or self._default_transport
        self.fallback_backend = fallback_backend or HeuristicPartDetectionBackend()

    async def analyze(self, image_path: str) -> JsonDict:
        fallback_result = await self.fallback_backend.analyze(image_path)
        if not self.api_url:
            fallback_result["backend_used"] = self.backend_name
            fallback_result["fallback_reason"] = self._merge_reasons(
                fallback_result.get("fallback_reason"),
                "LIVE2D_PART_API_URL is not configured; using heuristic fallback.",
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
                "Locate left_eye, right_eye, left_eyebrow, right_eyebrow, "
                "mouth, nose, and hair_front. Return strict JSON with "
                "parts[]. Each part needs name, bbox{x,y,width,height}, "
                "optional polygon, confidence, and side."
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
        with request.urlopen(req, timeout=30) as response:
            payload_text = response.read().decode("utf-8")
        parsed = json.loads(payload_text)
        if not isinstance(parsed, dict):
            raise ValueError("remote response must decode to a JSON object")
        return parsed

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
                    group=str(item.get("group", "face" if name in self.key_part_names else "body")),
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
