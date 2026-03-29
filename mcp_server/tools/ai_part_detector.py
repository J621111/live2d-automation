"""AI-guided semantic part detection with pluggable backends."""

from __future__ import annotations

import os
from typing import Any

from mcp_server.tools.part_detection_backends import (
    APIPartDetectionBackend,
    HeuristicPartDetectionBackend,
    PartDetectionBackend,
)

JsonDict = dict[str, Any]


class AIPartDetector:
    """Produce a structured semantic parts schema for downstream segmentation."""

    def __init__(
        self,
        *,
        backend_name: str | None = None,
        api_transport: Any | None = None,
    ) -> None:
        configured_backend = os.getenv("LIVE2D_PART_BACKEND") or "heuristic"
        selected_backend = backend_name if backend_name is not None else configured_backend
        self.backend_name = selected_backend.strip().lower()
        self.backend = self._build_backend(self.backend_name, api_transport)

    async def analyze(self, image_path: str) -> JsonDict:
        result = await self.backend.analyze(image_path)
        result.setdefault("backend_used", self.backend_name)
        return result

    def _build_backend(self, backend_name: str, api_transport: Any | None) -> PartDetectionBackend:
        if backend_name == "api":
            return APIPartDetectionBackend(transport=api_transport)
        return HeuristicPartDetectionBackend()
