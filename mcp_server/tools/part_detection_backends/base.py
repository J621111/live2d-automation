"""Abstract backend contract for part detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

JsonDict = dict[str, Any]


class PartDetectionBackend(ABC):
    """Base interface for structured part detection backends."""

    @abstractmethod
    async def analyze(self, image_path: str) -> JsonDict:
        """Return structured part detection payload."""
