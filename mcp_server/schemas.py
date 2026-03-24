"""Structured schemas for AI-guided Live2D preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class DetectedPart:
    name: str
    group: str
    side: str | None
    bbox: BoundingBox
    confidence: float
    detector: str
    polygon: list[dict[str, int]] = field(default_factory=list)
    occluded: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload


@dataclass(slots=True)
class PartMask:
    name: str
    bbox: BoundingBox
    mask_path: str
    alpha_path: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload


@dataclass(slots=True)
class LayerAsset:
    name: str
    group: str
    side: str | None
    path: str
    mask_path: str
    bounds: BoundingBox
    z_order: int
    confidence: float
    detector: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bounds"] = self.bounds.to_dict()
        return payload


@dataclass(slots=True)
class TemplateMapping:
    template_id: str
    layer_name: str
    target_part: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CubismJob:
    job_id: str
    template_id: str
    model_name: str
    output_dir: str
    interactive_mode: str = "assisted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CubismExportResult:
    status: str
    output_dir: str
    exported_files: dict[str, str]
    logs: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
