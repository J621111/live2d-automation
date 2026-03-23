"""Layer generation for normalized Live2D parts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from loguru import logger

JsonDict = dict[str, Any]
JsonList = list[JsonDict]


class LayerGenerator:
    """Live2D layer generator with normalized part handling."""

    PRECISE_SEGMENT_TIMEOUT_SECONDS = 15

    def __init__(self) -> None:
        self.layer_order = [
            "back_hair",
            "body",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
            "head",
            "face_base",
            "mouth",
            "nose",
            "left_eye",
            "right_eye",
            "left_eyebrow",
            "right_eyebrow",
            "front_hair",
            "left_hand",
            "right_hand",
            "accessories",
        ]
        self.segmenter: Any | None = None
        self.last_generation_metadata: JsonDict = {
            "detector_used": "unknown",
            "fallback_reason": None,
            "confidence_summary": {
                "count": 0,
                "average": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
            },
        }

    async def initialize(self) -> None:
        if self.segmenter is None:
            try:
                from mcp_server.tools.segmentation import PreciseSegmenter

                self.segmenter = PreciseSegmenter()
                await self.segmenter.initialize()
            except ImportError as exc:
                logger.warning(f"Could not load precise segmenter: {exc}")
                self.segmenter = None

    def _confidence_summary(self, parts: JsonDict) -> JsonDict:
        confidences = [
            float(part.get("confidence", 0.0))
            for part in parts.values()
            if isinstance(part, dict) and "confidence" in part
        ]
        if not confidences:
            return {"count": 0, "average": 0.0, "minimum": 0.0, "maximum": 0.0}
        return {
            "count": len(confidences),
            "average": round(sum(confidences) / len(confidences), 3),
            "minimum": round(min(confidences), 3),
            "maximum": round(max(confidences), 3),
        }

    async def generate(
        self,
        image_path: str,
        segments: JsonDict | None = None,
        output_dir: str | None = None,
    ) -> JsonList:
        output_path = Path(output_dir) if output_dir else Path("output")
        output_path.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGBA")
        layers: JsonList = []
        normalized_segments = self._normalize_segments(segments)
        if not normalized_segments.get("parts"):
            logger.info("Using precise segmentation...")
            normalized_segments = await self._precise_segment(image_path)

        parts = dict(normalized_segments.get("parts", {}))
        self.last_generation_metadata = {
            "detector_used": normalized_segments.get("detector_used")
            or normalized_segments.get("segmenter")
            or normalized_segments.get("detector")
            or "provided_parts",
            "fallback_reason": normalized_segments.get("fallback_reason"),
            "confidence_summary": normalized_segments.get("confidence_summary")
            or self._confidence_summary(parts),
        }

        for part_name, part_data in parts.items():
            if isinstance(part_data, dict) and "bounds" in part_data:
                layer = await self._create_layer_from_bounds(
                    image,
                    dict(part_data["bounds"]),
                    str(part_name),
                    output_path,
                    cast(np.ndarray | None, part_data.get("mask")),
                    cast(float | None, part_data.get("confidence")),
                )
                if layer is not None:
                    layers.append(layer)

        layers = self._sort_layers(layers)
        logger.info(f"Generated {len(layers)} layers")
        return layers

    async def _precise_segment(self, image_path: str) -> JsonDict:
        try:
            return await asyncio.wait_for(
                self._run_precise_segment(image_path),
                timeout=self.PRECISE_SEGMENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Precise segmentation timed out, falling back to empty parts")
            return {
                "parts": {},
                "detector_used": "fallback",
                "fallback_reason": "precise segmentation timed out",
                "confidence_summary": {
                    "count": 0,
                    "average": 0.0,
                    "minimum": 0.0,
                    "maximum": 0.0,
                },
            }

    async def _run_precise_segment(self, image_path: str) -> JsonDict:
        await self.initialize()
        if self.segmenter is None:
            logger.warning("Precise segmenter unavailable, returning empty parts")
            return {
                "parts": {},
                "detector_used": "fallback",
                "fallback_reason": "precise segmenter unavailable",
                "confidence_summary": {
                    "count": 0,
                    "average": 0.0,
                    "minimum": 0.0,
                    "maximum": 0.0,
                },
            }
        return cast(JsonDict, await self.segmenter.segment_character(image_path))

    def _normalize_segments(self, segments: JsonDict | None = None) -> JsonDict:
        if not segments:
            return {"parts": {}}
        if segments.get("parts"):
            return segments

        normalized_parts: JsonDict = {}
        for name, data in dict(segments.get("body_parts", {})).items():
            if not isinstance(data, dict):
                continue
            if "bounds" in data:
                normalized_parts[str(name)] = data
                continue
            for nested_name, nested_data in data.items():
                if isinstance(nested_data, dict) and "bounds" in nested_data:
                    normalized_parts[str(nested_name)] = nested_data
        return {**segments, "parts": normalized_parts}

    async def _create_layer_from_bounds(
        self,
        image: Image.Image,
        bounds: JsonDict,
        part_name: str,
        output_dir: Path,
        mask: np.ndarray | None = None,
        confidence: float | None = None,
    ) -> JsonDict | None:
        x = int(bounds.get("x", 0))
        y = int(bounds.get("y", 0))
        w = int(bounds.get("width", image.width))
        h = int(bounds.get("height", image.height))

        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        w = min(w, image.width - x)
        h = min(h, image.height - y)
        if w <= 0 or h <= 0:
            return None

        part_img = image.crop((x, y, x + w, y + h))
        if mask is not None:
            try:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8) * 255,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                part_img.putalpha(Image.fromarray(mask_resized, mode="L"))
            except Exception as exc:
                logger.warning(f"Mask application failed for {part_name}: {exc}")

        layer_filename = f"layer_{part_name}.png"
        layer_path = output_dir / layer_filename
        if part_img.mode != "RGBA":
            part_img = part_img.convert("RGBA")
        part_img.save(layer_path, "PNG")

        return {
            "name": part_name,
            "type": "body_part",
            "filename": layer_filename,
            "path": str(layer_path),
            "bounds": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "original_x": x,
                "original_y": y,
            },
            "texture_path": str(layer_path),
            "z_order": self._get_z_order(part_name),
            "confidence": confidence if confidence is not None else 0.5,
        }

    def _get_z_order(self, part_name: str) -> int:
        order_map = {name: index for index, name in enumerate(self.layer_order)}
        name_mapping = {
            "head": "head",
            "torso": "body",
            "left_arm": "left_arm",
            "right_arm": "right_arm",
            "left_leg": "left_leg",
            "right_leg": "right_leg",
            "left_eye": "left_eye",
            "right_eye": "right_eye",
            "left_eyebrow": "left_eyebrow",
            "right_eyebrow": "right_eyebrow",
            "nose": "nose",
            "mouth": "mouth",
        }
        return order_map.get(name_mapping.get(part_name, part_name), 50)

    def _sort_layers(self, layers: JsonList) -> JsonList:
        return sorted(layers, key=lambda item: int(item.get("z_order", 50)))

    async def generate_with_face_details(
        self,
        image_path: str,
        face_landmarks: JsonDict | None = None,
    ) -> JsonList:
        layers = await self.generate(image_path)
        if face_landmarks is None:
            return layers

        image = Image.open(image_path).convert("RGBA")
        for part_name in ("left_eye", "right_eye", "mouth", "nose"):
            points = list(face_landmarks.get(part_name, []))
            if not points:
                continue
            xs = [point["x"] for point in points]
            ys = [point["y"] for point in points]
            bounds = {
                "x": min(xs) - 10,
                "y": min(ys) - 10,
                "width": max(xs) - min(xs) + 20,
                "height": max(ys) - min(ys) + 20,
            }
            layer = await self._create_layer_from_bounds(
                image,
                bounds,
                part_name,
                Path("output"),
                None,
                0.6,
            )
            if layer is not None:
                layers.append(layer)

        return layers
