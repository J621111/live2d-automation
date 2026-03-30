"""Mask-driven part segmentation for AI-detected semantic parts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from mcp_server.schemas import BoundingBox, DetectedPart, LayerAsset

JsonDict = dict[str, Any]


class PartSegmenter:
    """Extract alpha-isolated layer assets from detected semantic parts."""

    def __init__(self) -> None:
        self._z_order = {
            "hair_back": 10,
            "left_arm": 20,
            "right_arm": 21,
            "left_leg": 22,
            "right_leg": 23,
            "torso": 30,
            "head": 40,
            "face_base": 50,
            "left_eyebrow": 60,
            "right_eyebrow": 61,
            "left_eye_white": 62,
            "left_iris": 63,
            "left_eye_highlight": 64,
            "left_eye": 65,
            "right_eye_white": 66,
            "right_iris": 67,
            "right_eye_highlight": 68,
            "right_eye": 69,
            "nose": 70,
            "mouth": 71,
            "hair_side_left": 72,
            "hair_side_right": 73,
            "hair_front": 74,
        }
        self._crop_padding = {
            "left_eye": (0.08, 0.12),
            "right_eye": (0.08, 0.12),
            "left_eye_white": (0.06, 0.1),
            "right_eye_white": (0.06, 0.1),
            "left_iris": (0.04, 0.08),
            "right_iris": (0.04, 0.08),
            "left_eye_highlight": (0.03, 0.05),
            "right_eye_highlight": (0.03, 0.05),
            "left_eyebrow": (0.08, 0.18),
            "right_eyebrow": (0.08, 0.18),
            "mouth": (0.12, 0.18),
            "nose": (0.1, 0.15),
        }

    async def segment(
        self,
        image_path: str,
        detected_parts: list[JsonDict],
        output_dir: str,
    ) -> JsonDict:
        base_output = Path(output_dir)
        parts_dir = base_output / "ai_parts"
        masks_dir = base_output / "ai_masks"
        parts_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path).convert("RGBA") as image:
            rgba = np.array(image)

        layers: list[LayerAsset] = []
        for payload in detected_parts:
            part = self._from_dict(payload)
            layer = self._extract_layer(rgba, part, parts_dir, masks_dir)
            if layer is not None:
                layers.append(layer)

        ordered = sorted(layers, key=lambda item: item.z_order)
        return {
            "status": "success",
            "layers": [layer.to_dict() for layer in ordered],
            "layers_generated": len(ordered),
            "output_dir": str(base_output),
        }

    def _extract_layer(
        self,
        rgba: np.ndarray,
        part: DetectedPart,
        parts_dir: Path,
        masks_dir: Path,
    ) -> LayerAsset | None:
        x, y, w, h = part.bbox.x, part.bbox.y, part.bbox.width, part.bbox.height
        crop = rgba[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        alpha = crop[:, :, 3]
        if np.count_nonzero(alpha) == 0:
            return None

        polygon_mask = self._mask_from_polygon(part, crop.shape[1], crop.shape[0])
        color_mask = self._foreground_mask(crop, part.name)
        mask = self._refine_mask(crop, part, polygon_mask, color_mask)
        if np.count_nonzero(mask) == 0:
            mask = (alpha > 0).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        min_x = int(xs.min())
        max_x = int(xs.max()) + 1
        min_y = int(ys.min())
        max_y = int(ys.max()) + 1
        min_x, max_x, min_y, max_y = self._expand_trim_bounds(
            part.name, min_x, max_x, min_y, max_y, crop.shape[1], crop.shape[0]
        )
        trimmed = crop[min_y:max_y, min_x:max_x].copy()
        trimmed_mask: np.ndarray = mask[min_y:max_y, min_x:max_x] * 255
        trimmed[:, :, 3] = trimmed_mask

        image_path = parts_dir / f"{part.name}.png"
        mask_path = masks_dir / f"{part.name}.png"
        Image.fromarray(trimmed, mode="RGBA").save(image_path, "PNG")
        Image.fromarray(trimmed_mask.astype(np.uint8), mode="L").save(mask_path, "PNG")

        bounds = BoundingBox(
            x=x + min_x,
            y=y + min_y,
            width=max(1, max_x - min_x),
            height=max(1, max_y - min_y),
        )
        return LayerAsset(
            name=part.name,
            group=part.group,
            side=part.side,
            path=str(image_path),
            mask_path=str(mask_path),
            bounds=bounds,
            z_order=self._z_order.get(part.name, 80),
            confidence=part.confidence,
            detector=part.detector,
            metadata={"occluded": part.occluded, "attributes": dict(part.attributes)},
        )

    def _refine_mask(
        self,
        crop: np.ndarray,
        part: DetectedPart,
        polygon_mask: np.ndarray | None,
        color_mask: np.ndarray,
    ) -> np.ndarray:
        alpha_mask = (crop[:, :, 3] > 0).astype(np.uint8)
        if polygon_mask is None:
            return cast(np.ndarray, color_mask if np.count_nonzero(color_mask) else alpha_mask)

        if part.name in {
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
        }:
            guide_mask = cv2.dilate(polygon_mask, np.ones((3, 3), np.uint8), iterations=1)
            guided_color = ((color_mask > 0) & (guide_mask > 0)).astype(np.uint8)
            if np.count_nonzero(guided_color) >= 6:
                return cast(np.ndarray, guided_color)
            return cast(np.ndarray, guide_mask)

        merged = ((polygon_mask > 0) & (alpha_mask > 0)).astype(np.uint8)
        if np.count_nonzero(merged) > 0:
            return cast(np.ndarray, merged)
        return cast(np.ndarray, alpha_mask)

    def _expand_trim_bounds(
        self,
        part_name: str,
        min_x: int,
        max_x: int,
        min_y: int,
        max_y: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        pad_x_ratio, pad_y_ratio = self._crop_padding.get(part_name, (0.04, 0.06))
        pad_x = max(1, int((max_x - min_x) * pad_x_ratio))
        pad_y = max(1, int((max_y - min_y) * pad_y_ratio))
        return (
            max(0, min_x - pad_x),
            min(width, max_x + pad_x),
            max(0, min_y - pad_y),
            min(height, max_y + pad_y),
        )

    def _mask_from_polygon(self, part: DetectedPart, width: int, height: int) -> np.ndarray | None:
        if not part.polygon:
            return None
        points = []
        for point in part.polygon:
            px = int(point["x"]) - part.bbox.x
            py = int(point["y"]) - part.bbox.y
            points.append([max(0, min(px, width - 1)), max(0, min(py, height - 1))])
        if len(points) < 3:
            return None
        polygon = np.array([points], dtype=np.int32)
        mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, polygon, 1)
        return cast(np.ndarray, np.asarray(mask, dtype=np.uint8))

    def _foreground_mask(self, crop: np.ndarray, part_name: str) -> np.ndarray:
        alpha = (crop[:, :, 3] > 0).astype(np.uint8)
        if alpha.shape[0] < 4 or alpha.shape[1] < 4:
            return cast(np.ndarray, alpha)

        rgb = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if "highlight" in part_name:
            threshold = int(np.quantile(gray[alpha > 0], 0.8)) if np.count_nonzero(alpha) else 200
            mask: np.ndarray = ((gray >= threshold) & (alpha > 0)).astype(np.uint8)
        elif "iris" in part_name:
            blue_strength = crop[:, :, 2].astype(np.int16) - crop[:, :, 1].astype(np.int16)
            threshold = (
                int(np.quantile(blue_strength[alpha > 0], 0.65)) if np.count_nonzero(alpha) else 10
            )
            mask = ((blue_strength >= threshold) & (alpha > 0)).astype(np.uint8)
        elif "eye_white" in part_name:
            threshold = int(np.quantile(gray[alpha > 0], 0.7)) if np.count_nonzero(alpha) else 180
            mask = ((gray >= threshold) & (alpha > 0)).astype(np.uint8)
        elif "eye" in part_name or "eyebrow" in part_name or part_name == "nose":
            threshold = (
                max(10, int(np.quantile(gray[alpha > 0], 0.25))) if np.count_nonzero(alpha) else 60
            )
            mask = ((gray <= threshold) & (alpha > 0)).astype(np.uint8)
        elif part_name == "mouth":
            red_strength = crop[:, :, 0].astype(np.int16) - crop[:, :, 1].astype(np.int16)
            threshold = (
                int(np.quantile(red_strength[alpha > 0], 0.65)) if np.count_nonzero(alpha) else 10
            )
            mask = ((red_strength >= threshold) & (alpha > 0)).astype(np.uint8)
        else:
            mask = alpha

        kernel: np.ndarray = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        if np.count_nonzero(mask) >= 6:
            return cast(np.ndarray, np.asarray(mask, dtype=np.uint8))
        return cast(np.ndarray, np.asarray(alpha, dtype=np.uint8))

    def _from_dict(self, payload: JsonDict) -> DetectedPart:
        bbox_payload = dict(payload.get("bbox", {}))
        return DetectedPart(
            name=str(payload["name"]),
            group=str(payload.get("group", "unknown")),
            side=payload.get("side"),
            bbox=BoundingBox(
                x=int(bbox_payload.get("x", 0)),
                y=int(bbox_payload.get("y", 0)),
                width=int(bbox_payload.get("width", 1)),
                height=int(bbox_payload.get("height", 1)),
            ),
            confidence=float(payload.get("confidence", 0.0)),
            detector=str(payload.get("detector", "unknown")),
            polygon=list(payload.get("polygon", [])),
            occluded=bool(payload.get("occluded", False)),
            attributes=dict(payload.get("attributes", {})),
        )
