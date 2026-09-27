"""Mask-driven part segmentation for AI-detected semantic parts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
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
            "left_eye_white": (0.06, 0.10),
            "right_eye_white": (0.06, 0.10),
            "left_iris": (0.05, 0.08),
            "right_iris": (0.05, 0.08),
            "left_eye_highlight": (0.03, 0.05),
            "right_eye_highlight": (0.03, 0.05),
            "left_eyebrow": (0.08, 0.18),
            "right_eyebrow": (0.08, 0.18),
            "mouth": (0.12, 0.18),
            "nose": (0.10, 0.15),
        }
        self._detail_guided_parts = {
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
        }

    async def segment(
        self,
        image_path: str,
        detected_parts: list[JsonDict],
        output_dir: str,
        corrections: JsonDict | None = None,
    ) -> JsonDict:
        base_output = Path(output_dir)
        parts_dir = base_output / "ai_parts"
        masks_dir = base_output / "ai_masks"
        parts_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path).convert("RGBA") as image:
            rgba = np.array(image)

        layers: list[LayerAsset] = []
        corrections = corrections or {}
        for payload in detected_parts:
            part = self._from_dict(payload)
            correction = corrections.get(part.name)
            if isinstance(correction, dict):
                part = self._apply_correction(part, correction)
            layer = self._extract_layer(rgba, part, parts_dir, masks_dir)
            if layer is not None:
                layers.append(layer)

        ordered = sorted(layers, key=lambda item: item.z_order)
        quality = [
            dict(layer.metadata.get("mask_quality", {}), name=layer.name) for layer in ordered
        ]
        by_name = {layer.name: layer for layer in ordered}
        for item in quality:
            layer = by_name[item["name"]]
            b = layer.bounds
            for other in ordered:
                if other.name == layer.name or other.group != layer.group:
                    continue
                ob = other.bounds
                ix = max(0, min(b.x + b.width, ob.x + ob.width) - max(b.x, ob.x))
                iy = max(0, min(b.y + b.height, ob.y + ob.height) - max(b.y, ob.y))
                intersection = ix * iy
                union = b.width * b.height + ob.width * ob.height - intersection
                if union and intersection / union > 0.85:
                    item.setdefault("reasons", []).append("severe_overlap")
                    item["needs_review"] = True
                    break
        review_items = [item for item in quality if item.get("needs_review")]
        correction_manifest = base_output / "correction_manifest.json"
        correction_manifest.write_text(
            json.dumps({"version": 1, "items": review_items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "status": "success",
            "layers": [layer.to_dict() for layer in ordered],
            "layers_generated": len(ordered),
            "output_dir": str(base_output),
            "mask_quality": quality,
            "quality_summary": {
                "total": len(quality),
                "high_quality": len(quality) - len(review_items),
                "low_quality": len(review_items),
            },
            "needs_review": bool(review_items),
            "review_items": review_items,
            "correction_manifest_path": str(correction_manifest),
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

        polygon_mask = self._mask_from_file(part, crop.shape[1], crop.shape[0])
        if polygon_mask is None:
            polygon_mask = self._mask_from_polygon(part, crop.shape[1], crop.shape[0])
        color_mask = self._foreground_mask(crop, part.name)
        mask = self._refine_mask(crop, part, polygon_mask, color_mask)
        if "highlight" in part.name and np.count_nonzero(color_mask) >= 2:
            mask = ((mask > 0) & (color_mask > 0)).astype(np.uint8)
            bright = (crop[:, :, :3].mean(axis=2) >= 220).astype(np.uint8)
            if np.count_nonzero(bright) >= 2:
                mask = bright
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
        trimmed_binary = mask[min_y:max_y, min_x:max_x].astype(np.uint8)
        trimmed_binary = np.asarray(
            cv2.morphologyEx(trimmed_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
            dtype=np.uint8,
        )
        # 高光、虹膜等细小部件不能让羽化扩散到整个裁剪框。
        if trimmed_binary.shape[0] <= 12 or trimmed_binary.shape[1] <= 12:
            trimmed_mask = np.asarray(trimmed_binary * 255, dtype=np.uint8)
        else:
            trimmed_mask = np.asarray(
                cv2.GaussianBlur(trimmed_binary * 255, (3, 3), 0),
                dtype=np.uint8,
            )
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
            metadata={
                "occluded": part.occluded,
                "attributes": dict(part.attributes),
                "mask_source": (
                    "model_mask" if part.mask_path else ("polygon" if part.polygon else "heuristic")
                ),
                "mask_quality": self._assess_quality(
                    part, mask, x, y, rgba.shape[1], rgba.shape[0]
                ),
            },
        )

    def _assess_quality(
        self, part: DetectedPart, mask: np.ndarray, x: int, y: int, width: int, height: int
    ) -> JsonDict:
        area = int(np.count_nonzero(mask))
        bbox_area = max(1, part.bbox.width * part.bbox.height)
        ratio = area / bbox_area
        touches_edge = (
            part.bbox.x <= 0
            or part.bbox.y <= 0
            or part.bbox.x + part.bbox.width >= width
            or part.bbox.y + part.bbox.height >= height
        )
        reasons: list[str] = []
        if not part.polygon and not part.mask_path:
            reasons.append("missing_polygon_or_mask")
        if area == 0:
            reasons.append("empty_mask")
        if ratio < 0.01 or ratio > 0.98:
            reasons.append("mask_area_mismatch")
        if touches_edge:
            reasons.append("touches_image_boundary")
        if part.occluded:
            reasons.append("occluded")
        confidence = max(0.0, min(1.0, float(part.confidence)))
        if confidence < 0.6:
            reasons.append("low_confidence")
        return {
            "confidence": round(confidence, 3),
            "mask_area": area,
            "mask_area_ratio": round(ratio, 4),
            "bbox_area": bbox_area,
            "touches_image_boundary": touches_edge,
            "occluded": part.occluded,
            "mask_source": (
                "model_mask" if part.mask_path else ("polygon" if part.polygon else "heuristic")
            ),
            "needs_review": bool(reasons),
            "reasons": reasons,
        }

    def _apply_correction(self, part: DetectedPart, correction: JsonDict) -> DetectedPart:
        bbox = correction.get("bbox")
        if isinstance(bbox, dict):
            part.bbox = BoundingBox(
                x=int(bbox.get("x", part.bbox.x)),
                y=int(bbox.get("y", part.bbox.y)),
                width=max(1, int(bbox.get("width", part.bbox.width))),
                height=max(1, int(bbox.get("height", part.bbox.height))),
            )
        polygon = correction.get("polygon")
        if isinstance(polygon, list):
            part.polygon = [p for p in polygon if isinstance(p, dict) and "x" in p and "y" in p]
        part.attributes = {
            **part.attributes,
            "correction": {k: v for k, v in correction.items() if k != "mask"},
        }
        return part

    def _refine_mask(
        self,
        crop: np.ndarray,
        part: DetectedPart,
        polygon_mask: np.ndarray | None,
        color_mask: np.ndarray,
    ) -> np.ndarray:
        alpha_mask = (crop[:, :, 3] > 0).astype(np.uint8)
        focus = self._focus_point(part, crop.shape[1], crop.shape[0])

        if polygon_mask is None:
            selected = self._select_component(color_mask, focus)
            if np.count_nonzero(selected) >= 6:
                return selected
            return cast(np.ndarray, color_mask if np.count_nonzero(color_mask) else alpha_mask)

        guide_mask = cv2.dilate(polygon_mask, np.ones((3, 3), np.uint8), iterations=1)
        guided_alpha = ((guide_mask > 0) & (alpha_mask > 0)).astype(np.uint8)
        guided_color = ((guide_mask > 0) & (color_mask > 0) & (alpha_mask > 0)).astype(np.uint8)

        if part.name in self._detail_guided_parts:
            selected = self._select_component(guided_color, focus)
            if np.count_nonzero(selected) >= 4:
                return selected
            selected = self._select_component(guided_alpha, focus)
            if np.count_nonzero(selected) >= 4:
                return selected
            return cast(np.ndarray, guide_mask)

        if part.name in {"left_eye", "right_eye"}:
            if np.count_nonzero(guided_alpha) >= 4:
                return self._select_component(guided_alpha, focus)
            return cast(np.ndarray, guide_mask)

        merged = ((polygon_mask > 0) & (alpha_mask > 0)).astype(np.uint8)
        if np.count_nonzero(merged) > 0:
            return self._select_component(merged, focus)
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
        polygon = np.asarray(points, dtype=np.int32)
        mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        return cast(np.ndarray, np.asarray(mask, dtype=np.uint8))

    def _mask_from_file(self, part: DetectedPart, width: int, height: int) -> np.ndarray | None:
        if not part.mask_path:
            return None
        path = Path(part.mask_path)
        if not path.exists():
            return None
        try:
            with Image.open(path).convert("L") as image:
                resized = image.resize((width, height), Image.Resampling.BILINEAR)
                return (np.asarray(resized) > 16).astype(np.uint8)
        except (OSError, ValueError):
            return None

    def _foreground_mask(self, crop: np.ndarray, part_name: str) -> np.ndarray:
        alpha = (crop[:, :, 3] > 0).astype(np.uint8)
        if alpha.shape[0] < 4 or alpha.shape[1] < 4:
            return cast(np.ndarray, alpha)

        rgb: np.ndarray = crop[:, :, :3].astype(np.uint8)
        gray = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.int16)
        value = hsv[:, :, 2].astype(np.int16)
        active = alpha > 0
        if not np.count_nonzero(active):
            return cast(np.ndarray, alpha)

        if "highlight" in part_name:
            value_threshold = max(225, int(np.quantile(value[active], 0.92)))
            saturation_threshold = min(80, int(np.quantile(saturation[active], 0.35)) + 8)
            mask = (
                (value >= value_threshold) & (saturation <= saturation_threshold) & active
            ).astype(np.uint8)
        elif "eye_white" in part_name:
            value_threshold = max(190, int(np.quantile(value[active], 0.72)))
            saturation_threshold = min(95, int(np.quantile(saturation[active], 0.45)) + 10)
            mask = (
                (value >= value_threshold) & (saturation <= saturation_threshold) & active
            ).astype(np.uint8)
        elif "iris" in part_name:
            red: np.ndarray = rgb[:, :, 0].astype(np.int16)
            green: np.ndarray = rgb[:, :, 1].astype(np.int16)
            blue: np.ndarray = rgb[:, :, 2].astype(np.int16)
            blue_delta = blue - ((red + green) // 2)
            delta_threshold = max(8, int(np.quantile(blue_delta[active], 0.60)))
            saturation_threshold = max(50, int(np.quantile(saturation[active], 0.60)))
            lower_value = max(25, int(np.quantile(value[active], 0.15)))
            upper_value = min(235, int(np.quantile(value[active], 0.85)) + 8)
            mask = (
                (blue_delta >= delta_threshold)
                & (saturation >= saturation_threshold)
                & (value >= lower_value)
                & (value <= upper_value)
                & active
            ).astype(np.uint8)
        elif "eye" in part_name or "eyebrow" in part_name or part_name == "nose":
            threshold = max(10, int(np.quantile(gray[active], 0.25)))
            mask = ((gray <= threshold) & active).astype(np.uint8)
        elif part_name == "mouth":
            red_strength = rgb[:, :, 0].astype(np.int16) - rgb[:, :, 1].astype(np.int16)
            threshold = max(8, int(np.quantile(red_strength[active], 0.65)))
            mask = ((red_strength >= threshold) & active).astype(np.uint8)
        else:
            mask = alpha

        kernel: np.ndarray = np.ones((3, 3), np.uint8)
        cleaned_mask = np.asarray(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1),
            dtype=np.uint8,
        )
        cleaned_mask = np.asarray(
            cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1),
            dtype=np.uint8,
        )
        if np.count_nonzero(cleaned_mask) >= 6:
            return self._select_component(cleaned_mask, None)
        return cast(np.ndarray, np.asarray(alpha, dtype=np.uint8))

    def _focus_point(
        self,
        part: DetectedPart,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        if part.polygon:
            xs = [max(0, min(int(point["x"]) - part.bbox.x, width - 1)) for point in part.polygon]
            ys = [max(0, min(int(point["y"]) - part.bbox.y, height - 1)) for point in part.polygon]
            if xs and ys:
                return (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))
        return (width / 2.0, height / 2.0)

    def _select_component(
        self,
        mask: np.ndarray,
        focus: tuple[float, float] | None,
    ) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary) < 2:
            return cast(np.ndarray, binary)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return cast(np.ndarray, binary)

        best_label = 0
        best_score = -1.0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            score = float(area)
            if focus is not None:
                centroid = centroids[label]
                distance = float(np.hypot(centroid[0] - focus[0], centroid[1] - focus[1]))
                score -= distance * 0.75
            if score > best_score:
                best_score = score
                best_label = label

        if best_label == 0:
            return cast(np.ndarray, binary)
        return cast(np.ndarray, (labels == best_label).astype(np.uint8))

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
            mask_path=str(payload.get("mask_path")) if payload.get("mask_path") else None,
            occluded=bool(payload.get("occluded", False)),
            attributes=dict(payload.get("attributes", {})),
        )
