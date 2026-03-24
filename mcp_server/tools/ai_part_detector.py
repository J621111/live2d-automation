"""AI-guided semantic part detection for anime character images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from mcp_server.schemas import BoundingBox, DetectedPart
from mcp_server.tools.facial_detector import FacialFeatureDetector
from mcp_server.tools.image_processor import ImageProcessor

JsonDict = dict[str, Any]


class AIPartDetector:
    """Produce a structured semantic parts schema for downstream segmentation."""

    def __init__(self) -> None:
        self.body_detector = ImageProcessor()
        self.face_detector = FacialFeatureDetector()

    async def analyze(self, image_path: str) -> JsonDict:
        body_result = await self.body_detector.analyze(image_path)
        face_result = await self.face_detector.detect_features(image_path)

        with Image.open(image_path).convert("RGBA") as image:
            width, height = image.size
            rgba = np.array(image)

        parts: list[DetectedPart] = []
        parts.extend(self._collect_body_parts(body_result, width, height))
        parts.extend(self._collect_face_parts(face_result, rgba))
        parts.extend(self._infer_hair_parts(face_result, width, height))

        deduped = self._dedupe_parts(parts)
        return {
            "schema_version": 1,
            "detector_used": "semantic_refine_v1",
            "fallback_reason": self._fallback_reason(body_result, face_result),
            "parts": [part.to_dict() for part in deduped],
            "part_count": len(deduped),
            "confidence_summary": self._confidence_summary(deduped),
            "image_info": {"path": image_path, "width": width, "height": height},
        }

    def _collect_body_parts(
        self, result: JsonDict, image_width: int, image_height: int
    ) -> list[DetectedPart]:
        parts: list[DetectedPart] = []
        body_parts = dict(result.get("body_parts", {}))
        for name in ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"):
            part = body_parts.get(name)
            if not isinstance(part, dict):
                continue
            bounds = part.get("bounds")
            if not isinstance(bounds, dict):
                continue
            parts.append(
                DetectedPart(
                    name=name,
                    group="body",
                    side=self._side_from_name(name),
                    bbox=self._box(bounds, image_width, image_height),
                    confidence=float(part.get("confidence", 0.35)),
                    detector=str(result.get("detector_used", "heuristic")),
                    attributes={"source": "body_detector", "refined": False},
                )
            )
        return parts

    def _collect_face_parts(self, result: JsonDict, rgba: np.ndarray) -> list[DetectedPart]:
        image_height, image_width = rgba.shape[:2]
        parts: list[DetectedPart] = []
        face_bounds = result.get("face_bounds")
        if isinstance(face_bounds, dict):
            parts.append(
                DetectedPart(
                    name="face_base",
                    group="face",
                    side=None,
                    bbox=self._box(face_bounds, image_width, image_height),
                    confidence=max(
                        0.4,
                        float(result.get("confidence_summary", {}).get("average", 0.4)),
                    ),
                    detector=str(result.get("detector_used", "heuristic")),
                    attributes={"source": "face_bounds", "refined": False},
                )
            )
        for name, part in dict(result.get("parts", {})).items():
            if not isinstance(part, dict):
                continue
            bounds = part.get("bounds")
            if not isinstance(bounds, dict):
                continue

            box = self._box(bounds, image_width, image_height)
            refined = self._refine_part_region(rgba, box, str(name))
            if refined is not None:
                refined_box, polygon = refined
                box = refined_box
                polygon_points = polygon
                refined_flag = True
            else:
                polygon_points = self._rect_polygon(box)
                refined_flag = False

            parts.append(
                DetectedPart(
                    name=str(name),
                    group="face",
                    side=self._side_from_name(str(name)),
                    bbox=box,
                    confidence=min(
                        0.98,
                        float(part.get("confidence", 0.4)) + (0.1 if refined_flag else 0.0),
                    ),
                    detector="semantic_refine_v1" if refined_flag else str(result.get("detector_used", "heuristic")),
                    polygon=polygon_points,
                    attributes={"source": "face_detector", "refined": refined_flag},
                )
            )
        return parts

    def _refine_part_region(
        self,
        rgba: np.ndarray,
        box: BoundingBox,
        name: str,
    ) -> tuple[BoundingBox, list[dict[str, int]]] | None:
        crop = rgba[box.y : box.y + box.height, box.x : box.x + box.width]
        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        alpha = crop[:, :, 3]
        alpha_mask = alpha > 0
        if not np.count_nonzero(alpha_mask):
            return None

        if "eye" in name or "eyebrow" in name or name == "nose":
            threshold = max(10, int(np.quantile(gray[alpha_mask], 0.18)))
            mask = ((gray <= threshold) & alpha_mask).astype(np.uint8)
        elif name == "mouth":
            red_strength = crop[:, :, 0].astype(np.int16) - crop[:, :, 1].astype(np.int16)
            threshold = int(np.quantile(red_strength[alpha_mask], 0.78))
            mask = ((red_strength >= threshold) & alpha_mask).astype(np.uint8)
        elif "blush" in name:
            red_strength = crop[:, :, 0].astype(np.int16) - crop[:, :, 2].astype(np.int16)
            threshold = int(np.quantile(red_strength[alpha_mask], 0.72))
            mask = ((red_strength >= threshold) & alpha_mask).astype(np.uint8)
        else:
            return None

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        component = self._best_component(mask, name)
        if component is None:
            return None

        component_mask, component_box = component
        if np.count_nonzero(component_mask) < 6:
            return None

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        padding_x = max(1, int(component_box[2] * 0.12))
        padding_y = max(1, int(component_box[3] * 0.18))
        x = max(0, component_box[0] - padding_x)
        y = max(0, component_box[1] - padding_y)
        max_x = min(crop.shape[1], component_box[0] + component_box[2] + padding_x)
        max_y = min(crop.shape[0], component_box[1] + component_box[3] + padding_y)

        global_box = BoundingBox(
            x=box.x + x,
            y=box.y + y,
            width=max(1, max_x - x),
            height=max(1, max_y - y),
        )
        polygon = [
            {"x": int(point[0][0]) + box.x, "y": int(point[0][1]) + box.y}
            for point in contour
        ]
        return global_box, polygon

    def _best_component(
        self,
        mask: np.ndarray,
        name: str,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None

        expected_y = 0.5
        if "eyebrow" in name:
            expected_y = 0.25
        elif "eye" in name:
            expected_y = 0.5
        elif name == "mouth":
            expected_y = 0.72
        elif name == "nose":
            expected_y = 0.55

        best_label = None
        best_score = None
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]
            norm_dx = abs((cx / mask.shape[1]) - 0.5)
            norm_dy = abs((cy / mask.shape[0]) - expected_y)
            score = area - (norm_dx * 45.0 + norm_dy * 60.0)
            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        component_mask = (labels == best_label).astype(np.uint8)
        box = (
            int(stats[best_label, cv2.CC_STAT_LEFT]),
            int(stats[best_label, cv2.CC_STAT_TOP]),
            int(stats[best_label, cv2.CC_STAT_WIDTH]),
            int(stats[best_label, cv2.CC_STAT_HEIGHT]),
        )
        return component_mask, box

    def _infer_hair_parts(
        self, face_result: JsonDict, image_width: int, image_height: int
    ) -> list[DetectedPart]:
        face_bounds = face_result.get("face_bounds")
        if not isinstance(face_bounds, dict):
            return []

        face_box = self._box(face_bounds, image_width, image_height)
        hair_top = max(0, face_box.y - int(face_box.height * 0.45))
        hair_bottom = min(image_height, face_box.y + int(face_box.height * 0.45))
        side_width = max(12, int(face_box.width * 0.28))
        front_height = max(12, int(face_box.height * 0.35))
        confidence = max(
            0.25,
            float(face_result.get("confidence_summary", {}).get("average", 0.35)) - 0.1,
        )
        detector = "semantic_refine_v1"
        front_x = max(0, face_box.x - int(face_box.width * 0.08))
        front_width = min(image_width - front_x, int(face_box.width * 1.16))
        right_x = max(0, face_box.x + face_box.width - int(face_box.width * 0.15))
        right_width = min(image_width - right_x, side_width + int(face_box.width * 0.15))

        return [
            DetectedPart(
                name="hair_front",
                group="hair",
                side=None,
                bbox=BoundingBox(
                    x=front_x,
                    y=hair_top,
                    width=max(1, front_width),
                    height=max(1, min(image_height - hair_top, front_height)),
                ),
                confidence=confidence,
                detector=detector,
                polygon=[],
                attributes={"source": "hair_inference", "refined": False},
            ),
            DetectedPart(
                name="hair_side_left",
                group="hair",
                side="left",
                bbox=BoundingBox(
                    x=max(0, face_box.x - side_width),
                    y=hair_top,
                    width=max(
                        1,
                        min(
                            image_width - max(0, face_box.x - side_width),
                            side_width + int(face_box.width * 0.15),
                        ),
                    ),
                    height=max(1, hair_bottom - hair_top),
                ),
                confidence=confidence,
                detector=detector,
                polygon=[],
                attributes={"source": "hair_inference", "refined": False},
            ),
            DetectedPart(
                name="hair_side_right",
                group="hair",
                side="right",
                bbox=BoundingBox(
                    x=right_x,
                    y=hair_top,
                    width=max(1, right_width),
                    height=max(1, hair_bottom - hair_top),
                ),
                confidence=confidence,
                detector=detector,
                polygon=[],
                attributes={"source": "hair_inference", "refined": False},
            ),
            DetectedPart(
                name="hair_back",
                group="hair",
                side=None,
                bbox=BoundingBox(
                    x=max(0, face_box.x - int(face_box.width * 0.18)),
                    y=hair_top,
                    width=max(
                        1,
                        min(
                            image_width - max(0, face_box.x - int(face_box.width * 0.18)),
                            int(face_box.width * 1.36),
                        ),
                    ),
                    height=max(1, min(image_height - hair_top, int(face_box.height * 0.9))),
                ),
                confidence=max(0.2, confidence - 0.05),
                detector=detector,
                polygon=[],
                attributes={"source": "hair_inference", "refined": False},
            ),
        ]

    def _rect_polygon(self, box: BoundingBox) -> list[dict[str, int]]:
        return [
            {"x": box.x, "y": box.y},
            {"x": box.x + box.width, "y": box.y},
            {"x": box.x + box.width, "y": box.y + box.height},
            {"x": box.x, "y": box.y + box.height},
        ]

    def _dedupe_parts(self, parts: list[DetectedPart]) -> list[DetectedPart]:
        best_by_name: dict[str, DetectedPart] = {}
        for part in parts:
            existing = best_by_name.get(part.name)
            if existing is None or part.confidence > existing.confidence:
                best_by_name[part.name] = part
        return sorted(best_by_name.values(), key=lambda item: (item.group, item.name))

    def _box(self, bounds: JsonDict, image_width: int, image_height: int) -> BoundingBox:
        x = max(0, min(int(bounds.get("x", 0)), image_width - 1))
        y = max(0, min(int(bounds.get("y", 0)), image_height - 1))
        width = max(1, min(int(bounds.get("width", 1)), image_width - x))
        height = max(1, min(int(bounds.get("height", 1)), image_height - y))
        return BoundingBox(x=x, y=y, width=width, height=height)

    def _side_from_name(self, name: str) -> str | None:
        if "left" in name:
            return "left"
        if "right" in name:
            return "right"
        return None

    def _fallback_reason(self, body_result: JsonDict, face_result: JsonDict) -> str | None:
        reasons = [
            str(body_result.get("fallback_reason", "")).strip(),
            str(face_result.get("fallback_reason", "")).strip(),
        ]
        joined = "; ".join(reason for reason in reasons if reason)
        return joined or None

    def _confidence_summary(self, parts: list[DetectedPart]) -> JsonDict:
        if not parts:
            return {"count": 0, "average": 0.0, "minimum": 0.0, "maximum": 0.0}
        confidences = [part.confidence for part in parts]
        return {
            "count": len(confidences),
            "average": round(sum(confidences) / len(confidences), 3),
            "minimum": round(min(confidences), 3),
            "maximum": round(max(confidences), 3),
        }
