"""Image analysis helpers for Live2D automation."""

from __future__ import annotations

from typing import Any

import cv2
from loguru import logger

JsonDict = dict[str, Any]


class ImageProcessor:
    """Analyze character images and emit a normalized parts schema."""

    def _load_pose_backend(self) -> tuple[Any | None, str, str | None]:
        try:
            import mediapipe as mp
        except Exception as exc:  # pragma: no cover - import error varies by env
            return None, "heuristic", f"mediapipe import failed: {exc}"

        solutions = getattr(mp, "solutions", None)
        if solutions is not None:
            pose_module = getattr(solutions, "pose", None)
            pose_cls = getattr(pose_module, "Pose", None)
            if pose_cls is not None:
                return pose_cls, "mediapipe.solutions.pose", None

        pose_cls = getattr(mp, "Pose", None)
        if pose_cls is not None:
            return pose_cls, "mediapipe.Pose", None

        return None, "heuristic", "mediapipe pose API unavailable in installed package"

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

    async def analyze(self, image_path: str) -> JsonDict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        height, width = image.shape[:2]
        body_parts = self._detect_body_parts_simple(width, height)
        detector = "heuristic"
        fallback_reason: str | None = None
        pose_landmarks: list[JsonDict] | None = None

        pose_backend, backend_name, backend_reason = self._load_pose_backend()
        if pose_backend is not None:
            detector = backend_name
            try:
                logger.info(f"Using pose backend: {backend_name}")
                pose = pose_backend(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                )
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if getattr(results, "pose_landmarks", None):
                    pose_landmarks = []
                    for index, landmark in enumerate(results.pose_landmarks.landmark):
                        pose_landmarks.append(
                            {
                                "index": index,
                                "x": float(landmark.x),
                                "y": float(landmark.y),
                                "z": float(landmark.z),
                                "visibility": float(getattr(landmark, "visibility", 0.0)),
                            }
                        )
                    body_parts = self._merge_body_parts(
                        body_parts,
                        self._extract_body_from_landmarks(pose_landmarks, width, height),
                    )
                else:
                    detector = "heuristic"
                    fallback_reason = f"{backend_name} returned no pose landmarks"
                close = getattr(pose, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                detector = "heuristic"
                fallback_reason = f"{backend_name} failed: {exc}"
                logger.warning(f"Pose backend failed, using fallback: {exc}")
        else:
            fallback_reason = backend_reason
            if backend_reason:
                logger.warning(f"Pose backend unavailable, using fallback: {backend_reason}")

        parts = self._flatten_parts(body_parts)
        confidence_summary = self._confidence_summary(parts)
        return {
            "schema_version": 2,
            "detector": detector,
            "detector_used": detector,
            "fallback_reason": fallback_reason,
            "confidence_summary": confidence_summary,
            "image_info": {"width": width, "height": height, "path": image_path},
            "body_parts": body_parts,
            "parts": parts,
            "face_landmarks": None,
            "pose_landmarks": pose_landmarks,
        }

    def _detect_body_parts_simple(self, width: int, height: int) -> JsonDict:
        parts: JsonDict = {
            "head": {
                "center": (width * 0.5, height * 0.15),
                "bounds": {
                    "x": width * 0.35,
                    "y": height * 0.05,
                    "width": width * 0.3,
                    "height": height * 0.25,
                },
                "confidence": 0.35,
            },
            "torso": {
                "center": (width * 0.5, height * 0.45),
                "bounds": {
                    "x": width * 0.3,
                    "y": height * 0.3,
                    "width": width * 0.4,
                    "height": height * 0.35,
                },
                "confidence": 0.35,
            },
            "left_arm": {
                "center": (width * 0.2, height * 0.4),
                "bounds": {
                    "x": width * 0.05,
                    "y": height * 0.3,
                    "width": width * 0.2,
                    "height": height * 0.4,
                },
                "confidence": 0.3,
            },
            "right_arm": {
                "center": (width * 0.8, height * 0.4),
                "bounds": {
                    "x": width * 0.75,
                    "y": height * 0.3,
                    "width": width * 0.2,
                    "height": height * 0.4,
                },
                "confidence": 0.3,
            },
            "left_leg": {
                "center": (width * 0.4, height * 0.75),
                "bounds": {
                    "x": width * 0.3,
                    "y": height * 0.65,
                    "width": width * 0.15,
                    "height": height * 0.3,
                },
                "confidence": 0.3,
            },
            "right_leg": {
                "center": (width * 0.6, height * 0.75),
                "bounds": {
                    "x": width * 0.55,
                    "y": height * 0.65,
                    "width": width * 0.15,
                    "height": height * 0.3,
                },
                "confidence": 0.3,
            },
        }

        face_center_x = width * 0.5
        face_center_y = height * 0.15
        face_width = width * 0.2
        face_height = height * 0.15
        parts["face_parts"] = {
            "left_eye": {
                "center": (
                    face_center_x - face_width * 0.2,
                    face_center_y - face_height * 0.1,
                ),
                "bounds": {
                    "x": face_center_x - face_width * 0.3,
                    "y": face_center_y - face_height * 0.15,
                    "width": face_width * 0.2,
                    "height": face_height * 0.1,
                },
                "confidence": 0.3,
            },
            "right_eye": {
                "center": (
                    face_center_x + face_width * 0.2,
                    face_center_y - face_height * 0.1,
                ),
                "bounds": {
                    "x": face_center_x + face_width * 0.1,
                    "y": face_center_y - face_height * 0.15,
                    "width": face_width * 0.2,
                    "height": face_height * 0.1,
                },
                "confidence": 0.3,
            },
            "mouth": {
                "center": (face_center_x, face_center_y + face_height * 0.15),
                "bounds": {
                    "x": face_center_x - face_width * 0.15,
                    "y": face_center_y + face_height * 0.1,
                    "width": face_width * 0.3,
                    "height": face_height * 0.1,
                },
                "confidence": 0.25,
            },
            "nose": {
                "center": (face_center_x, face_center_y),
                "bounds": {
                    "x": face_center_x - face_width * 0.05,
                    "y": face_center_y - face_height * 0.05,
                    "width": face_width * 0.1,
                    "height": face_height * 0.1,
                },
                "confidence": 0.25,
            },
        }
        return parts

    def _extract_body_from_landmarks(
        self,
        landmarks: list[JsonDict],
        width: int,
        height: int,
    ) -> JsonDict:
        parts: JsonDict = {}
        if len(landmarks) < 33:
            return parts

        head = landmarks[0]
        head_confidence = max(0.0, min(1.0, float(head.get("visibility", 0.75))))
        parts["head"] = {
            "center": (head["x"] * width, head["y"] * height),
            "bounds": {
                "x": (head["x"] - 0.1) * width,
                "y": (head["y"] - 0.15) * height,
                "width": width * 0.2,
                "height": height * 0.25,
            },
            "confidence": head_confidence,
        }

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        torso_confidence = max(
            0.0,
            min(
                1.0,
                (
                    float(left_shoulder.get("visibility", 0.7))
                    + float(right_shoulder.get("visibility", 0.7))
                )
                / 2,
            ),
        )
        center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2 * width
        center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2 * height
        parts["torso"] = {
            "center": (center_x, center_y),
            "bounds": {
                "x": left_shoulder["x"] * width - width * 0.1,
                "y": center_y - height * 0.1,
                "width": (right_shoulder["x"] - left_shoulder["x"]) * width + width * 0.2,
                "height": height * 0.4,
            },
            "confidence": torso_confidence,
        }
        return parts

    def _merge_body_parts(self, base_parts: JsonDict, detected_parts: JsonDict) -> JsonDict:
        merged = dict(base_parts)
        merged.update(detected_parts)
        return merged

    def _flatten_parts(self, body_parts: JsonDict) -> JsonDict:
        flat_parts: JsonDict = {}
        for part_name, part_data in body_parts.items():
            if not isinstance(part_data, dict):
                continue
            if "bounds" in part_data:
                flat_parts[str(part_name)] = part_data
                continue
            for nested_name, nested_data in part_data.items():
                if isinstance(nested_data, dict) and "bounds" in nested_data:
                    flat_parts[str(nested_name)] = nested_data
        return flat_parts
