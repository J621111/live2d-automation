"""Facial feature detection helpers for Live2D automation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from loguru import logger
from PIL import Image

JsonDict = dict[str, Any]
JsonList = list[JsonDict]


class FacialFeatureDetector:
    """Detect face regions and extract simple facial parts."""

    def __init__(self) -> None:
        self.face_cascade: Any = None
        self.eye_cascade: Any = None
        self.face_recognition: Any = None
        self.initialized = False
        self.initialization_notes: list[str] = []

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

    def _parts_from_face_bounds(
        self, x: int, y: int, w: int, h: int, confidence: float
    ) -> JsonDict:
        eye_h = int(h * 0.16)
        brow_h = max(1, int(h * 0.08))
        mouth_h = max(1, int(h * 0.2))
        parts: JsonDict = {
            "left_eye": {
                "bounds": {
                    "x": x + int(w * 0.18),
                    "y": y + int(h * 0.28),
                    "width": max(1, int(w * 0.18)),
                    "height": eye_h,
                },
                "center": (x + int(w * 0.27), y + int(h * 0.36)),
                "confidence": confidence,
            },
            "right_eye": {
                "bounds": {
                    "x": x + int(w * 0.64),
                    "y": y + int(h * 0.28),
                    "width": max(1, int(w * 0.18)),
                    "height": eye_h,
                },
                "center": (x + int(w * 0.73), y + int(h * 0.36)),
                "confidence": confidence,
            },
            "mouth": {
                "bounds": {
                    "x": x + int(w * 0.28),
                    "y": y + int(h * 0.64),
                    "width": max(1, int(w * 0.44)),
                    "height": mouth_h,
                },
                "center": (x + w // 2, y + int(h * 0.74)),
                "confidence": max(0.0, confidence - 0.05),
            },
            "nose": {
                "bounds": {
                    "x": x + int(w * 0.4),
                    "y": y + int(h * 0.46),
                    "width": max(1, int(w * 0.2)),
                    "height": max(1, int(h * 0.12)),
                },
                "center": (x + w // 2, y + int(h * 0.52)),
                "confidence": max(0.0, confidence - 0.08),
            },
            "left_eyebrow": {
                "bounds": {
                    "x": x + int(w * 0.16),
                    "y": y + int(h * 0.18),
                    "width": max(1, int(w * 0.22)),
                    "height": brow_h,
                },
                "confidence": max(0.0, confidence - 0.1),
            },
            "right_eyebrow": {
                "bounds": {
                    "x": x + int(w * 0.62),
                    "y": y + int(h * 0.18),
                    "width": max(1, int(w * 0.22)),
                    "height": brow_h,
                },
                "confidence": max(0.0, confidence - 0.1),
            },
        }
        return parts

    async def initialize(self) -> None:
        if self.initialized:
            return

        logger.info("Initializing facial feature detector...")
        try:
            cv_data = getattr(cv2, "data", None)
            if cv_data is None:
                raise RuntimeError("OpenCV haarcascade data directory is unavailable")
            face_cascade = cv2.CascadeClassifier(
                cv_data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            eye_cascade = cv2.CascadeClassifier(cv_data.haarcascades + "haarcascade_eye.xml")
            if face_cascade.empty() or eye_cascade.empty():
                raise RuntimeError("OpenCV cascade files could not be loaded")
            self.face_cascade = face_cascade
            self.eye_cascade = eye_cascade
            logger.info("OpenCV cascades loaded")
        except Exception as exc:
            note = f"opencv cascade unavailable: {exc}"
            self.initialization_notes.append(note)
            logger.warning(note)

        try:
            import face_recognition

            self.face_recognition = face_recognition
            logger.info("face_recognition backend available")
        except Exception as exc:
            note = f"face_recognition unavailable: {exc}"
            self.initialization_notes.append(note)
            logger.info(note)

        self.initialized = True

    async def detect_features(self, image_path: str) -> JsonDict:
        await self.initialize()

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        features: JsonDict = {
            "has_face": False,
            "face_bounds": None,
            "parts": {},
            "detector_used": "heuristic",
            "fallback_reason": None,
        }
        parts: JsonDict = features["parts"]

        if self.face_cascade is not None and self.eye_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                features["has_face"] = True
                features["face_bounds"] = {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                }
                features["detector_used"] = "opencv_haar"

                roi_gray = gray[y : y + h, x : x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                left_eyes: list[tuple[int, int, int, int]] = []
                right_eyes: list[tuple[int, int, int, int]] = []
                for ex, ey, ew, eh in eyes:
                    if ex < w / 2:
                        left_eyes.append((ex, ey, ew, eh))
                    else:
                        right_eyes.append((ex, ey, ew, eh))

                if left_eyes:
                    ex, ey, ew, eh = max(left_eyes, key=lambda item: item[2] * item[3])
                    parts["left_eye"] = {
                        "bounds": {"x": x + ex, "y": y + ey, "width": ew, "height": eh},
                        "center": (x + ex + ew // 2, y + ey + eh // 2),
                        "confidence": 0.8,
                    }
                if right_eyes:
                    ex, ey, ew, eh = max(right_eyes, key=lambda item: item[2] * item[3])
                    parts["right_eye"] = {
                        "bounds": {"x": x + ex, "y": y + ey, "width": ew, "height": eh},
                        "center": (x + ex + ew // 2, y + ey + eh // 2),
                        "confidence": 0.8,
                    }
                parts.update(self._parts_from_face_bounds(int(x), int(y), int(w), int(h), 0.7))

        if not features["has_face"] and self.face_recognition is not None:
            try:
                locations = self.face_recognition.face_locations(
                    self.face_recognition.load_image_file(image_path)
                )
                if locations:
                    top, right, bottom, left = max(
                        locations,
                        key=lambda item: (item[2] - item[0]) * (item[1] - item[3]),
                    )
                    x = int(left)
                    y = int(top)
                    w = int(right - left)
                    h = int(bottom - top)
                    features["has_face"] = True
                    features["face_bounds"] = {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                    }
                    features["detector_used"] = "face_recognition"
                    parts.update(self._parts_from_face_bounds(x, y, w, h, 0.65))
            except Exception as exc:
                features["fallback_reason"] = f"face_recognition failed: {exc}"
                logger.warning(features["fallback_reason"])

        if not features["has_face"]:
            logger.info("Using heuristic facial detection for anime character...")
            features = self._detect_anime_face(width, height)
            features["detector_used"] = "heuristic"
            notes = "; ".join(self.initialization_notes)
            features["fallback_reason"] = (
                features.get("fallback_reason")
                or notes
                or ("no face detected by available backends")
            )

        features["confidence_summary"] = self._confidence_summary(features.get("parts", {}))
        return features

    def _detect_anime_face(self, width: int, height: int) -> JsonDict:
        face_x = int(width * 0.22)
        face_y = int(height * 0.02)
        face_w = int(width * 0.56)
        face_h = int(height * 0.48)
        eye_y = face_y + int(face_h * 0.35)
        eye_h = int(face_h * 0.12)
        brow_y = face_y + int(face_h * 0.25)
        brow_h = int(face_h * 0.08)

        return {
            "has_face": True,
            "face_bounds": {
                "x": face_x,
                "y": face_y,
                "width": face_w,
                "height": face_h,
            },
            "parts": {
                "left_eye": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.15),
                        "y": eye_y,
                        "width": int(face_w * 0.18),
                        "height": eye_h,
                    },
                    "center": (face_x + int(face_w * 0.24), eye_y + eye_h // 2),
                    "confidence": 0.4,
                },
                "right_eye": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.67),
                        "y": eye_y,
                        "width": int(face_w * 0.18),
                        "height": eye_h,
                    },
                    "center": (face_x + int(face_w * 0.76), eye_y + eye_h // 2),
                    "confidence": 0.4,
                },
                "left_eye_highlight": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.18),
                        "y": eye_y + int(eye_h * 0.2),
                        "width": int(face_w * 0.06),
                        "height": int(eye_h * 0.3),
                    },
                    "confidence": 0.32,
                },
                "right_eye_highlight": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.70),
                        "y": eye_y + int(eye_h * 0.2),
                        "width": int(face_w * 0.06),
                        "height": int(eye_h * 0.3),
                    },
                    "confidence": 0.32,
                },
                "mouth": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.35),
                        "y": face_y + int(face_h * 0.65),
                        "width": int(face_w * 0.3),
                        "height": int(face_h * 0.12),
                    },
                    "center": (
                        face_x + face_w // 2,
                        face_y + int(face_h * 0.65) + int(face_h * 0.12) // 2,
                    ),
                    "confidence": 0.35,
                },
                "nose": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.42),
                        "y": face_y + int(face_h * 0.45),
                        "width": int(face_w * 0.16),
                        "height": int(face_h * 0.1),
                    },
                    "confidence": 0.33,
                },
                "left_eyebrow": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.12),
                        "y": brow_y,
                        "width": int(face_w * 0.22),
                        "height": brow_h,
                    },
                    "confidence": 0.3,
                },
                "right_eyebrow": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.66),
                        "y": brow_y,
                        "width": int(face_w * 0.22),
                        "height": brow_h,
                    },
                    "confidence": 0.3,
                },
                "left_blush": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.10),
                        "y": face_y + int(face_h * 0.45),
                        "width": int(face_w * 0.15),
                        "height": int(face_h * 0.1),
                    },
                    "confidence": 0.25,
                },
                "right_blush": {
                    "bounds": {
                        "x": face_x + int(face_w * 0.75),
                        "y": face_y + int(face_h * 0.45),
                        "width": int(face_w * 0.15),
                        "height": int(face_h * 0.1),
                    },
                    "confidence": 0.25,
                },
            },
        }

    async def extract_face_parts(
        self,
        image_path: str,
        output_dir: str,
        features: JsonDict | None = None,
    ) -> JsonList:
        if features is None:
            features = await self.detect_features(image_path)
        if not features.get("has_face"):
            logger.warning("No face detected")
            return []

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image = Image.open(image_path).convert("RGBA")
        parts = dict(features.get("parts", {}))
        layers: JsonList = []

        for part_name, part_data in parts.items():
            if not isinstance(part_data, dict):
                continue
            bounds = part_data.get("bounds")
            if not isinstance(bounds, dict):
                continue

            x = max(0, int(bounds["x"]))
            y = max(0, int(bounds["y"]))
            w = min(int(bounds["width"]), image.width - x)
            h = min(int(bounds["height"]), image.height - y)
            if w <= 0 or h <= 0:
                continue

            output_path = Path(output_dir) / f"face_{part_name}.png"
            image.crop((x, y, x + w, y + h)).save(output_path, "PNG")
            layers.append(
                {
                    "name": part_name,
                    "path": str(output_path),
                    "bounds": bounds,
                    "z_order": self._get_z_order(part_name),
                    "confidence": float(part_data.get("confidence", 0.0)),
                }
            )

        logger.info(f"Extracted {len(layers)} face parts")
        return layers

    def _get_z_order(self, part_name: str) -> int:
        order = {
            "left_eyebrow": 101,
            "right_eyebrow": 102,
            "left_eye": 103,
            "right_eye": 104,
            "left_eye_highlight": 105,
            "right_eye_highlight": 106,
            "nose": 107,
            "mouth": 108,
            "left_blush": 109,
            "right_blush": 110,
        }
        return order.get(part_name, 110)
