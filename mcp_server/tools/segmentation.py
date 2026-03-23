"""Precise segmentation helpers for Live2D automation."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from loguru import logger
from PIL import Image

JsonDict = dict[str, Any]


class PreciseSegmenter:
    """High-level character segmentation with model-first fallback behavior."""

    def __init__(self) -> None:
        self.rembg_model: Any = None
        self.segmenter_name = "fallback"
        self.fallback_reason: str | None = None

    def _load_rembg_backend(self) -> tuple[Any | None, str, str | None]:
        try:
            from rembg import remove
        except Exception as exc:  # pragma: no cover - import error varies by env
            return None, "fallback", f"rembg import failed: {exc}"
        return remove, "rembg", None

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

    async def initialize(self) -> None:
        if self.rembg_model is not None:
            return

        logger.info("Initializing precise segmentation model...")
        model, segmenter_name, fallback_reason = self._load_rembg_backend()
        self.rembg_model = model
        self.segmenter_name = segmenter_name
        self.fallback_reason = fallback_reason
        if model is not None:
            logger.info("rembg model loaded successfully")
        elif fallback_reason is not None:
            logger.warning(fallback_reason)

    async def segment_character(self, image_path: str) -> JsonDict:
        await self.initialize()

        image = Image.open(image_path).convert("RGBA")
        img_array = np.array(image)
        full_mask = await self._segment_full_body(image)
        parts = await self._segment_body_parts(image, full_mask)

        return {
            "schema_version": 2,
            "segmenter": self.segmenter_name,
            "detector_used": self.segmenter_name,
            "fallback_reason": self.fallback_reason,
            "confidence_summary": self._confidence_summary(parts),
            "original": img_array,
            "full_body_mask": full_mask,
            "parts": parts,
        }

    async def _segment_full_body(self, image: Image.Image) -> np.ndarray:
        try:
            if self.rembg_model:
                output = self.rembg_model(image)
                mask = np.array(output)[:, :, 3]
                self.fallback_reason = None
                return cast(np.ndarray, mask > 10)
        except Exception as exc:
            self.segmenter_name = "fallback"
            self.fallback_reason = f"rembg segmentation failed: {exc}"
            logger.warning(self.fallback_reason)

        return self._fallback_segment(image)

    def _fallback_segment(self, image: Image.Image) -> np.ndarray:
        img_array = np.array(image.convert("RGBA"))
        rgb = img_array[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel: np.ndarray = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        from scipy import ndimage

        return cast(np.ndarray, ndimage.binary_fill_holes(thresh > 0))

    async def _segment_body_parts(self, image: Image.Image, full_mask: np.ndarray) -> JsonDict:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        base_confidence = 0.75 if self.segmenter_name == "rembg" else 0.4

        def part(bounds: dict[str, int], mask_slice: np.ndarray, confidence: float) -> JsonDict:
            return {
                "bounds": bounds,
                "mask": mask_slice,
                "confidence": round(confidence, 3),
            }

        head_top = int(height * 0.02)
        head_bottom = int(height * 0.50)
        head_left = int(width * 0.22)
        head_right = int(width * 0.78)
        body_top = int(height * 0.40)
        body_bottom = int(height * 0.75)
        left_arm_left = int(width * 0.02)
        left_arm_right = int(width * 0.28)
        right_arm_left = int(width * 0.72)
        right_arm_right = int(width * 0.98)
        legs_top = int(height * 0.72)

        return {
            "head": part(
                {
                    "x": head_left,
                    "y": head_top,
                    "width": head_right - head_left,
                    "height": head_bottom - head_top,
                },
                full_mask[head_top:head_bottom, head_left:head_right],
                base_confidence,
            ),
            "torso": part(
                {
                    "x": int(width * 0.28),
                    "y": body_top,
                    "width": int(width * 0.44),
                    "height": body_bottom - body_top,
                },
                full_mask[body_top:body_bottom, int(width * 0.28) : int(width * 0.72)],
                base_confidence,
            ),
            "left_arm": part(
                {
                    "x": left_arm_left,
                    "y": int(height * 0.38),
                    "width": left_arm_right - left_arm_left,
                    "height": int(height * 0.40),
                },
                full_mask[int(height * 0.38) : int(height * 0.78), left_arm_left:left_arm_right],
                base_confidence - 0.05,
            ),
            "right_arm": part(
                {
                    "x": right_arm_left,
                    "y": int(height * 0.38),
                    "width": right_arm_right - right_arm_left,
                    "height": int(height * 0.40),
                },
                full_mask[int(height * 0.38) : int(height * 0.78), right_arm_left:right_arm_right],
                base_confidence - 0.05,
            ),
            "left_leg": part(
                {
                    "x": int(width * 0.32),
                    "y": legs_top,
                    "width": int(width * 0.15),
                    "height": height - legs_top,
                },
                full_mask[legs_top:height, int(width * 0.32) : int(width * 0.47)],
                base_confidence - 0.1,
            ),
            "right_leg": part(
                {
                    "x": int(width * 0.53),
                    "y": legs_top,
                    "width": int(width * 0.15),
                    "height": height - legs_top,
                },
                full_mask[legs_top:height, int(width * 0.53) : int(width * 0.68)],
                base_confidence - 0.1,
            ),
        }

    async def extract_part_image(
        self,
        image: Image.Image,
        part_bounds: dict[str, Any],
        mask: np.ndarray | None = None,
    ) -> Image.Image:
        x = int(part_bounds.get("x", 0))
        y = int(part_bounds.get("y", 0))
        w = int(part_bounds.get("width", image.width))
        h = int(part_bounds.get("height", image.height))

        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        w = min(w, image.width - x)
        h = min(h, image.height - y)

        part_img = image.crop((x, y, x + w, y + h))
        if mask is not None:
            mask_h, mask_w = mask.shape[:2]
            if mask_h > 0 and mask_w > 0:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)
                mask_array = np.array(mask_img)

                part_array = np.array(part_img)
                if len(part_array.shape) == 3 and part_array.shape[2] == 4:
                    part_array[:, :, 3] = mask_array
                    part_img = Image.fromarray(part_array)
                else:
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    if len(part_array.shape) == 3:
                        rgba[:, :, :3] = part_array
                    else:
                        rgba[:, :, :3] = np.stack([part_array] * 3, axis=-1)
                    rgba[:, :, 3] = mask_array
                    part_img = Image.fromarray(rgba)

        return part_img

    async def refine_mask(self, mask: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        del image
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter

        kernel: np.ndarray = np.ones((3, 3), np.uint8)
        refined = ndimage.binary_closing(mask, structure=kernel)
        refined = ndimage.binary_opening(refined, structure=kernel)
        mask_float = gaussian_filter(refined.astype(float), sigma=0.5)
        return cast(np.ndarray, mask_float > 0.3)


SegmentEngine = PreciseSegmenter
