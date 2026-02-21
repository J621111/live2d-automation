"""
精确分割引擎
使用 rembg + 自研算法进行高精度分割
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import cv2
from loguru import logger


class PreciseSegmenter:
    """高精度分割器"""

    def __init__(self):
        self.rembg_model = None
        self.use_gpu = True

    async def initialize(self):
        """初始化分割模型"""
        if self.rembg_model is not None:
            return

        logger.info("Initializing precise segmentation model...")

        try:
            from rembg import remove

            self.rembg_model = remove
            logger.info("rembg model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load rembg: {e}")
            self.rembg_model = None

    async def segment_character(self, image_path: str) -> Dict[str, Any]:
        """
        精确分割人物

        Args:
            image_path: 输入图像路径

        Returns:
            分割结果，包含各个部位的掩码
        """
        await self.initialize()

        # 加载图像
        image = Image.open(image_path).convert("RGBA")
        img_array = np.array(image)

        results = {"original": img_array, "full_body_mask": None, "parts": {}}

        # 1. 分割完整人物
        logger.info("Segmenting full character...")
        full_mask = await self._segment_full_body(image)
        results["full_body_mask"] = full_mask

        # 2. 基于颜色/位置分割各部位
        logger.info("Segmenting body parts...")
        parts = await self._segment_body_parts(image, full_mask)
        results["parts"] = parts

        return results

    async def _segment_full_body(self, image: Image.Image) -> np.ndarray:
        """分割完整人物"""
        try:
            if self.rembg_model:
                # 使用 rembg 移除背景
                output = self.rembg_model(image)
                mask = np.array(output)[:, :, 3]
                # 二值化
                return mask > 10
        except Exception as e:
            logger.warning(f"rembg segmentation failed: {e}")

        # 备选方案：基于颜色分割
        return self._fallback_segment(image)

    def _fallback_segment(self, image: Image.Image) -> np.ndarray:
        """基于颜色的备选分割方案"""
        img_array = np.array(image.convert("RGBA"))

        # 转换到不同颜色空间
        rgb = img_array[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 使用自适应阈值
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 填充孔洞
        from scipy import ndimage

        filled = ndimage.binary_fill_holes(thresh > 0)

        return filled

    async def _segment_body_parts(
        self, image: Image.Image, full_mask: np.ndarray
    ) -> Dict[str, Any]:
        """基于位置和颜色分割身体部位"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 定义身体区域（基于图像比例 - 针对动漫角色优化）
        parts = {}

        # 头部区域（上半部分）- 动漫角色头较大
        head_top = int(height * 0.02)
        head_bottom = int(height * 0.50)
        head_left = int(width * 0.22)
        head_right = int(width * 0.78)

        # 身体区域（中间部分）
        body_top = int(height * 0.40)
        body_bottom = int(height * 0.75)

        # 手臂区域
        left_arm_left = int(width * 0.02)
        left_arm_right = int(width * 0.28)
        right_arm_left = int(width * 0.72)
        right_arm_right = int(width * 0.98)

        # 腿部区域
        legs_top = int(height * 0.72)

        # 为每个区域创建掩码
        parts["head"] = {
            "bounds": {
                "x": head_left,
                "y": head_top,
                "width": head_right - head_left,
                "height": head_bottom - head_top,
            },
            "mask": full_mask[head_top:head_bottom, head_left:head_right],
        }

        parts["torso"] = {
            "bounds": {
                "x": int(width * 0.28),
                "y": body_top,
                "width": int(width * 0.44),
                "height": body_bottom - body_top,
            },
            "mask": full_mask[
                body_top:body_bottom, int(width * 0.28) : int(width * 0.72)
            ],
        }

        parts["left_arm"] = {
            "bounds": {
                "x": left_arm_left,
                "y": int(height * 0.38),
                "width": left_arm_right - left_arm_left,
                "height": int(height * 0.40),
            },
            "mask": full_mask[
                int(height * 0.38) : int(height * 0.78), left_arm_left:left_arm_right
            ],
        }

        parts["right_arm"] = {
            "bounds": {
                "x": right_arm_left,
                "y": int(height * 0.38),
                "width": right_arm_right - right_arm_left,
                "height": int(height * 0.40),
            },
            "mask": full_mask[
                int(height * 0.38) : int(height * 0.78), right_arm_left:right_arm_right
            ],
        }

        parts["left_leg"] = {
            "bounds": {
                "x": int(width * 0.32),
                "y": legs_top,
                "width": int(width * 0.15),
                "height": height - legs_top,
            },
            "mask": full_mask[legs_top:height, int(width * 0.32) : int(width * 0.47)],
        }

        parts["right_leg"] = {
            "bounds": {
                "x": int(width * 0.53),
                "y": legs_top,
                "width": int(width * 0.15),
                "height": height - legs_top,
            },
            "mask": full_mask[legs_top:height, int(width * 0.53) : int(width * 0.68)],
        }

        return parts

    async def extract_part_image(
        self, image: Image.Image, part_bounds: Dict, mask: Optional[np.ndarray] = None
    ) -> Image.Image:
        """从原图中提取特定部位"""
        x = int(part_bounds.get("x", 0))
        y = int(part_bounds.get("y", 0))
        w = int(part_bounds.get("width", image.width))
        h = int(part_bounds.get("height", image.height))

        # 确保边界有效
        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        w = min(w, image.width - x)
        h = min(h, image.height - y)

        # 提取区域
        part_img = image.crop((x, y, x + w, y + h))

        # 应用掩码（如果有）
        if mask is not None:
            mask_h, mask_w = mask.shape[:2]
            if mask_h > 0 and mask_w > 0:
                # 调整掩码大小以匹配图像
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((w, h), Image.LANCZOS)
                mask_array = np.array(mask_img)

                # 应用掩码
                part_array = np.array(part_img)
                if len(part_array.shape) == 3 and part_array.shape[2] == 4:
                    part_array[:, :, 3] = mask_array
                else:
                    # 添加 alpha 通道
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    if len(part_array.shape) == 3:
                        rgba[:, :, :3] = part_array
                    else:
                        gray = np.stack([part_array] * 3, axis=-1)
                        rgba[:, :, :3] = gray
                    rgba[:, :, 3] = mask_array
                    part_img = Image.fromarray(rgba)

        return part_img

    async def refine_mask(
        self, mask: np.ndarray, image: np.ndarray = None
    ) -> np.ndarray:
        """优化掩码边缘"""
        from scipy import ndimage

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)

        # 闭运算 - 填充小孔洞
        mask = ndimage.binary_closing(mask, structure=kernel)

        # 开运算 - 去除小噪点
        mask = ndimage.binary_opening(mask, structure=kernel)

        # 高斯模糊使边缘更平滑
        from scipy.ndimage import gaussian_filter

        mask_float = gaussian_filter(mask.astype(float), sigma=0.5)
        mask = mask_float > 0.3

        return mask


# 导出主类
SegmentEngine = PreciseSegmenter
