"""
分层生成器 v2
使用精确分割生成 Live2D 所需图层文件
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from loguru import logger
import cv2
import io


class LayerGenerator:
    """Live2D 图层生成器 v2 - 精确版"""

    def __init__(self):
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
        self.segmenter = None

    async def initialize(self):
        """初始化分割器"""
        if self.segmenter is None:
            try:
                from mcp_server.tools.segmentation import PreciseSegmenter

                self.segmenter = PreciseSegmenter()
                await self.segmenter.initialize()
            except ImportError as e:
                logger.warning(f"Could not load precise segmenter: {e}")
                self.segmenter = None

    async def generate(
        self, image_path: str, segments: Dict = None, output_dir: str = None
    ) -> List[Dict]:
        """
        生成分层文件

        Args:
            image_path: 原始图像路径
            segments: 分割信息（可选）
            output_dir: 输出目录

        Returns:
            生成的图层列表
        """
        await self.initialize()

        output_path = Path(output_dir) if output_dir else Path("output")
        output_path.mkdir(parents=True, exist_ok=True)

        # 加载原始图像
        image = Image.open(image_path).convert("RGBA")

        layers = []

        # 如果没有传入 segments，使用精确分割
        if segments is None or not segments.get("parts"):
            logger.info("Using precise segmentation...")
            segments = await self._precise_segment(image_path)

        # 获取部位
        parts = segments.get("parts", {})

        # 生成各部位图层
        for part_name, part_data in parts.items():
            if part_data and "bounds" in part_data:
                layer = await self._create_layer_from_bounds(
                    image,
                    part_data["bounds"],
                    part_name,
                    output_path,
                    part_data.get("mask"),
                )
                if layer:
                    layers.append(layer)

        # 排序图层
        layers = self._sort_layers(layers)

        logger.info(f"Generated {len(layers)} layers")
        return layers

    async def _precise_segment(self, image_path: str) -> Dict:
        """使用精确分割"""
        result = await self.segmenter.segment_character(image_path)
        return result

    async def _create_layer_from_bounds(
        self,
        image: Image.Image,
        bounds: Dict,
        part_name: str,
        output_dir: Path,
        mask: np.ndarray = None,
    ) -> Dict:
        """从边界框创建图层"""
        x = int(bounds.get("x", 0))
        y = int(bounds.get("y", 0))
        w = int(bounds.get("width", image.width))
        h = int(bounds.get("height", image.height))

        # 确保边界有效
        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        w = min(w, image.width - x)
        h = min(h, image.height - y)

        if w <= 0 or h <= 0:
            return None

        # 提取部位图像
        part_img = image.crop((x, y, x + w, y + h))

        # 应用掩码
        if mask is not None:
            try:
                # 调整掩码大小
                mask_resized = cv2.resize(
                    mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR
                )

                # 转换为 PIL
                mask_pil = Image.fromarray(mask_resized, mode="L")

                # 应用掩码
                part_img.putalpha(mask_pil)
            except Exception as e:
                logger.warning(f"Mask application failed for {part_name}: {e}")

        # 保存图层
        layer_filename = f"layer_{part_name}.png"
        layer_path = output_dir / layer_filename

        # 确保是 RGBA 模式
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
        }

    def _get_z_order(self, part_name: str) -> int:
        """获取图层的 Z 顺序"""
        order_map = {name: i for i, name in enumerate(self.layer_order)}

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

        mapped_name = name_mapping.get(part_name, part_name)
        return order_map.get(mapped_name, 50)

    def _sort_layers(self, layers: List[Dict]) -> List[Dict]:
        """按 Z 顺序排序图层"""
        return sorted(layers, key=lambda x: x.get("z_order", 50))

    async def generate_with_face_details(
        self, image_path: str, face_landmarks: Dict = None
    ) -> List[Dict]:
        """生成包含面部细节的图层"""
        layers = await self.generate(image_path)

        if face_landmarks is None:
            return layers

        # 添加面部细节图层
        image = Image.open(image_path).convert("RGBA")

        face_parts = {
            "left_eye": face_landmarks.get("left_eye", []),
            "right_eye": face_landmarks.get("right_eye", []),
            "mouth": face_landmarks.get("mouth", []),
            "nose": face_landmarks.get("nose", []),
        }

        for part_name, points in face_parts.items():
            if not points:
                continue

            # 计算边界
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]

            bounds = {
                "x": min(xs) - 10,
                "y": min(ys) - 10,
                "width": max(xs) - min(xs) + 20,
                "height": max(ys) - min(ys) + 20,
            }

            layer = await self._create_layer_from_bounds(
                image, bounds, part_name, Path("output"), None
            )
            if layer:
                layers.append(layer)

        return layers
