"""
变形器系统
实现 Live2D 的各种变形效果
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from loguru import logger


class DeformerSystem:
    """Live2D 变形器系统"""

    def __init__(self):
        self.deformers = []

    def create_warp_deformer(self, mesh: Dict, control_points: List[Dict]) -> Dict:
        """
        创建弯曲变形器

        Args:
            mesh: ArtMesh 网格
            control_points: 控制点列表

        Returns:
            弯曲变形器配置
        """
        deformer = {
            "type": "warp",
            "id": f"warp_{len(self.deformers)}",
            "mesh": mesh,
            "control_points": control_points,
            "influence": 1.0,
        }

        self.deformers.append(deformer)
        return deformer

    def create_rotation_deformer(self, mesh: Dict, pivot: Dict) -> Dict:
        """
        创建旋转变形器

        Args:
            mesh: ArtMesh 网格
            pivot: 旋转中心点

        Returns:
            旋转变形器配置
        """
        deformer = {
            "type": "rotation",
            "id": f"rotation_{len(self.deformers)}",
            "mesh": mesh,
            "pivot": pivot,
            "angle": 0,
            "influence": 1.0,
        }

        self.deformers.append(deformer)
        return deformer

    def apply_deformation(
        self, vertices: List[Dict], deformer: Dict, parameter_value: float
    ) -> List[Dict]:
        """
        应用变形到顶点

        Args:
            vertices: 原始顶点
            deformer: 变形器配置
            parameter_value: 参数值（-1 到 1）

        Returns:
            变形后的顶点
        """
        deformed = []

        if deformer["type"] == "warp":
            deformed = self._apply_warp(vertices, deformer, parameter_value)
        elif deformer["type"] == "rotation":
            deformed = self._apply_rotation(vertices, deformer, parameter_value)

        return deformed

    def _apply_warp(
        self, vertices: List[Dict], deformer: Dict, parameter_value: float
    ) -> List[Dict]:
        """应用弯曲变形"""
        control_points = deformer["control_points"]
        deformed = []

        for v in vertices:
            # 计算到控制点的影响
            displacement = {"x": 0, "y": 0}

            for cp in control_points:
                # 距离衰减
                dx = v["x"] - cp["x"]
                dy = v["y"] - cp["y"]
                distance = np.sqrt(dx**2 + dy**2)

                # 高斯衰减
                influence = np.exp(-(distance**2) / (2 * 50**2))

                # 应用位移
                displacement["x"] += cp.get("offset_x", 0) * influence * parameter_value
                displacement["y"] += cp.get("offset_y", 0) * influence * parameter_value

            deformed.append(
                {
                    "x": v["x"] + displacement["x"],
                    "y": v["y"] + displacement["y"],
                    "u": v["u"],
                    "v": v["v"],
                }
            )

        return deformed

    def _apply_rotation(
        self, vertices: List[Dict], deformer: Dict, parameter_value: float
    ) -> List[Dict]:
        """应用旋转变形"""
        pivot = deformer["pivot"]
        angle = parameter_value * 30  # 最大旋转 30 度

        # 转换为弧度
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        deformed = []

        for v in vertices:
            # 相对于旋转中心的坐标
            dx = v["x"] - pivot["x"]
            dy = v["y"] - pivot["y"]

            # 应用旋转
            new_x = pivot["x"] + dx * cos_a - dy * sin_a
            new_y = pivot["y"] + dx * sin_a + dy * cos_a

            deformed.append({"x": new_x, "y": new_y, "u": v["u"], "v": v["v"]})

        return deformed

    def create_bezier_deformer(self, mesh: Dict, curve_points: List[Dict]) -> Dict:
        """
        创建贝塞尔曲线变形器

        Args:
            mesh: ArtMesh 网格
            curve_points: 贝塞尔曲线控制点

        Returns:
            贝塞尔变形器配置
        """
        deformer = {
            "type": "bezier",
            "id": f"bezier_{len(self.deformers)}",
            "mesh": mesh,
            "curve": curve_points,
            "influence": 1.0,
        }

        self.deformers.append(deformer)
        return deformer

    def evaluate_bezier(self, t: float, points: List[Dict]) -> Tuple[float, float]:
        """
        计算贝塞尔曲线上的点

        Args:
            t: 参数 (0-1)
            points: 控制点列表

        Returns:
            (x, y) 坐标
        """
        n = len(points) - 1
        x, y = 0, 0

        for i, p in enumerate(points):
            # 伯恩斯坦多项式
            coef = self._binomial(n, i) * (t**i) * ((1 - t) ** (n - i))
            x += coef * p["x"]
            y += coef * p["y"]

        return x, y

    def _binomial(self, n: int, k: int) -> int:
        """计算二项式系数"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1

        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)

        return result
