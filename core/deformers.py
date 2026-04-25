"""
变形器系统
实现 Live2D 的各种变形效果
"""

from __future__ import annotations

from typing import Any

import numpy as np

JsonDict = dict[str, Any]


class DeformerSystem:
    """Live2D 变形器系统"""

    def __init__(self) -> None:
        self.deformers: list[JsonDict] = []

    def create_warp_deformer(self, mesh: JsonDict, control_points: list[JsonDict]) -> JsonDict:
        deformer = {
            "type": "warp",
            "id": f"warp_{len(self.deformers)}",
            "mesh": mesh,
            "control_points": control_points,
            "influence": 1.0,
        }
        self.deformers.append(deformer)
        return deformer

    def create_rotation_deformer(self, mesh: JsonDict, pivot: JsonDict) -> JsonDict:
        deformer = {
            "type": "rotation",
            "id": f"rotation_{len(self.deformers)}",
            "mesh": mesh,
            "pivot": pivot,
            "angle": 0.0,
            "influence": 1.0,
        }
        self.deformers.append(deformer)
        return deformer

    def create_bezier_deformer(self, mesh: JsonDict, curve_points: list[JsonDict]) -> JsonDict:
        deformer = {
            "type": "bezier",
            "id": f"bezier_{len(self.deformers)}",
            "mesh": mesh,
            "curve": curve_points,
            "influence": 1.0,
        }
        self.deformers.append(deformer)
        return deformer

    def apply_deformation(
        self, vertices: list[JsonDict], deformer: JsonDict, parameter_value: float
    ) -> list[JsonDict]:
        deformer_type = deformer.get("type")
        if deformer_type == "warp":
            return self._apply_warp(vertices, deformer, parameter_value)
        if deformer_type == "rotation":
            return self._apply_rotation(vertices, deformer, parameter_value)
        return list(vertices)

    def _apply_warp(
        self, vertices: list[JsonDict], deformer: JsonDict, parameter_value: float
    ) -> list[JsonDict]:
        control_points = list(deformer.get("control_points", []))
        deformed: list[JsonDict] = []
        for vertex in vertices:
            displacement_x = 0.0
            displacement_y = 0.0
            for control_point in control_points:
                dx = float(vertex["x"]) - float(control_point["x"])
                dy = float(vertex["y"]) - float(control_point["y"])
                distance = np.sqrt(dx**2 + dy**2)
                influence = np.exp(-(distance**2) / (2 * 50**2))
                displacement_x += float(control_point.get("offset_x", 0.0)) * influence
                displacement_y += float(control_point.get("offset_y", 0.0)) * influence
            deformed.append(
                {
                    "x": float(vertex["x"]) + displacement_x * parameter_value,
                    "y": float(vertex["y"]) + displacement_y * parameter_value,
                    "u": vertex["u"],
                    "v": vertex["v"],
                }
            )
        return deformed

    def _apply_rotation(
        self, vertices: list[JsonDict], deformer: JsonDict, parameter_value: float
    ) -> list[JsonDict]:
        pivot = dict(deformer.get("pivot", {"x": 0.0, "y": 0.0}))
        angle_rad = np.radians(parameter_value * 30.0)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        pivot_x = float(pivot.get("x", 0.0))
        pivot_y = float(pivot.get("y", 0.0))

        deformed: list[JsonDict] = []
        for vertex in vertices:
            dx = float(vertex["x"]) - pivot_x
            dy = float(vertex["y"]) - pivot_y
            deformed.append(
                {
                    "x": float(pivot_x + dx * cos_a - dy * sin_a),
                    "y": float(pivot_y + dx * sin_a + dy * cos_a),
                    "u": vertex["u"],
                    "v": vertex["v"],
                }
            )
        return deformed

    def evaluate_bezier(self, t: float, points: list[JsonDict]) -> tuple[float, float]:
        degree = len(points) - 1
        x_value = 0.0
        y_value = 0.0
        for index, point in enumerate(points):
            coefficient = self._binomial(degree, index) * (t**index) * ((1 - t) ** (degree - index))
            x_value += coefficient * float(point["x"])
            y_value += coefficient * float(point["y"])
        return x_value, y_value

    def _binomial(self, n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1

        reduced_k = min(k, n - k)
        result = 1
        for index in range(reduced_k):
            result = result * (n - index) // (index + 1)
        return result
