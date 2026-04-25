"""
参数系统
Live2D 参数管理和插值
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass(slots=True)
class Parameter:
    """Live2D 参数定义"""

    id: str
    name: str
    min_value: float
    max_value: float
    default_value: float
    current_value: float = 0.0

    def __post_init__(self) -> None:
        self.current_value = self.default_value

    def set_value(self, value: float) -> None:
        self.current_value = max(self.min_value, min(self.max_value, value))

    def normalize(self, value: float) -> float:
        span = self.max_value - self.min_value
        if span == 0:
            return 0.0
        return (value - self.min_value) / span

    def denormalize(self, normalized: float) -> float:
        return normalized * (self.max_value - self.min_value) + self.min_value


class ParameterSystem:
    """Live2D 参数系统"""

    def __init__(self) -> None:
        self.parameters: dict[str, Parameter] = {}
        self.groups: dict[str, list[str]] = {}
        self.blend_shapes: dict[str, dict[str, Any]] = {}

    def add_parameter(self, param: Parameter) -> None:
        self.parameters[param.id] = param
        logger.debug(f"添加参数: {param.id}")

    def add_parameters(self, param_list: list[dict[str, Any]]) -> None:
        for payload in param_list:
            self.add_parameter(
                Parameter(
                    id=str(payload["id"]),
                    name=str(payload["name"]),
                    min_value=float(payload.get("min", 0.0)),
                    max_value=float(payload.get("max", 1.0)),
                    default_value=float(payload.get("default", 0.0)),
                )
            )

    def get_parameter(self, param_id: str) -> Parameter | None:
        return self.parameters.get(param_id)

    def set_value(self, param_id: str, value: float) -> None:
        param = self.parameters.get(param_id)
        if param is not None:
            param.set_value(value)

    def set_values(self, values: dict[str, float]) -> None:
        for param_id, value in values.items():
            self.set_value(param_id, value)

    def create_group(self, name: str, param_ids: list[str]) -> None:
        self.groups[name] = param_ids
        logger.debug(f"创建参数组: {name} ({len(param_ids)} 个参数)")

    def interpolate(
        self,
        _param_id: str,
        start: float,
        end: float,
        progress: float,
        easing: str = "linear",
    ) -> float:
        t = self._apply_easing(max(0.0, min(1.0, progress)), easing)
        return start + (end - start) * t

    def _apply_easing(self, t: float, easing: str) -> float:
        if easing == "linear":
            return t
        if easing == "ease_in":
            return t * t
        if easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        if easing == "ease_in_out":
            return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
        if easing == "elastic":
            if t == 0 or t == 1:
                return t
            c4 = (2 * np.pi) / 3
            return float(-(2 ** (10 * t - 10)) * np.sin((t * 10 - 10.75) * c4))
        return t

    def create_blend_shape(self, name: str, targets: dict[str, float]) -> None:
        self.blend_shapes[name] = {"name": name, "targets": targets}
        logger.debug(f"创建混合形状: {name}")

    def apply_blend_shape(self, name: str, weight: float = 1.0) -> None:
        blend = self.blend_shapes.get(name)
        if blend is None:
            logger.warning(f"混合形状不存在: {name}")
            return

        clamped_weight = max(0.0, min(1.0, weight))
        targets = dict(blend.get("targets", {}))
        for param_id, target_value in targets.items():
            param = self.parameters.get(param_id)
            if param is None:
                continue
            new_value = (
                param.default_value + (float(target_value) - param.default_value) * clamped_weight
            )
            param.set_value(new_value)

    def reset_all(self) -> None:
        for param in self.parameters.values():
            param.current_value = param.default_value
        logger.debug("重置所有参数")

    def get_state(self) -> dict[str, float]:
        return {param_id: param.current_value for param_id, param in self.parameters.items()}

    def export_to_json(self) -> dict[str, Any]:
        return {
            "Version": 3,
            "Parameters": [
                {
                    "Id": param.id,
                    "Name": param.name,
                    "Min": param.min_value,
                    "Max": param.max_value,
                    "Default": param.default_value,
                    "Value": param.current_value,
                }
                for param in self.parameters.values()
            ],
            "Groups": [{"Name": name, "Ids": ids} for name, ids in self.groups.items()],
        }
