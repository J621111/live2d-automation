"""
参数系统
Live2D 参数管理和插值
"""

import numpy as np
from typing import Any, Dict, List, Optional, cast
from dataclasses import dataclass
from loguru import logger


@dataclass
class Parameter:
    """Live2D 参数定义"""

    id: str
    name: str
    min_value: float
    max_value: float
    default_value: float
    current_value: float = 0.0

    def __post_init__(self):
        self.current_value = self.default_value

    def set_value(self, value: float):
        """设置参数值（带限制）"""
        self.current_value = max(self.min_value, min(self.max_value, value))

    def normalize(self, value: float) -> float:
        """将值归一化到 0-1 范围"""
        return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, normalized: float) -> float:
        """从归一化值恢复"""
        return normalized * (self.max_value - self.min_value) + self.min_value


class ParameterSystem:
    """Live2D 参数系统"""

    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self.groups: Dict[str, List[str]] = {}
        self.blend_shapes: Dict[str, Dict] = {}

    def add_parameter(self, param: Parameter):
        """添加参数"""
        self.parameters[param.id] = param
        logger.debug(f"添加参数: {param.id}")

    def add_parameters(self, param_list: List[Dict]):
        """批量添加参数"""
        for p in param_list:
            param = Parameter(
                id=p["id"],
                name=p["name"],
                min_value=p.get("min", 0),
                max_value=p.get("max", 1),
                default_value=p.get("default", 0),
            )
            self.add_parameter(param)

    def get_parameter(self, param_id: str) -> Optional[Parameter]:
        """获取参数"""
        return self.parameters.get(param_id)

    def set_value(self, param_id: str, value: float):
        """设置参数值"""
        if param_id in self.parameters:
            self.parameters[param_id].set_value(value)

    def set_values(self, values: Dict[str, float]):
        """批量设置参数值"""
        for param_id, value in values.items():
            self.set_value(param_id, value)

    def create_group(self, name: str, param_ids: List[str]):
        """创建参数组"""
        self.groups[name] = param_ids
        logger.debug(f"创建参数组: {name} ({len(param_ids)} 个参数)")

    def interpolate(
        self,
        param_id: str,
        start: float,
        end: float,
        progress: float,
        easing: str = "linear",
    ) -> float:
        """
        参数插值

        Args:
            param_id: 参数 ID
            start: 起始值
            end: 结束值
            progress: 进度 (0-1)
            easing: 缓动函数类型

        Returns:
            插值后的值
        """
        t = self._apply_easing(progress, easing)
        value = start + (end - start) * t
        return value

    def _apply_easing(self, t: float, easing: str) -> float:
        """应用缓动函数"""
        if easing == "linear":
            return t
        elif easing == "ease_in":
            return t * t
        elif easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif easing == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - (-2 * t + 2) ** 2 / 2
        elif easing == "elastic":
            c4 = (2 * np.pi) / 3
            if t == 0:
                return 0
            elif t == 1:
                return 1
            else:
                value = -(2 ** (10 * t - 10)) * np.sin((t * 10 - 10.75) * c4)
                return cast(float, value)
        else:
            return t

    def create_blend_shape(self, name: str, targets: Dict[str, float]):
        """
        创建混合形状（Blend Shape）

        Args:
            name: 混合形状名称
            targets: 目标参数值字典
        """
        self.blend_shapes[name] = {"name": name, "targets": targets}
        logger.debug(f"创建混合形状: {name}")

    def apply_blend_shape(self, name: str, weight: float = 1.0):
        """
        应用混合形状

        Args:
            name: 混合形状名称
            weight: 权重 (0-1)
        """
        if name not in self.blend_shapes:
            logger.warning(f"混合形状不存在: {name}")
            return

        blend = self.blend_shapes[name]
        for param_id, target_value in blend["targets"].items():
            if param_id in self.parameters:
                param = self.parameters[param_id]
                # 插值到目标值
                new_value = (
                    param.default_value + (target_value - param.default_value) * weight
                )
                param.set_value(new_value)

    def reset_all(self):
        """重置所有参数到默认值"""
        for param in self.parameters.values():
            param.current_value = param.default_value
        logger.debug("重置所有参数")

    def get_state(self) -> Dict[str, float]:
        """获取当前所有参数值"""
        return {
            param_id: param.current_value for param_id, param in self.parameters.items()
        }

    def export_to_json(self) -> Dict:
        """导出为 Live2D JSON 格式"""
        return {
            "Version": 3,
            "Parameters": [
                {
                    "Id": p.id,
                    "Name": p.name,
                    "Min": p.min_value,
                    "Max": p.max_value,
                    "Default": p.default_value,
                    "Value": p.current_value,
                }
                for p in self.parameters.values()
            ],
            "Groups": [{"Name": name, "Ids": ids} for name, ids in self.groups.items()],
        }
