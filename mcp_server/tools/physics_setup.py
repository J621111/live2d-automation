"""
物理设置
配置 Live2D 物理效果，如头发、衣服摆动
"""

import numpy as np
from typing import Dict, List, Any
from loguru import logger


class PhysicsSetup:
    """Live2D 物理系统配置"""

    def __init__(self):
        self.gravity = 9.8
        self.default_damping = 0.9
        self.default_stiffness = 0.3

    async def configure(self, rigging: Dict, segments: Dict) -> Dict[str, Any]:
        """
        配置物理系统

        Args:
            rigging: 绑定信息
            segments: 身体部位信息

        Returns:
            物理配置
        """
        logger.info("配置物理系统...")

        physics_groups = []

        # 1. 头发物理
        hair_physics = self._create_hair_physics(rigging)
        if hair_physics:
            physics_groups.append(hair_physics)

        # 2. 衣服物理（如果有）
        clothing_physics = self._create_clothing_physics(rigging)
        if clothing_physics:
            physics_groups.append(clothing_physics)

        # 3. 配饰物理
        accessory_physics = self._create_accessory_physics(rigging)
        if accessory_physics:
            physics_groups.append(accessory_physics)

        physics_config = {
            "version": 3,
            "groups": physics_groups,
            "settings": {
                "gravity": self.gravity,
                "wind": {"x": 0, "y": 0},
                "enable": True,
            },
        }

        logger.info(f"物理系统配置完成: {len(physics_groups)} 个物理组")
        return physics_config

    def _create_hair_physics(self, rigging: Dict) -> Dict:
        """创建头发物理"""
        # 查找头发相关的骨骼或参数
        bones = rigging.get("bones", [])
        parameters = rigging.get("parameters", [])

        hair_group = {
            "name": "Hair",
            "id": "HairPhysics",
            "input": [],
            "output": [],
            "particles": [],
        }

        # 输入：头部旋转
        for param in parameters:
            if "Angle" in param["id"] and "Body" not in param["id"]:
                hair_group["input"].append(
                    {"id": param["id"], "weight": 0.5 if "X" in param["id"] else 0.3}
                )

        # 输出：头发参数
        hair_params = [
            {"id": "ParamHairFront", "name": "前发", "type": "angle"},
            {"id": "ParamHairSide", "name": "侧发", "type": "angle"},
            {"id": "ParamHairBack", "name": "后发", "type": "angle"},
        ]

        for hp in hair_params:
            hair_group["output"].append(
                {"id": hp["id"], "type": hp["type"], "weight": 1.0}
            )

        # 创建粒子系统
        hair_group["particles"] = self._create_hair_particles()

        return hair_group

    def _create_hair_particles(self) -> List[Dict]:
        """创建头发粒子系统"""
        particles = []

        # 创建链式粒子系统模拟头发
        num_segments = 5
        base_x, base_y = 0, -100  # 头发根部位置

        for i in range(num_segments):
            particle = {
                "index": i,
                "position": {"x": base_x, "y": base_y - i * 20},
                "mass": 0.5 + i * 0.1,  # 越往下越重
                "damping": self.default_damping,
                "stiffness": self.default_stiffness - i * 0.05,  # 越往下越软
                "radius": 10 + i * 2,
                "constraint": {
                    "type": "distance",
                    "target": i - 1 if i > 0 else None,
                    "distance": 20,
                }
                if i > 0
                else None,
            }
            particles.append(particle)

        return particles

    def _create_clothing_physics(self, rigging: Dict) -> Dict:
        """创建衣服物理"""
        clothing_group = {
            "name": "Clothing",
            "id": "ClothingPhysics",
            "input": [],
            "output": [],
            "particles": [],
        }

        # 输入：身体运动
        body_params = ["ParamBodyAngleX", "ParamBodyAngleY", "ParamBreath"]
        for param_id in body_params:
            clothing_group["input"].append({"id": param_id, "weight": 0.3})

        # 输出：衣服摆动
        clothing_params = [
            {"id": "ParamClothA", "name": "衣服摆动 A", "type": "angle"},
            {"id": "ParamClothB", "name": "衣服摆动 B", "type": "angle"},
            {"id": "ParamClothC", "name": "衣服摆动 C", "type": "angle"},
        ]

        for cp in clothing_params:
            clothing_group["output"].append(
                {"id": cp["id"], "type": cp["type"], "weight": 0.8}
            )

        # 创建布料粒子
        clothing_group["particles"] = self._create_cloth_particles()

        return clothing_group

    def _create_cloth_particles(self) -> List[Dict]:
        """创建布料粒子系统"""
        particles = []

        # 创建简单的布料网格
        rows, cols = 3, 3
        spacing = 30

        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                particle = {
                    "index": index,
                    "position": {"x": (col - cols // 2) * spacing, "y": row * spacing},
                    "mass": 0.3,
                    "damping": 0.85,
                    "stiffness": 0.4,
                    "fixed": row == 0,  # 顶部固定
                    "constraints": [],
                }

                # 添加约束
                if row > 0:
                    particle["constraints"].append(
                        {
                            "target": (row - 1) * cols + col,
                            "type": "distance",
                            "distance": spacing,
                        }
                    )
                if col > 0:
                    particle["constraints"].append(
                        {
                            "target": row * cols + (col - 1),
                            "type": "distance",
                            "distance": spacing,
                        }
                    )

                particles.append(particle)

        return particles

    def _create_accessory_physics(self, rigging: Dict) -> Dict:
        """创建配饰物理"""
        accessory_group = {
            "name": "Accessories",
            "id": "AccessoryPhysics",
            "input": [],
            "output": [],
            "particles": [],
        }

        # 输入：头部运动
        head_params = ["ParamAngleX", "ParamAngleY"]
        for param_id in head_params:
            accessory_group["input"].append({"id": param_id, "weight": 0.6})

        # 输出：配饰摆动
        accessory_params = [
            {"id": "ParamAccessoryA", "name": "配饰 A", "type": "angle"},
            {"id": "ParamAccessoryB", "name": "配饰 B", "type": "angle"},
        ]

        for ap in accessory_params:
            accessory_group["output"].append(
                {"id": ap["id"], "type": ap["type"], "weight": 1.0}
            )

        # 简单的单摆粒子
        accessory_group["particles"] = [
            {
                "index": 0,
                "position": {"x": 0, "y": -150},
                "mass": 0.2,
                "damping": 0.95,
                "stiffness": 0.5,
                "length": 50,
                "constraint": {"type": "pendulum", "pivot": {"x": 0, "y": -100}},
            }
        ]

        return accessory_group

    def _calculate_physics_params(self, bones: List[Dict]) -> Dict:
        """根据骨骼结构计算物理参数"""
        params = {
            "spring_constant": self.default_stiffness,
            "damping": self.default_damping,
            "gravity": self.gravity,
        }

        # 根据骨骼长度调整参数
        for bone in bones:
            if "length" in bone:
                length = bone["length"]
                # 更长的骨骼需要更低的刚度
                params["spring_constant"] = min(
                    params["spring_constant"], 1.0 / (length / 100 + 1)
                )

        return params
