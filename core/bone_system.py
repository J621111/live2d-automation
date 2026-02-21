"""
骨骼系统
Live2D 骨骼管理和动画
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Bone:
    """Live2D 骨骼定义"""

    id: str
    name: str
    parent_id: Optional[str] = None
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    rotation: float = 0.0
    scale: Dict[str, float] = field(default_factory=lambda: {"x": 1, "y": 1})

    # 运行时数据
    world_position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    world_rotation: float = 0.0
    world_scale: Dict[str, float] = field(default_factory=lambda: {"x": 1, "y": 1})

    # 子骨骼
    children: List[str] = field(default_factory=list)


class BoneSystem:
    """Live2D 骨骼系统"""

    def __init__(self):
        self.bones: Dict[str, Bone] = {}
        self.root_bones: List[str] = []
        self.bind_poses: Dict[str, Dict] = {}

    def add_bone(self, bone: Bone):
        """添加骨骼"""
        self.bones[bone.id] = bone

        # 更新父骨骼的子列表
        if bone.parent_id and bone.parent_id in self.bones:
            self.bones[bone.parent_id].children.append(bone.id)
        elif not bone.parent_id:
            self.root_bones.append(bone.id)

        logger.debug(f"添加骨骼: {bone.id}")

    def create_bone(self, bone_data: Dict) -> Bone:
        """从字典创建骨骼"""
        bone = Bone(
            id=bone_data["id"],
            name=bone_data["name"],
            parent_id=bone_data.get("parent"),
            position=bone_data.get("position", {"x": 0, "y": 0}),
            rotation=bone_data.get("rotation", 0),
            scale=bone_data.get("scale", {"x": 1, "y": 1}),
        )

        # 保存绑定姿势
        self.bind_poses[bone.id] = {
            "position": bone.position.copy(),
            "rotation": bone.rotation,
            "scale": bone.scale.copy(),
        }

        self.add_bone(bone)
        return bone

    def get_bone(self, bone_id: str) -> Optional[Bone]:
        """获取骨骼"""
        return self.bones.get(bone_id)

    def update_transforms(self):
        """更新所有骨骼的世界变换"""
        for root_id in self.root_bones:
            self._update_bone_transform(self.bones[root_id], None)

    def _update_bone_transform(self, bone: Bone, parent: Optional[Bone]):
        """递归更新骨骼变换"""
        if parent:
            # 计算世界变换
            # 旋转
            angle_rad = np.radians(parent.world_rotation)
            cos_r = np.cos(angle_rad)
            sin_r = np.sin(angle_rad)

            # 位置变换
            local_x = bone.position["x"] * parent.world_scale["x"]
            local_y = bone.position["y"] * parent.world_scale["y"]

            rotated_x = local_x * cos_r - local_y * sin_r
            rotated_y = local_x * sin_r + local_y * cos_r

            bone.world_position = {
                "x": parent.world_position["x"] + rotated_x,
                "y": parent.world_position["y"] + rotated_y,
            }

            # 旋转累积
            bone.world_rotation = parent.world_rotation + bone.rotation

            # 缩放累积
            bone.world_scale = {
                "x": parent.world_scale["x"] * bone.scale["x"],
                "y": parent.world_scale["y"] * bone.scale["y"],
            }
        else:
            # 根骨骼
            bone.world_position = bone.position.copy()
            bone.world_rotation = bone.rotation
            bone.world_scale = bone.scale.copy()

        # 递归更新子骨骼
        for child_id in bone.children:
            if child_id in self.bones:
                self._update_bone_transform(self.bones[child_id], bone)

    def set_bone_rotation(self, bone_id: str, rotation: float):
        """设置骨骼旋转"""
        if bone_id in self.bones:
            self.bones[bone_id].rotation = rotation
            self.update_transforms()

    def set_bone_position(self, bone_id: str, x: float, y: float):
        """设置骨骼位置"""
        if bone_id in self.bones:
            self.bones[bone_id].position = {"x": x, "y": y}
            self.update_transforms()

    def set_bone_scale(self, bone_id: str, x: float, y: float):
        """设置骨骼缩放"""
        if bone_id in self.bones:
            self.bones[bone_id].scale = {"x": x, "y": y}
            self.update_transforms()

    def get_bone_world_matrix(self, bone_id: str) -> np.ndarray:
        """
        获取骨骼的世界变换矩阵

        Returns:
            3x3 变换矩阵
        """
        if bone_id not in self.bones:
            return np.eye(3)

        bone = self.bones[bone_id]

        # 构建变换矩阵
        angle_rad = np.radians(bone.world_rotation)
        cos_r = np.cos(angle_rad)
        sin_r = np.sin(angle_rad)

        matrix = np.array(
            [
                [
                    cos_r * bone.world_scale["x"],
                    -sin_r * bone.world_scale["y"],
                    bone.world_position["x"],
                ],
                [
                    sin_r * bone.world_scale["x"],
                    cos_r * bone.world_scale["y"],
                    bone.world_position["y"],
                ],
                [0, 0, 1],
            ]
        )

        return matrix

    def transform_point(
        self, bone_id: str, local_point: Dict[str, float]
    ) -> Dict[str, float]:
        """
        将局部坐标转换为世界坐标

        Args:
            bone_id: 骨骼 ID
            local_point: 局部坐标 {x, y}

        Returns:
            世界坐标 {x, y}
        """
        matrix = self.get_bone_world_matrix(bone_id)

        point = np.array([local_point["x"], local_point["y"], 1])
        transformed = matrix @ point

        return {"x": transformed[0], "y": transformed[1]}

    def inverse_transform_point(
        self, bone_id: str, world_point: Dict[str, float]
    ) -> Dict[str, float]:
        """
        将世界坐标转换为局部坐标

        Args:
            bone_id: 骨骼 ID
            world_point: 世界坐标 {x, y}

        Returns:
            局部坐标 {x, y}
        """
        matrix = self.get_bone_world_matrix(bone_id)
        inv_matrix = np.linalg.inv(matrix)

        point = np.array([world_point["x"], world_point["y"], 1])
        transformed = inv_matrix @ point

        return {"x": transformed[0], "y": transformed[1]}

    def look_at(
        self, bone_id: str, target: Dict[str, float], up_vector: Optional[Dict] = None
    ):
        """
        让骨骼朝向目标点

        Args:
            bone_id: 骨骼 ID
            target: 目标点 {x, y}
            up_vector: 上方向向量（可选）
        """
        if bone_id not in self.bones:
            return

        bone = self.bones[bone_id]

        # 计算方向向量
        dx = target["x"] - bone.world_position["x"]
        dy = target["y"] - bone.world_position["y"]

        # 计算角度
        angle = np.degrees(np.arctan2(dy, dx))

        # 设置旋转
        bone.rotation = angle - bone.world_rotation + bone.rotation
        self.update_transforms()

    def reset_to_bind_pose(self):
        """重置到绑定姿势"""
        for bone_id, bind_pose in self.bind_poses.items():
            if bone_id in self.bones:
                bone = self.bones[bone_id]
                bone.position = bind_pose["position"].copy()
                bone.rotation = bind_pose["rotation"]
                bone.scale = bind_pose["scale"].copy()

        self.update_transforms()
        logger.debug("重置到绑定姿势")

    def export_to_json(self) -> Dict:
        """导出为 Live2D JSON 格式"""
        return {
            "Version": 3,
            "Bones": [
                {
                    "Id": bone.id,
                    "Name": bone.name,
                    "Parent": bone.parent_id,
                    "Position": bone.position,
                    "Rotation": bone.rotation,
                    "Scale": bone.scale,
                }
                for bone in self.bones.values()
            ],
        }
