"""
骨骼系统
Live2D 骨骼管理和动画
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger


@dataclass(slots=True)
class Bone:
    """Live2D 骨骼定义"""

    id: str
    name: str
    parent_id: str | None = None
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    rotation: float = 0.0
    scale: dict[str, float] = field(default_factory=lambda: {"x": 1.0, "y": 1.0})
    world_position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    world_rotation: float = 0.0
    world_scale: dict[str, float] = field(default_factory=lambda: {"x": 1.0, "y": 1.0})
    children: list[str] = field(default_factory=list)


class BoneSystem:
    """Live2D 骨骼系统"""

    def __init__(self) -> None:
        self.bones: dict[str, Bone] = {}
        self.root_bones: list[str] = []
        self.bind_poses: dict[str, dict[str, Any]] = {}

    def add_bone(self, bone: Bone) -> None:
        self.bones[bone.id] = bone
        if bone.parent_id and bone.parent_id in self.bones:
            self.bones[bone.parent_id].children.append(bone.id)
        elif not bone.parent_id and bone.id not in self.root_bones:
            self.root_bones.append(bone.id)
        logger.debug(f"添加骨骼: {bone.id}")

    def create_bone(self, bone_data: dict[str, Any]) -> Bone:
        bone = Bone(
            id=str(bone_data["id"]),
            name=str(bone_data["name"]),
            parent_id=bone_data.get("parent"),
            position=dict(bone_data.get("position", {"x": 0.0, "y": 0.0})),
            rotation=float(bone_data.get("rotation", 0.0)),
            scale=dict(bone_data.get("scale", {"x": 1.0, "y": 1.0})),
        )
        self.bind_poses[bone.id] = {
            "position": bone.position.copy(),
            "rotation": bone.rotation,
            "scale": bone.scale.copy(),
        }
        self.add_bone(bone)
        return bone

    def get_bone(self, bone_id: str) -> Bone | None:
        return self.bones.get(bone_id)

    def update_transforms(self) -> None:
        for root_id in self.root_bones:
            root = self.bones.get(root_id)
            if root is not None:
                self._update_bone_transform(root, None)

    def _update_bone_transform(self, bone: Bone, parent: Bone | None) -> None:
        if parent is None:
            bone.world_position = bone.position.copy()
            bone.world_rotation = bone.rotation
            bone.world_scale = bone.scale.copy()
        else:
            angle_rad = np.radians(parent.world_rotation)
            cos_r = np.cos(angle_rad)
            sin_r = np.sin(angle_rad)
            local_x = bone.position["x"] * parent.world_scale["x"]
            local_y = bone.position["y"] * parent.world_scale["y"]
            rotated_x = local_x * cos_r - local_y * sin_r
            rotated_y = local_x * sin_r + local_y * cos_r
            bone.world_position = {
                "x": float(parent.world_position["x"] + rotated_x),
                "y": float(parent.world_position["y"] + rotated_y),
            }
            bone.world_rotation = parent.world_rotation + bone.rotation
            bone.world_scale = {
                "x": parent.world_scale["x"] * bone.scale["x"],
                "y": parent.world_scale["y"] * bone.scale["y"],
            }

        for child_id in bone.children:
            child = self.bones.get(child_id)
            if child is not None:
                self._update_bone_transform(child, bone)

    def set_bone_rotation(self, bone_id: str, rotation: float) -> None:
        bone = self.bones.get(bone_id)
        if bone is not None:
            bone.rotation = rotation
            self.update_transforms()

    def set_bone_position(self, bone_id: str, x: float, y: float) -> None:
        bone = self.bones.get(bone_id)
        if bone is not None:
            bone.position = {"x": x, "y": y}
            self.update_transforms()

    def set_bone_scale(self, bone_id: str, x: float, y: float) -> None:
        bone = self.bones.get(bone_id)
        if bone is not None:
            bone.scale = {"x": x, "y": y}
            self.update_transforms()

    def get_bone_world_matrix(self, bone_id: str) -> npt.NDArray[np.float64]:
        bone = self.bones.get(bone_id)
        if bone is None:
            return np.eye(3, dtype=np.float64)

        angle_rad = np.radians(bone.world_rotation)
        cos_r = np.cos(angle_rad)
        sin_r = np.sin(angle_rad)
        matrix: npt.NDArray[np.float64] = np.array(
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
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return matrix

    def transform_point(self, bone_id: str, local_point: dict[str, float]) -> dict[str, float]:
        matrix = self.get_bone_world_matrix(bone_id)
        transformed = matrix @ np.array([local_point["x"], local_point["y"], 1.0])
        return {"x": float(transformed[0]), "y": float(transformed[1])}

    def inverse_transform_point(
        self, bone_id: str, world_point: dict[str, float]
    ) -> dict[str, float]:
        matrix = self.get_bone_world_matrix(bone_id)
        transformed = np.linalg.inv(matrix) @ np.array([world_point["x"], world_point["y"], 1.0])
        return {"x": float(transformed[0]), "y": float(transformed[1])}

    def look_at(
        self,
        bone_id: str,
        target: dict[str, float],
        _up_vector: dict[str, float] | None = None,
    ) -> None:
        bone = self.bones.get(bone_id)
        if bone is None:
            return
        dx = target["x"] - bone.world_position["x"]
        dy = target["y"] - bone.world_position["y"]
        angle = float(np.degrees(np.arctan2(dy, dx)))
        bone.rotation = angle - bone.world_rotation + bone.rotation
        self.update_transforms()

    def reset_to_bind_pose(self) -> None:
        for bone_id, bind_pose in self.bind_poses.items():
            bone = self.bones.get(bone_id)
            if bone is None:
                continue
            bone.position = dict(bind_pose["position"])
            bone.rotation = float(bind_pose["rotation"])
            bone.scale = dict(bind_pose["scale"])
        self.update_transforms()
        logger.debug("重置到绑定姿势")

    def export_to_json(self) -> dict[str, Any]:
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
