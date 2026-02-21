"""
自动绑定系统
自动生成 Live2D 的骨骼、变形器和参数
"""

import numpy as np
from typing import Dict, List, Any
from loguru import logger


class AutoRigger:
    """Live2D 自动绑定器"""

    def __init__(self):
        self.bone_hierarchy = {}
        self.parameters = []
        self.deformers = []

    async def setup(self, meshes: Dict, segments: Dict) -> Dict[str, Any]:
        """
        设置完整的绑定系统

        Args:
            meshes: ArtMesh 网格数据
            segments: 身体部位分割信息

        Returns:
            完整的绑定信息
        """
        logger.info("开始设置 Live2D 绑定...")

        # 1. 创建骨骼系统
        bones = self._create_bone_system(segments)

        # 2. 创建变形器
        deformers = self._create_deformers(meshes)

        # 3. 创建参数系统
        parameters = self._create_parameters(segments)

        # 4. 绑定骨骼到网格
        bone_weights = self._bind_bones_to_meshes(bones, meshes)

        # 5. 创建交互区域
        hit_areas = self._create_hit_areas(segments)

        # 6. 创建参数组
        groups = self._create_parameter_groups(parameters)

        rigging = {
            "bones": bones,
            "deformers": deformers,
            "parameters": parameters,
            "bone_weights": bone_weights,
            "hit_areas": hit_areas,
            "groups": groups,
        }

        logger.info(f"绑定完成: {len(bones)} 根骨骼, {len(parameters)} 个参数")
        return rigging

    def _create_bone_system(self, segments: Dict) -> List[Dict]:
        """创建骨骼系统"""
        bones = []
        body_parts = segments.get("body_parts", {})

        # 根骨骼（身体中心）
        root_bone = {
            "id": "root",
            "name": "Root",
            "parent": None,
            "position": {"x": 0, "y": 0},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(root_bone)

        # 头部骨骼
        if "head" in body_parts:
            head_info = body_parts["head"]
            if head_info and "center" in head_info:
                center = head_info["center"]
                head_bone = {
                    "id": "head",
                    "name": "Head",
                    "parent": "root",
                    "position": {"x": center[0], "y": center[1]},
                    "rotation": 0,
                    "scale": {"x": 1, "y": 1},
                }
                bones.append(head_bone)

        # 颈部骨骼
        neck_bone = {
            "id": "neck",
            "name": "Neck",
            "parent": "root",
            "position": {"x": 0, "y": -50},  # 相对于根骨骼
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(neck_bone)

        # 身体骨骼
        if "torso" in body_parts:
            torso_info = body_parts["torso"]
            if torso_info and "center" in torso_info:
                center = torso_info["center"]
                torso_bone = {
                    "id": "torso",
                    "name": "Torso",
                    "parent": "root",
                    "position": {"x": center[0], "y": center[1]},
                    "rotation": 0,
                    "scale": {"x": 1, "y": 1},
                }
                bones.append(torso_bone)

        # 左臂骨骼
        left_arm_bone = {
            "id": "left_arm",
            "name": "Left Arm",
            "parent": "torso",
            "position": {"x": -80, "y": -100},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(left_arm_bone)

        left_forearm_bone = {
            "id": "left_forearm",
            "name": "Left Forearm",
            "parent": "left_arm",
            "position": {"x": -60, "y": 80},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(left_forearm_bone)

        # 右臂骨骼
        right_arm_bone = {
            "id": "right_arm",
            "name": "Right Arm",
            "parent": "torso",
            "position": {"x": 80, "y": -100},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(right_arm_bone)

        right_forearm_bone = {
            "id": "right_forearm",
            "name": "Right Forearm",
            "parent": "right_arm",
            "position": {"x": 60, "y": 80},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(right_forearm_bone)

        # 左腿骨骼
        left_leg_bone = {
            "id": "left_leg",
            "name": "Left Leg",
            "parent": "root",
            "position": {"x": -40, "y": 150},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(left_leg_bone)

        left_shin_bone = {
            "id": "left_shin",
            "name": "Left Shin",
            "parent": "left_leg",
            "position": {"x": 0, "y": 100},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(left_shin_bone)

        # 右腿骨骼
        right_leg_bone = {
            "id": "right_leg",
            "name": "Right Leg",
            "parent": "root",
            "position": {"x": 40, "y": 150},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(right_leg_bone)

        right_shin_bone = {
            "id": "right_shin",
            "name": "Right Shin",
            "parent": "right_leg",
            "position": {"x": 0, "y": 100},
            "rotation": 0,
            "scale": {"x": 1, "y": 1},
        }
        bones.append(right_shin_bone)

        # 面部骨骼
        face_bones = self._create_face_bones()
        bones.extend(face_bones)

        return bones

    def _create_face_bones(self) -> List[Dict]:
        """创建面部骨骼"""
        face_bones = []

        # 左眼骨骼
        face_bones.append(
            {
                "id": "left_eye",
                "name": "Left Eye",
                "parent": "head",
                "position": {"x": -30, "y": -20},
                "rotation": 0,
                "scale": {"x": 1, "y": 1},
            }
        )

        # 右眼骨骼
        face_bones.append(
            {
                "id": "right_eye",
                "name": "Right Eye",
                "parent": "head",
                "position": {"x": 30, "y": -20},
                "rotation": 0,
                "scale": {"x": 1, "y": 1},
            }
        )

        # 嘴部骨骼
        face_bones.append(
            {
                "id": "mouth",
                "name": "Mouth",
                "parent": "head",
                "position": {"x": 0, "y": 20},
                "rotation": 0,
                "scale": {"x": 1, "y": 1},
            }
        )

        return face_bones

    def _create_deformers(self, meshes: Dict) -> List[Dict]:
        """创建变形器"""
        deformers = []

        for mesh_name, mesh_data in meshes.items():
            # 为每个部位创建弯曲变形器
            warp_deformer = {
                "id": f"{mesh_name}_warp",
                "name": f"{mesh_name} Warp",
                "type": "warp",
                "mesh": mesh_data["mesh"],
                "base_mesh": mesh_data["mesh"],
            }
            deformers.append(warp_deformer)

            # 创建旋转变形器
            rotation_deformer = {
                "id": f"{mesh_name}_rotation",
                "name": f"{mesh_name} Rotation",
                "type": "rotation",
                "mesh": mesh_data["mesh"],
                "center": self._calculate_mesh_center(mesh_data["mesh"]),
            }
            deformers.append(rotation_deformer)

        return deformers

    def _calculate_mesh_center(self, mesh: Dict) -> Dict:
        """计算网格中心点"""
        vertices = mesh["vertices"]
        if not vertices:
            return {"x": 0, "y": 0}

        x_coords = [v["x"] for v in vertices]
        y_coords = [v["y"] for v in vertices]

        return {"x": sum(x_coords) / len(x_coords), "y": sum(y_coords) / len(y_coords)}

    def _create_parameters(self, segments: Dict) -> List[Dict]:
        """创建参数系统"""
        parameters = []

        # 身体参数
        body_params = [
            {
                "id": "ParamBodyAngleX",
                "name": "身体 X 轴旋转",
                "min": -30,
                "max": 30,
                "default": 0,
            },
            {
                "id": "ParamBodyAngleY",
                "name": "身体 Y 轴旋转",
                "min": -30,
                "max": 30,
                "default": 0,
            },
            {
                "id": "ParamBodyAngleZ",
                "name": "身体 Z 轴旋转",
                "min": -30,
                "max": 30,
                "default": 0,
            },
            {"id": "ParamBreath", "name": "呼吸", "min": 0, "max": 1, "default": 0},
        ]
        parameters.extend(body_params)

        # 头部参数
        head_params = [
            {
                "id": "ParamAngleX",
                "name": "头部 X 轴旋转",
                "min": -30,
                "max": 30,
                "default": 0,
            },
            {
                "id": "ParamAngleY",
                "name": "头部 Y 轴旋转",
                "min": -30,
                "max": 30,
                "default": 0,
            },
            {
                "id": "ParamAngleZ",
                "name": "头部 Z 轴旋转",
                "min": -30,
                "max": 30,
                "default": 0,
            },
        ]
        parameters.extend(head_params)

        # 眼睛参数
        eye_params = [
            {
                "id": "ParamEyeLOpen",
                "name": "左眼开合",
                "min": 0,
                "max": 1,
                "default": 1,
            },
            {
                "id": "ParamEyeROpen",
                "name": "右眼开合",
                "min": 0,
                "max": 1,
                "default": 1,
            },
            {
                "id": "ParamEyeBallX",
                "name": "眼球 X",
                "min": -1,
                "max": 1,
                "default": 0,
            },
            {
                "id": "ParamEyeBallY",
                "name": "眼球 Y",
                "min": -1,
                "max": 1,
                "default": 0,
            },
            {
                "id": "ParamEyeLSmile",
                "name": "左眼微笑",
                "min": 0,
                "max": 1,
                "default": 0,
            },
            {
                "id": "ParamEyeRSmile",
                "name": "右眼微笑",
                "min": 0,
                "max": 1,
                "default": 0,
            },
        ]
        parameters.extend(eye_params)

        # 眉毛参数
        eyebrow_params = [
            {"id": "ParamBrowLY", "name": "左眉 Y", "min": -1, "max": 1, "default": 0},
            {"id": "ParamBrowRY", "name": "右眉 Y", "min": -1, "max": 1, "default": 0},
            {
                "id": "ParamBrowLAngle",
                "name": "左眉角度",
                "min": -1,
                "max": 1,
                "default": 0,
            },
            {
                "id": "ParamBrowRAngle",
                "name": "右眉角度",
                "min": -1,
                "max": 1,
                "default": 0,
            },
            {
                "id": "ParamBrowLForm",
                "name": "左眉形状",
                "min": -1,
                "max": 1,
                "default": 0,
            },
            {
                "id": "ParamBrowRForm",
                "name": "右眉形状",
                "min": -1,
                "max": 1,
                "default": 0,
            },
        ]
        parameters.extend(eyebrow_params)

        # 嘴部参数
        mouth_params = [
            {
                "id": "ParamMouthOpenY",
                "name": "嘴巴开合",
                "min": 0,
                "max": 1,
                "default": 0,
            },
            {"id": "ParamMouthForm", "name": "嘴型", "min": -1, "max": 1, "default": 0},
        ]
        parameters.extend(mouth_params)

        # 手臂参数
        arm_params = [
            {
                "id": "ParamArmLA",
                "name": "左臂角度",
                "min": -120,
                "max": 120,
                "default": 0,
            },
            {
                "id": "ParamArmRA",
                "name": "右臂角度",
                "min": -120,
                "max": 120,
                "default": 0,
            },
        ]
        parameters.extend(arm_params)

        # 腿参数
        leg_params = [
            {"id": "ParamLegL", "name": "左腿", "min": -90, "max": 90, "default": 0},
            {"id": "ParamLegR", "name": "右腿", "min": -90, "max": 90, "default": 0},
        ]
        parameters.extend(leg_params)

        return parameters

    def _bind_bones_to_meshes(self, bones: List[Dict], meshes: Dict) -> Dict:
        """将骨骼绑定到网格"""
        weights = {}

        bone_mesh_mapping = {
            "head": ["head", "face_base", "back_hair", "front_hair"],
            "left_eye": ["left_eye"],
            "right_eye": ["right_eye"],
            "mouth": ["mouth"],
            "torso": ["body"],
            "left_arm": ["left_arm"],
            "right_arm": ["right_arm"],
            "left_leg": ["left_leg"],
            "right_leg": ["right_leg"],
        }

        for bone in bones:
            bone_id = bone["id"]
            if bone_id in bone_mesh_mapping:
                for mesh_name in bone_mesh_mapping[bone_id]:
                    if mesh_name in meshes:
                        weights[mesh_name] = {"bone": bone_id, "weight": 1.0}

        return weights

    def _create_hit_areas(self, segments: Dict) -> List[Dict]:
        """创建交互区域"""
        hit_areas = []

        body_parts = segments.get("body_parts", {})

        # 头部交互区
        if "head" in body_parts:
            head_info = body_parts["head"]
            if head_info and "bounds" in head_info:
                hit_areas.append(
                    {"id": "HitHead", "name": "Head", "bounds": head_info["bounds"]}
                )

        # 身体交互区
        if "torso" in body_parts:
            torso_info = body_parts["torso"]
            if torso_info and "bounds" in torso_info:
                hit_areas.append(
                    {"id": "HitBody", "name": "Body", "bounds": torso_info["bounds"]}
                )

        # 手部交互区
        for side in ["left", "right"]:
            arm_key = f"{side}_arm"
            if arm_key in body_parts:
                arm_info = body_parts[arm_key]
                if arm_info and "bounds" in arm_info:
                    hit_areas.append(
                        {
                            "id": f"Hit{side.capitalize()}Hand",
                            "name": f"{side.capitalize()} Hand",
                            "bounds": arm_info["bounds"],
                        }
                    )

        return hit_areas

    def _create_parameter_groups(self, parameters: List[Dict]) -> List[Dict]:
        """创建参数组"""
        groups = [
            {
                "name": "头部",
                "ids": [
                    p["id"]
                    for p in parameters
                    if "Angle" in p["id"] and "Body" not in p["id"]
                ],
            },
            {"name": "身体", "ids": [p["id"] for p in parameters if "Body" in p["id"]]},
            {"name": "眼睛", "ids": [p["id"] for p in parameters if "Eye" in p["id"]]},
            {"name": "眉毛", "ids": [p["id"] for p in parameters if "Brow" in p["id"]]},
            {
                "name": "嘴巴",
                "ids": [p["id"] for p in parameters if "Mouth" in p["id"]],
            },
            {"name": "手臂", "ids": [p["id"] for p in parameters if "Arm" in p["id"]]},
            {"name": "腿部", "ids": [p["id"] for p in parameters if "Leg" in p["id"]]},
        ]

        return groups
