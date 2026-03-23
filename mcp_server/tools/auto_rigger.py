"""Automatic rigging helpers for generated Live2D meshes."""

from __future__ import annotations

from typing import Any

from loguru import logger

JsonDict = dict[str, Any]
JsonList = list[JsonDict]


class AutoRigger:
    """Create a lightweight rig, parameters, and bindings from segmented parts."""

    def __init__(self) -> None:
        self.bone_hierarchy: JsonDict = {}
        self.parameters: JsonList = []
        self.deformers: JsonList = []
        self.warnings: list[str] = []

    async def setup(self, meshes: JsonDict, segments: JsonDict) -> JsonDict:
        logger.info("Starting Live2D rig setup...")

        bones = self._normalize_bones(self._create_bone_system(segments))
        deformers = self._create_deformers(meshes)
        parameters = self._create_parameters()
        bone_weights = self._bind_bones_to_meshes(bones, meshes)
        hit_areas = self._create_hit_areas(segments)
        groups = self._create_parameter_groups(parameters)

        rigging: JsonDict = {
            "bones": bones,
            "deformers": deformers,
            "parameters": parameters,
            "bone_weights": bone_weights,
            "hit_areas": hit_areas,
            "groups": groups,
            "warnings": list(self.warnings),
        }

        logger.info(f"Rigging complete: {len(bones)} bones, {len(parameters)} parameters")
        return rigging

    def _bone(
        self,
        bone_id: str,
        name: str,
        parent: str | None,
        x: float,
        y: float,
    ) -> JsonDict:
        return {
            "id": bone_id,
            "name": name,
            "parent": parent,
            "position": {"x": x, "y": y},
            "rotation": 0.0,
            "scale": {"x": 1.0, "y": 1.0},
        }

    def _create_bone_system(self, segments: JsonDict) -> JsonList:
        bones: JsonList = [self._bone("root", "Root", None, 0.0, 0.0)]
        body_parts = dict(segments.get("body_parts", {}))

        head_info = body_parts.get("head")
        if isinstance(head_info, dict) and "center" in head_info:
            center = head_info["center"]
            bones.append(self._bone("head", "Head", "root", float(center[0]), float(center[1])))

        bones.append(self._bone("neck", "Neck", "head", 0.0, -50.0))

        torso_info = body_parts.get("torso")
        if isinstance(torso_info, dict) and "center" in torso_info:
            center = torso_info["center"]
            bones.append(self._bone("torso", "Torso", "root", float(center[0]), float(center[1])))

        bones.extend(
            [
                self._bone("left_arm", "Left Arm", "torso", -80.0, -100.0),
                self._bone("left_forearm", "Left Forearm", "left_arm", -60.0, 80.0),
                self._bone("right_arm", "Right Arm", "torso", 80.0, -100.0),
                self._bone("right_forearm", "Right Forearm", "right_arm", 60.0, 80.0),
                self._bone("left_leg", "Left Leg", "root", -40.0, 150.0),
                self._bone("left_shin", "Left Shin", "left_leg", 0.0, 100.0),
                self._bone("right_leg", "Right Leg", "root", 40.0, 150.0),
                self._bone("right_shin", "Right Shin", "right_leg", 0.0, 100.0),
            ]
        )
        bones.extend(self._create_face_bones())
        return bones

    def _create_face_bones(self) -> JsonList:
        return [
            self._bone("left_eye", "Left Eye", "head", -30.0, -20.0),
            self._bone("right_eye", "Right Eye", "head", 30.0, -20.0),
            self._bone("mouth", "Mouth", "head", 0.0, 20.0),
        ]

    def _normalize_bones(self, bones: JsonList) -> JsonList:
        bone_ids = {str(bone["id"]) for bone in bones}
        normalized: JsonList = []
        self.warnings = []
        for bone in bones:
            parent = bone.get("parent")
            if parent and parent not in bone_ids:
                fallback_parent = "neck" if parent == "head" and "neck" in bone_ids else "root"
                self.warnings.append(
                    f"Bone {bone['id']} referenced missing parent {parent}; falling back to {fallback_parent}."
                )
                bone = {**bone, "parent": fallback_parent}
            normalized.append(bone)
        return normalized

    def _create_deformers(self, meshes: JsonDict) -> JsonList:
        deformers: JsonList = []
        for mesh_name, mesh_data in meshes.items():
            mesh = mesh_data["mesh"]
            deformers.append(
                {
                    "id": f"{mesh_name}_warp",
                    "name": f"{mesh_name} Warp",
                    "type": "warp",
                    "mesh": mesh,
                    "base_mesh": mesh,
                }
            )
            deformers.append(
                {
                    "id": f"{mesh_name}_rotation",
                    "name": f"{mesh_name} Rotation",
                    "type": "rotation",
                    "mesh": mesh,
                    "center": self._calculate_mesh_center(mesh),
                }
            )
        return deformers

    def _calculate_mesh_center(self, mesh: JsonDict) -> JsonDict:
        vertices = list(mesh.get("vertices", []))
        if not vertices:
            return {"x": 0.0, "y": 0.0}

        x_coords = [float(vertex["x"]) for vertex in vertices]
        y_coords = [float(vertex["y"]) for vertex in vertices]
        return {"x": sum(x_coords) / len(x_coords), "y": sum(y_coords) / len(y_coords)}

    def _create_parameters(self) -> JsonList:
        return [
            {"id": "ParamBodyAngleX", "name": "Body Angle X", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamBodyAngleY", "name": "Body Angle Y", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamBodyAngleZ", "name": "Body Angle Z", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamBreath", "name": "Breath", "min": 0.0, "max": 1.0, "default": 0.0},
            {"id": "ParamAngleX", "name": "Head Angle X", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamAngleY", "name": "Head Angle Y", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamAngleZ", "name": "Head Angle Z", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamEyeLOpen", "name": "Left Eye Open", "min": 0.0, "max": 1.0, "default": 1.0},
            {"id": "ParamEyeROpen", "name": "Right Eye Open", "min": 0.0, "max": 1.0, "default": 1.0},
            {"id": "ParamEyeBallX", "name": "Eye Ball X", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamEyeBallY", "name": "Eye Ball Y", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamEyeLSmile", "name": "Left Eye Smile", "min": 0.0, "max": 1.0, "default": 0.0},
            {"id": "ParamEyeRSmile", "name": "Right Eye Smile", "min": 0.0, "max": 1.0, "default": 0.0},
            {"id": "ParamBrowLY", "name": "Left Brow Y", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamBrowRY", "name": "Right Brow Y", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamBrowLAngle", "name": "Left Brow Angle", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamBrowRAngle", "name": "Right Brow Angle", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamBrowLForm", "name": "Left Brow Form", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamBrowRForm", "name": "Right Brow Form", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamMouthOpenY", "name": "Mouth Open", "min": 0.0, "max": 1.0, "default": 0.0},
            {"id": "ParamMouthForm", "name": "Mouth Form", "min": -1.0, "max": 1.0, "default": 0.0},
            {"id": "ParamArmLA", "name": "Left Arm Angle", "min": -120.0, "max": 120.0, "default": 0.0},
            {"id": "ParamArmRA", "name": "Right Arm Angle", "min": -120.0, "max": 120.0, "default": 0.0},
            {"id": "ParamLegL", "name": "Left Leg", "min": -90.0, "max": 90.0, "default": 0.0},
            {"id": "ParamLegR", "name": "Right Leg", "min": -90.0, "max": 90.0, "default": 0.0},
            {"id": "ParamHairFront", "name": "Hair Front", "min": -45.0, "max": 45.0, "default": 0.0},
            {"id": "ParamHairSide", "name": "Hair Side", "min": -45.0, "max": 45.0, "default": 0.0},
            {"id": "ParamHairBack", "name": "Hair Back", "min": -45.0, "max": 45.0, "default": 0.0},
            {"id": "ParamClothA", "name": "Cloth A", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamClothB", "name": "Cloth B", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamClothC", "name": "Cloth C", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamAccessoryA", "name": "Accessory A", "min": -30.0, "max": 30.0, "default": 0.0},
            {"id": "ParamAccessoryB", "name": "Accessory B", "min": -30.0, "max": 30.0, "default": 0.0},
        ]

    def _bind_bones_to_meshes(self, bones: JsonList, meshes: JsonDict) -> JsonDict:
        weights: JsonDict = {}
        bone_mesh_mapping: dict[str, list[str]] = {
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
            bone_id = str(bone.get("id", ""))
            for mesh_name in bone_mesh_mapping.get(bone_id, []):
                if mesh_name in meshes:
                    weights[mesh_name] = {"bone": bone_id, "weight": 1.0}

        return weights

    def _create_hit_areas(self, segments: JsonDict) -> JsonList:
        hit_areas: JsonList = []
        body_parts = dict(segments.get("body_parts", {}))

        head_info = body_parts.get("head")
        if isinstance(head_info, dict) and "bounds" in head_info:
            hit_areas.append({"id": "HitHead", "name": "Head", "bounds": head_info["bounds"]})

        torso_info = body_parts.get("torso")
        if isinstance(torso_info, dict) and "bounds" in torso_info:
            hit_areas.append({"id": "HitBody", "name": "Body", "bounds": torso_info["bounds"]})

        for side in ("left", "right"):
            arm_info = body_parts.get(f"{side}_arm")
            if isinstance(arm_info, dict) and "bounds" in arm_info:
                hit_areas.append(
                    {
                        "id": f"Hit{side.capitalize()}Hand",
                        "name": f"{side.capitalize()} Hand",
                        "bounds": arm_info["bounds"],
                    }
                )

        return hit_areas

    def _create_parameter_groups(self, parameters: JsonList) -> JsonList:
        return [
            {
                "name": "Head",
                "ids": [
                    param["id"]
                    for param in parameters
                    if "Angle" in str(param["id"]) and "Body" not in str(param["id"])
                ],
            },
            {"name": "Body", "ids": [param["id"] for param in parameters if "Body" in str(param["id"])]},
            {"name": "Eyes", "ids": [param["id"] for param in parameters if "Eye" in str(param["id"])]},
            {"name": "Brows", "ids": [param["id"] for param in parameters if "Brow" in str(param["id"])]},
            {"name": "Mouth", "ids": [param["id"] for param in parameters if "Mouth" in str(param["id"])]},
            {"name": "Arms", "ids": [param["id"] for param in parameters if "Arm" in str(param["id"])]},
            {"name": "Legs", "ids": [param["id"] for param in parameters if "Leg" in str(param["id"])]},
            {"name": "Hair", "ids": [param["id"] for param in parameters if "Hair" in str(param["id"])]},
            {"name": "Clothing", "ids": [param["id"] for param in parameters if "Cloth" in str(param["id"])]},
            {
                "name": "Accessories",
                "ids": [param["id"] for param in parameters if "Accessory" in str(param["id"])],
            },
        ]
