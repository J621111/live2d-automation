"""Physics configuration helpers for generated Live2D rigs."""

from __future__ import annotations

from typing import Any

from loguru import logger

JsonDict = dict[str, Any]
JsonList = list[JsonDict]


class PhysicsSetup:
    """Create simplified physics groups for generated rigs."""

    def __init__(self) -> None:
        self.gravity = 9.8
        self.default_damping = 0.9
        self.default_stiffness = 0.3

    def _parameter_ids(self, rigging: JsonDict) -> set[str]:
        return {str(param.get("id", "")) for param in rigging.get("parameters", [])}

    def _filter_known_ids(self, parameter_ids: set[str], candidates: JsonList) -> JsonList:
        return [
            candidate for candidate in candidates if str(candidate.get("id", "")) in parameter_ids
        ]

    async def configure(self, rigging: JsonDict, segments: JsonDict) -> JsonDict:
        del segments
        logger.info("Configuring Live2D physics...")

        parameter_ids = self._parameter_ids(rigging)
        physics_groups: JsonList = []
        for group in (
            self._create_hair_physics(parameter_ids),
            self._create_clothing_physics(parameter_ids),
            self._create_accessory_physics(parameter_ids),
        ):
            if group:
                physics_groups.append(group)

        physics_config: JsonDict = {
            "version": 3,
            "groups": physics_groups,
            "settings": {
                "gravity": self.gravity,
                "wind": {"x": 0.0, "y": 0.0},
                "enable": True,
            },
        }

        logger.info(f"Physics configuration complete: {len(physics_groups)} groups")
        return physics_config

    def _create_hair_physics(self, parameter_ids: set[str]) -> JsonDict:
        hair_group: JsonDict = {
            "name": "Hair",
            "id": "HairPhysics",
            "input": self._filter_known_ids(
                parameter_ids,
                [
                    {"id": "ParamAngleX", "weight": 0.5},
                    {"id": "ParamAngleY", "weight": 0.3},
                    {"id": "ParamAngleZ", "weight": 0.3},
                ],
            ),
            "output": self._filter_known_ids(
                parameter_ids,
                [
                    {"id": "ParamHairFront", "type": "angle", "weight": 1.0},
                    {"id": "ParamHairSide", "type": "angle", "weight": 1.0},
                    {"id": "ParamHairBack", "type": "angle", "weight": 1.0},
                ],
            ),
            "particles": self._create_hair_particles(),
        }
        return hair_group if hair_group["output"] else {}

    def _create_hair_particles(self) -> JsonList:
        particles: JsonList = []
        base_x, base_y = 0.0, -100.0

        for index in range(5):
            particle: JsonDict = {
                "index": index,
                "position": {"x": base_x, "y": base_y - index * 20.0},
                "mass": 0.5 + index * 0.1,
                "damping": self.default_damping,
                "stiffness": self.default_stiffness - index * 0.05,
                "radius": 10.0 + index * 2.0,
                "constraint": None,
            }
            if index > 0:
                particle["constraint"] = {
                    "type": "distance",
                    "target": index - 1,
                    "distance": 20.0,
                }
            particles.append(particle)

        return particles

    def _create_clothing_physics(self, parameter_ids: set[str]) -> JsonDict:
        clothing_group: JsonDict = {
            "name": "Clothing",
            "id": "ClothingPhysics",
            "input": self._filter_known_ids(
                parameter_ids,
                [
                    {"id": "ParamBodyAngleX", "weight": 0.3},
                    {"id": "ParamBodyAngleY", "weight": 0.3},
                    {"id": "ParamBreath", "weight": 0.3},
                ],
            ),
            "output": self._filter_known_ids(
                parameter_ids,
                [
                    {"id": "ParamClothA", "type": "angle", "weight": 0.8},
                    {"id": "ParamClothB", "type": "angle", "weight": 0.8},
                    {"id": "ParamClothC", "type": "angle", "weight": 0.8},
                ],
            ),
            "particles": self._create_cloth_particles(),
        }
        return clothing_group if clothing_group["output"] else {}

    def _create_cloth_particles(self) -> JsonList:
        particles: JsonList = []
        rows, cols = 3, 3
        spacing = 30.0

        for row in range(rows):
            for col in range(cols):
                constraints: JsonList = []
                if row > 0:
                    constraints.append(
                        {
                            "target": (row - 1) * cols + col,
                            "type": "distance",
                            "distance": spacing,
                        }
                    )
                if col > 0:
                    constraints.append(
                        {
                            "target": row * cols + (col - 1),
                            "type": "distance",
                            "distance": spacing,
                        }
                    )

                particles.append(
                    {
                        "index": row * cols + col,
                        "position": {"x": (col - cols // 2) * spacing, "y": row * spacing},
                        "mass": 0.3,
                        "damping": 0.85,
                        "stiffness": 0.4,
                        "fixed": row == 0,
                        "constraints": constraints,
                    }
                )

        return particles

    def _create_accessory_physics(self, parameter_ids: set[str]) -> JsonDict:
        accessory_group: JsonDict = {
            "name": "Accessories",
            "id": "AccessoryPhysics",
            "input": self._filter_known_ids(
                parameter_ids,
                [
                    {"id": "ParamAngleX", "weight": 0.6},
                    {"id": "ParamAngleY", "weight": 0.6},
                ],
            ),
            "output": self._filter_known_ids(
                parameter_ids,
                [
                    {"id": "ParamAccessoryA", "type": "angle", "weight": 1.0},
                    {"id": "ParamAccessoryB", "type": "angle", "weight": 1.0},
                ],
            ),
            "particles": [
                {
                    "index": 0,
                    "position": {"x": 0.0, "y": -150.0},
                    "mass": 0.2,
                    "damping": 0.95,
                    "stiffness": 0.5,
                    "length": 50.0,
                    "constraint": {"type": "pendulum", "pivot": {"x": 0.0, "y": -100.0}},
                }
            ],
        }
        return accessory_group if accessory_group["output"] else {}
