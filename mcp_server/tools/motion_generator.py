"""Motion generation helpers for Live2D automation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from loguru import logger

JsonDict = dict[str, Any]
JsonList = list[JsonDict]
ParameterList = list[JsonDict]
ParameterSpec = tuple[float, float, float]


class MotionGenerator:
    """Generate simple motion curves from parameter definitions."""

    def __init__(self) -> None:
        self.fps = 30
        self.default_duration = 3.0

    def _parameter_specs(self, parameters: ParameterList) -> dict[str, ParameterSpec]:
        return {
            str(param.get("id", "")): (
                float(param.get("min", 0.0)),
                float(param.get("max", 1.0)),
                float(param.get("default", 0.0)),
            )
            for param in parameters
        }

    def _clamp(self, parameter_id: str, value: float, specs: dict[str, ParameterSpec]) -> float:
        if parameter_id not in specs:
            return value
        minimum, maximum, _ = specs[parameter_id]
        return max(minimum, min(maximum, value))

    def _set_value(
        self,
        values: dict[str, float],
        parameter_id: str,
        value: float,
        specs: dict[str, ParameterSpec],
    ) -> None:
        values[parameter_id] = float(self._clamp(parameter_id, value, specs))

    async def generate(self, rigging: JsonDict, motion_types: list[str]) -> JsonList:
        motions: JsonList = []
        parameters = list(rigging.get("parameters", []))
        specs = self._parameter_specs(parameters)

        logger.info(f"Generating motions: {motion_types}")
        for motion_type in motion_types:
            if motion_type == "idle":
                motions.extend(self._generate_idle_motions(parameters, specs))
            elif motion_type == "tap":
                motions.extend(self._generate_tap_motions(parameters, specs))
            elif motion_type == "move":
                motions.extend(self._generate_movement_motions(parameters, specs))
            elif motion_type == "emotional":
                motions.extend(self._generate_emotional_motions(parameters, specs))

        logger.info(f"Motion generation complete: {len(motions)} motions")
        return motions

    def _new_frame(self, timestamp: float) -> JsonDict:
        return {"time": timestamp, "values": {}}

    def _frame_values(self, frame: JsonDict) -> dict[str, float]:
        return cast(dict[str, float], frame["values"])

    def _build_motion(
        self,
        name: str,
        motion_type: str,
        duration: float,
        loop: bool,
        frames: JsonList,
    ) -> JsonDict:
        return {
            "name": name,
            "type": motion_type,
            "duration": duration,
            "fps": self.fps,
            "loop": loop,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": loop},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _generate_idle_motions(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonList:
        return [
            self._create_breathing_motion(parameters, specs),
            self._create_blink_motion(parameters, specs),
            self._create_sway_motion(parameters, specs),
        ]

    def _create_breathing_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonDict:
        frames: JsonList = []
        duration = 3.0
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            breath_value = float((np.sin(t * 2 * np.pi * 0.5) + 1) / 2)
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if param_id == "ParamBreath":
                    self._set_value(values, param_id, breath_value, specs)
                elif param_id == "ParamBodyAngleY":
                    self._set_value(values, param_id, breath_value * 2.0, specs)

            frames.append(frame)

        return self._build_motion("Idle_Breath", "idle", duration, True, frames)

    def _create_blink_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonDict:
        frames: JsonList = []
        duration = 0.3
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            eye_open = 1.0 - t * 2.0 if t < 0.5 else (t - 0.5) * 2.0
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if "Eye" in param_id and "Open" in param_id:
                    self._set_value(values, param_id, eye_open, specs)

            frames.append(frame)

        return self._build_motion("Idle_Blink", "idle", duration, False, frames)

    def _create_sway_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonDict:
        frames: JsonList = []
        duration = 4.0
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            sway_x = float(np.sin(t * 2 * np.pi * 0.25) * 3)
            sway_z = float(np.cos(t * 2 * np.pi * 0.2) * 2)
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if param_id == "ParamAngleX":
                    self._set_value(values, param_id, sway_x, specs)
                elif param_id == "ParamAngleZ":
                    self._set_value(values, param_id, sway_z, specs)

            frames.append(frame)

        return self._build_motion("Idle_Sway", "idle", duration, True, frames)

    def _generate_tap_motions(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonList:
        return [
            self._create_tap_motion(parameters, specs, "head", "Head"),
            self._create_tap_motion(parameters, specs, "body", "Body"),
            self._create_tap_motion(parameters, specs, "hand", "Hand"),
        ]

    def _create_tap_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
        area: str,
        area_name: str,
    ) -> JsonDict:
        frames: JsonList = []
        duration = 0.5
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if area == "head" and "Angle" in param_id and "Body" not in param_id:
                    self._set_value(values, param_id, float(np.sin(t * np.pi) * 5), specs)
                elif area == "body" and "Body" in param_id:
                    self._set_value(values, param_id, float(np.sin(t * np.pi) * 3), specs)

            frames.append(frame)

        return self._build_motion(f"Tap_{area_name}", "tap", duration, False, frames)

    def _generate_movement_motions(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonList:
        return [
            self._create_walk_motion(parameters, specs),
            self._create_wave_motion(parameters, specs),
            self._create_sit_motion(parameters, specs),
        ]

    def _create_walk_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonDict:
        frames: JsonList = []
        duration = 1.0
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            leg_angle = float(np.sin(t * 2 * np.pi) * 30)
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if param_id == "ParamLegL":
                    self._set_value(values, param_id, leg_angle, specs)
                elif param_id == "ParamLegR":
                    self._set_value(values, param_id, -leg_angle, specs)
                elif param_id == "ParamBodyAngleY":
                    self._set_value(values, param_id, float(abs(np.sin(t * 2 * np.pi)) * 3), specs)

            frames.append(frame)

        return self._build_motion("Move_Walk", "move", duration, True, frames)

    def _create_wave_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonDict:
        frames: JsonList = []
        duration = 2.0
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            arm_angle = -t * 200.0 if t < 0.3 else -60.0 + np.sin((t - 0.3) * 2 * np.pi * 2) * 20
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if param_id == "ParamArmRA":
                    self._set_value(values, param_id, float(arm_angle), specs)

            frames.append(frame)

        return self._build_motion("Move_Wave", "move", duration, False, frames)

    def _create_sit_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonDict:
        frames: JsonList = []
        duration = 1.5
        num_frames = int(duration * self.fps)

        for index in range(num_frames):
            t = index / num_frames
            body_y = t * 50.0
            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)

            for param in parameters:
                param_id = str(param.get("id", ""))
                if param_id == "ParamBodyAngleY":
                    self._set_value(values, param_id, body_y, specs)
                elif param_id in {"ParamLegL", "ParamLegR"}:
                    self._set_value(values, param_id, t * 90.0, specs)

            frames.append(frame)

        return self._build_motion("Move_Sit", "move", duration, False, frames)

    def _generate_emotional_motions(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
    ) -> JsonList:
        return [
            self._create_emotion_motion(parameters, specs, "happy", "Happy"),
            self._create_emotion_motion(parameters, specs, "surprised", "Surprised"),
            self._create_emotion_motion(parameters, specs, "thinking", "Thinking"),
        ]

    def _create_emotion_motion(
        self,
        parameters: ParameterList,
        specs: dict[str, ParameterSpec],
        emotion: str,
        emotion_name: str,
    ) -> JsonDict:
        frames: JsonList = []
        duration = 2.0
        num_frames = int(duration * self.fps)
        emotion_keyframes: dict[str, dict[str, float]] = {
            "happy": {
                "ParamEyeLSmile": 1.0,
                "ParamEyeRSmile": 1.0,
                "ParamMouthOpenY": 0.3,
                "ParamMouthForm": 1.0,
                "ParamBrowLY": -0.3,
                "ParamBrowRY": -0.3,
                "ParamAngleX": 0.0,
                "ParamAngleY": -5.0,
            },
            "surprised": {
                "ParamEyeLOpen": 1.5,
                "ParamEyeROpen": 1.5,
                "ParamMouthOpenY": 0.8,
                "ParamMouthForm": 0.0,
                "ParamBrowLY": 0.5,
                "ParamBrowRY": 0.5,
                "ParamBrowLAngle": -0.5,
                "ParamBrowRAngle": -0.5,
                "ParamAngleX": 0.0,
                "ParamAngleY": -10.0,
            },
            "thinking": {
                "ParamEyeLOpen": 0.6,
                "ParamEyeROpen": 0.6,
                "ParamMouthOpenY": 0.0,
                "ParamMouthForm": -0.3,
                "ParamBrowLY": 0.2,
                "ParamBrowRY": -0.3,
                "ParamBrowLAngle": 0.3,
                "ParamBrowRAngle": -0.3,
                "ParamAngleX": 5.0,
                "ParamAngleY": 5.0,
            },
        }
        target_values = emotion_keyframes.get(emotion, {})

        for index in range(num_frames):
            t = index / num_frames
            if t < 0.3:
                blend = t / 0.3
            elif t > 0.7:
                blend = (1 - t) / 0.3
            else:
                blend = 1.0

            frame = self._new_frame(t * duration)
            values = self._frame_values(frame)
            for param in parameters:
                param_id = str(param.get("id", ""))
                if param_id in target_values:
                    _, _, default = specs.get(param_id, (0.0, 1.0, 0.0))
                    target = target_values[param_id]
                    self._set_value(values, param_id, default + (target - default) * blend, specs)

            frames.append(frame)

        return self._build_motion(
            f"Emotion_{emotion_name}",
            "emotional",
            duration,
            False,
            frames,
        )

    def _convert_to_curves(self, frames: JsonList) -> JsonList:
        curves: JsonList = []
        all_params: set[str] = set()
        for frame in frames:
            all_params.update(self._frame_values(frame).keys())

        for param_id in all_params:
            curve: JsonDict = {"Target": "Model", "Id": param_id, "Segments": []}
            segments = cast(list[float], curve["Segments"])
            keyframes: list[dict[str, float]] = []
            for frame in frames:
                values = self._frame_values(frame)
                if param_id in values:
                    keyframes.append({"time": float(frame["time"]), "value": values[param_id]})

            if keyframes:
                segments.append(keyframes[0]["time"])
                segments.append(keyframes[0]["value"])
                for keyframe in keyframes[1:]:
                    segments.append(keyframe["time"])
                    segments.append(keyframe["value"])

            curves.append(curve)

        return curves
