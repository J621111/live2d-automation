"""
动作生成器
自动生成 Live2D 动作文件
"""

import numpy as np
from typing import Dict, List, Any
from loguru import logger


class MotionGenerator:
    """Live2D 动作生成器"""

    def __init__(self):
        self.fps = 30
        self.default_duration = 3.0  # 默认动作时长（秒）

    async def generate(self, rigging: Dict, motion_types: List[str]) -> List[Dict]:
        """
        生成动作文件

        Args:
            rigging: 绑定信息
            motion_types: 动作类型列表

        Returns:
            生成的动作文件列表
        """
        motions = []
        parameters = rigging.get("parameters", [])

        logger.info(f"生成动作: {motion_types}")

        for motion_type in motion_types:
            if motion_type == "idle":
                idle_motions = self._generate_idle_motions(parameters)
                motions.extend(idle_motions)
            elif motion_type == "tap":
                tap_motions = self._generate_tap_motions(parameters)
                motions.extend(tap_motions)
            elif motion_type == "move":
                move_motions = self._generate_movement_motions(parameters)
                motions.extend(move_motions)
            elif motion_type == "emotional":
                emotional_motions = self._generate_emotional_motions(parameters)
                motions.extend(emotional_motions)

        logger.info(f"动作生成完成: {len(motions)} 个动作")
        return motions

    def _generate_idle_motions(self, parameters: List[Dict]) -> List[Dict]:
        """生成空闲动作"""
        motions = []

        # 1. 呼吸动作
        breath_motion = self._create_breathing_motion(parameters)
        motions.append(breath_motion)

        # 2. 眨眼动作
        blink_motion = self._create_blink_motion(parameters)
        motions.append(blink_motion)

        # 3. 待机晃动
        sway_motion = self._create_sway_motion(parameters)
        motions.append(sway_motion)

        return motions

    def _create_breathing_motion(self, parameters: List[Dict]) -> Dict:
        """创建呼吸动作"""
        frames = []
        duration = 3.0
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames
            # 正弦波呼吸
            breath_value = (np.sin(t * 2 * np.pi * 0.5) + 1) / 2

            frame = {"time": t * duration, "values": {}}

            # 设置呼吸参数
            for param in parameters:
                if param["id"] == "ParamBreath":
                    frame["values"][param["id"]] = breath_value
                elif param["id"] in ["ParamBodyAngleY"]:
                    # 身体轻微上下移动
                    frame["values"][param["id"]] = breath_value * 2

            frames.append(frame)

        return {
            "name": "Idle_Breath",
            "type": "idle",
            "duration": duration,
            "fps": self.fps,
            "loop": True,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": True},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _create_blink_motion(self, parameters: List[Dict]) -> Dict:
        """创建眨眼动作"""
        frames = []
        duration = 0.3  # 眨眼很快
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames

            # 眨眼曲线：闭眼睁眼
            if t < 0.5:
                eye_open = 1 - t * 2  # 闭眼
            else:
                eye_open = (t - 0.5) * 2  # 睁眼

            frame = {"time": t * duration, "values": {}}

            # 设置眼睛开合
            for param in parameters:
                if "Eye" in param["id"] and "Open" in param["id"]:
                    frame["values"][param["id"]] = max(0, min(1, eye_open))

            frames.append(frame)

        return {
            "name": "Idle_Blink",
            "type": "idle",
            "duration": duration,
            "fps": self.fps,
            "loop": False,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": False},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _create_sway_motion(self, parameters: List[Dict]) -> Dict:
        """创建待机晃动动作"""
        frames = []
        duration = 4.0
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames

            # 轻微的身体晃动
            sway_x = np.sin(t * 2 * np.pi * 0.25) * 3
            sway_z = np.cos(t * 2 * np.pi * 0.2) * 2

            frame = {"time": t * duration, "values": {}}

            for param in parameters:
                if param["id"] == "ParamAngleX":
                    frame["values"][param["id"]] = sway_x
                elif param["id"] == "ParamAngleZ":
                    frame["values"][param["id"]] = sway_z

            frames.append(frame)

        return {
            "name": "Idle_Sway",
            "type": "idle",
            "duration": duration,
            "fps": self.fps,
            "loop": True,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": True},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _generate_tap_motions(self, parameters: List[Dict]) -> List[Dict]:
        """生成点击互动动作"""
        motions = []

        # 点击头部
        head_tap = self._create_tap_motion(parameters, "head", "Head")
        motions.append(head_tap)

        # 点击身体
        body_tap = self._create_tap_motion(parameters, "body", "Body")
        motions.append(body_tap)

        # 点击手
        hand_tap = self._create_tap_motion(parameters, "hand", "Hand")
        motions.append(hand_tap)

        return motions

    def _create_tap_motion(
        self, parameters: List[Dict], area: str, area_name: str
    ) -> Dict:
        """创建点击动作"""
        frames = []
        duration = 0.5
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames

            # 缩放效果
            if t < 0.3:
                scale = 1 - t * 0.2  # 缩小
            else:
                scale = 0.94 + (t - 0.3) * 0.1  # 恢复

            frame = {"time": t * duration, "values": {}}

            # 根据点击区域设置参数
            for param in parameters:
                if (
                    area == "head"
                    and "Angle" in param["id"]
                    and "Body" not in param["id"]
                ):
                    frame["values"][param["id"]] = np.sin(t * np.pi) * 5
                elif area == "body" and "Body" in param["id"]:
                    frame["values"][param["id"]] = np.sin(t * np.pi) * 3

            frames.append(frame)

        return {
            "name": f"Tap_{area_name}",
            "type": "tap",
            "duration": duration,
            "fps": self.fps,
            "loop": False,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": False},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _generate_movement_motions(self, parameters: List[Dict]) -> List[Dict]:
        """生成移动动作"""
        motions = []

        # 行走动作
        walk_motion = self._create_walk_motion(parameters)
        motions.append(walk_motion)

        # 挥手动作
        wave_motion = self._create_wave_motion(parameters)
        motions.append(wave_motion)

        # 坐下动作
        sit_motion = self._create_sit_motion(parameters)
        motions.append(sit_motion)

        return motions

    def _create_walk_motion(self, parameters: List[Dict]) -> Dict:
        """创建行走动作"""
        frames = []
        duration = 1.0
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames

            frame = {"time": t * duration, "values": {}}

            # 腿部摆动
            leg_angle = np.sin(t * 2 * np.pi) * 30

            for param in parameters:
                if param["id"] == "ParamLegL":
                    frame["values"][param["id"]] = leg_angle
                elif param["id"] == "ParamLegR":
                    frame["values"][param["id"]] = -leg_angle
                elif param["id"] == "ParamBodyAngleY":
                    # 身体轻微上下
                    frame["values"][param["id"]] = abs(np.sin(t * 2 * np.pi)) * 3

            frames.append(frame)

        return {
            "name": "Move_Walk",
            "type": "move",
            "duration": duration,
            "fps": self.fps,
            "loop": True,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": True},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _create_wave_motion(self, parameters: List[Dict]) -> Dict:
        """创建挥手动作"""
        frames = []
        duration = 2.0
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames

            frame = {"time": t * duration, "values": {}}

            # 手臂抬起和挥动
            if t < 0.3:
                arm_angle = -t * 200  # 抬起
            else:
                arm_angle = -60 + np.sin((t - 0.3) * 2 * np.pi * 2) * 20  # 挥动

            for param in parameters:
                if param["id"] == "ParamArmRA":
                    frame["values"][param["id"]] = arm_angle

            frames.append(frame)

        return {
            "name": "Move_Wave",
            "type": "move",
            "duration": duration,
            "fps": self.fps,
            "loop": False,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": False},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _create_sit_motion(self, parameters: List[Dict]) -> Dict:
        """创建坐下动作"""
        frames = []
        duration = 1.5
        num_frames = int(duration * self.fps)

        for i in range(num_frames):
            t = i / num_frames

            frame = {"time": t * duration, "values": {}}

            # 身体下蹲
            body_y = t * 50

            for param in parameters:
                if param["id"] == "ParamBodyAngleY":
                    frame["values"][param["id"]] = body_y
                elif param["id"] in ["ParamLegL", "ParamLegR"]:
                    # 腿部弯曲
                    frame["values"][param["id"]] = t * 90

            frames.append(frame)

        return {
            "name": "Move_Sit",
            "type": "move",
            "duration": duration,
            "fps": self.fps,
            "loop": False,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": False},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _generate_emotional_motions(self, parameters: List[Dict]) -> List[Dict]:
        """生成情感动作"""
        motions = []

        # 开心
        happy_motion = self._create_emotion_motion(parameters, "happy", "Happy")
        motions.append(happy_motion)

        # 惊讶
        surprised_motion = self._create_emotion_motion(
            parameters, "surprised", "Surprised"
        )
        motions.append(surprised_motion)

        # 思考
        thinking_motion = self._create_emotion_motion(
            parameters, "thinking", "Thinking"
        )
        motions.append(thinking_motion)

        return motions

    def _create_emotion_motion(
        self, parameters: List[Dict], emotion: str, emotion_name: str
    ) -> Dict:
        """创建情感动作"""
        frames = []
        duration = 2.0
        num_frames = int(duration * self.fps)

        # 定义不同情感的关键帧
        emotion_keyframes = {
            "happy": {
                "ParamEyeLSmile": 1.0,
                "ParamEyeRSmile": 1.0,
                "ParamMouthOpenY": 0.3,
                "ParamMouthForm": 1.0,
                "ParamBrowLY": -0.3,
                "ParamBrowRY": -0.3,
                "ParamAngleX": 0,
                "ParamAngleY": -5,
            },
            "surprised": {
                "ParamEyeLOpen": 1.5,
                "ParamEyeROpen": 1.5,
                "ParamMouthOpenY": 0.8,
                "ParamMouthForm": 0,
                "ParamBrowLY": 0.5,
                "ParamBrowRY": 0.5,
                "ParamBrowLAngle": -0.5,
                "ParamBrowRAngle": -0.5,
                "ParamAngleX": 0,
                "ParamAngleY": -10,
            },
            "thinking": {
                "ParamEyeLOpen": 0.6,
                "ParamEyeROpen": 0.6,
                "ParamMouthOpenY": 0,
                "ParamMouthForm": -0.3,
                "ParamBrowLY": 0.2,
                "ParamBrowRY": -0.3,
                "ParamBrowLAngle": 0.3,
                "ParamBrowRAngle": -0.3,
                "ParamAngleX": 5,
                "ParamAngleY": 5,
            },
        }

        target_values = emotion_keyframes.get(emotion, {})

        for i in range(num_frames):
            t = i / num_frames

            # 缓入缓出
            if t < 0.3:
                blend = t / 0.3
            elif t > 0.7:
                blend = (1 - t) / 0.3
            else:
                blend = 1

            frame = {"time": t * duration, "values": {}}

            for param in parameters:
                if param["id"] in target_values:
                    target = target_values[param["id"]]
                    default = param.get("default", 0)
                    frame["values"][param["id"]] = default + (target - default) * blend

            frames.append(frame)

        return {
            "name": f"Emotion_{emotion_name}",
            "type": "emotional",
            "duration": duration,
            "fps": self.fps,
            "loop": False,
            "data": {
                "Version": 3,
                "Meta": {"Duration": duration, "Fps": self.fps, "Loop": False},
                "Curves": self._convert_to_curves(frames),
            },
        }

    def _convert_to_curves(self, frames: List[Dict]) -> List[Dict]:
        """将帧数据转换为 Live2D 曲线格式"""
        curves = []

        # 收集所有参数
        all_params = set()
        for frame in frames:
            all_params.update(frame["values"].keys())

        for param_id in all_params:
            curve = {"Target": "Model", "Id": param_id, "Segments": []}

            # 提取该参数的关键帧
            keyframes = []
            for frame in frames:
                if param_id in frame["values"]:
                    keyframes.append(
                        {"time": frame["time"], "value": frame["values"][param_id]}
                    )

            # 生成曲线段（线性插值）
            if keyframes:
                curve["Segments"].append(keyframes[0]["time"])
                curve["Segments"].append(keyframes[0]["value"])

                for i in range(1, len(keyframes)):
                    kf = keyframes[i]
                    curve["Segments"].append(kf["time"])
                    curve["Segments"].append(kf["value"])

            curves.append(curve)

        return curves
