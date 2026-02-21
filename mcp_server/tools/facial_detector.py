"""
面部详细检测器
检测眼睛、嘴巴、眉毛、鼻子等面部特征
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger


class FacialFeatureDetector:
    """面部特征检测器"""

    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.initialized = False

    async def initialize(self):
        """初始化检测器"""
        if self.initialized:
            return

        logger.info("Initializing facial feature detector...")

        # 尝试使用 OpenCV 的级联分类器
        try:
            # OpenCV 自带的人脸检测
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            self.initialized = True
            logger.info("OpenCV cascades loaded")
        except Exception as e:
            logger.warning(f"Failed to load OpenCV cascades: {e}")

        # 尝试使用 face_recognition 库
        try:
            import face_recognition

            self.face_recognition = face_recognition
            self.use_fr = True
            logger.info("face_recognition available")
        except ImportError:
            self.use_fr = False
            logger.info("face_recognition not available, using fallback")

    async def detect_features(self, image_path: str) -> Dict[str, Any]:
        """
        检测面部特征

        Returns:
            面部特征位置信息
        """
        await self.initialize()

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]

        features = {"has_face": False, "face_bounds": None, "parts": {}}

        # 方法1: 使用 OpenCV 级联分类器
        if self.face_cascade is not None and self.eye_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # 取最大的脸
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                features["has_face"] = True
                features["face_bounds"] = {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                }

                # 在面部区域内检测眼睛
                roi_gray = gray[y : y + h, x : x + w]
                roi_color = image[y : y + h, x : x + w]

                eyes = self.eye_cascade.detectMultiScale(roi_gray)

                # 分类左右眼
                left_eyes = []
                right_eyes = []

                for ex, ey, ew, eh in eyes:
                    if ex < w / 2:
                        left_eyes.append((ex, ey, ew, eh))
                    else:
                        right_eyes.append((ex, ey, ew, eh))

                # 取最大的眼睛
                if left_eyes:
                    ex, ey, ew, eh = max(left_eyes, key=lambda e: e[2] * e[3])
                    features["parts"]["left_eye"] = {
                        "bounds": {"x": x + ex, "y": y + ey, "width": ew, "height": eh},
                        "center": (x + ex + ew // 2, y + ey + eh // 2),
                    }

            if right_eyes:
                ex, ey, ew, eh = max(right_eyes, key=lambda e: e[2] * e[3])
                features["parts"]["right_eye"] = {
                    "bounds": {"x": x + ex, "y": y + ey, "width": ew, "height": eh},
                    "center": (x + ex + ew // 2, y + ey + eh // 2),
                }

            # 估计嘴巴位置（面部下三分之一）
            mouth_y = y + int(h * 0.6)
            mouth_h = int(h * 0.25)
            features["parts"]["mouth"] = {
                "bounds": {
                    "x": x + int(w * 0.2),
                    "y": mouth_y,
                    "width": int(w * 0.6),
                    "height": mouth_h,
                },
                "center": (x + w // 2, mouth_y + mouth_h // 2),
            }

            # 估计鼻子位置（面部中间）
            nose_y = y + int(h * 0.4)
            nose_h = int(h * 0.2)
            features["parts"]["nose"] = {
                "bounds": {
                    "x": x + int(w * 0.35),
                    "y": nose_y,
                    "width": int(w * 0.3),
                    "height": nose_h,
                },
                "center": (x + w // 2, nose_y + nose_h // 2),
            }

            # 估计眉毛位置（眼睛上方）
            if "left_eye" in features["parts"]:
                le = features["parts"]["left_eye"]
                features["parts"]["left_eyebrow"] = {
                    "bounds": {
                        "x": le["bounds"]["x"],
                        "y": le["bounds"]["y"] - int(le["bounds"]["height"] * 0.8),
                        "width": le["bounds"]["width"],
                        "height": int(le["bounds"]["height"] * 0.6),
                    }
                }

            if "right_eye" in features["parts"]:
                re = features["parts"]["right_eye"]
                features["parts"]["right_eyebrow"] = {
                    "bounds": {
                        "x": re["bounds"]["x"],
                        "y": re["bounds"]["y"] - int(re["bounds"]["height"] * 0.8),
                        "width": re["bounds"]["width"],
                        "height": int(re["bounds"]["height"] * 0.6),
                    }
                }

        # 方法2: 使用 face_recognition (如果可用)
        if not features["has_face"] and self.use_fr:
            try:
                import face_recognition

                img = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(img)

                if len(encodings) > 0:
                    features["has_face"] = True
                    # 使用默认位置
                    features["face_bounds"] = {
                        "x": width // 4,
                        "y": height // 6,
                        "width": width // 2,
                        "height": height // 2,
                    }
            except Exception as e:
                logger.warning(f"face_recognition error: {e}")

        # 方法3: 基于面部特征的启发式方法（动漫角色）
        if not features["has_face"]:
            logger.info("Using heuristic facial detection for anime character...")
            features = self._detect_anime_face(image, width, height)

        return features

    def _detect_anime_face(self, image: np.ndarray, width: int, height: int) -> Dict:
        """动漫角色面部检测（启发式方法）"""
        features = {
            "has_face": True,  # 假设有脸
            "face_bounds": {
                "x": int(width * 0.22),
                "y": int(height * 0.02),
                "width": int(width * 0.56),
                "height": int(height * 0.48),
            },
            "parts": {},
        }

        # 动漫角色面部比例（头部占比较大）
        face_x = int(width * 0.22)
        face_y = int(height * 0.02)
        face_w = int(width * 0.56)
        face_h = int(height * 0.48)

        # 眼睛（面部上三分之一，两侧）
        eye_y = face_y + int(face_h * 0.35)
        eye_h = int(face_h * 0.12)

        # 左眼
        features["parts"]["left_eye"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.15),
                "y": eye_y,
                "width": int(face_w * 0.18),
                "height": eye_h,
            },
            "center": (face_x + int(face_w * 0.24), eye_y + eye_h // 2),
        }

        # 右眼
        features["parts"]["right_eye"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.67),
                "y": eye_y,
                "width": int(face_w * 0.18),
                "height": eye_h,
            },
            "center": (face_x + int(face_w * 0.76), eye_y + eye_h // 2),
        }

        # 眼睛下方的高光（眨眼用）
        features["parts"]["left_eye_highlight"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.18),
                "y": eye_y + int(eye_h * 0.2),
                "width": int(face_w * 0.06),
                "height": int(eye_h * 0.3),
            }
        }

        features["parts"]["right_eye_highlight"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.70),
                "y": eye_y + int(eye_h * 0.2),
                "width": int(face_w * 0.06),
                "height": int(eye_h * 0.3),
            }
        }

        # 嘴巴（面部下四分之一）
        mouth_y = face_y + int(face_h * 0.65)
        mouth_h = int(face_h * 0.12)
        features["parts"]["mouth"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.35),
                "y": mouth_y,
                "width": int(face_w * 0.3),
                "height": mouth_h,
            },
            "center": (face_x + face_w // 2, mouth_y + mouth_h // 2),
        }

        # 鼻子（面部中间偏上）
        nose_y = face_y + int(face_h * 0.45)
        nose_h = int(face_h * 0.1)
        features["parts"]["nose"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.42),
                "y": nose_y,
                "width": int(face_w * 0.16),
                "height": nose_h,
            }
        }

        # 眉毛（眼睛上方）
        brow_y = face_y + int(face_h * 0.25)
        brow_h = int(face_h * 0.08)

        features["parts"]["left_eyebrow"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.12),
                "y": brow_y,
                "width": int(face_w * 0.22),
                "height": brow_h,
            }
        }

        features["parts"]["right_eyebrow"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.66),
                "y": brow_y,
                "width": int(face_w * 0.22),
                "height": brow_h,
            }
        }

        # 脸颊红晕（动漫特色）
        features["parts"]["left_blush"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.10),
                "y": face_y + int(face_h * 0.45),
                "width": int(face_w * 0.15),
                "height": int(face_h * 0.1),
            }
        }

        features["parts"]["right_blush"] = {
            "bounds": {
                "x": face_x + int(face_w * 0.75),
                "y": face_y + int(face_h * 0.45),
                "width": int(face_w * 0.15),
                "height": int(face_h * 0.1),
            }
        }

        return features

    async def extract_face_parts(self, image_path: str, output_dir: str) -> List[Dict]:
        """提取面部各部位图像"""
        features = await self.detect_features(image_path)

        if not features["has_face"]:
            logger.warning("No face detected")
            return []

        # 加载原图
        image = Image.open(image_path).convert("RGBA")

        layers = []

        for part_name, part_data in features["parts"].items():
            bounds = part_data.get("bounds")
            if bounds is None:
                continue

            # 提取区域
            x = max(0, int(bounds["x"]))
            y = max(0, int(bounds["y"]))
            w = min(int(bounds["width"]), image.width - x)
            h = min(int(bounds["height"]), image.height - y)

            if w <= 0 or h <= 0:
                continue

            # 裁剪
            part_img = image.crop((x, y, x + w, y + h))

            # 保存
            output_path = f"{output_dir}/face_{part_name}.png"
            part_img.save(output_path, "PNG")

            layers.append(
                {
                    "name": part_name,
                    "path": output_path,
                    "bounds": bounds,
                    "z_order": self._get_z_order(part_name),
                }
            )

        logger.info(f"Extracted {len(layers)} face parts")
        return layers

    def _get_z_order(self, part_name: str) -> int:
        """获取 Z 顺序"""
        order = {
            "left_eyebrow": 101,
            "right_eyebrow": 102,
            "left_eye": 103,
            "right_eye": 104,
            "left_eye_highlight": 105,
            "right_eye_highlight": 106,
            "nose": 107,
            "mouth": 108,
            "left_blush": 109,
            "right_blush": 110,
        }
        return order.get(part_name, 110)
