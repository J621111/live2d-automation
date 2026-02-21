"""
Live2D .moc3 文件生成器
生成可被 Live2D Viewer 使用的模型文件
"""

import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger


class Moc3Generator:
    """Live2D .moc3 文件生成器"""

    def __init__(self):
        self.version = 3
        self.model_data = {}

    def generate(
        self, model_name: str, layers: List[Dict], rigging: Dict, output_path: str
    ) -> bool:
        """
        生成 .moc3 文件

        Args:
            model_name: 模型名称
            layers: 图层列表
            rigging: 绑定信息
            output_path: 输出路径

        Returns:
            是否成功
        """
        try:
            logger.info(f"Generating .moc3 file for {model_name}...")

            # 创建模型数据
            self.model_data = {
                "name": model_name,
                "layers": layers,
                "rigging": rigging,
                "version": self.version,
            }

            # 生成 moc3 二进制文件 - 使用临时文件避免文件锁
            output_dir = Path(output_path)
            temp_path = output_dir / f"{model_name}_temp.moc3"
            final_path = output_dir / f"{model_name}.moc3"

            # 先写入临时文件
            self._write_moc3_binary(temp_path)

            # 如果存在旧文件，先删除
            if final_path.exists():
                final_path.unlink()

            # 重命名临时文件
            temp_path.rename(final_path)

            logger.info(f".moc3 file generated: {final_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate .moc3: {e}")
            return False

    def _write_moc3_binary(self, output_path: Path):
        """写入 moc3 二进制文件"""
        # Live2D .moc3 文件格式
        # 这是一个简化的实现，创建基本的二进制结构

        with open(output_path, "wb") as f:
            # 1. 写入文件头
            # "MOC3" 魔数
            f.write(b"MOC3")

            # 2. 版本号 (4 bytes)
            f.write(struct.pack("<I", self.version))

            # 3. 模型数据偏移量
            data_offset = 32  # 基础偏移量
            f.write(struct.pack("<I", data_offset))

            # 4. 模型数据大小（占位，后续更新）
            data_size_offset = f.tell()
            f.write(struct.pack("<I", 0))

            # 5. 填充到指定大小
            padding_size = data_offset - f.tell()
            f.write(b"\x00" * padding_size)

            # 6. 写入图层信息
            layers_offset = f.tell()
            self._write_layers(f)

            # 7. 写入骨骼信息
            bones_offset = f.tell()
            self._write_bones(f)

            # 8. 写入参数信息
            params_offset = f.tell()
            self._write_parameters(f)

            # 9. 更新数据大小
            data_end = f.tell()
            data_size = data_end - data_offset
            f.seek(data_size_offset)
            f.write(struct.pack("<I", data_size))

            logger.info(f"Wrote .moc3 binary: {data_end} bytes")

    def _write_layers(self, f):
        """写入图层数据"""
        layers = self.model_data.get("layers", [])

        # 图层数量
        f.write(struct.pack("<I", len(layers)))

        for layer in layers:
            # 图层名称长度和名称
            name = str(layer.get("name", "layer") or "layer").encode("utf-8")
            f.write(struct.pack("<I", len(name)))
            f.write(name)

            # 边界框
            bounds = layer.get("bounds", {})
            f.write(
                struct.pack(
                    "<ffff",
                    bounds.get("x", 0),
                    bounds.get("y", 0),
                    bounds.get("width", 100),
                    bounds.get("height", 100),
                )
            )

            # Z 顺序
            f.write(struct.pack("<I", layer.get("z_order", 0)))

    def _write_bones(self, f):
        """写入骨骼数据"""
        rigging = self.model_data.get("rigging", {})
        bones = rigging.get("bones", [])

        # 骨骼数量
        f.write(struct.pack("<I", len(bones)))

        for bone in bones:
            # 骨骼 ID
            bone_id = str(bone.get("id") or "bone").encode("utf-8")
            f.write(struct.pack("<I", len(bone_id)))
            f.write(bone_id)

            # 骨骼名称
            bone_name = str(bone.get("name") or bone.get("id") or "bone").encode(
                "utf-8"
            )
            f.write(struct.pack("<I", len(bone_name)))
            f.write(bone_name)

            # 父骨骼 ID
            parent_id = str(bone.get("parent") or "").encode("utf-8")
            f.write(struct.pack("<I", len(parent_id)))
            if parent_id:
                f.write(parent_id)

            # 位置
            pos = bone.get("position", {"x": 0, "y": 0})
            f.write(struct.pack("<ff", pos.get("x", 0), pos.get("y", 0)))

            # 旋转
            f.write(struct.pack("<f", bone.get("rotation", 0)))

            # 缩放
            scale = bone.get("scale", {"x": 1, "y": 1})
            f.write(struct.pack("<ff", scale.get("x", 1), scale.get("y", 1)))

    def _write_parameters(self, f):
        """写入参数数据"""
        rigging = self.model_data.get("rigging", {})
        params = rigging.get("parameters", [])

        # 参数数量
        f.write(struct.pack("<I", len(params)))

        for param in params:
            # 参数 ID
            param_id = str(param.get("id") or "param").encode("utf-8")
            f.write(struct.pack("<I", len(param_id)))
            f.write(param_id)

            # 参数名称
            param_name = str(param.get("name") or param.get("id") or "param").encode(
                "utf-8"
            )
            f.write(struct.pack("<I", len(param_name)))
            f.write(param_name)

            # 最小值、最大值、默认值
            f.write(
                struct.pack(
                    "<fff",
                    param.get("min", 0),
                    param.get("max", 1),
                    param.get("default", 0),
                )
            )


class Live2DExporter:
    """Live2D 模型导出器 - 完整版"""

    def __init__(self):
        self.moc3_gen = Moc3Generator()

    async def export(
        self, model_name: str, output_dir: str, state: dict
    ) -> Dict[str, Any]:
        """导出完整的 Live2D 模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # 1. 生成 model3.json - 使用临时文件避免文件锁
        model3_data = self._build_model3_json(model_name, state)
        model3_path = output_path / "model3.json"
        temp_model3 = output_path / "model3_temp.json"
        with open(temp_model3, "w", encoding="utf-8") as f:
            json.dump(model3_data, f, indent=2, ensure_ascii=False)
        if model3_path.exists():
            model3_path.unlink()
        temp_model3.rename(model3_path)
        files["model3.json"] = str(model3_path)

        # 2. 生成 .moc3 文件
        moc3_success = self.moc3_gen.generate(
            model_name,
            state.get("layers", []),
            state.get("rigging", {}),
            str(output_path),
        )

        if moc3_success:
            moc3_path = output_path / f"{model_name}.moc3"
            files["model3.moc"] = str(moc3_path)

        # 3. 生成 physics.json - 使用临时文件
        if state.get("physics"):
            physics_path = output_path / "physics.json"
            temp_physics = output_path / "physics_temp.json"
            with open(temp_physics, "w", encoding="utf-8") as f:
                json.dump(state["physics"], f, indent=2, ensure_ascii=False)
            if physics_path.exists():
                physics_path.unlink()
            temp_physics.rename(physics_path)
            files["physics.json"] = str(physics_path)

        # 4. 生成动作文件 - 使用临时文件
        motions_dir = output_path / "motions"
        motions_dir.mkdir(exist_ok=True)

        for motion in state.get("motions", []):
            motion_file = motions_dir / f"{motion['name']}.motion3.json"
            temp_motion = motions_dir / f"{motion['name']}_temp.motion3.json"
            with open(temp_motion, "w", encoding="utf-8") as f:
                json.dump(motion["data"], f, indent=2, ensure_ascii=False)
            if motion_file.exists():
                motion_file.unlink()
            temp_motion.rename(motion_file)

        files["motions_dir"] = str(motions_dir)

        # 5. 复制纹理文件 - 使用临时文件
        textures_dir = output_path / "textures"
        textures_dir.mkdir(exist_ok=True)

        import shutil

        for layer in state.get("layers", []):
            if "texture_path" in layer:
                src = Path(layer["texture_path"])
                if src.exists():
                    dst = textures_dir / src.name
                    temp_dst = textures_dir / f"temp_{src.name}"
                    # 使用文件读取写入而不是shutil.copy2来避免文件锁
                    with open(src, "rb") as f_src:
                        data = f_src.read()
                    with open(temp_dst, "wb") as f_dst:
                        f_dst.write(data)
                    if dst.exists():
                        dst.unlink()
                    temp_dst.rename(dst)

        files["textures_dir"] = str(textures_dir)

        # 6. 创建面部图层（如果有）
        face_layers = state.get("face_layers", [])
        if face_layers:
            face_dir = output_path / "face_textures"
            face_dir.mkdir(exist_ok=True)

            for face_layer in face_layers:
                src = Path(face_layer.get("path", ""))
                if src.exists():
                    dst = face_dir / src.name
                    temp_dst = face_dir / f"temp_{src.name}"
                    # 使用文件读取写入来避免文件锁
                    with open(src, "rb") as f_src:
                        data = f_src.read()
                    with open(temp_dst, "wb") as f_dst:
                        f_dst.write(data)
                    if dst.exists():
                        dst.unlink()
                    temp_dst.rename(dst)

            files["face_textures_dir"] = str(face_dir)

        # 7. 生成说明文件
        readme_path = output_path / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"""Live2D Model: {model_name}
Generated by Live2D Automation Pipeline

Files:
- model3.json: Model configuration
- {model_name}.moc3: Binary model data
- physics.json: Physics settings
- motions/: Animation files
- textures/: Body part textures
- face_textures/: Facial feature textures

To use in Live2D Viewer or Cubism:
1. Open model3.json in Live2D Viewer
2. Or import into Cubism Editor for further editing

Note: The .moc3 file is generated with basic structure.
For full compatibility, use Cubism Editor to export proper .moc3 files.
""")
        files["readme"] = str(readme_path)

        return files

    def _build_model3_json(self, model_name: str, state: dict) -> dict:
        """构建 model3.json"""
        layers = state.get("layers", [])

        # 纹理列表
        textures = [
            f"textures/{Path(l['texture_path']).name}"
            for l in layers
            if "texture_path" in l
        ]

        # 添加面部纹理
        face_layers = state.get("face_layers", [])
        for fl in face_layers:
            if "path" in fl:
                textures.append(f"face_textures/{Path(fl['path']).name}")

        # 动作列表
        motions = {}
        for motion in state.get("motions", []):
            motion_type = motion.get("type", "Idle")
            if motion_type not in motions:
                motions[motion_type] = []
            motions[motion_type].append(
                {"File": f"motions/{motion['name']}.motion3.json"}
            )

        # 参数列表（用于面部动画）
        rigging = state.get("rigging", {})

        return {
            "Version": 3,
            "FileReferences": {
                "Moc": f"{model_name}.moc3",
                "Textures": textures,
                "Physics": "physics.json" if state.get("physics") else None,
                "Motions": motions,
            },
            "Groups": rigging.get("groups", []),
            "HitAreas": rigging.get("hit_areas", []),
            "Layout": {"center_x": 0, "center_y": 0, "width": 512, "height": 512},
        }
