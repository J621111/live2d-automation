"""
Live2D .moc3 file generator
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


class Moc3Generator:
    """Live2D .moc3 file generator."""

    def __init__(self):
        self.version = 3
        self.model_data = {}

    def generate(
        self, model_name: str, layers: List[Dict], rigging: Dict, output_path: str
    ) -> bool:
        try:
            logger.info(f"Generating .moc3 file for {model_name}...")
            self.model_data = {
                "name": model_name,
                "layers": layers,
                "rigging": rigging,
                "version": self.version,
            }

            output_dir = Path(output_path)
            temp_path = output_dir / f"{model_name}_temp.moc3"
            final_path = output_dir / f"{model_name}.moc3"
            self._write_moc3_binary(temp_path)

            if final_path.exists():
                final_path.unlink()

            temp_path.rename(final_path)
            logger.info(f".moc3 file generated: {final_path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to generate .moc3: {exc}")
            return False

    def _write_moc3_binary(self, output_path: Path):
        with open(output_path, "wb") as handle:
            handle.write(b"MOC3")
            handle.write(struct.pack("<I", self.version))

            data_offset = 32
            handle.write(struct.pack("<I", data_offset))
            data_size_offset = handle.tell()
            handle.write(struct.pack("<I", 0))

            padding_size = data_offset - handle.tell()
            handle.write(b"\x00" * padding_size)

            self._write_layers(handle)
            self._write_bones(handle)
            self._write_parameters(handle)

            data_end = handle.tell()
            data_size = data_end - data_offset
            handle.seek(data_size_offset)
            handle.write(struct.pack("<I", data_size))
            logger.info(f"Wrote .moc3 binary: {data_end} bytes")

    def _write_layers(self, handle):
        layers = self.model_data.get("layers", [])
        handle.write(struct.pack("<I", len(layers)))

        for layer in layers:
            name = str(layer.get("name", "layer") or "layer").encode("utf-8")
            handle.write(struct.pack("<I", len(name)))
            handle.write(name)

            bounds = layer.get("bounds", {})
            handle.write(
                struct.pack(
                    "<ffff",
                    bounds.get("x", 0),
                    bounds.get("y", 0),
                    bounds.get("width", 100),
                    bounds.get("height", 100),
                )
            )
            handle.write(struct.pack("<I", layer.get("z_order", 0)))

    def _write_bones(self, handle):
        rigging = self.model_data.get("rigging", {})
        bones = rigging.get("bones", [])
        handle.write(struct.pack("<I", len(bones)))

        for bone in bones:
            bone_id = str(bone.get("id") or "bone").encode("utf-8")
            handle.write(struct.pack("<I", len(bone_id)))
            handle.write(bone_id)

            bone_name = str(bone.get("name") or bone.get("id") or "bone").encode("utf-8")
            handle.write(struct.pack("<I", len(bone_name)))
            handle.write(bone_name)

            parent_id = str(bone.get("parent") or "").encode("utf-8")
            handle.write(struct.pack("<I", len(parent_id)))
            if parent_id:
                handle.write(parent_id)

            pos = bone.get("position", {"x": 0, "y": 0})
            handle.write(struct.pack("<ff", pos.get("x", 0), pos.get("y", 0)))
            handle.write(struct.pack("<f", bone.get("rotation", 0)))
            scale = bone.get("scale", {"x": 1, "y": 1})
            handle.write(struct.pack("<ff", scale.get("x", 1), scale.get("y", 1)))

    def _write_parameters(self, handle):
        rigging = self.model_data.get("rigging", {})
        params = rigging.get("parameters", [])
        handle.write(struct.pack("<I", len(params)))

        for param in params:
            param_id = str(param.get("id") or "param").encode("utf-8")
            handle.write(struct.pack("<I", len(param_id)))
            handle.write(param_id)

            param_name = str(param.get("name") or param.get("id") or "param").encode("utf-8")
            handle.write(struct.pack("<I", len(param_name)))
            handle.write(param_name)

            handle.write(
                struct.pack(
                    "<fff",
                    param.get("min", 0),
                    param.get("max", 1),
                    param.get("default", 0),
                )
            )


class Live2DExporter:
    """Live2D model exporter with explicit success/failure reporting."""

    def __init__(self):
        self.moc3_gen = Moc3Generator()

    async def export(
        self, model_name: str, output_dir: str, state: dict
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files: Dict[str, str] = {}
        errors: List[str] = []

        try:
            model3_data = self._build_model3_json(model_name, state)
            model3_path = output_path / "model3.json"
            temp_model3 = output_path / "model3_temp.json"
            with open(temp_model3, "w", encoding="utf-8") as handle:
                json.dump(model3_data, handle, indent=2, ensure_ascii=False)
            if model3_path.exists():
                model3_path.unlink()
            temp_model3.rename(model3_path)
            files["model3.json"] = str(model3_path)
        except Exception as exc:
            logger.error(f"Failed to export model3.json: {exc}")
            errors.append("model3.json export failed")

        moc3_success = self.moc3_gen.generate(
            model_name,
            state.get("layers", []),
            state.get("rigging", {}),
            str(output_path),
        )
        if moc3_success:
            files["model3.moc"] = str(output_path / f"{model_name}.moc3")
        else:
            errors.append(".moc3 export failed")

        if state.get("physics"):
            try:
                physics_path = output_path / "physics.json"
                temp_physics = output_path / "physics_temp.json"
                with open(temp_physics, "w", encoding="utf-8") as handle:
                    json.dump(state["physics"], handle, indent=2, ensure_ascii=False)
                if physics_path.exists():
                    physics_path.unlink()
                temp_physics.rename(physics_path)
                files["physics.json"] = str(physics_path)
            except Exception as exc:
                logger.error(f"Failed to export physics.json: {exc}")
                errors.append("physics.json export failed")

        motions_dir = output_path / "motions"
        motions_dir.mkdir(exist_ok=True)
        for motion in state.get("motions", []):
            try:
                motion_file = motions_dir / f"{motion['name']}.motion3.json"
                temp_motion = motions_dir / f"{motion['name']}_temp.motion3.json"
                with open(temp_motion, "w", encoding="utf-8") as handle:
                    json.dump(motion["data"], handle, indent=2, ensure_ascii=False)
                if motion_file.exists():
                    motion_file.unlink()
                temp_motion.rename(motion_file)
            except Exception as exc:
                logger.error(f"Failed to export motion {motion.get('name')}: {exc}")
                errors.append(f"motion export failed: {motion.get('name', 'unknown')}")
        files["motions_dir"] = str(motions_dir)

        textures_dir = output_path / "textures"
        textures_dir.mkdir(exist_ok=True)
        for layer in state.get("layers", []):
            if "texture_path" not in layer:
                continue
            src = Path(layer["texture_path"])
            if not src.exists():
                continue
            try:
                dst = textures_dir / src.name
                temp_dst = textures_dir / f"temp_{src.name}"
                with open(src, "rb") as src_handle:
                    data = src_handle.read()
                with open(temp_dst, "wb") as dst_handle:
                    dst_handle.write(data)
                if dst.exists():
                    dst.unlink()
                temp_dst.rename(dst)
            except Exception as exc:
                logger.error(f"Failed to export texture {src}: {exc}")
                errors.append(f"texture export failed: {src.name}")
        files["textures_dir"] = str(textures_dir)

        face_layers = state.get("face_layers", [])
        if face_layers:
            face_dir = output_path / "face_textures"
            face_dir.mkdir(exist_ok=True)
            for face_layer in face_layers:
                src = Path(face_layer.get("path", ""))
                if not src.exists():
                    continue
                try:
                    dst = face_dir / src.name
                    temp_dst = face_dir / f"temp_{src.name}"
                    with open(src, "rb") as src_handle:
                        data = src_handle.read()
                    with open(temp_dst, "wb") as dst_handle:
                        dst_handle.write(data)
                    if dst.exists():
                        dst.unlink()
                    temp_dst.rename(dst)
                except Exception as exc:
                    logger.error(f"Failed to export face texture {src}: {exc}")
                    errors.append(f"face texture export failed: {src.name}")
            files["face_textures_dir"] = str(face_dir)

        try:
            readme_path = output_path / "README.txt"
            with open(readme_path, "w", encoding="utf-8") as handle:
                handle.write(f"""Live2D Model: {model_name}
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
        except Exception as exc:
            logger.error(f"Failed to write README.txt: {exc}")
            errors.append("README export failed")

        critical_outputs = {"model3.json", "model3.moc"}
        missing_critical = sorted(critical_outputs.difference(files))

        return {
            "status": "success" if not missing_critical and not errors else "error",
            "files": files,
            "errors": errors,
            "missing_critical_outputs": missing_critical,
        }

    def _build_model3_json(self, model_name: str, state: dict) -> dict:
        layers = state.get("layers", [])
        textures = [
            f"textures/{Path(layer['texture_path']).name}"
            for layer in layers
            if "texture_path" in layer
        ]

        face_layers = state.get("face_layers", [])
        for face_layer in face_layers:
            if "path" in face_layer:
                textures.append(f"face_textures/{Path(face_layer['path']).name}")

        motions: Dict[str, List[Dict[str, str]]] = {}
        for motion in state.get("motions", []):
            motion_type = motion.get("type", "Idle")
            motions.setdefault(motion_type, []).append(
                {"File": f"motions/{motion['name']}.motion3.json"}
            )

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
