"""Live2D .moc3 file generator."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

from loguru import logger

JsonDict = dict[str, Any]
JsonList = list[JsonDict]


class Moc3Generator:
    """Live2D .moc3 file generator."""

    def __init__(self) -> None:
        self.version = 3
        self.model_data: JsonDict = {}

    def generate(
        self,
        model_name: str,
        layers: JsonList,
        rigging: JsonDict,
        output_path: str,
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

    def _write_moc3_binary(self, output_path: Path) -> None:
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

    def _write_layers(self, handle: Any) -> None:
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
                    float(bounds.get("x", 0.0)),
                    float(bounds.get("y", 0.0)),
                    float(bounds.get("width", 100.0)),
                    float(bounds.get("height", 100.0)),
                )
            )
            handle.write(struct.pack("<I", int(layer.get("z_order", 0))))

    def _write_bones(self, handle: Any) -> None:
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

            pos = bone.get("position", {"x": 0.0, "y": 0.0})
            handle.write(struct.pack("<ff", float(pos.get("x", 0.0)), float(pos.get("y", 0.0))))
            handle.write(struct.pack("<f", float(bone.get("rotation", 0.0))))
            scale = bone.get("scale", {"x": 1.0, "y": 1.0})
            handle.write(struct.pack("<ff", float(scale.get("x", 1.0)), float(scale.get("y", 1.0))))

    def _write_parameters(self, handle: Any) -> None:
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
                    float(param.get("min", 0.0)),
                    float(param.get("max", 1.0)),
                    float(param.get("default", 0.0)),
                )
            )


class Live2DExporter:
    """Live2D model exporter with explicit contract validation."""

    def __init__(self) -> None:
        self.moc3_gen = Moc3Generator()

    async def export(self, model_name: str, output_dir: str, state: JsonDict) -> JsonDict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}
        errors: list[str] = []
        warnings: list[str] = []

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
            model3_data = {"FileReferences": {}}

        moc3_success = self.moc3_gen.generate(
            model_name,
            state.get("layers", []),
            state.get("rigging", {}),
            str(output_path),
        )
        if moc3_success:
            files["model3.moc3"] = str(output_path / f"{model_name}.moc3")
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

Bundle Type: Mock intermediate asset package

Files:
- model3.json: Model configuration
- {model_name}.moc3: Placeholder binary model data
- physics.json: Physics settings
- motions/: Animation files
- textures/: Body part textures
- face_textures/: Facial feature textures

Use in Cubism:
1. Open model3.json in Cubism Editor for inspection
2. Rebuild and export a production-ready model from Cubism Editor

Important:
- This repository exports a mock .moc3 package for tooling validation
- Direct runtime compatibility is not guaranteed until Cubism Editor finalization
""")
            files["readme"] = str(readme_path)
        except Exception as exc:
            logger.error(f"Failed to write README.txt: {exc}")
            errors.append("README export failed")

        validation = self._validate_export_bundle(
            model_name, output_path, files, model3_data, state
        )
        warnings.extend(validation["warnings"])
        errors.extend(validation["errors"])

        critical_outputs = {"model3.json", "model3.moc3"}
        missing_critical = sorted(critical_outputs.difference(files))

        return {
            "status": "success" if not missing_critical and not errors else "error",
            "files": files,
            "errors": errors,
            "warnings": warnings,
            "missing_critical_outputs": missing_critical,
            "validation": validation,
            "artifact_stage": "mock-intermediate",
            "direct_viewer_compatible": False,
            "ready_for_cubism_editor": False,
        }

    def _parameter_ranges(self, rigging: JsonDict) -> dict[str, tuple[float, float]]:
        return {
            str(param.get("id", "")): (
                float(param.get("min", 0.0)),
                float(param.get("max", 1.0)),
            )
            for param in rigging.get("parameters", [])
        }

    def _validate_export_bundle(
        self,
        model_name: str,
        output_path: Path,
        files: dict[str, str],
        model3_data: JsonDict,
        state: JsonDict,
    ) -> JsonDict:
        warnings: list[str] = []
        errors: list[str] = []
        file_refs = model3_data.get("FileReferences", {})
        rigging = state.get("rigging", {})
        parameter_ranges = self._parameter_ranges(rigging)
        parameter_ids = set(parameter_ranges)

        expected_moc_name = f"{model_name}.moc3"
        moc_name = file_refs.get("Moc")
        if moc_name != expected_moc_name:
            errors.append("model3.json does not reference the expected .moc3 file")
        else:
            moc_path = output_path / moc_name
            if not moc_path.exists():
                errors.append(f"Referenced moc3 file is missing: {moc_name}")
            else:
                with open(moc_path, "rb") as handle:
                    if handle.read(4) != b"MOC3":
                        errors.append(
                            "Exported .moc3 file does not contain the expected magic header"
                        )

        textures = list(file_refs.get("Textures", []))
        if len(textures) != len(set(textures)):
            errors.append("model3.json contains duplicate texture references")
        for texture in textures:
            if not (output_path / texture).exists():
                errors.append(f"Missing referenced texture: {texture}")

        physics_ref = file_refs.get("Physics")
        if physics_ref and not (output_path / physics_ref).exists():
            errors.append(f"Missing referenced physics file: {physics_ref}")

        for motion_entries in file_refs.get("Motions", {}).values():
            for motion_entry in motion_entries:
                motion_file = motion_entry.get("File")
                if motion_file and not (output_path / motion_file).exists():
                    errors.append(f"Missing referenced motion file: {motion_file}")

        bone_ids = {str(bone.get("id", "")) for bone in rigging.get("bones", [])}
        for bone in rigging.get("bones", []):
            parent = bone.get("parent")
            if parent and parent not in bone_ids:
                errors.append(f"Bone {bone.get('id')} references missing parent {parent}")

        for group in state.get("physics", {}).get("groups", []):
            for side in ("input", "output"):
                for entry in group.get(side, []):
                    parameter_id = str(entry.get("id", ""))
                    if parameter_id not in parameter_ids:
                        errors.append(f"Physics references unknown parameter: {parameter_id}")

        for motion in state.get("motions", []):
            for curve in motion.get("data", {}).get("Curves", []):
                parameter_id = str(curve.get("Id", ""))
                if parameter_id not in parameter_ids:
                    errors.append(
                        f"Motion {motion.get('name', 'unknown')} references unknown parameter: {parameter_id}"
                    )
                    continue
                minimum, maximum = parameter_ranges[parameter_id]
                values = list(cast(list[float], curve.get("Segments", []))[1::2])
                if any(value < minimum or value > maximum for value in values):
                    errors.append(
                        f"Motion {motion.get('name', 'unknown')} has out-of-range values for {parameter_id}"
                    )

        warnings.append(
            "Exported bundle is a mock intermediate artifact and must be rebuilt in Cubism Editor."
        )

        return {
            "contract_valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "expected_moc": expected_moc_name,
            "direct_viewer_compatible": False,
        }

    def _build_model3_json(self, model_name: str, state: JsonDict) -> JsonDict:
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

        motions: dict[str, list[dict[str, str]]] = {}
        for motion in state.get("motions", []):
            motion_type = str(motion.get("type", "Idle"))
            motions.setdefault(motion_type, []).append(
                {"File": f"motions/{motion['name']}.motion3.json"}
            )

        rigging = state.get("rigging", {})
        file_references: JsonDict = {
            "Moc": f"{model_name}.moc3",
            "Textures": textures,
        }
        if state.get("physics"):
            file_references["Physics"] = "physics.json"
        if motions:
            file_references["Motions"] = motions

        return {
            "Version": 3,
            "FileReferences": file_references,
            "Groups": rigging.get("groups", []),
            "HitAreas": rigging.get("hit_areas", []),
            "Layout": {"center_x": 0, "center_y": 0, "width": 512, "height": 512},
        }
