"""Application services for image-to-model and Cubism pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from core.mesh_generator import ArtMeshGenerator
from mcp_server.session_store import InMemorySessionStore, empty_session_state
from mcp_server.tools.ai_part_detector import AIPartDetector
from mcp_server.tools.auto_rigger import AutoRigger
from mcp_server.tools.cubism_automation import CubismAutomationManager
from mcp_server.tools.cubism_bridge import CubismBridge
from mcp_server.tools.export_validator import CubismExportValidator
from mcp_server.tools.facial_detector import FacialFeatureDetector
from mcp_server.tools.image_processor import ImageProcessor
from mcp_server.tools.layer_generator import LayerGenerator
from mcp_server.tools.moc3_generator import Live2DExporter as Moc3Exporter
from mcp_server.tools.motion_generator import MotionGenerator
from mcp_server.tools.part_segmenter import PartSegmenter
from mcp_server.tools.physics_setup import PhysicsSetup
from mcp_server.tools.psd_builder import CubismPSDBuilder
from mcp_server.tools.template_mapper import TemplateMapper
from mcp_server.validation import InputValidationError

JsonDict = dict[str, Any]


class ImagePipelineService:
    """Run image analysis and mock Live2D model generation stages."""

    def __init__(self, session_store: InMemorySessionStore) -> None:
        self._session_store = session_store

    async def analyze_photo(self, image_path: Path, session_id: str) -> JsonDict:
        logger.info(f"Analyzing photo for session {session_id}: {image_path.name}")
        processor = ImageProcessor()
        segments = await processor.analyze(str(image_path))
        state = self._session_store.get_state(session_id)
        state.clear()
        state.update(empty_session_state())
        state["input_image"] = str(image_path)
        state["segments"] = segments
        state["analysis_metadata"] = {
            "detector_used": segments.get("detector_used"),
            "fallback_reason": segments.get("fallback_reason"),
            "confidence_summary": segments.get("confidence_summary"),
        }
        detected_parts = segments.get("parts") or {}
        return {
            "status": "success",
            "session_id": session_id,
            "parts_detected": len(detected_parts),
            "segments": segments,
            "detector_used": segments.get("detector_used"),
            "fallback_reason": segments.get("fallback_reason"),
            "confidence_summary": segments.get("confidence_summary"),
            "message": f"Detected {len(detected_parts)} candidate parts.",
        }

    async def ensure_face_features(
        self,
        session_id: str,
        output_dir: Path,
    ) -> tuple[JsonDict, list[JsonDict], bool]:
        state = self._session_store.require_state_field(
            session_id,
            "input_image",
            "Run analyze_photo before face extraction.",
        )
        face_output_dir = output_dir / "face_textures"
        face_output_dir_str = str(face_output_dir)
        if (
            state.get("face_features_complete") is True
            and state.get("face_output_dir") == face_output_dir_str
        ):
            return state["face_features"], state["face_layers"], False
        logger.info(f"Detecting face features for session {session_id}")
        detector = FacialFeatureDetector()
        face_features = await detector.detect_features(state["input_image"])
        face_output_dir.mkdir(parents=True, exist_ok=True)
        face_layers = await detector.extract_face_parts(
            state["input_image"],
            str(face_output_dir),
            features=face_features,
        )
        state["face_features"] = face_features
        state["face_layers"] = face_layers
        state["face_output_dir"] = face_output_dir_str
        state["face_features_complete"] = True
        return face_features, face_layers, True

    async def detect_face_features(self, session_id: str, output_dir: Path) -> JsonDict:
        face_features, face_layers, _ = await self.ensure_face_features(session_id, output_dir)
        return {
            "status": "success",
            "session_id": session_id,
            "parts_detected": len(face_features.get("parts", {})),
            "layers_extracted": len(face_layers),
            "detector_used": face_features.get("detector_used"),
            "fallback_reason": face_features.get("fallback_reason"),
            "confidence_summary": face_features.get("confidence_summary"),
            "message": f"Detected {len(face_features.get('parts', {}))} face parts.",
        }

    async def analyze_parts_with_ai(self, image_path: Path, session_id: str) -> JsonDict:
        logger.info(f"Analyzing semantic parts for session {session_id}: {image_path.name}")
        detector = AIPartDetector()
        result = await detector.analyze(str(image_path))
        state = self._session_store.get_state(session_id)
        state.clear()
        state.update(empty_session_state())
        state["input_image"] = str(image_path)
        state["ai_parts"] = result.get("parts", [])
        state["analysis_metadata"] = {
            "backend_used": result.get("backend_used"),
            "detector_used": result.get("detector_used"),
            "fallback_reason": result.get("fallback_reason"),
            "confidence_summary": result.get("confidence_summary"),
        }
        return {
            "status": "success",
            "session_id": session_id,
            "parts": result.get("parts", []),
            "part_count": result.get("part_count", 0),
            "backend_used": result.get("backend_used"),
            "detector_used": result.get("detector_used"),
            "fallback_reason": result.get("fallback_reason"),
            "confidence_summary": result.get("confidence_summary"),
            "message": f"Detected {result.get('part_count', 0)} semantic parts.",
        }

    async def segment_detected_parts(self, session_id: str, output_dir: Path) -> JsonDict:
        state = self._session_store.require_state_field(
            session_id,
            "ai_parts",
            "Run analyze_parts_with_ai before segment_detected_parts.",
        )
        state["output_dir"] = str(output_dir)
        segmenter = PartSegmenter()
        result = await segmenter.segment(
            state["input_image"],
            state["ai_parts"],
            str(output_dir),
        )
        state["ai_part_layers"] = result.get("layers", [])
        return {
            "status": "success",
            "session_id": session_id,
            "layers_generated": result.get("layers_generated", 0),
            "layers": result.get("layers", []),
            "message": f"Generated {result.get('layers_generated', 0)} AI-guided part layers.",
        }

    async def generate_layers(self, session_id: str, output_dir: Path) -> JsonDict:
        state = self._session_store.require_state_field(
            session_id,
            "segments",
            "Run analyze_photo before generate_layers.",
        )
        logger.info(f"Generating layers for session {session_id}")
        state["output_dir"] = str(output_dir)
        face_features, face_layers, _ = await self.ensure_face_features(
            session_id,
            output_dir,
        )

        ai_detector = AIPartDetector()
        ai_result = await ai_detector.analyze(state["input_image"])
        state["ai_parts"] = ai_result.get("parts", [])
        ai_segmenter = PartSegmenter()
        ai_layer_result = await ai_segmenter.segment(
            state["input_image"],
            state["ai_parts"],
            str(output_dir),
        )
        ai_layers = ai_layer_result.get("layers", [])
        if ai_layers:
            state["ai_part_layers"] = ai_layers
            state["layers"] = ai_layers
            state["layer_generation_metadata"] = {
                "backend_used": ai_result.get("backend_used"),
                "detector_used": ai_result.get("detector_used"),
                "fallback_reason": ai_result.get("fallback_reason"),
                "confidence_summary": ai_result.get("confidence_summary"),
                "source": "semantic_refine",
            }
            return {
                "status": "success",
                "session_id": session_id,
                "layers_generated": len(ai_layers),
                "layers": ai_layers,
                "face_layers_extracted": len(face_layers),
                "backend_used": ai_result.get("backend_used"),
                "detector_used": ai_result.get("detector_used"),
                "fallback_reason": ai_result.get("fallback_reason"),
                "confidence_summary": ai_result.get("confidence_summary"),
                "face_detector_used": face_features.get("detector_used"),
                "message": f"Generated {len(ai_layers)} AI-guided layers.",
            }

        generator = LayerGenerator()
        layers = await generator.generate(
            image_path=state["input_image"],
            segments=state["segments"],
            output_dir=str(output_dir),
        )
        state["layers"] = layers
        state["layer_generation_metadata"] = generator.last_generation_metadata
        return {
            "status": "success",
            "session_id": session_id,
            "layers_generated": len(layers),
            "layers": layers,
            "face_layers_extracted": len(face_layers),
            "detector_used": generator.last_generation_metadata.get("detector_used"),
            "fallback_reason": generator.last_generation_metadata.get("fallback_reason"),
            "confidence_summary": generator.last_generation_metadata.get("confidence_summary"),
            "face_detector_used": face_features.get("detector_used"),
            "message": f"Generated {len(layers)} layers.",
        }

    async def create_mesh(self, session_id: str) -> JsonDict:
        state = self._session_store.require_state_field(
            session_id,
            "layers",
            "Run generate_layers before create_mesh.",
        )
        logger.info(f"Creating meshes for session {session_id}")
        mesh_gen = ArtMeshGenerator()
        meshes = await mesh_gen.generate_from_layers(state["layers"])
        state["meshes"] = meshes
        return {
            "status": "success",
            "session_id": session_id,
            "meshes_created": len(meshes),
            "meshes": meshes,
            "message": f"Created {len(meshes)} art meshes.",
        }

    async def setup_rigging(self, session_id: str) -> JsonDict:
        state = self._session_store.require_state_field(
            session_id,
            "meshes",
            "Run create_mesh before setup_rigging.",
        )
        logger.info(f"Setting up rigging for session {session_id}")
        rigger = AutoRigger()
        rigging = await rigger.setup(meshes=state["meshes"], segments=state["segments"])
        state["rigging"] = rigging
        return {
            "status": "success",
            "session_id": session_id,
            "bones_created": len(rigging.get("bones", [])),
            "parameters_created": len(rigging.get("parameters", [])),
            "rigging": rigging,
            "message": (
                f"Created {len(rigging.get('bones', []))} bones and "
                f"{len(rigging.get('parameters', []))} parameters."
            ),
        }

    async def configure_physics(self, session_id: str) -> JsonDict:
        state = self._session_store.require_state_field(
            session_id,
            "rigging",
            "Run setup_rigging before configure_physics.",
        )
        logger.info(f"Configuring physics for session {session_id}")
        physics = PhysicsSetup()
        config = await physics.configure(rigging=state["rigging"], segments=state["segments"])
        state["physics"] = config
        return {
            "status": "success",
            "session_id": session_id,
            "physics_groups": len(config.get("groups", [])),
            "physics": config,
            "message": f"Configured {len(config.get('groups', []))} physics groups.",
        }

    async def generate_motions(self, session_id: str, motion_types: list[str]) -> JsonDict:
        state = self._session_store.require_state_field(
            session_id,
            "rigging",
            "Run setup_rigging before generate_motions.",
        )
        logger.info(f"Generating motions for session {session_id}: {motion_types}")
        motion_gen = MotionGenerator()
        motions = await motion_gen.generate(
            rigging=state["rigging"],
            motion_types=motion_types,
        )
        state["motions"] = motions
        return {
            "status": "success",
            "session_id": session_id,
            "motions_generated": len(motions),
            "motions": motions,
            "message": f"Generated {len(motions)} motions.",
        }

    async def export_model(
        self,
        session_id: str,
        output_dir: Path,
        model_name: str,
    ) -> JsonDict:
        state = self._session_store.get_state(session_id)
        state["output_dir"] = str(output_dir)
        exporter = Moc3Exporter()
        export_result = await exporter.export(
            model_name=model_name,
            output_dir=str(output_dir),
            state=state,
        )
        state["model_files"] = export_result.get("files", {})
        warnings = export_result.get("warnings") or []
        return {
            "status": export_result.get("status", "error"),
            "session_id": session_id,
            "model_files": export_result.get("files", {}),
            "export_result": export_result,
            "message": warnings[0] if warnings else "Export complete.",
        }


class CubismPipelineService:
    """Run Cubism package preparation, dispatch, and validation stages."""

    def __init__(
        self,
        session_store: InMemorySessionStore,
        *,
        template_dirs: list[Path],
        output_root: Path,
    ) -> None:
        self._session_store = session_store
        self._template_dirs = template_dirs
        self._output_root = output_root

    async def build_psd(
        self,
        session_id: str,
        output_dir: Path,
        template_id: str,
        model_name: str,
    ) -> JsonDict:
        state = self._session_store.get_state(session_id)
        layers = list(state.get("ai_part_layers") or state.get("layers") or [])
        if not layers:
            raise InputValidationError("Run generate_layers before build_cubism_psd.")

        mapper = TemplateMapper(self._template_dirs)
        mapping = mapper.map_layers(layers, template_id)
        builder = CubismPSDBuilder()
        result = await builder.build(layers, mapping, str(output_dir), model_name)
        state["cubism_template_mapping"] = mapping
        state["cubism_psd_path"] = result.get("psd_path")
        state["output_dir"] = str(output_dir)
        return {
            "status": result.get("status", "error"),
            "session_id": session_id,
            "template_id": template_id,
            "psd_path": result.get("psd_path"),
            "preview_path": result.get("preview_path"),
            "manifest_path": result.get("manifest_path"),
            "mapping_path": result.get("mapping_path"),
            "coverage": result.get("coverage", 0.0),
            "missing_required": result.get("missing_required", []),
            "message": (
                f"Built Cubism PSD package for template '{template_id}'."
                if not result.get("missing_required")
                else (
                    "Built Cubism PSD package with missing required parts: "
                    f"{result.get('missing_required', [])}."
                )
            ),
        }

    async def prepare_automation(
        self,
        session_id: str,
        output_dir: Path,
        template_id: str,
        model_name: str,
        editor_path: str | None,
        automation_backend: str | None,
    ) -> JsonDict:
        state = self._session_store.get_state(session_id)
        psd_path = state.get("cubism_psd_path")
        if not psd_path:
            raise InputValidationError("Run build_cubism_psd before prepare_cubism_automation.")

        psd_file = Path(str(psd_path))
        if not psd_file.exists() or not psd_file.is_file():
            raise InputValidationError(
                "The prepared Cubism PSD package is missing. Re-run "
                "build_cubism_psd before prepare_cubism_automation."
            )

        manager = CubismAutomationManager()
        try:
            descriptor = manager.resolve_backend(automation_backend)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        bridge = CubismBridge()
        editor_info = bridge.discover_editor(editor_path)
        plan = bridge.build_plan(
            psd_path=str(psd_path),
            output_dir=str(output_dir),
            template_id=template_id,
            model_name=model_name,
            editor_info=editor_info,
            automation_backend=descriptor.name,
        )
        execution = manager.prepare_execution(
            descriptor.name,
            editor_info=editor_info,
            plan=plan,
        )
        dispatch_bundle = manager.build_dispatch_bundle(
            descriptor.name,
            plan=plan,
            execution=execution,
            template_id=template_id,
            model_name=model_name,
            psd_path=str(psd_path),
            output_dir=str(output_dir),
            editor_info=editor_info,
        )
        plan["status"] = execution.get("status", plan.get("status", "blocked"))
        plan["automation_mode"] = execution.get(
            "automation_mode",
            plan.get("automation_mode"),
        )
        plan["execution"] = execution
        plan_path = bridge.write_plan(plan, str(output_dir), model_name)
        dispatch_bundle_path = manager.write_dispatch_bundle(
            dispatch_bundle,
            str(output_dir),
            model_name,
        )
        state["cubism_automation_plan"] = {**plan, "plan_path": plan_path}
        state["cubism_dispatch_bundle"] = {
            **dispatch_bundle,
            "bundle_path": dispatch_bundle_path,
        }
        return {
            "status": plan.get("status", "blocked"),
            "session_id": session_id,
            "template_id": template_id,
            "editor": editor_info,
            "automation_backend": descriptor.name,
            "automation_mode": plan.get("automation_mode"),
            "execution_supported": execution.get("execution_supported", False),
            "backend_capabilities": execution.get("capabilities", []),
            "missing_requirements": execution.get("missing_requirements", []),
            "plan_path": plan_path,
            "dispatch_bundle_path": dispatch_bundle_path,
            "steps": plan.get("steps", []),
            "message": (
                f"Cubism automation plan is ready for backend '{descriptor.name}'."
                if plan.get("status") == "ready"
                else (
                    f"Cubism automation plan was created for backend '{descriptor.name}', "
                    "but dispatch execution is unavailable."
                    if not execution.get("execution_supported", False)
                    else (
                        "Cubism automation plan was created for backend "
                        f"'{descriptor.name}', but required tools are missing."
                    )
                )
            ),
        }

    async def execute_dispatch(self, session_id: str) -> JsonDict:
        state = self._session_store.get_state(session_id)
        bundle = dict(state.get("cubism_dispatch_bundle") or {})
        if not bundle:
            raise InputValidationError(
                "Run prepare_cubism_automation before execute_cubism_dispatch."
            )

        manager = CubismAutomationManager()
        execution = manager.execute_dispatch_bundle(bundle)
        output_dir = Path(
            str(bundle.get("output_dir", state.get("output_dir") or self._output_root))
        )
        model_name = str(bundle.get("model_name", "ATRI"))
        execution_path = manager.write_dispatch_execution(
            execution,
            str(output_dir),
            model_name,
        )
        calibration_report = manager.build_profile_calibration_report(bundle, execution)
        calibration_report_path = manager.write_profile_calibration_report(
            calibration_report,
            str(output_dir),
            model_name,
        )
        state["cubism_dispatch_execution"] = {
            **execution,
            "execution_path": execution_path,
            "calibration_report": calibration_report,
            "calibration_report_path": calibration_report_path,
        }
        return {
            "status": execution.get("status", "error"),
            "session_id": session_id,
            "automation_backend": bundle.get("backend"),
            "execution_path": execution_path,
            "calibration_report_path": calibration_report_path,
            "calibration_report": calibration_report,
            "executed_steps": execution.get("executed_steps", []),
            "artifacts": execution.get("artifacts", []),
            "resume": execution.get("resume", {}),
            "message": execution.get("message", "Cubism dispatch execution finished."),
        }

    async def resume_dispatch(self, session_id: str) -> JsonDict:
        state = self._session_store.get_state(session_id)
        bundle = dict(state.get("cubism_dispatch_bundle") or {})
        if not bundle:
            raise InputValidationError(
                "Run prepare_cubism_automation before resume_cubism_dispatch."
            )
        previous_execution = dict(state.get("cubism_dispatch_execution") or {})
        if not previous_execution:
            raise InputValidationError("Run execute_cubism_dispatch before resume_cubism_dispatch.")

        manager = CubismAutomationManager()
        output_dir = Path(
            str(bundle.get("output_dir", state.get("output_dir") or self._output_root))
        )
        refreshed_execution = manager.prepare_execution(
            str(bundle.get("backend", "native_gui")),
            editor_info=dict(bundle.get("editor", {})),
            plan=dict(state.get("cubism_automation_plan") or {}),
        )
        bundle["native_controller"] = refreshed_execution.get("native_controller")
        bundle["native_adapter"] = refreshed_execution.get("native_adapter")
        bundle["preflight"] = {
            "commands": refreshed_execution.get("preflight_commands", []),
            "results": refreshed_execution.get("preflight_results", []),
        }
        execution = manager.execute_dispatch_bundle(
            bundle,
            previous_execution=previous_execution,
            resume=True,
        )
        model_name = str(bundle.get("model_name", "ATRI"))
        execution_path = manager.write_dispatch_execution(
            execution,
            str(output_dir),
            model_name,
            suffix="_resume",
        )
        calibration_report = manager.build_profile_calibration_report(bundle, execution)
        calibration_report_path = manager.write_profile_calibration_report(
            calibration_report,
            str(output_dir),
            model_name,
            suffix="_resume",
        )
        state["cubism_dispatch_execution"] = {
            **execution,
            "execution_path": execution_path,
            "calibration_report": calibration_report,
            "calibration_report_path": calibration_report_path,
        }
        return {
            "status": execution.get("status", "error"),
            "session_id": session_id,
            "automation_backend": bundle.get("backend"),
            "execution_path": execution_path,
            "calibration_report_path": calibration_report_path,
            "calibration_report": calibration_report,
            "executed_steps": execution.get("executed_steps", []),
            "artifacts": execution.get("artifacts", []),
            "resume": execution.get("resume", {}),
            "message": execution.get("message", "Cubism dispatch execution resumed."),
        }

    async def validate_export(self, output_dir: Path, model_name: str) -> JsonDict:
        validator = CubismExportValidator()
        result = validator.validate(str(output_dir), model_name)
        return {
            "status": result.get("status", "error"),
            "output_dir": str(output_dir),
            "model_name": model_name,
            "missing": result.get("missing", []),
            "warnings": result.get("warnings", []),
            "errors": result.get("errors", []),
            "checks": result.get("checks", {}),
            "message": (
                "Cubism export bundle validated successfully."
                if result.get("status") == "success"
                else "Cubism export bundle is incomplete."
            ),
        }
