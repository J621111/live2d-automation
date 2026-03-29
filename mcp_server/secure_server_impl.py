"""Hardened Live2D Automation MCP server implementation."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from fastmcp import FastMCP
from loguru import logger
from PIL import Image as PILImage
from PIL import ImageFile, UnidentifiedImageError

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.mesh_generator import ArtMeshGenerator
from mcp_server.tools.ai_part_detector import AIPartDetector
from mcp_server.tools.auto_rigger import AutoRigger
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

mcp = FastMCP("live2d-automation")

OUTPUT_ROOT = (project_root / "output").resolve()
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_777_216
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_MOTION_TYPES = ["idle", "tap", "move", "emotional"]
ALLOWED_MOTION_TYPES = tuple(DEFAULT_MOTION_TYPES)
MAX_MOTION_TYPES = len(DEFAULT_MOTION_TYPES)

PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def _template_dirs() -> list[Path]:
    return [project_root / "templates", project_root / "mcp_server" / "templates"]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning(f"Invalid integer for {name}: {raw!r}; using default {default}")
        return default


MAX_SESSIONS = _env_int("LIVE2D_MAX_SESSIONS", 32)
SESSION_TTL_SECONDS = _env_int("LIVE2D_SESSION_TTL_SECONDS", 60 * 60)
MAX_CONCURRENT_OPERATIONS = _env_int("LIVE2D_MAX_CONCURRENT_OPERATIONS", 4)


class InputValidationError(ValueError):
    """Raised when a user-controlled input is unsafe or invalid."""


@dataclass
class SessionMetrics:
    created_sessions: int = 0
    rejected_sessions: int = 0
    expired_sessions: int = 0
    closed_sessions: int = 0
    completed_sessions: int = 0


@dataclass
class SessionRecord:
    """In-memory state for a single MCP session."""

    session_id: str
    state: dict[str, Any] = field(default_factory=lambda: _empty_state())
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    active_operation: bool = False


def _empty_state() -> dict[str, Any]:
    return {
        "input_image": None,
        "output_dir": None,
        "segments": {},
        "face_features": {},
        "face_layers": [],
        "layers": [],
        "meshes": {},
        "rigging": {},
        "physics": {},
        "motions": [],
        "model_files": {},
        "face_output_dir": None,
        "analysis_metadata": {},
        "layer_generation_metadata": {},
        "ai_parts": [],
        "ai_part_layers": [],
        "cubism_template_mapping": {},
        "cubism_psd_path": None,
        "cubism_automation_plan": {},
    }


session_store: dict[str, SessionRecord] = {}
session_lock = RLock()
session_metrics = SessionMetrics()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _new_session_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def _validate_session_id(session_id: str) -> str:
    if not SESSION_ID_RE.match(session_id):
        raise InputValidationError("session_id contains unsupported characters.")
    return session_id


def _prune_expired_sessions_locked(now: float | None = None) -> None:
    current_time = now or time.time()
    expired = [
        session_id
        for session_id, record in session_store.items()
        if current_time - record.last_accessed > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        session_store.pop(session_id, None)
        session_metrics.expired_sessions += 1


def _create_session() -> str:
    with session_lock:
        _prune_expired_sessions_locked()
        if len(session_store) >= MAX_SESSIONS:
            session_metrics.rejected_sessions += 1
            raise InputValidationError(
                "Too many active sessions. Wait for an existing job to expire or finish."
            )
        session_id = _new_session_id()
        session_store[session_id] = SessionRecord(session_id=session_id)
        session_metrics.created_sessions += 1
        return session_id


def _remove_session(session_id: str, *, reason: str = "manual") -> bool:
    with session_lock:
        record = session_store.pop(session_id, None)
        if record is None:
            return False
        if reason == "manual":
            session_metrics.closed_sessions += 1
        elif reason == "completed":
            session_metrics.completed_sessions += 1
        elif reason == "expired":
            session_metrics.expired_sessions += 1
        return True


def _get_session_record(session_id: str, *, touch: bool = True) -> SessionRecord:
    _validate_session_id(session_id)
    with session_lock:
        _prune_expired_sessions_locked()
        record = session_store.get(session_id)
        if record is None:
            raise InputValidationError("Unknown or expired session_id. Run analyze_photo first.")
        if touch:
            record.last_accessed = time.time()
        return record


def _get_session_state(session_id: str) -> dict[str, Any]:
    return _get_session_record(session_id).state


def _require_state_field(session_id: str, field: str, message: str) -> dict[str, Any]:
    state = _get_session_state(session_id)
    if not state.get(field):
        raise InputValidationError(message)
    return state


def _active_operation_count() -> int:
    return sum(1 for record in session_store.values() if record.active_operation)


@contextmanager
def _session_operation(session_id: str):
    with session_lock:
        _prune_expired_sessions_locked()
        record = _get_session_record(session_id, touch=False)
        if record.active_operation:
            session_metrics.rejected_sessions += 1
            raise InputValidationError("This session is already busy running another operation.")
        if _active_operation_count() >= MAX_CONCURRENT_OPERATIONS:
            session_metrics.rejected_sessions += 1
            raise InputValidationError("Server is busy. Retry after another job completes.")
        record.active_operation = True
        record.last_accessed = time.time()

    try:
        yield record
    finally:
        with session_lock:
            existing = session_store.get(session_id)
            if existing is not None:
                existing.active_operation = False
                existing.last_accessed = time.time()


def _partial_outputs(session_id: str | None) -> dict[str, Any]:
    if not session_id:
        return {}
    try:
        state = _get_session_state(session_id)
    except InputValidationError:
        return {}
    return {
        "output_dir": state.get("output_dir"),
        "layers_generated": len(state.get("layers", [])),
        "meshes_created": len(state.get("meshes", {})),
        "motions_generated": len(state.get("motions", [])),
        "model_files": state.get("model_files", {}),
    }


def _build_error_payload(
    *,
    error_code: str,
    message: str,
    session_id: str | None = None,
    validation_errors: list[str] | None = None,
    partial_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "error", "error_code": error_code, "message": message}
    if session_id:
        payload["session_id"] = session_id
    if validation_errors:
        payload["validation_errors"] = validation_errors
    if partial_outputs:
        payload["partial_outputs"] = partial_outputs
    return payload


def _resolve_image_path(image_path: str) -> Path:
    if not image_path or not image_path.strip():
        raise InputValidationError("image_path is required.")
    raw_path = Path(image_path)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise InputValidationError("image_path does not point to an existing file.")
    if resolved.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise InputValidationError("Unsupported image format.")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise InputValidationError("Image file is too large.")
    try:
        with PILImage.open(resolved) as image:
            width, height = image.size
    except (PILImage.DecompressionBombError, OSError, UnidentifiedImageError) as exc:
        raise InputValidationError(f"Unable to safely open image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise InputValidationError("Image dimensions are invalid.")
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise InputValidationError(
            f"Image dimensions exceed the {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT} limit."
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise InputValidationError("Image contains too many pixels.")
    return resolved


def _resolve_output_dir(output_dir: str) -> Path:
    if not output_dir or not output_dir.strip():
        raise InputValidationError("output_dir is required.")

    normalized_output_dir = output_dir.strip().replace("\\", "/")
    raw_path = Path(normalized_output_dir)

    if ".." in raw_path.parts:
        raise InputValidationError("output_dir must stay inside the project output directory.")

    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    elif raw_path.parts and raw_path.parts[0] == "output":
        resolved = (project_root / raw_path).resolve()
    else:
        resolved = (OUTPUT_ROOT / raw_path).resolve()
    if not _is_relative_to(resolved, OUTPUT_ROOT):
        raise InputValidationError("output_dir must stay inside the project output directory.")
    return resolved


def _validate_model_name(model_name: str) -> str:
    if not MODEL_NAME_RE.match(model_name):
        raise InputValidationError(
            "model_name may only contain letters, numbers, underscores, and hyphens."
        )
    return model_name


def _validate_motion_types(motion_types: list[str] | None) -> list[str]:
    normalized = motion_types or DEFAULT_MOTION_TYPES
    if not isinstance(normalized, list):
        raise InputValidationError("motion_types must be an array of supported motion names.")
    if not normalized:
        raise InputValidationError("motion_types must contain at least one motion type.")
    if len(normalized) > MAX_MOTION_TYPES:
        raise InputValidationError(
            f"At most {MAX_MOTION_TYPES} motion types may be requested at once."
        )
    safe_motion_types: list[str] = []
    for motion_type in normalized:
        if not isinstance(motion_type, str):
            raise InputValidationError("motion_types entries must be strings.")
        normalized_type = motion_type.strip().lower()
        if normalized_type not in ALLOWED_MOTION_TYPES:
            allowed = ", ".join(ALLOWED_MOTION_TYPES)
            raise InputValidationError(
                f"Unsupported motion_type '{motion_type}'. Allowed values: {allowed}."
            )
        if normalized_type not in safe_motion_types:
            safe_motion_types.append(normalized_type)
    return safe_motion_types


def _status_summary(record: SessionRecord) -> dict[str, Any]:
    state = record.state
    return {
        "has_image": bool(state.get("input_image")),
        "segments_ready": bool(state.get("segments")),
        "layers_ready": len(state.get("layers", [])),
        "meshes_ready": len(state.get("meshes", {})),
        "motions_ready": len(state.get("motions", [])),
        "has_export": bool(state.get("model_files")),
        "busy": record.active_operation,
    }


def _append_step(results: dict[str, Any], name: str, result: dict[str, Any]) -> None:
    results.setdefault("steps", []).append({"name": name, "result": result})
    results["partial_outputs"] = _partial_outputs(results.get("session_id"))


async def _analyze_photo_impl(image_path: Path, session_id: str) -> dict[str, Any]:
    logger.info(f"Analyzing photo for session {session_id}: {image_path.name}")
    processor = ImageProcessor()
    segments = await processor.analyze(str(image_path))
    state = _get_session_state(session_id)
    state.clear()
    state.update(_empty_state())
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


async def _ensure_face_features_impl(
    session_id: str, output_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    state = _require_state_field(
        session_id, "input_image", "Run analyze_photo before face extraction."
    )
    face_output_dir = output_dir / "face_textures"
    face_output_dir_str = str(face_output_dir)
    if (
        state.get("face_features")
        and state.get("face_layers")
        and state.get("face_output_dir") == face_output_dir_str
    ):
        return state["face_features"], state["face_layers"], False
    logger.info(f"Detecting face features for session {session_id}")
    detector = FacialFeatureDetector()
    face_features = await detector.detect_features(state["input_image"])
    face_output_dir.mkdir(parents=True, exist_ok=True)
    face_layers = await detector.extract_face_parts(
        state["input_image"], str(face_output_dir), features=face_features
    )
    state["face_features"] = face_features
    state["face_layers"] = face_layers
    state["face_output_dir"] = face_output_dir_str
    return face_features, face_layers, True


async def _detect_face_features_impl(session_id: str, output_dir: Path) -> dict[str, Any]:
    face_features, face_layers, _ = await _ensure_face_features_impl(session_id, output_dir)
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


async def _analyze_parts_with_ai_impl(image_path: Path, session_id: str) -> dict[str, Any]:
    logger.info(f"Analyzing semantic parts for session {session_id}: {image_path.name}")
    detector = AIPartDetector()
    result = await detector.analyze(str(image_path))
    state = _get_session_state(session_id)
    state.clear()
    state.update(_empty_state())
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


async def _segment_detected_parts_impl(session_id: str, output_dir: Path) -> dict[str, Any]:
    state = _require_state_field(
        session_id, "ai_parts", "Run analyze_parts_with_ai before segment_detected_parts."
    )
    state["output_dir"] = str(output_dir)
    segmenter = PartSegmenter()
    result = await segmenter.segment(state["input_image"], state["ai_parts"], str(output_dir))
    state["ai_part_layers"] = result.get("layers", [])
    return {
        "status": "success",
        "session_id": session_id,
        "layers_generated": result.get("layers_generated", 0),
        "layers": result.get("layers", []),
        "message": f"Generated {result.get('layers_generated', 0)} AI-guided part layers.",
    }


async def _build_cubism_psd_impl(
    session_id: str,
    output_dir: Path,
    template_id: str,
    model_name: str,
) -> dict[str, Any]:
    state = _get_session_state(session_id)
    layers = list(state.get("ai_part_layers") or state.get("layers") or [])
    if not layers:
        raise InputValidationError("Run generate_layers before build_cubism_psd.")

    mapper = TemplateMapper(_template_dirs())
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
            else f"Built Cubism PSD package with missing required parts: {result.get('missing_required', [])}."
        ),
    }


async def _prepare_cubism_automation_impl(
    session_id: str,
    output_dir: Path,
    template_id: str,
    model_name: str,
    editor_path: str | None,
) -> dict[str, Any]:
    state = _get_session_state(session_id)
    psd_path = state.get("cubism_psd_path")
    if not psd_path:
        raise InputValidationError("Run build_cubism_psd before prepare_cubism_automation.")

    psd_file = Path(str(psd_path))
    if not psd_file.exists() or not psd_file.is_file():
        raise InputValidationError(
            "The prepared Cubism PSD package is missing. Re-run build_cubism_psd before prepare_cubism_automation."
        )

    bridge = CubismBridge()
    editor_info = bridge.discover_editor(editor_path)
    plan = bridge.build_plan(
        psd_path=str(psd_path),
        output_dir=str(output_dir),
        template_id=template_id,
        model_name=model_name,
        editor_info=editor_info,
    )
    plan_path = bridge.write_plan(plan, str(output_dir), model_name)
    state["cubism_automation_plan"] = {**plan, "plan_path": plan_path}
    return {
        "status": plan.get("status", "blocked"),
        "session_id": session_id,
        "template_id": template_id,
        "editor": editor_info,
        "plan_path": plan_path,
        "steps": plan.get("steps", []),
        "message": (
            "Cubism automation plan is ready."
            if plan.get("status") == "ready"
            else "Cubism automation plan was created, but the editor executable was not found."
        ),
    }


async def _validate_cubism_export_impl(output_dir: Path, model_name: str) -> dict[str, Any]:
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


async def _generate_layers_impl(session_id: str, output_dir: Path) -> dict[str, Any]:
    state = _require_state_field(
        session_id, "segments", "Run analyze_photo before generate_layers."
    )
    logger.info(f"Generating layers for session {session_id}")
    state["output_dir"] = str(output_dir)
    face_features, face_layers, _ = await _ensure_face_features_impl(session_id, output_dir)

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
            "detector_used": ai_result.get("detector_used"),
            "fallback_reason": ai_result.get("fallback_reason"),
            "confidence_summary": ai_result.get("confidence_summary"),
            "face_detector_used": face_features.get("detector_used"),
            "message": f"Generated {len(ai_layers)} AI-guided layers.",
        }

    generator = LayerGenerator()
    layers = await generator.generate(
        image_path=state["input_image"], segments=state["segments"], output_dir=str(output_dir)
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


async def _create_mesh_impl(session_id: str) -> dict[str, Any]:
    state = _require_state_field(session_id, "layers", "Run generate_layers before create_mesh.")
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


async def _setup_rigging_impl(session_id: str) -> dict[str, Any]:
    state = _require_state_field(session_id, "meshes", "Run create_mesh before setup_rigging.")
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
        "message": f"Created {len(rigging.get('bones', []))} bones and {len(rigging.get('parameters', []))} parameters.",
    }


async def _configure_physics_impl(session_id: str) -> dict[str, Any]:
    state = _require_state_field(
        session_id, "rigging", "Run setup_rigging before configure_physics."
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


async def _generate_motions_impl(session_id: str, motion_types: list[str]) -> dict[str, Any]:
    state = _require_state_field(
        session_id, "rigging", "Run setup_rigging before generate_motions."
    )
    logger.info(f"Generating motions for session {session_id}: {motion_types}")
    motion_gen = MotionGenerator()
    motions = await motion_gen.generate(rigging=state["rigging"], motion_types=motion_types)
    state["motions"] = motions
    return {
        "status": "success",
        "session_id": session_id,
        "motions_generated": len(motions),
        "motions": motions,
        "message": f"Generated {len(motions)} motions.",
    }


async def _export_model_impl(session_id: str, output_dir: Path, model_name: str) -> dict[str, Any]:
    state = _get_session_state(session_id)
    state["output_dir"] = str(output_dir)
    exporter = Moc3Exporter()
    export_result = await exporter.export(
        model_name=model_name, output_dir=str(output_dir), state=state
    )
    state["model_files"] = export_result.get("files", {})
    return {
        "status": export_result.get("status", "error"),
        "session_id": session_id,
        "model_files": export_result.get("files", {}),
        "export_result": export_result,
        "message": export_result.get("warnings", ["Export complete."])[0],
    }


@mcp.tool()
async def analyze_parts_with_ai(image_path: str) -> dict[str, Any]:
    session_id = _create_session()
    try:
        resolved_image = _resolve_image_path(image_path)
        with _session_operation(session_id):
            return await _analyze_parts_with_ai_impl(resolved_image, session_id)
    except InputValidationError as exc:
        _remove_session(session_id, reason="completed")
        logger.warning(f"Validation error in AI part analysis for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
        )
    except Exception:
        _remove_session(session_id, reason="completed")
        logger.exception(f"AI part analysis failed for session {session_id}")
        return _build_error_payload(
            error_code="ai_part_analysis_failed",
            message="AI part analysis failed. Check server logs for details.",
            session_id=session_id,
        )


@mcp.tool()
async def analyze_photo(image_path: str) -> dict[str, Any]:
    session_id = _create_session()
    try:
        resolved_image = _resolve_image_path(image_path)
        with _session_operation(session_id):
            return await _analyze_photo_impl(resolved_image, session_id)
    except InputValidationError as exc:
        _remove_session(session_id, reason="completed")
        logger.warning(f"Validation error in session {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
        )
    except Exception:
        _remove_session(session_id, reason="completed")
        logger.exception(f"Photo analysis failed for session {session_id}")
        return _build_error_payload(
            error_code="analysis_failed",
            message="Photo analysis failed. Check server logs for details.",
            session_id=session_id,
        )


@mcp.tool()
async def segment_detected_parts(session_id: str, output_dir: str) -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        with _session_operation(session_id):
            return await _segment_detected_parts_impl(session_id, resolved_output_dir)
    except InputValidationError as exc:
        logger.warning(f"Validation error in segment_detected_parts for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"AI part segmentation failed for session {session_id}")
        return _build_error_payload(
            error_code="ai_part_segmentation_failed",
            message="AI part segmentation failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def build_cubism_psd(
    session_id: str,
    output_dir: str,
    template_id: str = "standard_bust_up",
    model_name: str = "ATRI",
) -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        safe_model_name = _validate_model_name(model_name)
        with _session_operation(session_id):
            return await _build_cubism_psd_impl(
                session_id,
                resolved_output_dir,
                template_id,
                safe_model_name,
            )
    except InputValidationError as exc:
        logger.warning(f"Validation error in build_cubism_psd for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except FileNotFoundError as exc:
        logger.warning(f"Template error in build_cubism_psd for {session_id}: {exc}")
        return _build_error_payload(
            error_code="template_not_found",
            message=str(exc),
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Cubism PSD build failed for session {session_id}")
        return _build_error_payload(
            error_code="cubism_psd_build_failed",
            message="Cubism PSD build failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def prepare_cubism_automation(
    session_id: str,
    output_dir: str,
    template_id: str = "standard_bust_up",
    model_name: str = "ATRI",
    editor_path: str | None = None,
) -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        safe_model_name = _validate_model_name(model_name)
        with _session_operation(session_id):
            return await _prepare_cubism_automation_impl(
                session_id,
                resolved_output_dir,
                template_id,
                safe_model_name,
                editor_path,
            )
    except InputValidationError as exc:
        logger.warning(f"Validation error in prepare_cubism_automation for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Cubism automation planning failed for session {session_id}")
        return _build_error_payload(
            error_code="cubism_automation_prepare_failed",
            message="Cubism automation planning failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def validate_cubism_export(output_dir: str, model_name: str = "ATRI") -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        safe_model_name = _validate_model_name(model_name)
        return await _validate_cubism_export_impl(resolved_output_dir, safe_model_name)
    except InputValidationError as exc:
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            validation_errors=[str(exc)],
        )
    except Exception:
        logger.exception("Cubism export validation failed")
        return _build_error_payload(
            error_code="cubism_export_validation_failed",
            message="Cubism export validation failed. Check server logs for details.",
        )


@mcp.tool()
async def detect_face_features(session_id: str, output_dir: str) -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        with _session_operation(session_id):
            return await _detect_face_features_impl(session_id, resolved_output_dir)
    except InputValidationError as exc:
        logger.warning(f"Validation error in detect_face_features for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Face feature detection failed for session {session_id}")
        return _build_error_payload(
            error_code="face_detection_failed",
            message="Face feature detection failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def generate_layers(session_id: str, output_dir: str) -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        with _session_operation(session_id):
            return await _generate_layers_impl(session_id, resolved_output_dir)
    except InputValidationError as exc:
        logger.warning(f"Validation error in generate_layers for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Layer generation failed for session {session_id}")
        return _build_error_payload(
            error_code="layer_generation_failed",
            message="Layer generation failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def create_mesh(session_id: str) -> dict[str, Any]:
    try:
        with _session_operation(session_id):
            return await _create_mesh_impl(session_id)
    except InputValidationError as exc:
        logger.warning(f"Validation error in create_mesh for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Mesh generation failed for session {session_id}")
        return _build_error_payload(
            error_code="mesh_generation_failed",
            message="Mesh generation failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def setup_rigging(session_id: str) -> dict[str, Any]:
    try:
        with _session_operation(session_id):
            return await _setup_rigging_impl(session_id)
    except InputValidationError as exc:
        logger.warning(f"Validation error in setup_rigging for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Rigging failed for session {session_id}")
        return _build_error_payload(
            error_code="rigging_failed",
            message="Rigging failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def configure_physics(session_id: str) -> dict[str, Any]:
    try:
        with _session_operation(session_id):
            return await _configure_physics_impl(session_id)
    except InputValidationError as exc:
        logger.warning(f"Validation error in configure_physics for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Physics configuration failed for session {session_id}")
        return _build_error_payload(
            error_code="physics_configuration_failed",
            message="Physics configuration failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def generate_motions(session_id: str, motion_types: list[str]) -> dict[str, Any]:
    try:
        safe_motion_types = _validate_motion_types(motion_types)
        with _session_operation(session_id):
            return await _generate_motions_impl(session_id, safe_motion_types)
    except InputValidationError as exc:
        logger.warning(f"Validation error in generate_motions for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Motion generation failed for session {session_id}")
        return _build_error_payload(
            error_code="motion_generation_failed",
            message="Motion generation failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def export_model(
    session_id: str, output_dir: str, model_name: str = "ATRI"
) -> dict[str, Any]:
    try:
        resolved_output_dir = _resolve_output_dir(output_dir)
        safe_model_name = _validate_model_name(model_name)
        with _session_operation(session_id):
            return await _export_model_impl(session_id, resolved_output_dir, safe_model_name)
    except InputValidationError as exc:
        logger.warning(f"Validation error in export_model for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Export failed for session {session_id}")
        return _build_error_payload(
            error_code="export_failed",
            message="Export failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def close_session(session_id: str) -> dict[str, Any]:
    try:
        _validate_session_id(session_id)
        partial_outputs = _partial_outputs(session_id)
        if not _remove_session(session_id, reason="manual"):
            raise InputValidationError("Unknown or expired session_id. Nothing to close.")
        return {
            "status": "success",
            "session_id": session_id,
            "partial_outputs": partial_outputs,
            "message": "Session closed.",
        }
    except InputValidationError as exc:
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
        )


@mcp.tool()
async def full_pipeline(
    image_path: str,
    output_dir: str,
    model_name: str = "ATRI",
    motion_types: list[str] | None = None,
) -> dict[str, Any]:
    session_id = _create_session()
    results: dict[str, Any] = {
        "model_name": model_name,
        "session_id": session_id,
        "steps": [],
        "partial_outputs": {},
    }
    try:
        resolved_image = _resolve_image_path(image_path)
        resolved_output_dir = _resolve_output_dir(output_dir)
        safe_model_name = _validate_model_name(model_name)
        safe_motion_types = _validate_motion_types(motion_types)
        with _session_operation(session_id):
            step1 = await _analyze_photo_impl(resolved_image, session_id)
            _append_step(results, "analyze_photo", step1)
            step15 = await _detect_face_features_impl(session_id, resolved_output_dir)
            _append_step(results, "detect_face_features", step15)
            step2 = await _generate_layers_impl(session_id, resolved_output_dir)
            _append_step(results, "generate_layers", step2)
            step3 = await _create_mesh_impl(session_id)
            _append_step(results, "create_mesh", step3)
            step4 = await _setup_rigging_impl(session_id)
            _append_step(results, "setup_rigging", step4)
            step5 = await _configure_physics_impl(session_id)
            _append_step(results, "configure_physics", step5)
            step6 = await _generate_motions_impl(session_id, safe_motion_types)
            _append_step(results, "generate_motions", step6)
            export_result = await _export_model_impl(
                session_id, resolved_output_dir, safe_model_name
            )
            results["model_files"] = export_result.get("model_files", {})
            results["export_result"] = export_result.get("export_result", {})
            _append_step(results, "export_model", export_result)
        if results["export_result"].get("status") != "success":
            results.update(
                _build_error_payload(
                    error_code="export_failed",
                    message="Model export failed contract validation.",
                    session_id=session_id,
                    partial_outputs=_partial_outputs(session_id),
                )
            )
            return results
        results["status"] = "success"
        results["output_path"] = str(resolved_output_dir)
        results["motion_types"] = safe_motion_types
        results["message"] = (
            f"Mock intermediate Live2D package '{safe_model_name}' generated successfully. Finalize the exported bundle in Cubism Editor before production use."
        )
        return results
    except InputValidationError as exc:
        logger.warning(f"Validation error in session {session_id}: {exc}")
        results.update(
            _build_error_payload(
                error_code="invalid_input",
                message=str(exc),
                session_id=session_id,
                validation_errors=[str(exc)],
                partial_outputs=_partial_outputs(session_id),
            )
        )
    except Exception:
        logger.exception(f"Pipeline execution failed for session {session_id}")
        results.update(
            _build_error_payload(
                error_code="pipeline_failed",
                message="Pipeline execution failed. Check server logs for details.",
                session_id=session_id,
                partial_outputs=_partial_outputs(session_id),
            )
        )
    finally:
        _remove_session(session_id, reason="completed")
    return results


@mcp.resource("live2d://status")
def get_status() -> str:
    with session_lock:
        _prune_expired_sessions_locked()
        summaries = [_status_summary(record) for record in session_store.values()]
    return json.dumps(
        {
            "active_sessions": len(summaries),
            "busy_sessions": sum(1 for summary in summaries if summary["busy"]),
            "max_sessions": MAX_SESSIONS,
            "max_concurrent_operations": MAX_CONCURRENT_OPERATIONS,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "session_ids_hidden": True,
            "metrics": {
                "created_sessions": session_metrics.created_sessions,
                "rejected_sessions": session_metrics.rejected_sessions,
                "expired_sessions": session_metrics.expired_sessions,
                "closed_sessions": session_metrics.closed_sessions,
                "completed_sessions": session_metrics.completed_sessions,
            },
        },
        indent=2,
    )


@mcp.resource("live2d://templates")
def get_templates() -> str:
    mapper = TemplateMapper(_template_dirs())
    return json.dumps({"templates": mapper.available_templates()}, indent=2)


@mcp.prompt()
def photo_to_live2d_guide() -> str:
    return """
# Live2D Automation Guide

## Full Pipeline
Use `full_pipeline` for one-shot generation:

```json
{
  "image_path": "ATRI.png",
  "output_dir": "output/ATRI",
  "model_name": "ATRI",
  "motion_types": ["idle", "tap", "move", "emotional"]
}
```

## Step-by-Step
1. `analyze_photo(image_path)` to start a new session and receive a server-issued `session_id`
2. `detect_face_features(session_id, output_dir)`
3. `generate_layers(session_id, output_dir)`
4. `create_mesh(session_id)`
5. `setup_rigging(session_id)`
6. `configure_physics(session_id)`
7. `generate_motions(session_id, motion_types)`
8. `export_model(session_id, output_dir, model_name)`
9. `close_session(session_id)` when the step flow is complete

## Safety Rules
- `output_dir` must stay inside the project `output/` directory
- `model_name` only supports letters, numbers, `_`, and `-`
- supported image types: png, jpg, jpeg, webp
- maximum image size: 20 MiB, 4096x4096 pixels, 16,777,216 total pixels
- supported motion types: idle, tap, move, emotional

## Export Contract
- the exporter produces a mock intermediate `.moc3` package for Cubism Editor finalization
- `model3.json` and the exported file list always reference `{model_name}.moc3`
- `full_pipeline` closes its session automatically, while step-by-step flows should call `close_session`
"""


def main() -> None:
    mcp.run()
