"""Hardened Live2D Automation MCP server implementation."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from loguru import logger
from PIL import Image as PILImage
from PIL import ImageFile, UnidentifiedImageError

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from mcp_server.pipeline_services import CubismPipelineService, ImagePipelineService
from mcp_server.session_store import (
    InMemorySessionStore,
    SessionRecord,
    SessionRemovalReason,
    empty_session_state,
)
from mcp_server.tools.template_mapper import TemplateMapper
from mcp_server.validation import (
    OUTPUT_ROOT,
    InputValidationError,
)
from mcp_server.validation import (
    is_relative_to as _shared_is_relative_to,
)
from mcp_server.validation import (
    resolve_input_image_path as _shared_resolve_input_image_path,
)
from mcp_server.validation import (
    resolve_output_dir as _shared_resolve_output_dir,
)
from mcp_server.validation import (
    validate_model_name as _shared_validate_model_name,
)

mcp = FastMCP("live2d-automation")

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_777_216
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
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


_empty_state = empty_session_state
_session_store_manager = InMemorySessionStore(
    max_sessions=MAX_SESSIONS,
    ttl_seconds=SESSION_TTL_SECONDS,
    max_concurrent_operations=MAX_CONCURRENT_OPERATIONS,
)
session_store = _session_store_manager.records
session_lock = _session_store_manager.lock
session_metrics = _session_store_manager.metrics
_image_pipeline_service = ImagePipelineService(_session_store_manager)
_cubism_pipeline_service = CubismPipelineService(
    _session_store_manager,
    template_dirs=_template_dirs(),
    output_root=OUTPUT_ROOT,
)


def _is_relative_to(path: Path, root: Path) -> bool:
    return _shared_is_relative_to(path, root)


def _new_session_id() -> str:
    return _session_store_manager.new_session_id()


def _validate_session_id(session_id: str) -> str:
    return _session_store_manager.validate_session_id(session_id)


def _prune_expired_sessions_locked(now: float | None = None) -> None:
    _session_store_manager.prune_expired(now)


def _create_session() -> str:
    return _session_store_manager.create()


def _remove_session(
    session_id: str,
    *,
    reason: SessionRemovalReason = "manual",
) -> bool:
    return _session_store_manager.remove(session_id, reason=reason)


def _get_session_record(session_id: str, *, touch: bool = True) -> SessionRecord:
    return _session_store_manager.get(session_id, touch=touch)


def _get_session_state(session_id: str) -> dict[str, Any]:
    return _session_store_manager.get_state(session_id)


def _require_state_field(session_id: str, field: str, message: str) -> dict[str, Any]:
    return _session_store_manager.require_state_field(session_id, field, message)


def _active_operation_count() -> int:
    return _session_store_manager.active_operation_count()


@contextmanager
def _session_operation(session_id: str) -> Iterator[SessionRecord]:
    with _session_store_manager.operation(session_id) as record:
        yield record


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
    resolved = _shared_resolve_input_image_path(image_path)
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
    return _shared_resolve_output_dir(output_dir)


def _validate_model_name(model_name: str) -> str:
    return _shared_validate_model_name(model_name)


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
    return await _image_pipeline_service.analyze_photo(image_path, session_id)


async def _ensure_face_features_impl(
    session_id: str, output_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    return await _image_pipeline_service.ensure_face_features(session_id, output_dir)


async def _detect_face_features_impl(session_id: str, output_dir: Path) -> dict[str, Any]:
    return await _image_pipeline_service.detect_face_features(session_id, output_dir)


async def _analyze_parts_with_ai_impl(image_path: Path, session_id: str) -> dict[str, Any]:
    return await _image_pipeline_service.analyze_parts_with_ai(image_path, session_id)


async def _segment_detected_parts_impl(session_id: str, output_dir: Path) -> dict[str, Any]:
    return await _image_pipeline_service.segment_detected_parts(session_id, output_dir)


async def _build_cubism_psd_impl(
    session_id: str,
    output_dir: Path,
    template_id: str,
    model_name: str,
) -> dict[str, Any]:
    return await _cubism_pipeline_service.build_psd(
        session_id,
        output_dir,
        template_id,
        model_name,
    )


async def _prepare_cubism_automation_impl(
    session_id: str,
    output_dir: Path,
    template_id: str,
    model_name: str,
    editor_path: str | None,
    automation_backend: str | None,
) -> dict[str, Any]:
    return await _cubism_pipeline_service.prepare_automation(
        session_id,
        output_dir,
        template_id,
        model_name,
        editor_path,
        automation_backend,
    )


async def _execute_cubism_dispatch_impl(session_id: str) -> dict[str, Any]:
    return await _cubism_pipeline_service.execute_dispatch(session_id)


async def _resume_cubism_dispatch_impl(session_id: str) -> dict[str, Any]:
    return await _cubism_pipeline_service.resume_dispatch(session_id)


async def _validate_cubism_export_impl(output_dir: Path, model_name: str) -> dict[str, Any]:
    return await _cubism_pipeline_service.validate_export(output_dir, model_name)


async def _generate_layers_impl(session_id: str, output_dir: Path) -> dict[str, Any]:
    return await _image_pipeline_service.generate_layers(session_id, output_dir)


async def _create_mesh_impl(session_id: str) -> dict[str, Any]:
    return await _image_pipeline_service.create_mesh(session_id)


async def _setup_rigging_impl(session_id: str) -> dict[str, Any]:
    return await _image_pipeline_service.setup_rigging(session_id)


async def _configure_physics_impl(session_id: str) -> dict[str, Any]:
    return await _image_pipeline_service.configure_physics(session_id)


async def _generate_motions_impl(session_id: str, motion_types: list[str]) -> dict[str, Any]:
    return await _image_pipeline_service.generate_motions(session_id, motion_types)


async def _export_model_impl(session_id: str, output_dir: Path, model_name: str) -> dict[str, Any]:
    return await _image_pipeline_service.export_model(session_id, output_dir, model_name)


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
    automation_backend: str | None = None,
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
                automation_backend,
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
async def execute_cubism_dispatch(session_id: str) -> dict[str, Any]:
    try:
        with _session_operation(session_id):
            return await _execute_cubism_dispatch_impl(session_id)
    except InputValidationError as exc:
        logger.warning(f"Validation error in execute_cubism_dispatch for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Cubism dispatch execution failed for session {session_id}")
        return _build_error_payload(
            error_code="cubism_dispatch_execution_failed",
            message="Cubism dispatch execution failed. Check server logs for details.",
            session_id=session_id,
            partial_outputs=_partial_outputs(session_id),
        )


@mcp.tool()
async def resume_cubism_dispatch(session_id: str) -> dict[str, Any]:
    try:
        with _session_operation(session_id):
            return await _resume_cubism_dispatch_impl(session_id)
    except InputValidationError as exc:
        logger.warning(f"Validation error in resume_cubism_dispatch for {session_id}: {exc}")
        return _build_error_payload(
            error_code="invalid_input",
            message=str(exc),
            session_id=session_id,
            validation_errors=[str(exc)],
            partial_outputs=_partial_outputs(session_id),
        )
    except Exception:
        logger.exception(f"Cubism dispatch resume failed for session {session_id}")
        return _build_error_payload(
            error_code="cubism_dispatch_resume_failed",
            message="Cubism dispatch resume failed. Check server logs for details.",
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
