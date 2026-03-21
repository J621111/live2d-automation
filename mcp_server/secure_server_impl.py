"""
Hardened Live2D Automation MCP server implementation.
"""

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.mesh_generator import ArtMeshGenerator
from mcp_server.tools.auto_rigger import AutoRigger
from mcp_server.tools.facial_detector import FacialFeatureDetector
from mcp_server.tools.image_processor import ImageProcessor
from mcp_server.tools.layer_generator import LayerGenerator
from mcp_server.tools.moc3_generator import Live2DExporter as Moc3Exporter
from mcp_server.tools.motion_generator import MotionGenerator
from mcp_server.tools.physics_setup import PhysicsSetup

mcp = FastMCP("live2d-automation")

OUTPUT_ROOT = (project_root / "output").resolve()
MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

session_store: Dict[str, Dict[str, Any]] = {}


class InputValidationError(ValueError):
    """Raised when a user-controlled input is unsafe or invalid."""


def _empty_state() -> Dict[str, Any]:
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
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _new_session_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def _ensure_session(session_id: Optional[str] = None, reset: bool = False) -> str:
    if session_id is None:
        session_id = _new_session_id()
    if not SESSION_ID_RE.match(session_id):
        raise InputValidationError("session_id contains unsupported characters.")
    if reset or session_id not in session_store:
        session_store[session_id] = _empty_state()
    return session_id


def _get_session_state(session_id: str) -> Dict[str, Any]:
    if session_id not in session_store:
        raise InputValidationError("Unknown session_id. Run analyze_photo first.")
    return session_store[session_id]


def _require_state_field(session_id: str, field: str, message: str) -> Dict[str, Any]:
    state = _get_session_state(session_id)
    if not state.get(field):
        raise InputValidationError(message)
    return state


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

    return resolved


def _resolve_output_dir(output_dir: str) -> Path:
    if not output_dir or not output_dir.strip():
        raise InputValidationError("output_dir is required.")

    raw_path = Path(output_dir)
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


def _status_summary(session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "has_image": bool(state.get("input_image")),
        "segments_ready": bool(state.get("segments")),
        "layers_ready": len(state.get("layers", [])),
        "meshes_ready": len(state.get("meshes", {})),
        "motions_ready": len(state.get("motions", [])),
        "has_export": bool(state.get("model_files")),
    }


async def _analyze_photo_impl(image_path: Path, session_id: str) -> Dict[str, Any]:
    logger.info(f"Analyzing photo for session {session_id}: {image_path.name}")
    processor = ImageProcessor()
    segments = await processor.analyze(str(image_path))

    state = _get_session_state(session_id)
    state.clear()
    state.update(_empty_state())
    state["input_image"] = str(image_path)
    state["segments"] = segments

    return {
        "status": "success",
        "session_id": session_id,
        "parts_detected": len(segments.get("body_parts", {})),
        "segments": segments,
        "message": f"Detected {len(segments.get('body_parts', {}))} body parts.",
    }


async def _detect_face_features_impl(session_id: str, output_dir: Path) -> Dict[str, Any]:
    state = _require_state_field(session_id, "input_image", "Run analyze_photo before face extraction.")
    logger.info(f"Detecting face features for session {session_id}")

    detector = FacialFeatureDetector()
    face_features = await detector.detect_features(state["input_image"])
    face_output_dir = output_dir / "face_textures"
    face_output_dir.mkdir(parents=True, exist_ok=True)
    face_layers = await detector.extract_face_parts(state["input_image"], str(face_output_dir))

    state["face_features"] = face_features
    state["face_layers"] = face_layers

    return {
        "status": "success",
        "session_id": session_id,
        "parts_detected": len(face_features.get("parts", {})),
        "layers_extracted": len(face_layers),
        "message": f"Detected {len(face_features.get('parts', {}))} face parts.",
    }


async def _generate_layers_impl(session_id: str, output_dir: Path) -> Dict[str, Any]:
    state = _require_state_field(session_id, "segments", "Run analyze_photo before generate_layers.")
    logger.info(f"Generating layers for session {session_id}")

    generator = LayerGenerator()
    layers = await generator.generate(
        image_path=state["input_image"],
        segments=state["segments"],
        output_dir=str(output_dir),
    )

    state["output_dir"] = str(output_dir)
    state["layers"] = layers

    return {
        "status": "success",
        "session_id": session_id,
        "layers_generated": len(layers),
        "layers": layers,
        "message": f"Generated {len(layers)} layers.",
    }


async def _create_mesh_impl(session_id: str) -> Dict[str, Any]:
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


async def _setup_rigging_impl(session_id: str) -> Dict[str, Any]:
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
        "message": (
            f"Created {len(rigging.get('bones', []))} bones and "
            f"{len(rigging.get('parameters', []))} parameters."
        ),
    }


async def _configure_physics_impl(session_id: str) -> Dict[str, Any]:
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


async def _generate_motions_impl(session_id: str, motion_types: List[str]) -> Dict[str, Any]:
    state = _require_state_field(session_id, "rigging", "Run setup_rigging before generate_motions.")
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


@mcp.tool()
async def analyze_photo(image_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    session_id = _ensure_session(session_id, reset=True)
    resolved_image = _resolve_image_path(image_path)
    return await _analyze_photo_impl(resolved_image, session_id)


@mcp.tool()
async def generate_layers(session_id: str, output_dir: str) -> Dict[str, Any]:
    resolved_output_dir = _resolve_output_dir(output_dir)
    return await _generate_layers_impl(session_id, resolved_output_dir)


@mcp.tool()
async def create_mesh(session_id: str) -> Dict[str, Any]:
    return await _create_mesh_impl(session_id)


@mcp.tool()
async def setup_rigging(session_id: str) -> Dict[str, Any]:
    return await _setup_rigging_impl(session_id)


@mcp.tool()
async def configure_physics(session_id: str) -> Dict[str, Any]:
    return await _configure_physics_impl(session_id)


@mcp.tool()
async def generate_motions(session_id: str, motion_types: List[str]) -> Dict[str, Any]:
    return await _generate_motions_impl(session_id, motion_types)


@mcp.tool()
async def full_pipeline(
    image_path: str,
    output_dir: str,
    model_name: str = "ATRI",
    motion_types: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    if motion_types is None:
        motion_types = ["idle", "tap", "move", "emotional"]

    session_id = _ensure_session(session_id, reset=True)
    results: Dict[str, Any] = {"model_name": model_name, "session_id": session_id, "steps": []}

    try:
        resolved_image = _resolve_image_path(image_path)
        resolved_output_dir = _resolve_output_dir(output_dir)
        safe_model_name = _validate_model_name(model_name)

        step1 = await _analyze_photo_impl(resolved_image, session_id)
        results["steps"].append({"name": "analyze_photo", "result": step1})

        step15 = await _detect_face_features_impl(session_id, resolved_output_dir)
        results["steps"].append({"name": "detect_face_features", "result": step15})

        step2 = await _generate_layers_impl(session_id, resolved_output_dir)
        results["steps"].append({"name": "generate_layers", "result": step2})

        step3 = await _create_mesh_impl(session_id)
        results["steps"].append({"name": "create_mesh", "result": step3})

        step4 = await _setup_rigging_impl(session_id)
        results["steps"].append({"name": "setup_rigging", "result": step4})

        step5 = await _configure_physics_impl(session_id)
        results["steps"].append({"name": "configure_physics", "result": step5})

        step6 = await _generate_motions_impl(session_id, motion_types)
        results["steps"].append({"name": "generate_motions", "result": step6})

        exporter = Moc3Exporter()
        state = _get_session_state(session_id)
        model_files = await exporter.export(
            model_name=safe_model_name,
            output_dir=str(resolved_output_dir),
            state=state,
        )
        state["model_files"] = model_files

        results["model_files"] = model_files
        results["steps"].append({"name": "export_model", "result": model_files})
        results["status"] = "success"
        results["message"] = f"Live2D model '{safe_model_name}' generated successfully."
        results["output_path"] = str(resolved_output_dir)

    except InputValidationError as exc:
        logger.warning(f"Validation error in session {session_id}: {exc}")
        results["status"] = "error"
        results["error"] = "invalid_input"
        results["message"] = str(exc)
    except Exception:
        logger.exception(f"Pipeline execution failed for session {session_id}")
        results["status"] = "error"
        results["error"] = "pipeline_failed"
        results["message"] = "Pipeline execution failed. Check server logs for details."

    return results


@mcp.resource("live2d://status")
def get_status() -> str:
    summaries = [
        _status_summary(session_id, state)
        for session_id, state in session_store.items()
    ]
    return json.dumps({"active_sessions": len(summaries), "sessions": summaries}, indent=2)


@mcp.resource("live2d://templates")
def get_templates() -> str:
    templates_dir = project_root / "templates"
    templates = []
    if templates_dir.exists():
        templates = [f.stem for f in templates_dir.glob("*.json")]
    return json.dumps({"templates": templates}, indent=2)


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
1. `analyze_photo` to start a session and get a `session_id`
2. `generate_layers(session_id, output_dir)`
3. `create_mesh(session_id)`
4. `setup_rigging(session_id)`
5. `configure_physics(session_id)`
6. `generate_motions(session_id, motion_types)`

## Safety Rules
- `output_dir` must stay inside the project `output/` directory
- `model_name` only supports letters, numbers, `_`, and `-`
- supported image types: png, jpg, jpeg, webp
"""


def main() -> None:
    mcp.run()
