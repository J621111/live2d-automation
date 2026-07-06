from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp_server.tools import native_gui_scripts

JsonDict = dict[str, Any]
PowerShellRunner = Callable[[Path], subprocess.CompletedProcess[str]]
ArtifactSerializer = Callable[[JsonDict], str]


def _run_probe_script(
    *,
    script_path: Path,
    artifact_path: Path,
    profile: JsonDict,
    run_powershell: PowerShellRunner,
    artifact_json: ArtifactSerializer,
    extra_fragments: list[str] | None = None,
) -> JsonDict:
    script_path.write_text(
        native_gui_scripts.window_probe_script(profile, extra_fragments=extra_fragments),
        encoding="utf-8",
    )
    try:
        completed = run_powershell(script_path)
        payload: JsonDict = {
            "script_path": str(script_path),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        payload.update(native_gui_scripts.parse_probe_stdout(payload["stdout"]))
    except subprocess.TimeoutExpired as exc:
        payload = {
            "script_path": str(script_path),
            "returncode": None,
            "stdout": "",
            "stderr": f"Timed out after {exc.timeout} seconds.",
            "timed_out": True,
        }
    artifact_path.write_text(artifact_json(payload), encoding="utf-8")
    return payload


def probe_window(
    *,
    controller: JsonDict,
    output_dir: Path,
    run_powershell: PowerShellRunner,
    artifact_json: ArtifactSerializer,
) -> JsonDict | None:
    if controller.get("status") != "ready" or controller.get("mode") != "execute":
        return None
    profile = dict(controller.get("profile") or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "native_gui_builtin_window_probe.ps1"
    artifact_path = output_dir / "native_gui_builtin_window_probe_result.json"
    payload = _run_probe_script(
        script_path=script_path,
        artifact_path=artifact_path,
        profile=profile,
        run_powershell=run_powershell,
        artifact_json=artifact_json,
    )
    return {
        "status": "success" if payload.get("returncode") == 0 else "error",
        "artifact_path": str(artifact_path),
        "script_path": str(script_path),
        "returncode": payload.get("returncode"),
        "timed_out": payload.get("timed_out", False),
        **({"diagnostics": payload["diagnostics"]} if "diagnostics" in payload else {}),
        **({"all_titles": payload["all_titles"]} if "all_titles" in payload else {}),
        **({"all_diagnostics": payload["all_diagnostics"]} if "all_diagnostics" in payload else {}),
    }


def capture_failure_context(
    *,
    source_action: str,
    output_dir: Path,
    profile: JsonDict,
    run_powershell: PowerShellRunner,
    artifact_json: ArtifactSerializer,
) -> JsonDict:
    if profile.get("capture_screenshot_on_error", True) is False:
        return {
            "status": "disabled",
            "details": "Failure screenshot capture is disabled in the controller profile.",
        }

    stem = f"native_gui_builtin_{source_action}_failure_capture"
    script_path = output_dir / f"{stem}.ps1"
    screenshot_path = output_dir / f"{stem}.png"
    artifact_path = output_dir / f"{stem}.json"
    script_path.write_text(
        native_gui_scripts.capture_script(screenshot_path, profile),
        encoding="utf-8",
    )
    completed = run_powershell(script_path)
    payload: JsonDict = {
        "script_path": str(script_path),
        "screenshot_path": str(screenshot_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    artifact_path.write_text(artifact_json(payload), encoding="utf-8")
    return {
        "status": "success" if completed.returncode == 0 else "error",
        "artifact_path": str(artifact_path),
        "script_path": str(script_path),
        "screenshot_path": str(screenshot_path),
        "returncode": completed.returncode,
    }


def capture_success_context(
    *,
    source_action: str,
    output_dir: Path,
    profile: JsonDict,
    run_powershell: PowerShellRunner,
    artifact_json: ArtifactSerializer,
) -> JsonDict:
    if profile.get("capture_screenshot_on_success", True) is False:
        return {
            "status": "disabled",
            "details": "Success screenshot capture is disabled in the controller profile.",
        }

    stem = f"native_gui_builtin_{source_action}_success_capture"
    script_path = output_dir / f"{stem}.ps1"
    screenshot_path = output_dir / f"{stem}.png"
    artifact_path = output_dir / f"{stem}.json"
    script_path.write_text(
        native_gui_scripts.capture_script(
            screenshot_path,
            profile,
            wait_key="success_capture_wait_seconds",
        ),
        encoding="utf-8",
    )
    completed = run_powershell(script_path)
    payload: JsonDict = {
        "script_path": str(script_path),
        "screenshot_path": str(screenshot_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    artifact_path.write_text(artifact_json(payload), encoding="utf-8")
    return {
        "status": "success" if completed.returncode == 0 else "error",
        "artifact_path": str(artifact_path),
        "script_path": str(script_path),
        "screenshot_path": str(screenshot_path),
        "returncode": completed.returncode,
    }


def capture_post_action_probe(
    *,
    source_action: str,
    output_dir: Path,
    profile: JsonDict,
    run_powershell: PowerShellRunner,
    artifact_json: ArtifactSerializer,
) -> JsonDict:
    stem = f"native_gui_builtin_{source_action}_post_probe"
    script_path = output_dir / f"{stem}.ps1"
    artifact_path = output_dir / f"{stem}.json"
    payload = _run_probe_script(
        script_path=script_path,
        artifact_path=artifact_path,
        profile=profile,
        run_powershell=run_powershell,
        artifact_json=artifact_json,
    )
    return {
        "status": "success" if payload.get("returncode") == 0 else "error",
        "artifact_path": str(artifact_path),
        "script_path": str(script_path),
        "returncode": payload.get("returncode"),
        "timed_out": payload.get("timed_out", False),
        **({"matched_titles": payload["matched_titles"]} if "matched_titles" in payload else {}),
        **({"all_titles": payload["all_titles"]} if "all_titles" in payload else {}),
    }


def await_export_outputs(
    *,
    output_dir: Path,
    model_name: str,
    profile: JsonDict,
) -> JsonDict:
    timeout_seconds = max(0.0, float(profile.get("export_output_timeout_seconds", 6.0)))
    poll_seconds = max(0.05, float(profile.get("export_output_poll_seconds", 0.25)))
    deadline = time.time() + timeout_seconds
    moc3_path = output_dir / f"{model_name}.moc3"
    model3_path = output_dir / "model3.json"
    textures_dir = output_dir / "textures"
    while time.time() <= deadline:
        if moc3_path.exists() and model3_path.exists() and textures_dir.exists():
            return {
                "status": "success",
                "details": "Detected exported Cubism files in the target output directory.",
                "observed": {
                    "moc3": str(moc3_path),
                    "model3": str(model3_path),
                    "textures_dir": str(textures_dir),
                },
            }
        time.sleep(poll_seconds)
    missing: list[str] = []
    if not moc3_path.exists():
        missing.append("moc3")
    if not model3_path.exists():
        missing.append("model3")
    if not textures_dir.exists():
        missing.append("textures_dir")
    return {
        "status": "error",
        "details": (
            "Export command finished, but the expected files did not appear in the target "
            f"directory: {', '.join(missing)}."
        ),
        "missing": missing,
    }


def await_window_title_fragment(
    *,
    output_dir: Path,
    profile: JsonDict,
    fragment: str,
    stem: str,
    run_powershell: PowerShellRunner,
    artifact_json: ArtifactSerializer,
) -> JsonDict:
    timeout_seconds = max(0.0, float(profile.get("document_window_timeout_seconds", 8.0)))
    poll_seconds = max(0.05, float(profile.get("document_window_poll_seconds", 0.25)))
    deadline = time.time() + timeout_seconds
    script_path = output_dir / f"{stem}.ps1"
    artifact_path = output_dir / f"{stem}.json"
    target_fragment = fragment.strip()
    payload: JsonDict = {}
    while time.time() <= deadline:
        payload = _run_probe_script(
            script_path=script_path,
            artifact_path=artifact_path,
            profile=profile,
            run_powershell=run_powershell,
            artifact_json=artifact_json,
            extra_fragments=[target_fragment],
        )
        matched_titles = []
        for entry in payload.get("all_diagnostics", []):
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("Title", ""))
            process_name = str(entry.get("ProcessName", "")).lower()
            title_lower = title.lower()
            if target_fragment.lower() not in title_lower:
                continue
            if "cubism" in title_lower or "editor" in title_lower or process_name == "java":
                matched_titles.append(title)
        if matched_titles:
            return {
                "status": "success",
                "details": f"Detected a Cubism window containing '{target_fragment}' after import.",
                "artifact_path": str(artifact_path),
                "matched_titles": matched_titles,
            }
        time.sleep(poll_seconds)
    return {
        "status": "error",
        "details": f"Did not observe a Cubism window containing '{target_fragment}' after import.",
        "artifact_path": str(artifact_path),
        "observed_titles": payload.get("all_titles", []),
    }
