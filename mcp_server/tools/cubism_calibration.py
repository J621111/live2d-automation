from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from mcp_server.artifacts import ArtifactStore

JsonDict = dict[str, Any]


def build_profile_calibration_report(
    bundle: JsonDict,
    execution: JsonDict,
) -> JsonDict:
    backend = str(bundle.get("backend", "native_gui"))
    profile = dict(bundle.get("native_controller", {}).get("profile") or {})
    probe_candidates = [
        str(item).strip()
        for item in profile.get("window_probe_candidates", [])
        if str(item).strip()
    ]
    report: JsonDict = {
        "backend": backend,
        "status": "ready" if backend == "native_gui" else "blocked",
        "model_name": bundle.get("model_name"),
        "window_title_contains": profile.get("window_title_contains"),
        "configured_window_probe_candidates": probe_candidates,
        "observed_titles": [],
        "resume_probe_titles": [],
        "visible_window_titles": [],
        "likely_window_titles": [],
        "missing_probe_candidates": [],
        "action_diagnostics": {},
        "suggestions": [],
        "message": (
            "Profile calibration data collected from native GUI execution artifacts."
            if backend == "native_gui"
            else "Profile calibration is currently available only for native_gui."
        ),
    }
    if backend != "native_gui":
        return report

    observed_titles: list[str] = []
    visible_titles: list[str] = []
    resume_probe = dict(execution.get("resume", {}).get("window_probe") or {})
    resume_titles = _titles_from_probe_like(resume_probe)
    observed_titles.extend(resume_titles)
    visible_titles.extend(_all_titles_from_probe_like(resume_probe))
    report["resume_probe_titles"] = resume_titles

    action_diagnostics: dict[str, Any] = {}
    for step in execution.get("executed_steps", []):
        action = str(step.get("source_action", ""))
        if not action:
            continue
        diagnostics: JsonDict = {
            "status": str(step.get("status", "pending")),
            "details": str(step.get("details", "")).strip(),
        }
        if action == "apply_template":
            diagnostics.update(_apply_template_profile_diagnostics(profile))
        elif action == "export_embedded_data":
            diagnostics.update(_export_profile_diagnostics(profile))
        artifact_path = step.get("artifact_path")
        if not artifact_path:
            if diagnostics:
                action_diagnostics[action] = diagnostics
            continue
        artifact = _read_json_artifact(artifact_path)
        if not artifact:
            if diagnostics:
                action_diagnostics[action] = diagnostics
            continue
        attempts = artifact.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            if diagnostics:
                action_diagnostics[action] = diagnostics
            continue
        action_titles: list[str] = []
        recovery_plans: list[JsonDict] = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            recovery = attempt.get("recovery")
            if not isinstance(recovery, dict):
                continue
            plan = recovery.get("dialog_recovery_plan")
            if isinstance(plan, dict):
                recovery_plans.append(plan)
            action_titles.extend(_titles_from_probe_like(recovery.get("probe")))
            visible_titles.extend(_all_titles_from_probe_like(recovery.get("probe")))
        if not action_titles and not recovery_plans:
            continue
        observed_titles.extend(action_titles)
        configured_dialogs = [
            str(entry.get("title_contains", "")).strip()
            for entry in _configured_dialog_recovery(action, profile)
            if str(entry.get("title_contains", "")).strip()
        ]
        inferred_dialog_titles = [
            title
            for title in _unique_items(action_titles)
            if title and title != str(profile.get("window_title_contains", "")).strip()
        ]
        missing_dialog_titles = [
            title
            for title in inferred_dialog_titles
            if title not in configured_dialogs and title not in probe_candidates
        ]
        diagnostics.update(
            {
                "configured_dialog_titles": configured_dialogs,
                "observed_titles": _unique_items(action_titles),
                "recovery_plan_sources": [
                    {
                        "dialog_source": plan.get("dialog_source"),
                        "sequence_source": plan.get("sequence_source"),
                    }
                    for plan in recovery_plans
                ],
                "suggested_dialog_titles": missing_dialog_titles,
            }
        )
        action_diagnostics[action] = diagnostics

    target_title = str(profile.get("window_title_contains", "")).strip()
    unique_observed = _unique_items(
        [title for title in observed_titles if title and title != target_title]
    )
    unique_visible = _unique_items(
        [title for title in visible_titles if title and title != target_title]
    )
    likely_window_titles = _likely_window_titles(unique_visible)
    report["observed_titles"] = unique_observed
    report["visible_window_titles"] = unique_visible
    report["likely_window_titles"] = likely_window_titles
    report["missing_probe_candidates"] = [
        title
        for title in (unique_observed or likely_window_titles)
        if title not in probe_candidates
    ]
    report["action_diagnostics"] = action_diagnostics
    report["suggestions"] = _calibration_suggestions(
        report["missing_probe_candidates"],
        action_diagnostics,
        likely_window_titles,
    )
    return report


def write_profile_calibration_report(
    report: JsonDict,
    output_dir: str,
    model_name: str,
    suffix: str = "",
) -> str:
    artifact_store = ArtifactStore(output_dir)
    report_path = artifact_store.write_json(
        f"{model_name}_cubism_profile_calibration{suffix}.json", report
    )
    return str(report_path)


def _read_json_artifact(artifact_path: str | os.PathLike[str]) -> JsonDict | None:
    try:
        loaded = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return cast(JsonDict, loaded)
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _titles_from_probe_like(probe: Any) -> list[str]:
    if not isinstance(probe, dict):
        return []
    diagnostics = probe.get("diagnostics")
    titles: list[str] = []
    if isinstance(diagnostics, list):
        for item in diagnostics:
            if isinstance(item, dict):
                title = str(item.get("Title", "")).strip()
                if title:
                    titles.append(title)
    matched_titles = probe.get("matched_titles")
    if isinstance(matched_titles, list):
        titles.extend(str(item).strip() for item in matched_titles if str(item).strip())
    return _unique_items(titles)


def _all_titles_from_probe_like(probe: Any) -> list[str]:
    if not isinstance(probe, dict):
        return []
    titles = _titles_from_probe_like(probe)
    diagnostics = probe.get("all_diagnostics")
    if isinstance(diagnostics, list):
        for item in diagnostics:
            if isinstance(item, dict):
                title = str(item.get("Title", "")).strip()
                if title:
                    titles.append(title)
    all_titles = probe.get("all_titles")
    if isinstance(all_titles, list):
        titles.extend(str(item).strip() for item in all_titles if str(item).strip())
    return _unique_items(titles)


def _likely_window_titles(values: list[str]) -> list[str]:
    keywords = (
        "cubism",
        "live2d",
        "editor",
        "import",
        "open",
        "template",
        "confirm",
        "export",
        "overwrite",
        "psd",
    )
    return [value for value in values if any(keyword in value.lower() for keyword in keywords)]


def _configured_dialog_recovery(action: str, profile: JsonDict) -> list[JsonDict]:
    dialogs_map = dict(profile.get("known_dialog_recovery") or {})
    selected = dialogs_map.get(action) or dialogs_map.get("default") or []
    return [entry for entry in selected if isinstance(entry, dict)]


def _unique_items(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _apply_template_profile_diagnostics(profile: JsonDict) -> JsonDict:
    shortcut = str(profile.get("template_shortcut", "")).strip()
    raw_menu_sequence = profile.get("template_menu_sequence", [])
    menu_sequence = (
        [entry for entry in raw_menu_sequence if isinstance(entry, dict)]
        if isinstance(raw_menu_sequence, list)
        else []
    )
    executable_sequence = [entry for entry in menu_sequence if str(entry.get("keys", "")).strip()]
    return {
        "template_shortcut_configured": bool(shortcut),
        "template_menu_sequence_configured": bool(executable_sequence),
        "template_menu_sequence_length": len(executable_sequence),
        "template_menu_sequence_entries": len(menu_sequence),
        "apply_template_ready": bool(shortcut or executable_sequence),
        "recommended_menu_path": ["Modeling", "Model template", "Apply template"],
        "documentation_hint": (
            "Calibrate this action against the Cubism menu path "
            "[Modeling] -> [Model template] -> [Apply template]."
        ),
    }


def _export_profile_diagnostics(profile: JsonDict) -> JsonDict:
    shortcut = str(profile.get("export_shortcut", "")).strip()
    raw_menu_sequence = profile.get("export_menu_sequence", [])
    menu_sequence = (
        [entry for entry in raw_menu_sequence if isinstance(entry, dict)]
        if isinstance(raw_menu_sequence, list)
        else []
    )
    executable_sequence = [entry for entry in menu_sequence if str(entry.get("keys", "")).strip()]
    return {
        "export_shortcut_configured": bool(shortcut),
        "export_menu_sequence_configured": bool(executable_sequence),
        "export_menu_sequence_length": len(executable_sequence),
        "export_menu_sequence_entries": len(menu_sequence),
        "export_embedded_data_ready": bool(shortcut or executable_sequence),
        "recommended_menu_path": ["File", "Export Embedded File", "Export as MOC3 file"],
        "documentation_hint": (
            "Calibrate this action against the Cubism menu path "
            "[File] -> [Export Embedded File] -> [Export as MOC3 file]."
        ),
    }


def _calibration_suggestions(
    missing_probe_candidates: list[str],
    action_diagnostics: dict[str, Any],
    likely_window_titles: list[str],
) -> list[str]:
    suggestions: list[str] = []
    if missing_probe_candidates:
        suggestions.append(
            "Consider adding these observed window titles to "
            f"`window_probe_candidates`: {', '.join(missing_probe_candidates)}."
        )
    elif likely_window_titles:
        suggestions.append(
            "Probe did not match the current `window_title_contains`, but these visible "
            f"window titles look relevant: {', '.join(likely_window_titles)}."
        )
    for action, diagnostics in action_diagnostics.items():
        suggested = diagnostics.get("suggested_dialog_titles", [])
        if suggested:
            suggestions.append(
                "Consider adding these titles to "
                f"`known_dialog_recovery.{action}`: {', '.join(suggested)}."
            )
        if action == "apply_template" and diagnostics.get("apply_template_ready") is False:
            suggestions.append(
                "Configure `template_menu_sequence` in the native GUI profile for "
                "`apply_template` using the Cubism menu path "
                "[Modeling] -> [Model template] -> [Apply template], or provide an "
                "adapter-backed implementation, before expecting template application "
                "to run in execute mode."
            )
        if (
            action == "export_embedded_data"
            and diagnostics.get("export_embedded_data_ready") is False
        ):
            suggestions.append(
                "Configure `export_menu_sequence` in the native GUI profile for "
                "`export_embedded_data` using the Cubism menu path "
                "[File] -> [Export Embedded File] -> [Export as MOC3 file], or provide "
                "an adapter-backed implementation, before expecting embedded export "
                "to run in execute mode."
            )
    if not suggestions:
        suggestions.append(
            "No new profile suggestions were inferred from this execution. "
            "If the run still failed, compare the probe diagnostics against "
            "your local Cubism window titles."
        )
    return suggestions
