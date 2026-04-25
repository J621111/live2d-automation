from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp_server.secure_server_impl import (
    analyze_photo,
    build_cubism_psd,
    close_session,
    execute_cubism_dispatch,
    generate_layers,
    prepare_cubism_automation,
    validate_cubism_export,
)
from mcp_server.tools.cubism_automation import CubismAutomationManager
from mcp_server.tools.cubism_bridge import CubismBridge
from mcp_server.validation import (
    OUTPUT_ROOT,
    InputValidationError,
    resolve_output_dir,
    validate_model_name,
)

JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_ADAPTER = PROJECT_ROOT / "scripts" / "native_gui_adapter_demo.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live2d-run",
        description="Run the Cubism-ready Live2D pipeline without an MCP client.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run image analysis, PSD packaging, automation preparation, dispatch, and validation.",
    )
    run_parser.add_argument("--image-path", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--model-name", default="ATRI")
    run_parser.add_argument("--template-id", default="standard_bust_up")
    run_parser.add_argument("--editor-path")
    run_parser.add_argument(
        "--automation-backend",
        choices=["native_gui", "opencli"],
        default="native_gui",
    )
    run_parser.add_argument("--adapter-command")
    run_parser.add_argument(
        "--native-gui-controller-mode",
        choices=["disabled", "dry_run", "execute"],
        default="disabled",
        help="Enable the built-in Windows GUI controller.",
    )
    run_parser.add_argument("--native-gui-profile")
    run_parser.add_argument(
        "--demo-adapter-mode",
        choices=["partial", "full", "fail"],
        help="Use the bundled demo adapter with the selected mode.",
    )
    run_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after writing the Cubism automation plan and dispatch bundle.",
    )
    run_parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the final explicit validate_cubism_export call.",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate-template",
        help=(
            "Prepare and execute the Cubism automation flow against an existing PSD package "
            "without re-running image analysis."
        ),
    )
    calibrate_parser.add_argument("--output-dir", required=True)
    calibrate_parser.add_argument("--model-name", default="ATRI")
    calibrate_parser.add_argument("--template-id", default="standard_bust_up")
    calibrate_parser.add_argument("--psd-path")
    calibrate_parser.add_argument("--editor-path")
    calibrate_parser.add_argument(
        "--automation-backend",
        choices=["native_gui", "opencli"],
        default="native_gui",
    )
    calibrate_parser.add_argument("--adapter-command")
    calibrate_parser.add_argument(
        "--native-gui-controller-mode",
        choices=["disabled", "dry_run", "execute"],
        default="disabled",
        help="Enable the built-in Windows GUI controller.",
    )
    calibrate_parser.add_argument("--native-gui-profile")
    calibrate_parser.add_argument(
        "--demo-adapter-mode",
        choices=["partial", "full", "fail"],
        help="Use the bundled demo adapter with the selected mode.",
    )
    calibrate_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after writing the Cubism automation plan and dispatch bundle.",
    )
    calibrate_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest dispatch execution artifact in the output directory.",
    )
    return parser


def _apply_cli_environment(args: argparse.Namespace) -> dict[str, str | None]:
    previous = {
        "LIVE2D_NATIVE_GUI_ADAPTER_COMMAND": os.getenv("LIVE2D_NATIVE_GUI_ADAPTER_COMMAND"),
        "LIVE2D_NATIVE_GUI_CONTROLLER_MODE": os.getenv("LIVE2D_NATIVE_GUI_CONTROLLER_MODE"),
        "LIVE2D_NATIVE_GUI_PROFILE": os.getenv("LIVE2D_NATIVE_GUI_PROFILE"),
    }
    os.environ["LIVE2D_NATIVE_GUI_CONTROLLER_MODE"] = args.native_gui_controller_mode
    if args.native_gui_profile:
        os.environ["LIVE2D_NATIVE_GUI_PROFILE"] = args.native_gui_profile
    if args.adapter_command:
        os.environ["LIVE2D_NATIVE_GUI_ADAPTER_COMMAND"] = args.adapter_command
    elif args.demo_adapter_mode:
        os.environ["LIVE2D_NATIVE_GUI_ADAPTER_COMMAND"] = (
            f'"{sys.executable}" "{DEMO_ADAPTER}" --mode {args.demo_adapter_mode}'
        )
    return previous


def _restore_cli_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _report_path(output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_name}_cli_report.json"


def _status_code(status: str) -> int:
    if status == "success":
        return 0
    if status in {"partial", "blocked"}:
        return 2
    return 1


def _invalid_input_report(command: str, message: str) -> JsonDict:
    return {
        "command": command,
        "output_dir": str(OUTPUT_ROOT / "cli_errors"),
        "model_name": "invalid_input",
        "status": "error",
        "steps": {
            "validate_cli_inputs": {
                "status": "error",
                "message": message,
            }
        },
    }


def _latest_execution_artifact(output_dir: Path, model_name: str) -> JsonDict | None:
    pattern = f"{model_name}_cubism_dispatch_execution*.json"
    candidates = sorted(output_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    for candidate in reversed(candidates):
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _resolve_calibration_psd_path(
    args: argparse.Namespace, output_dir: Path, model_name: str
) -> Path:
    if args.psd_path:
        return Path(args.psd_path)
    return output_dir / f"{model_name}.psd"


def _build_calibration_resume_context(
    args: argparse.Namespace,
    psd_path: Path,
    editor_info: JsonDict,
) -> JsonDict:
    stat = psd_path.stat()
    return {
        "command": "calibrate-template",
        "automation_backend": args.automation_backend,
        "editor_path": editor_info.get("editor_path"),
        "model_name": args.model_name,
        "native_gui_controller_mode": args.native_gui_controller_mode,
        "native_gui_profile": args.native_gui_profile,
        "psd_path": str(psd_path.resolve()),
        "psd_size": stat.st_size,
        "psd_mtime_ns": stat.st_mtime_ns,
        "template_id": args.template_id,
    }


def _validate_resume_context(
    previous_execution: JsonDict | None,
    current_context: JsonDict,
) -> tuple[bool, str]:
    if not previous_execution:
        return False, "No previous dispatch execution artifact was found."
    previous_context = previous_execution.get("resume_context")
    if not isinstance(previous_context, dict):
        return False, "The previous dispatch execution is missing resume metadata."
    if previous_context != current_context:
        return False, (
            "The previous dispatch execution does not match the current PSD/template/editor "
            "combination."
        )
    return True, "Resume metadata matches the current calibration inputs."


def _pick_status(*results: JsonDict) -> str:
    statuses = [str(result.get("status", "error")) for result in results if result]
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "blocked" for status in statuses):
        return "blocked"
    if any(status == "partial" for status in statuses):
        return "partial"
    return "success"


async def _run_pipeline(args: argparse.Namespace) -> JsonDict:
    try:
        output_dir = resolve_output_dir(args.output_dir)
        model_name = validate_model_name(args.model_name)
    except InputValidationError as exc:
        return _invalid_input_report("run", str(exc))
    semantic_dir = output_dir / "semantic_layers"
    report: JsonDict = {
        "command": "run",
        "image_path": args.image_path,
        "output_dir": str(output_dir),
        "model_name": model_name,
        "template_id": args.template_id,
        "automation_backend": args.automation_backend,
        "demo_adapter_mode": args.demo_adapter_mode,
        "native_gui_controller_mode": args.native_gui_controller_mode,
        "native_gui_profile": args.native_gui_profile,
        "prepare_only": args.prepare_only,
        "skip_validate": args.skip_validate,
        "steps": {},
    }

    analyze_result = await analyze_photo(args.image_path)
    report["steps"]["analyze_photo"] = analyze_result
    session_id = analyze_result.get("session_id")
    if analyze_result.get("status") != "success" or not session_id:
        report["status"] = "error"
        return report

    try:
        layers_result = await generate_layers(session_id, str(semantic_dir))
        report["steps"]["generate_layers"] = layers_result
        if layers_result.get("status") != "success":
            report["status"] = "error"
            return report

        psd_result = await build_cubism_psd(
            session_id,
            str(output_dir),
            template_id=args.template_id,
            model_name=model_name,
        )
        report["steps"]["build_cubism_psd"] = psd_result
        if psd_result.get("status") != "success":
            report["status"] = "error"
            return report

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(output_dir),
            template_id=args.template_id,
            model_name=model_name,
            editor_path=args.editor_path,
            automation_backend=args.automation_backend,
        )
        report["steps"]["prepare_cubism_automation"] = prepare_result

        if args.prepare_only:
            report["status"] = prepare_result.get("status", "blocked")
            return report

        execute_result = await execute_cubism_dispatch(session_id)
        report["steps"]["execute_cubism_dispatch"] = execute_result

        validate_result: JsonDict = {}
        if not args.skip_validate and execute_result.get("status") == "success":
            validate_result = await validate_cubism_export(str(output_dir), model_name)
            report["steps"]["validate_cubism_export"] = validate_result

        report["status"] = _pick_status(prepare_result, execute_result, validate_result)
        return report
    finally:
        close_result = await close_session(session_id)
        report["steps"]["close_session"] = close_result


async def _run_template_calibration(args: argparse.Namespace) -> JsonDict:
    try:
        output_dir = resolve_output_dir(args.output_dir)
        model_name = validate_model_name(args.model_name)
    except InputValidationError as exc:
        return _invalid_input_report("calibrate-template", str(exc))
    psd_path = _resolve_calibration_psd_path(args, output_dir, model_name)
    report: JsonDict = {
        "command": "calibrate-template",
        "output_dir": str(output_dir),
        "model_name": model_name,
        "template_id": args.template_id,
        "psd_path": str(psd_path),
        "editor_path": args.editor_path,
        "automation_backend": args.automation_backend,
        "demo_adapter_mode": args.demo_adapter_mode,
        "native_gui_controller_mode": args.native_gui_controller_mode,
        "native_gui_profile": args.native_gui_profile,
        "prepare_only": args.prepare_only,
        "resume": args.resume,
        "steps": {},
    }

    if not psd_path.exists():
        report["status"] = "error"
        report["steps"]["prepare_cubism_automation"] = {
            "status": "error",
            "message": f"PSD path does not exist: {psd_path}",
        }
        return report
    resolved_psd_path = psd_path.resolve()
    report["psd_path"] = str(resolved_psd_path)

    bridge = CubismBridge()
    manager = CubismAutomationManager()
    editor_info = bridge.discover_editor(args.editor_path)
    report["steps"]["discover_editor"] = editor_info

    plan = bridge.build_plan(
        psd_path=str(resolved_psd_path),
        output_dir=str(output_dir),
        template_id=args.template_id,
        model_name=model_name,
        editor_info=editor_info,
        automation_backend=args.automation_backend,
    )
    execution_prep = manager.prepare_execution(
        args.automation_backend,
        editor_info=editor_info,
        plan=plan,
    )
    plan["execution"] = execution_prep
    plan_path = bridge.write_plan(plan, str(output_dir), model_name)
    report["steps"]["prepare_cubism_automation"] = {
        "status": execution_prep.get("status", "blocked"),
        "automation_backend": args.automation_backend,
        "automation_mode": execution_prep.get("automation_mode"),
        "missing_requirements": execution_prep.get("missing_requirements", []),
        "warnings": execution_prep.get("warnings", []),
        "editor": editor_info,
        "plan_path": plan_path,
    }

    bundle = manager.build_dispatch_bundle(
        args.automation_backend,
        plan=plan,
        execution=execution_prep,
        template_id=args.template_id,
        model_name=model_name,
        psd_path=str(resolved_psd_path),
        output_dir=str(output_dir),
        editor_info=editor_info,
    )
    bundle_path = manager.write_dispatch_bundle(bundle, str(output_dir), model_name)
    report["steps"]["prepare_cubism_automation"]["dispatch_bundle_path"] = bundle_path

    if args.prepare_only:
        report["status"] = execution_prep.get("status", "blocked")
        return report

    resume_context = _build_calibration_resume_context(args, resolved_psd_path, editor_info)
    resume_context["model_name"] = model_name
    previous_execution = None
    effective_resume = False
    if args.resume:
        candidate_execution = _latest_execution_artifact(output_dir, model_name)
        can_resume, resume_message = _validate_resume_context(
            candidate_execution,
            resume_context,
        )
        resume_status = "ready" if can_resume else "ignored"
        report["steps"]["resume_validation"] = {
            "status": resume_status,
            "message": resume_message,
        }
        if can_resume:
            previous_execution = candidate_execution
            effective_resume = True

    execution = manager.execute_dispatch_bundle(
        bundle,
        previous_execution=previous_execution,
        resume=effective_resume,
    )
    execution["resume_context"] = resume_context
    execution_suffix = "_resume" if effective_resume else ""
    execution_path = manager.write_dispatch_execution(
        execution, str(output_dir), model_name, suffix=execution_suffix
    )
    calibration_report = manager.build_profile_calibration_report(bundle, execution)
    calibration_report_path = manager.write_profile_calibration_report(
        calibration_report,
        str(output_dir),
        model_name,
        suffix=execution_suffix,
    )
    report["steps"]["execute_cubism_dispatch"] = {
        "status": execution.get("status", "error"),
        "message": execution.get("message"),
        "execution_path": execution_path,
        "calibration_report_path": calibration_report_path,
        "executed_steps": execution.get("executed_steps", []),
    }
    report["status"] = _pick_status(execution_prep, execution)
    return report


def _write_report(report: JsonDict) -> Path:
    path = _report_path(Path(str(report["output_dir"])), str(report["model_name"]))
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


async def _run_command(args: argparse.Namespace) -> int:
    previous_env = _apply_cli_environment(args)
    try:
        report = await _run_pipeline(args)
    finally:
        _restore_cli_environment(previous_env)

    report_path = _write_report(report)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report.get("status", "error"),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return _status_code(str(report.get("status", "error")))


async def _run_calibration_command(args: argparse.Namespace) -> int:
    previous_env = _apply_cli_environment(args)
    try:
        report = await _run_template_calibration(args)
    finally:
        _restore_cli_environment(previous_env)

    report_path = _write_report(report)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report.get("status", "error"),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return _status_code(str(report.get("status", "error")))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(asyncio.run(_run_command(args)))
    if args.command == "calibrate-template":
        raise SystemExit(asyncio.run(_run_calibration_command(args)))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
