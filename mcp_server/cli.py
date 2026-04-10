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
    output_dir = Path(args.output_dir)
    semantic_dir = output_dir / "semantic_layers"
    report: JsonDict = {
        "command": "run",
        "image_path": args.image_path,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
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
            model_name=args.model_name,
        )
        report["steps"]["build_cubism_psd"] = psd_result
        if psd_result.get("status") != "success":
            report["status"] = "error"
            return report

        prepare_result = await prepare_cubism_automation(
            session_id,
            str(output_dir),
            template_id=args.template_id,
            model_name=args.model_name,
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
            validate_result = await validate_cubism_export(str(output_dir), args.model_name)
            report["steps"]["validate_cubism_export"] = validate_result

        report["status"] = _pick_status(prepare_result, execute_result, validate_result)
        return report
    finally:
        close_result = await close_session(session_id)
        report["steps"]["close_session"] = close_result


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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(asyncio.run(_run_command(args)))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
