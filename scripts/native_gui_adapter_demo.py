from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UNSUPPORTED_EXIT_CODE = 64
SUPPORTED_ACTIONS = {
    "launch_editor",
    "import_psd",
    "apply_template",
    "export_embedded_data",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo native GUI adapter for the Live2D Cubism execution PoC."
    )
    parser.add_argument("action", choices=sorted(SUPPORTED_ACTIONS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--editor-path")
    parser.add_argument("--psd-path")
    parser.add_argument(
        "--mode",
        choices=["partial", "full", "fail"],
        default="partial",
        help=(
            "partial: only launch/import succeed, later actions return unsupported; "
            "full: all actions succeed and export writes a minimal bundle; "
            "fail: later actions return a hard failure."
        ),
    )
    return parser.parse_args()


def write_receipt(output_dir: Path, action: str, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = output_dir / f"demo_adapter_{action}.json"
    receipt.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_mock_bundle(output_dir: Path, model_name: str) -> None:
    textures_dir = output_dir / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_name}.moc3").write_bytes(b"demo-moc3")
    (textures_dir / "texture_00.png").write_bytes(b"demo-texture")
    model3 = {
        "FileReferences": {
            "Moc": f"{model_name}.moc3",
            "Textures": ["textures/texture_00.png"],
        }
    }
    (output_dir / "model3.json").write_text(
        json.dumps(model3, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = {
        "action": args.action,
        "mode": args.mode,
        "model_name": args.model_name,
        "template_id": args.template_id,
        "editor_path": args.editor_path,
        "psd_path": args.psd_path,
    }
    write_receipt(output_dir, args.action, payload)

    if args.mode == "fail" and args.action in {"apply_template", "export_embedded_data"}:
        return 2

    if args.mode == "partial" and args.action in {"apply_template", "export_embedded_data"}:
        return UNSUPPORTED_EXIT_CODE

    if args.action == "export_embedded_data":
        write_mock_bundle(output_dir, args.model_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
