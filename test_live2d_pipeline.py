"""Test Live2D automation full pipeline contract."""

import asyncio
import sys
from pathlib import Path

from PIL import Image, ImageDraw

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_server.server import full_pipeline


def _create_sample_character_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (768, 1024), (245, 247, 252, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((220, 340, 548, 980), radius=48, fill=(72, 114, 196, 255))
    draw.ellipse((244, 84, 524, 380), fill=(255, 224, 197, 255))
    draw.ellipse((296, 178, 356, 238), fill=(255, 255, 255, 255))
    draw.ellipse((412, 178, 472, 238), fill=(255, 255, 255, 255))
    draw.ellipse((318, 198, 340, 220), fill=(48, 67, 110, 255))
    draw.ellipse((434, 198, 456, 220), fill=(48, 67, 110, 255))
    draw.arc((332, 248, 436, 314), start=15, end=165, fill=(180, 82, 102, 255), width=5)
    draw.rectangle((250, 118, 518, 164), fill=(34, 45, 92, 255))
    image.save(path, format="PNG")
    return path


async def test_pipeline() -> None:
    print("=" * 60)
    print("Live2D Automation Pipeline Test")
    print("=" * 60)

    input_image = _create_sample_character_image(Path("output/test_inputs/test_pipeline.png"))
    output_dir = "output/ATRI"
    model_name = "ATRI"

    print(f"
Input Image: {input_image}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Name: {model_name}")
    print("
" + "-" * 60)

    result = await full_pipeline(
        image_path=str(input_image),
        output_dir=output_dir,
        model_name=model_name,
        motion_types=["idle", "tap", "move"],
    )

    print("
" + "=" * 60)
    print("[OK] Test Result")
    print("=" * 60)
    print(f"
Status: {result.get('status', 'unknown')}")
    print(f"Message: {result.get('message', '')}")

    if result.get("status") != "success":
        raise AssertionError(f"Pipeline failed: {result.get('error_code', 'unknown_error')}")

    model_files = result.get("model_files", {})
    export_result = result.get("export_result", {})
    required_files = {"model3.json", "model3.moc3"}
    missing = sorted(required_files.difference(model_files))

    print(f"
Session ID: {result.get('session_id', '')}")
    print(f"Output Path: {result.get('output_path', '')}")
    print(f"Executed Steps: {len(result.get('steps', []))}")
    print(f"Model Files: {sorted(model_files.keys())}")
    print(f"Export Status: {export_result.get('status', 'unknown')}")
    print(f"Export Validation: {export_result.get('validation', {})}")

    if missing:
        raise AssertionError(f"Missing expected model files: {missing}")
    if export_result.get("status") != "success":
        raise AssertionError("Export result did not report success")

    print("
" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
