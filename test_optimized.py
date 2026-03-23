"""Optimized Live2D pipeline smoke test."""

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
    print("Live2D Automation Pipeline - Optimized Test")
    print("=" * 60)

    input_image = _create_sample_character_image(Path("output/test_inputs/test_optimized.png"))
    output_dir = "output/ATRI_final"
    model_name = "ATRI"

    print(f"
Input: {input_image}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_name}")
    print("-" * 60)

    result = await full_pipeline(
        image_path=str(input_image),
        output_dir=output_dir,
        model_name=model_name,
        motion_types=["idle", "tap", "move"],
    )

    print("
" + "=" * 60)
    status = result.get("status", "unknown")
    print(f"Status: {status}")

    if status != "success":
        error = result.get("error_code", result.get("message", "Unknown error"))
        raise AssertionError(f"Pipeline did not succeed: {error}")

    print(f"Message: {result.get('message', '')}")
    print(f"Output: {result.get('output_path', '')}")
    print(f"Session: {result.get('session_id', '')}")
    print(f"Steps: {len(result.get('steps', []))}")

    model_files = result.get("model_files", {})
    export_result = result.get("export_result", {})
    required_files = {"model3.json", "model3.moc3"}
    missing = sorted(required_files.difference(model_files))
    print(f"Model files: {sorted(model_files.keys())}")
    print(f"Export status: {export_result.get('status', 'unknown')}")

    if missing:
        raise AssertionError(f"Missing expected model files: {missing}")
    if export_result.get("status") != "success":
        raise AssertionError("Export result did not report success")

    print("
" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
