"""Simplified Live2D pipeline smoke test."""

import asyncio
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))


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


async def test_simple() -> None:
    print("=" * 60)
    print("Live2D Automation Pipeline Test")
    print("=" * 60)

    print("
[Test 1] Analyzing image...")
    from mcp_server.server import analyze_photo, create_mesh, generate_layers

    input_image = _create_sample_character_image(Path("output/test_inputs/test_simple.png"))
    result = await analyze_photo(str(input_image))
    if result.get("status") != "success":
        raise AssertionError(f"Photo analysis failed: {result}")
    session_id = result["session_id"]
    print(f"  [OK] Session created: {session_id}")

    print("
[Test 2] Generating layers...")
    layers = await generate_layers(session_id, "output/test_simple")
    if layers.get("status") != "success":
        raise AssertionError(f"Layer generation failed: {layers}")
    print(f"  [OK] Layers generated: {layers['layers_generated']}")

    print("
[Test 3] Creating mesh...")
    meshes = await create_mesh(session_id)
    if meshes.get("status") != "success":
        raise AssertionError(f"Mesh creation failed: {meshes}")
    print(f"  [OK] Meshes created: {meshes['meshes_created']}")

    print("
" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_simple())
