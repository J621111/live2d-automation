# -*- coding: utf-8 -*-
"""
Simplified Live2D Pipeline Test
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_simple():
    print("=" * 60)
    print("Live2D Automation Pipeline Test")
    print("=" * 60)

    print("\n[Test 1] Analyzing image...")
    from mcp_server.server import analyze_photo, create_mesh, generate_layers

    result = await analyze_photo("ATRI.png")
    if result.get("status") != "success":
        raise AssertionError(f"Photo analysis failed: {result}")
    session_id = result["session_id"]
    print(f"  [OK] Session created: {session_id}")

    print("\n[Test 2] Generating layers...")
    layers = await generate_layers(session_id, "output/test_simple")
    if layers.get("status") != "success":
        raise AssertionError(f"Layer generation failed: {layers}")
    print(f"  [OK] Layers generated: {layers['layers_generated']}")

    print("\n[Test 3] Creating mesh...")
    meshes = await create_mesh(session_id)
    if meshes.get("status") != "success":
        raise AssertionError(f"Mesh creation failed: {meshes}")
    print(f"  [OK] Meshes created: {meshes['meshes_created']}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_simple())
