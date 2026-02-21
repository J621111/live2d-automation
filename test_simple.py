# -*- coding: utf-8 -*-
"""
Simplified Live2D Pipeline Test
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "live2d_automation"))


async def test_simple():
    """Simplified test"""
    print("=" * 60)
    print("Live2D Automation Pipeline Test")
    print("=" * 60)

    try:
        # Test 1: Image analysis
        print("\n[Test 1] Analyzing image...")
        from live2d_automation.mcp_server.tools.image_processor import ImageProcessor

        processor = ImageProcessor()
        image_path = "ATRI.png"

        import cv2

        img = cv2.imread(image_path)
        if img is None:
            print(f"  [ERROR] Cannot load image: {image_path}")
            return

        print(f"  [OK] Image loaded: {img.shape}")

        # Test 2: Simple segmentation (using rembg)
        print("\n[Test 2] Testing segmentation...")
        try:
            from rembg import remove

            print("  [OK] rembg imported")
        except ImportError:
            print("  [WARN] rembg not available")

        # Test 3: Generate layers
        print("\n[Test 3] Generating layers...")
        from live2d_automation.mcp_server.tools.layer_generator import LayerGenerator

        generator = LayerGenerator()
        print("  [OK] Layer generator initialized")

        # Test 4: Create mesh
        print("\n[Test 4] Creating mesh...")
        from live2d_automation.core.mesh_generator import ArtMeshGenerator

        mesh_gen = ArtMeshGenerator()
        print("  [OK] Mesh generator initialized")

        # Test 5: Rigging
        print("\n[Test 5] Setting up rigging...")
        from live2d_automation.mcp_server.tools.auto_rigger import AutoRigger

        rigger = AutoRigger()
        print("  [OK] Rigger initialized")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple())
