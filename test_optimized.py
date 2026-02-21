#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized Live2D Pipeline Test
Fixed encoding and improved segmentation
"""

import asyncio
import sys
import os
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "live2d_automation"))

from live2d_automation.mcp_server.server import full_pipeline


async def test_pipeline():
    """Test the complete pipeline"""
    print("=" * 60)
    print("Live2D Automation Pipeline - Optimized Test")
    print("=" * 60)

    input_image = "ATRI.png"
    output_dir = "live2d_automation/output/ATRI_final"
    model_name = "ATRI"

    print(f"\nInput: {input_image}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_name}")
    print("-" * 60)

    try:
        result = await full_pipeline(
            image_path=input_image,
            output_dir=output_dir,
            model_name=model_name,
            motion_types=["idle", "tap", "move"],
        )

        print("\n" + "=" * 60)
        status = result.get("status", "unknown")
        print(f"Status: {status}")

        if status == "success":
            print(f"Message: {result.get('message', '')}")
            print(f"Output: {result.get('output_path', '')}")
            print(f"Steps: {len(result.get('steps', []))}")

            for i, step in enumerate(result.get("steps", []), 1):
                step_name = step.get("name", "")
                step_result = step.get("result", {})
                msg = step_result.get("message", "")
                print(f"  Step {i}: {step_name} - {msg}")

            # Copy to live2d_view
            print("\nCopying to live2d_view...")
            import shutil

            src = Path(output_dir) / model_name
            dst = Path("live2d_view/models/ATRI_Live2D")

            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"Copied to: {dst}")

                # List files
                files = list(dst.glob("*"))
                print(f"Total files: {len(files)}")

        else:
            error = result.get("error", result.get("message", "Unknown error"))
            print(f"Error: {error}")

    except Exception as e:
        print(f"\nException: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
