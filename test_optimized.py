#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized Live2D Pipeline Test
Fixed encoding and improved segmentation
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_server.server import full_pipeline


async def test_pipeline():
    print("=" * 60)
    print("Live2D Automation Pipeline - Optimized Test")
    print("=" * 60)

    input_image = "ATRI.png"
    output_dir = "output/ATRI_final"
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
            print(f"Session: {result.get('session_id', '')}")
            print(f"Steps: {len(result.get('steps', []))}")
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
