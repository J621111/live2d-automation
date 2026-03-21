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

    result = await full_pipeline(
        image_path=input_image,
        output_dir=output_dir,
        model_name=model_name,
        motion_types=["idle", "tap", "move"],
    )

    print("\n" + "=" * 60)
    status = result.get("status", "unknown")
    print(f"Status: {status}")

    if status != "success":
        error = result.get("error", result.get("message", "Unknown error"))
        raise AssertionError(f"Pipeline did not succeed: {error}")

    print(f"Message: {result.get('message', '')}")
    print(f"Output: {result.get('output_path', '')}")
    print(f"Session: {result.get('session_id', '')}")
    print(f"Steps: {len(result.get('steps', []))}")

    model_files = result.get("model_files", {})
    export_result = result.get("export_result", {})
    required_files = {"model3.json", "model3.moc"}
    missing = sorted(required_files.difference(model_files))
    print(f"Model files: {sorted(model_files.keys())}")
    print(f"Export status: {export_result.get('status', 'unknown')}")

    if missing:
        raise AssertionError(f"Missing expected model files: {missing}")
    if export_result.get("status") != "success":
        raise AssertionError("Export result did not report success")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
