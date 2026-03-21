"""
Test Live2D automation pipeline.
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_server.server import full_pipeline


async def test_pipeline():
    print("=" * 60)
    print("Live2D Automation Pipeline Test")
    print("=" * 60)

    input_image = "ATRI.png"
    output_dir = "output/ATRI"
    model_name = "ATRI"

    print(f"\nInput Image: {input_image}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Name: {model_name}")
    print("\n" + "-" * 60)

    result = await full_pipeline(
        image_path=input_image,
        output_dir=output_dir,
        model_name=model_name,
        motion_types=["idle", "tap", "move"],
    )

    print("\n" + "=" * 60)
    print("[OK] Test Result")
    print("=" * 60)
    print(f"\nStatus: {result.get('status', 'unknown')}")
    print(f"Message: {result.get('message', '')}")

    if result.get("status") != "success":
        raise AssertionError(f"Pipeline failed: {result.get('error', 'unknown_error')}")

    model_files = result.get("model_files", {})
    export_result = result.get("export_result", {})
    required_files = {"model3.json", "model3.moc"}
    missing = sorted(required_files.difference(model_files))

    print(f"\nSession ID: {result.get('session_id', '')}")
    print(f"Output Path: {result.get('output_path', '')}")
    print(f"Executed Steps: {len(result.get('steps', []))}")
    print(f"Model Files: {sorted(model_files.keys())}")
    print(f"Export Status: {export_result.get('status', 'unknown')}")

    if missing:
        raise AssertionError(f"Missing expected model files: {missing}")
    if export_result.get("status") != "success":
        raise AssertionError("Export result did not report success")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
