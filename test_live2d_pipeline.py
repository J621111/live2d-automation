"""
娴嬭瘯 Live2D 鑷姩鍖栨祦姘寸嚎
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

    try:
        result = await full_pipeline(
            image_path=input_image,
            output_dir=output_dir,
            model_name=model_name,
            motion_types=["idle", "tap", "move"],
        )

        print("\n" + "=" * 60)
        print("[OK] 娴嬭瘯缁撴灉")
        print("=" * 60)
        print(f"\n鐘舵€? {result.get('status', 'unknown')}")
        print(f"娑堟伅: {result.get('message', '')}")

        if result.get("status") == "success":
            print(f"\nSession ID: {result.get('session_id', '')}")
            print(f"杈撳嚭璺緞: {result.get('output_path', '')}")
            print(f"鎵ц姝ラ: {len(result.get('steps', []))}")
        else:
            print(f"\n[ERROR] 閿欒: {result.get('error', '鏈煡閿欒')}")

    except Exception as e:
        print(f"\n[ERROR] 娴嬭瘯澶辫触: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
