"""
测试 Live2D 自动化流水线
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "live2d_automation"))

from live2d_automation.mcp_server.server import full_pipeline


async def test_pipeline():
    """测试完整流水线"""
    print("=" * 60)
    print("Live2D Automation Pipeline Test")
    print("=" * 60)

    # 输入照片路径
    input_image = "ATRI.png"  # 根目录下的照片
    output_dir = "live2d_automation/output/ATRI"
    model_name = "ATRI"

    print(f"\nInput Image: {input_image}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Name: {model_name}")
    print("\n" + "-" * 60)

    try:
        # 运行完整流水线
        result = await full_pipeline(
            image_path=input_image,
            output_dir=output_dir,
            model_name=model_name,
            motion_types=["idle", "tap", "move"],
        )

        print("\n" + "=" * 60)
        print("[OK] 测试结果")
        print("=" * 60)
        print(f"\n状态: {result.get('status', 'unknown')}")
        print(f"消息: {result.get('message', '')}")

        if result.get("status") == "success":
            print(f"\n📁 输出路径: {result.get('output_path', '')}")
            print(f"📊 执行步骤: {len(result.get('steps', []))} 个")

            for i, step in enumerate(result.get("steps", []), 1):
                step_name = step.get("name", "")
                step_result = step.get("result", {})
                print(f"\n  步骤 {i}: {step_name}")
                if "message" in step_result:
                    print(f"    → {step_result['message']}")

            # 复制到 live2d_view
            print("\n🔄 复制模型到 live2d_view...")
            import shutil

            src = Path(output_dir)
            dst = Path("live2d_view/models/ATRI_Live2D")

            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"[OK] 模型已复制到: {dst}")
        else:
            print(f"\n[ERROR] 错误: {result.get('error', '未知错误')}")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
