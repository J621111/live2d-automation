"""
Live2D Automation MCP Server
从单张照片自动生成完整的 Live2D 模型
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.mesh_generator import ArtMeshGenerator
from core.deformers import DeformerSystem
from core.parameter_system import ParameterSystem
from core.bone_system import BoneSystem
from mcp_server.tools.image_processor import ImageProcessor
from mcp_server.tools.segmentation import PreciseSegmenter as SegmentationEngine
from mcp_server.tools.layer_generator import LayerGenerator
from mcp_server.tools.auto_rigger import AutoRigger
from mcp_server.tools.physics_setup import PhysicsSetup
from mcp_server.tools.motion_generator import MotionGenerator
from mcp_server.tools.facial_detector import FacialFeatureDetector
from mcp_server.tools.moc3_generator import Live2DExporter as Moc3Exporter

# 初始化 MCP Server
mcp = FastMCP("live2d-automation")

# 全局状态
pipeline_state: Dict[str, Any] = {
    "input_image": None,
    "segments": {},
    "layers": [],
    "meshes": {},
    "rigging": {},
    "physics": {},
    "motions": [],
}


@mcp.tool()
async def analyze_photo(image_path: str) -> Dict[str, Any]:
    """
    分析照片，检测人物姿态、分割部位

    Args:
        image_path: 输入照片的路径

    Returns:
        分析结果，包括检测到的部位列表和关键点位置
    """
    logger.info(f"开始分析照片: {image_path}")

    processor = ImageProcessor()
    segments = await processor.analyze(image_path)

    pipeline_state["input_image"] = image_path
    pipeline_state["segments"] = segments

    return {
        "status": "success",
        "parts_detected": len(segments),
        "segments": segments,
        "message": f"成功检测到 {len(segments)} 个身体部位",
    }


@mcp.tool()
async def generate_layers(output_dir: str) -> Dict[str, Any]:
    """
    生成 Live2D 分层 PSD/PNG

    Args:
        output_dir: 输出目录路径

    Returns:
        生成的图层文件列表
    """
    logger.info("开始生成分层文件")

    if not pipeline_state["segments"]:
        return {"status": "error", "message": "请先调用 analyze_photo 分析照片"}

    generator = LayerGenerator()
    layers = await generator.generate(
        image_path=pipeline_state["input_image"],
        segments=pipeline_state["segments"],
        output_dir=output_dir,
    )

    pipeline_state["layers"] = layers

    return {
        "status": "success",
        "layers_generated": len(layers),
        "layers": layers,
        "message": f"成功生成 {len(layers)} 个图层",
    }


@mcp.tool()
async def create_mesh() -> Dict[str, Any]:
    """
    为每个部位生成 ArtMesh 网格

    Returns:
        生成的网格信息
    """
    logger.info("开始生成 ArtMesh 网格")

    if not pipeline_state["layers"]:
        return {"status": "error", "message": "请先调用 generate_layers 生成分层"}

    mesh_gen = ArtMeshGenerator()
    meshes = await mesh_gen.generate_from_layers(pipeline_state["layers"])

    pipeline_state["meshes"] = meshes

    return {
        "status": "success",
        "meshes_created": len(meshes),
        "meshes": meshes,
        "message": f"成功创建 {len(meshes)} 个 ArtMesh",
    }


@mcp.tool()
async def setup_rigging() -> Dict[str, Any]:
    """
    设置骨骼、变形器、参数系统

    Returns:
        绑定信息
    """
    logger.info("开始设置绑定")

    if not pipeline_state["meshes"]:
        return {"status": "error", "message": "请先调用 create_mesh 创建网格"}

    rigger = AutoRigger()
    rigging = await rigger.setup(
        meshes=pipeline_state["meshes"], segments=pipeline_state["segments"]
    )

    pipeline_state["rigging"] = rigging

    return {
        "status": "success",
        "bones_created": len(rigging.get("bones", [])),
        "parameters_created": len(rigging.get("parameters", [])),
        "rigging": rigging,
        "message": f"成功创建绑定：{len(rigging.get('bones', []))} 根骨骼，{len(rigging.get('parameters', []))} 个参数",
    }


@mcp.tool()
async def configure_physics() -> Dict[str, Any]:
    """
    配置头发、衣服物理效果

    Returns:
        物理设置信息
    """
    logger.info("开始配置物理系统")

    if not pipeline_state["rigging"]:
        return {"status": "error", "message": "请先调用 setup_rigging 设置绑定"}

    physics = PhysicsSetup()
    config = await physics.configure(
        rigging=pipeline_state["rigging"], segments=pipeline_state["segments"]
    )

    pipeline_state["physics"] = config

    return {
        "status": "success",
        "physics_groups": len(config.get("groups", [])),
        "physics": config,
        "message": f"成功配置物理系统：{len(config.get('groups', []))} 个物理组",
    }


@mcp.tool()
async def generate_motions(motion_types: List[str]) -> Dict[str, Any]:
    """
    生成空闲、点击、移动等动作

    Args:
        motion_types: 动作类型列表，可选值：["idle", "tap", "move", "emotional"]

    Returns:
        生成的动作文件列表
    """
    logger.info(f"开始生成动作: {motion_types}")

    if not pipeline_state["rigging"]:
        return {"status": "error", "message": "请先调用 setup_rigging 设置绑定"}

    motion_gen = MotionGenerator()
    motions = await motion_gen.generate(
        rigging=pipeline_state["rigging"], motion_types=motion_types
    )

    pipeline_state["motions"] = motions

    return {
        "status": "success",
        "motions_generated": len(motions),
        "motions": motions,
        "message": f"成功生成 {len(motions)} 个动作",
    }


@mcp.tool()
async def full_pipeline(
    image_path: str,
    output_dir: str,
    model_name: str = "ATRI",
    motion_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    一键完整流水线：照片→Live2D模型

    Args:
        image_path: 输入照片路径
        output_dir: 输出目录
        model_name: 模型名称
        motion_types: 动作类型列表，默认 ["idle", "tap", "move"]

    Returns:
        完整的模型生成结果
    """
    if motion_types is None:
        motion_types = ["idle", "tap", "move", "emotional"]

    logger.info(f"开始完整流水线处理: {model_name}")

    results = {"model_name": model_name, "steps": []}

    try:
        # 步骤 1: 分析照片
        logger.info("步骤 1/7: 分析照片...")
        step1 = await analyze_photo(image_path)
        results["steps"].append({"name": "analyze_photo", "result": step1})

        # 步骤 1.5: 面部详细检测
        logger.info("步骤 1.5/8: 检测面部特征...")
        detector = FacialFeatureDetector()
        face_features = await detector.detect_features(image_path)
        pipeline_state["face_features"] = face_features

        # 提取面部部位图像
        face_output_dir = f"{output_dir}/face_textures"
        import os

        os.makedirs(face_output_dir, exist_ok=True)
        face_layers = await detector.extract_face_parts(image_path, face_output_dir)
        pipeline_state["face_layers"] = face_layers
        results["steps"].append(
            {
                "name": "detect_face_features",
                "result": {
                    "status": "success",
                    "parts_detected": len(face_features.get("parts", {})),
                    "layers_extracted": len(face_layers),
                    "message": f"成功检测 {len(face_features.get('parts', {}))} 个面部特征",
                },
            }
        )

        # 步骤 2: 生成分层
        logger.info("步骤 2/8: 生成分层...")
        step2 = await generate_layers(output_dir)
        results["steps"].append({"name": "generate_layers", "result": step2})

        # 步骤 3: 创建网格
        logger.info("步骤 3/8: 创建 ArtMesh...")
        step3 = await create_mesh()
        results["steps"].append({"name": "create_mesh", "result": step3})

        # 步骤 4: 设置绑定
        logger.info("步骤 4/8: 设置绑定...")
        step4 = await setup_rigging()
        results["steps"].append({"name": "setup_rigging", "result": step4})

        # 步骤 5: 配置物理
        logger.info("步骤 5/8: 配置物理...")
        step5 = await configure_physics()
        results["steps"].append({"name": "configure_physics", "result": step5})

        # 步骤 6: 生成动作
        logger.info("步骤 6/8: 生成动作...")
        step6 = await generate_motions(motion_types)
        results["steps"].append({"name": "generate_motions", "result": step6})

        # 步骤 7: 导出模型
        logger.info("步骤 7/8: 导出 Live2D 模型...")
        exporter = Moc3Exporter()
        model_files = await exporter.export(
            model_name=model_name, output_dir=output_dir, state=pipeline_state
        )
        results["model_files"] = model_files
        results["steps"].append({"name": "export_model", "result": model_files})

        results["status"] = "success"
        results["message"] = f"Live2D Model '{model_name}' Generated Successfully!"
        results["output_path"] = output_dir

    except Exception as e:
        import traceback

        logger.error(f"流水线执行失败: {e}")
        logger.error(f"详细错误: {traceback.format_exc()}")
        results["status"] = "error"
        results["message"] = f"Error: {str(e)}"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


@mcp.resource("live2d://status")
def get_status() -> str:
    """获取当前流水线状态"""
    return json.dumps(pipeline_state, indent=2, ensure_ascii=False)


@mcp.resource("live2d://templates")
def get_templates() -> str:
    """获取可用的模板列表"""
    templates_dir = Path(__file__).parent.parent / "templates"
    templates = []
    if templates_dir.exists():
        templates = [f.stem for f in templates_dir.glob("*.json")]
    return json.dumps({"templates": templates}, indent=2)


@mcp.prompt()
def photo_to_live2d_guide() -> str:
    """生成 Live2D 模型的操作指南"""
    return """
# Live2D 自动化生成指南

## 快速开始
使用 `full_pipeline` 工具可以一键完成整个流程：

```json
{
  "image_path": "path/to/your/photo.png",
  "output_dir": "path/to/output",
  "model_name": "CharacterName",
  "motion_types": ["idle", "tap", "move", "emotional"]
}
```

## 分步执行
如果需要更精细的控制，可以按顺序调用：
1. `analyze_photo` - 分析照片结构
2. `generate_layers` - 生成分层文件
3. `create_mesh` - 创建 ArtMesh 网格
4. `setup_rigging` - 设置骨骼绑定
5. `configure_physics` - 配置物理效果
6. `generate_motions` - 生成动作文件

## 注意事项
- 照片最好是正面全身照
- 背景越简单越好
- 分辨率建议 1024x1024 以上
"""


class Live2DExporter:
    """Live2D 模型导出器"""

    async def export(
        self, model_name: str, output_dir: str, state: dict
    ) -> Dict[str, Any]:
        """导出完整的 Live2D 模型文件"""
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)

        files = {
            "model3.json": output_path / "model3.json",
            "physics.json": output_path / "physics.json",
            "motions_dir": output_path / "motions",
            "textures_dir": output_path / "textures",
        }

        # 创建 motions 目录
        files["motions_dir"].mkdir(exist_ok=True)
        files["textures_dir"].mkdir(exist_ok=True)

        # 导出 model3.json
        model3_data = self._build_model3_json(model_name, state)
        with open(files["model3.json"], "w", encoding="utf-8") as f:
            json.dump(model3_data, f, indent=2, ensure_ascii=False)

        # 导出 physics.json
        if state.get("physics"):
            with open(files["physics.json"], "w", encoding="utf-8") as f:
                json.dump(state["physics"], f, indent=2, ensure_ascii=False)

        # 导出动作文件
        for motion in state.get("motions", []):
            motion_file = files["motions_dir"] / f"{motion['name']}.motion3.json"
            with open(motion_file, "w", encoding="utf-8") as f:
                json.dump(motion["data"], f, indent=2, ensure_ascii=False)

        # 复制纹理文件 - 使用临时文件避免文件锁
        import shutil

        for layer in state.get("layers", []):
            if "texture_path" in layer:
                src = Path(layer["texture_path"])
                if src.exists():
                    dest = files["textures_dir"] / src.name
                    # 先复制到临时文件，再重命名
                    temp_dest = files["textures_dir"] / f"temp_{src.name}"
                    shutil.copy2(src, temp_dest)
                    if dest.exists():
                        dest.unlink()
                    temp_dest.rename(dest)

        return {
            "model_name": model_name,
            "output_path": str(output_path),
            "files": {k: str(v) for k, v in files.items()},
        }

    def _build_model3_json(self, model_name: str, state: dict) -> dict:
        """构建 model3.json 结构"""
        return {
            "Version": 3,
            "FileReferences": {
                "Moc": f"{model_name}.moc3",
                "Textures": [
                    f"textures/{Path(l['texture_path']).name}"
                    for l in state.get("layers", [])
                    if "texture_path" in l
                ],
                "Physics": "physics.json",
                "Motions": {
                    "Idle": [
                        {"File": f"motions/{m['name']}.motion3.json"}
                        for m in state.get("motions", [])
                        if m.get("type") == "idle"
                    ],
                    "Tap": [
                        {"File": f"motions/{m['name']}.motion3.json"}
                        for m in state.get("motions", [])
                        if m.get("type") == "tap"
                    ],
                    "Move": [
                        {"File": f"motions/{m['name']}.motion3.json"}
                        for m in state.get("motions", [])
                        if m.get("type") == "move"
                    ],
                },
            },
            "Groups": state.get("rigging", {}).get("groups", []),
            "HitAreas": state.get("rigging", {}).get("hit_areas", []),
        }


if __name__ == "__main__":
    mcp.run()
