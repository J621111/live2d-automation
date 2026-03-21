# Live2D Automation MCP Server

从单张照片自动生成完整的 Live2D 模型。

## 功能特点

- AI 自动分析角色图片
- 自动生成 Live2D 分层
- 自动生成基础网格与绑定
- 自动配置物理与动作
- 支持 MCP 一键流水线调用

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 方法 1：在 VS Code 中使用 MCP

1. 打开 VS Code 命令面板 (`Ctrl+Shift+P`)
2. 运行 `MCP: Add Server`
3. 选择 `Command`
4. 输入：`python -m mcp_server.server`

### 方法 2：直接运行

```bash
python -m mcp_server.server
```

### 方法 3：调用完整流水线

```python
from mcp_server.server import full_pipeline

result = await full_pipeline(
    image_path="path/to/photo.png",
    output_dir="output/MyCharacter",
    model_name="MyCharacter",
    motion_types=["idle", "tap", "move", "emotional"],
)
```

## MCP Tools

| 工具 | 说明 |
|-----|------|
| `analyze_photo` | 分析图片并创建隔离的 `session_id` |
| `generate_layers` | 生成 Live2D 分层 |
| `create_mesh` | 创建 ArtMesh 网格 |
| `setup_rigging` | 设置骨骼绑定 |
| `configure_physics` | 配置物理效果 |
| `generate_motions` | 生成动作文件 |
| `full_pipeline` | 一键完成完整流水线 |

## 安全约束

- `output_dir` 必须位于项目 `output/` 目录内
- `model_name` 仅支持字母、数字、`_`、`-`
- 支持的输入图片格式：`png`、`jpg`、`jpeg`、`webp`

## 系统要求

- Python 3.8+
- NVIDIA GPU（推荐 8GB+ 显存）
- CUDA 11.8+（如使用 GPU）

## License

MIT
