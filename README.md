# Live2D 自动化 MCP Server

从单张照片自动生成完整的 Live2D 模型。

## 功能特点

- 🤖 AI 自动分割身体部位
- 🎨 自动生成 Live2D 分层
- 🔧 自动绑定骨骼和变形器
- 🌊 自动配置物理效果
- 🎬 自动生成多种动作
- ⚡ 一键完整流水线

## 安装

```bash
cd live2d_automation
pip install -r requirements.txt
```

## 使用

### 方法 1：在 VS Code 中使用（推荐）

1. 打开 VS Code 命令面板 (`Ctrl+Shift+P`)
2. 运行 `MCP: Add Server`
3. 选择 `Command`
4. 输入: `python -m live2d_automation.mcp_server.server`

### 方法 2：直接运行

```bash
python -m live2d_automation.mcp_server.server
```

### 方法 3：使用完整流水线

```python
# 一键生成 Live2D 模型
from live2d_automation.mcp_server.server import full_pipeline

result = await full_pipeline(
    image_path="path/to/photo.png",
    output_dir="output/",
    model_name="MyCharacter",
    motion_types=["idle", "tap", "move", "emotional"]
)
```

## MCP Tools

| 工具 | 说明 |
|-----|------|
| `analyze_photo` | 分析照片，检测人物姿态 |
| `generate_layers` | 生成 Live2D 分层 |
| `create_mesh` | 创建 ArtMesh 网格 |
| `setup_rigging` | 设置骨骼绑定 |
| `configure_physics` | 配置物理效果 |
| `generate_motions` | 生成动作文件 |
| `full_pipeline` | 一键完整流水线 |

## 输出文件结构

```
output/
└── [model_name]/
    ├── model3.json       # 模型配置文件
    ├── physics.json      # 物理配置文件
    ├── textures/         # 贴图目录
    │   ├── layer_head.png
│   ├── layer_body.png
│   └── ...
└── motions/            # 动作目录
    ├── Idle_Breath.motion3.json
    ├── Idle_Blink.motion3.json
    ├── Tap_Head.motion3.json
    └── ...
```

## 系统要求

- Python 3.8+
- NVIDIA GPU (推荐 8GB+ 显存)
- CUDA 11.8+ (如果使用 GPU)

## License

MIT
