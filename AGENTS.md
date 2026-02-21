# Live2D Automation 项目维护指南

## 项目结构

```
live2d_automation/
├── core/                      # 核心模块
│   ├── bone_system.py         # 骨骼系统
│   ├── deformers.py           # 变形器系统
│   ├── mesh_generator.py     # ArtMesh 网格生成
│   └── parameter_system.py   # 参数系统
├── mcp_server/                # MCP 服务器
│   ├── server.py              # 主服务器入口
│   └── tools/                 # 工具模块
│       ├── image_processor.py    # 图像处理
│       ├── segmentation.py        # 语义分割
│       ├── layer_generator.py    # 分层生成
│       ├── auto_rigger.py         # 自动绑定
│       ├── physics_setup.py       # 物理设置
│       ├── motion_generator.py    # 动作生成
│       ├── facial_detector.py     # 面部检测
│       └── moc3_generator.py      # 模型导出
├── README.md                  # 项目说明
├── requirements.txt           # 依赖列表
└── pyproject.toml             # 项目配置
```

## 开发指南

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 代码规范

- 使用 Black 格式化代码: `black .`
- 使用 Ruff 检查: `ruff check .`
- 使用 MyPy 类型检查: `mypy .`

### 测试

```bash
pytest tests/
```

## 发布流程

1. 更新版本号在 `pyproject.toml`
2. 创建 git tag: `git tag v0.x.x`
3. 构建包: `python -m build`
4. 上传: `twine upload dist/*`

## 注意事项

- 本项目依赖深度学习模型，首次运行需要下载模型权重
- 需要 GPU 才能流畅运行（推荐 8GB+ 显存）
- 输出目录需要写入权限
