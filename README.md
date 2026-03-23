# Live2D Automation MCP Server

Generate a mock intermediate Live2D package from a single character image.

## Features

- MCP tools for image analysis, face extraction, layer generation, rigging, physics, motions, and export
- Server-issued session IDs with TTL, concurrency limits, explicit close support, and status metrics
- Output directory confinement under `output/`
- Mock `.moc3` export contract validated before success is reported
- Explicit `detector_used`, `fallback_reason`, and `confidence_summary` metadata on analysis steps

## Installation

Minimal runtime:

```bash
pip install -e .
```

CPU-assisted vision stack:

```bash
pip install -e ".[vision-cpu]"
```

GPU-assisted vision stack:

```bash
pip install -e ".[vision-gpu]"
```

Development tools:

```bash
pip install -e ".[dev]"
```

## Usage

### Run the MCP server

```bash
python -m mcp_server.server
```

### Run the full pipeline

```python
from mcp_server.server import full_pipeline

result = await full_pipeline(
    image_path="ATRI.png",
    output_dir="output/ATRI",
    model_name="ATRI",
    motion_types=["idle", "tap", "move", "emotional"],
)
```

### Step-by-step flow

1. Call `analyze_photo(image_path)` and store the returned `session_id`
2. Call `detect_face_features(session_id, output_dir)`
3. Call `generate_layers(session_id, output_dir)`
4. Call `create_mesh(session_id)`
5. Call `setup_rigging(session_id)`
6. Call `configure_physics(session_id)`
7. Call `generate_motions(session_id, motion_types)`
8. Call `export_model(session_id, output_dir, model_name)`
9. Call `close_session(session_id)` when the step flow is complete

## Safety constraints

- `output_dir` must remain inside the project `output/` directory
- `model_name` only supports letters, digits, `_`, and `-`
- input image formats: `png`, `jpg`, `jpeg`, `webp`
- input image limits: 20 MiB, 4096x4096, 16,777,216 total pixels
- supported motion types: `idle`, `tap`, `move`, `emotional`

## Export notes

- The exporter writes a mock intermediate bundle, not a production-ready Live2D runtime model
- `model3.json` and the returned file manifest always reference `{model_name}.moc3`
- `ready_for_cubism_editor` remains `false` until a real Cubism-compatible exporter exists
- Final validation and export should happen in Cubism Editor before production use

## License

MIT
