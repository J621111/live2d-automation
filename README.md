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

### Run the local CLI workflow

```bash
live2d-run run --image-path ATRI.png --output-dir output/ATRI --demo-adapter-mode full
```

Or without the console script:

```bash
python -m mcp_server.cli run --image-path ATRI.png --output-dir output/ATRI --demo-adapter-mode full
```

The CLI writes a `<model_name>_cli_report.json` file into the output directory.

If you already have a Cubism-ready PSD and only want to tune the Cubism automation half, use the
calibration command instead of re-running image analysis:

```bash
python -m mcp_server.cli calibrate-template --output-dir output/ATRI_real --model-name ATRI --psd-path output/ATRI_real/ATRI.psd --editor-path "C:\Program Files\Live2D\Cubism5\Cubism Editor 5\CubismEditor5.exe" --native-gui-controller-mode execute
```

If `--psd-path` is omitted, the CLI will look for `<output_dir>/<model_name>.psd`. This is the
fastest loop for calibrating `template_menu_sequence`, because it only rebuilds the Cubism plan,
dispatch bundle, execution report, and profile calibration report.

Add `--resume` when you want to continue from the latest compatible dispatch execution in the same
output directory. The CLI only resumes when the PSD file, template id, editor path, and controller
mode still match; otherwise it falls back to a fresh execution and records that decision in the
CLI report.


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

## Native GUI adapter

The minimal Cubism execution PoC can call an external native GUI adapter through `LIVE2D_NATIVE_GUI_ADAPTER_COMMAND`. The adapter contract is documented in [docs/native_gui_adapter_contract.md](docs/native_gui_adapter_contract.md).

In short:

- MCP appends an action name such as `launch_editor`, `import_psd`, `apply_template`, or `export_embedded_data`
- exit code `0` means success
- exit code `64` means "unsupported, please fall back" for later PoC steps
- other non-zero codes are treated as execution failures

You can test the PoC with the bundled demo adapter:

```bash
set LIVE2D_NATIVE_GUI_ADAPTER_COMMAND=python scripts/native_gui_adapter_demo.py --mode partial
```

Use `--mode full` to let the demo adapter emit a minimal mock export bundle, or `--mode fail` to simulate hard adapter failures.

You can also enable the built-in Windows GUI controller for the first two steps:

```bash
live2d-run run --image-path ATRI.png --output-dir output/ATRI --editor-path "C:\Program Files\Live2D\Cubism5\Cubism Editor 5\CubismEditor5.exe" --native-gui-controller-mode dry_run
```

`dry_run` writes PowerShell scripts and receipts for `launch_editor` / `import_psd`; `execute` will attempt to run those scripts on Windows using the bundled profile.

The bundled Windows profile now includes conservative seed rules for common dialog recovery during retries:

- `import_psd`: tries `Open` and `Import PSD`
- `apply_template`: tries `Template` and `Confirm`
- `export_embedded_data`: tries `Export` and `Overwrite`

Each recovery artifact also records a `dialog_recovery_plan` section so you can see which action-specific or default recovery rules were selected. These seeds are meant to be tuned against your local Cubism window titles before production use.

The built-in probe now also records matched window titles and lightweight diagnostics in the probe artifact. When real Cubism runs do not behave as expected, check the probe JSON first to see which window titles were actually visible to the controller.

Each dispatch execution now also writes a `{model_name}_cubism_profile_calibration*.json` report that summarizes:

- observed probe window titles
- missing `window_probe_candidates`
- per-action dialog recovery observations
- suggested `known_dialog_recovery` additions

Use this report as the primary guide when tuning the built-in Windows profile against a real Cubism installation.

For `apply_template`, the built-in controller now expects an explicit profile-driven invocation. The bundled default profile leaves this empty on purpose, because Cubism's template workflow is UI-version dependent and a wrong shortcut is worse than no shortcut at all.

Use `template_menu_sequence` in [mcp_server/profiles/windows_cubism_default.json](mcp_server/profiles/windows_cubism_default.json) to define a menu-driven action sequence such as:

```json
"template_menu_sequence": [
  { "keys": "%m", "wait_seconds": 0.2 },
  { "keys": "t", "wait_seconds": 0.2 },
  { "keys": "a", "wait_seconds": 0.2 }
]
```

Calibrate that sequence against the Cubism menu path documented in the official editor manual:
[Modeling] -> [Model template] -> [Apply template](https://docs.live2d.com/en/cubism-editor-manual/applying-the-model-template/).

If `apply_template` fails without an artifact, the calibration report will now explicitly tell you whether `template_menu_sequence` or `template_shortcut` is still missing, and it will repeat that recommended menu path in the diagnostics.

## Export notes

- The exporter writes a mock intermediate bundle, not a production-ready Live2D runtime model
- `model3.json` and the returned file manifest always reference `{model_name}.moc3`
- `ready_for_cubism_editor` remains `false` until a real Cubism-compatible exporter exists
- Final validation and export should happen in Cubism Editor before production use

## License

MIT
