# macOS Cubism authoring

These tools operate an installed Cubism Editor and validate its exported files. The image-to-model pipeline still produces its documented intermediate format. Native serialization and SDK export are performed by Cubism.

The widget adapter was developed against **Cubism 5.3.04 with English UI**. Numeric controls use version-specific widget classes; custom layouts, localization, and other editor releases may require adapters. Start with a copy of an editable model and inspect snapshots before writing.

## Install

Use a Python environment with the same architecture as the installed editor JVM:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[macos-cubism]'
```

The default package can import and start MCP without the optional JPype/WebSocket dependencies. Install `[dev,macos-cubism]` to run the development checks.

No Cubism JVM, SDK/Core, model, token, or compiled bridge binary is bundled. You need your own compatible editor installation and rights to your model assets.

## State and installation paths

| Setting | Default | Purpose |
| --- | --- | --- |
| `LIVE2D_AUTOMATION_STATE_DIR` | `output/` under current working directory | Generated bridge classes and API token |
| `LIVE2D_CUBISM_SESSION_DIR` | `<state>/swing-bridge/session` | Private file request queue |
| `LIVE2D_CUBISM_HOME` | Discover one installation under `/Applications` or `~/Applications` | Cubism distribution or app path |
| `LIVE2D_CUBISM_BUNDLE_ID` | Match process name `CubismEditor` | Optional native accessibility process disambiguation |

Set absolute state paths when MCP and command-line clients run from different directories. CLI `--editor-root`, `--build-dir`, `--session-dir`, and `--token-path` override their corresponding defaults where offered. Existing bridge/session directories must be private to their current user (`0700`); queue files and API tokens must also be private (`0600`).

## Compile and check the widget bridge

```bash
live2d-cubism-bridge self-test --editor-root '/Applications/Live2D Cubism 5.3'
```

This compiles the packaged Java source using the installed editor's JVM and exercises an unrealized Swing control. It does not attach to the editor or edit a model. In a development checkout, run the fuller Java harness:

```bash
live2d-cubism-bridge self-test \
  --editor-root '/Applications/Live2D Cubism 5.3' \
  --test-source tests/java/SwingBridgeTest.java
```

Launch the editor and open your model, then attach:

```bash
live2d-cubism-bridge attach --editor-root '/Applications/Live2D Cubism 5.3'
```

If multiple matching editors are running, specify `--pid`. The process must belong to the current user and run the exact executable from the selected installation. macOS accessibility access may be needed by the native menu tools; grant it to the terminal or MCP host in System Settings.

The attached bridge uses Java instrumentation and a local file queue, with no network listener. Reattachment invalidates widget IDs. Use a single authoring workflow at a time; transport serialization prevents simultaneous clients from overwriting requests, but it does not turn a multi-step editing workflow into a transaction.

Inspect the current model:

```bash
printf '%s\n' '{"action":"snapshot","depth":8}' | \
  live2d-cubism-widget --expected-document Avatar.cmo3
```

Widget actions use IDs from the current snapshot. `close_dialog` requests the normal close event for a dialog, honoring editor confirmation handlers; it cannot close the main frame. Pass `--expected-document` for mutations so both the Python client and Swing execution thread verify the document and front modal. The low-level JSON transport is intended for trusted local callers, including attach/shutdown operations.

Guarded snapshot requests honor `depth`, including `0` for top-level window details. The default depth is 42 and the bridge caps it at 48. Mutations always use a depth-42 ownership snapshot regardless of a supplied snapshot depth. Table selection validates the complete row list before changing the current selection.

Widget IDs are opaque strings containing an attachment UUID and a local sequence. Reattaching creates a new namespace; do not convert IDs to numbers or reuse cached IDs. Document guards compare the complete filename, including any ` - ` within it.

Each top-level window reports `blocked_by`: the ID of its actual AWT modal blocker, or `null` when unblocked. Controls in blocked windows cannot be operated. The bridge reads this state again immediately before acting, independent of window creation order. Reattach after updating the bridge; clients reject mutations when this metadata is missing. Attaching opens the JVM's `java.awt` package to the bridge for this check and fails if the required blocker information is unavailable.

Disabled controls and read-only text fields reject edits. Widget IDs expire when their components are collected. Table snapshots include the first 150 rows; `find_table_rows` looks up exact values across the current table for `select_parts`, including rows beyond that limit. Collapsed or filtered-out Parts must first be expanded or made visible in the table.

Captures preserve the component's aspect ratio, use up to 2× scaling, and limit the longest output edge to 2048 pixels. Encoded responses must fit the 32 MiB transport limit.

```python
from mcp_server.tools.cubism_ui import EditorUI

ui = EditorUI('Avatar.cmo3')
state, widgets = ui.snapshot()
ui.select_parts(['Face'])
ui.parameter_value('Angle X', 0)
```

Helpers resolve menu ownership and exact object names. Single-object selection waits until the visible Inspector reports that object. Pointer events include a hover before press/drag so the editor receives its normal interaction sequence.

## Authorize the parameter API

Enable external application integration in Cubism, then register this client:

```bash
live2d-cubism-api register
```

Approve **Live2D Automation** in Cubism. Registration stores a private token file and prints only its path and approval status. Tokens are not printed. If approval was revoked or a token became stale, explicitly run `register --renew` and approve again; ordinary parameter reads do not replace tokens.

The API pins the loopback connection, rejects cross-origin redirects, disables proxy routing, and suppresses WebSocket frame logging for this authenticated connection; its default is `ws://127.0.0.1:22033`. You can configure `--endpoint` and `--token-path` at registration and pass the same values to `CubismEditorAPI`.

```python
import asyncio
import json
from pathlib import Path
from mcp_server.tools.cubism_editor_api import CubismEditorAPI

async def inspect():
    async with CubismEditorAPI('Avatar.cmo3') as editor:
        parameters = await editor.parameters()
    Path('parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')

asyncio.run(inspect())
```

Temporary previews restore only the parameters changed through `set_preview()`, using each parameter's value before its first preview change. Changes to other parameters are preserved.

## Native exports

Keep the expected model open and dismiss existing dialogs before exporting. Destination directories/files must be new unless you explicitly pass `--overwrite`. MOC export resets the displayed parameters to their defaults; PSD export preserves the current pose.

Native export commands are for trusted local operators. MOC3 and PSD destinations must resolve inside the same output root as the image pipeline: the checkout's `output/` directory by default. The existing `LIVE2D_OUTPUT_ROOT` override also applies and must remain inside the project directory. Relative paths are resolved from the current working directory, including when calling from a parent checkout. Paths and symbolic links that resolve outside the output root are rejected before editor interaction; `--overwrite` does not bypass this check.

Before MOC3 overwrite, existing entries in the bundle and resolved MOC destination directory are also checked. This includes atlas folders, descriptors, and other companion files, following directory aliases inside the output root and visiting each resolved directory once. A companion link that escapes the output root is rejected even if its target does not exist yet.

Exports write directly to the destination and do not provide rollback for a whole bundle; interruptions with `--overwrite` can leave mixed old and new files. Prefer a new directory, validate its resources, and replace an existing bundle only after verification.

```bash
live2d-cubism-export moc3 --document Avatar.cmo3 --output output/Avatar
live2d-cubism-export psd --document Avatar.cmo3 --output output/Avatar-source/neutral.psd
```

MOC export defaults to SDK 4.2 and the document filename stem. Use `--sdk-version` or `--basename` when needed. The requested SDK option must exist in the running editor. Exports wait for dialogs to finish and require a fresh nonempty output; an old file is not counted as a successful export.

For save/reopen verification, export to one directory, save and reopen the model in Cubism, then export to another directory:

```bash
live2d-cubism-export compare \
  --before output/Avatar-before/Avatar.model3.json \
  --after output/Avatar-after/Avatar.model3.json
```

Comparison checks MOC bytes and decoded RGBA atlas pixels, returning exit status `1` on differences. It does not interpret MOC data or prove SDK compatibility. `CubismExportValidator` separately checks declared resources, path confinement, and available readiness metadata; an actual SDK/viewer smoke remains necessary for runtime acceptance.

## MCP tools

Run `live2d-mcp` or `python -m mcp_server.server` using the same environment/state directory.

| Tool | Purpose |
| --- | --- |
| `cubism_editor_status()` | Read the active native document/windows |
| `cubism_editor_open_menu(action, expected_document)` | Open a whitelisted native menu |
| `cubism_editor_widgets(expected_document)` | Inspect the attached widget tree |
| `cubism_editor_widget_action(expected_document, action)` | Execute one guarded widget action |
| `cubism_editor_parameters(expected_document)` | Read authenticated parameter ranges/keyforms |

The menu action names are `auto_deformer`, `face_deformer`, `face_motion`, `sway_motion`, `eye_lip_settings`, `texture_atlas`, `model_template`, and `export_moc3`.

## Runtime companion builders

`mcp_server.tools.runtime_assets` provides `motion`, `expression`, `physics_group`, and `version_textures`. They validate authored parameter ranges, exact motion allocation counts, loop endpoints and local texture references. Texture filenames include a content hash; unrelated files are preserved. Character-specific curves and asset recipes belong in the consuming project.

Texture versioning publishes each file atomically after writing its complete contents. A failed write leaves the original references intact, and completed textures can be reused on retry.

## Recovery and verification boundaries

- A timed-out request remains pending. Inspect the editor for a modal or stalled operation before retrying. Do not replay a request until its outcome is understood.
- A private session belonging to another/stale process is rejected. Use a fresh session directory after verifying the intended editor process.
- Unexpected dialogs stop export rather than accepting arbitrary prompts. Resolve them in Cubism and retry with a new destination or an explicit overwrite choice.
- Synthetic tests and the Swing self-test do not replace visual model checks. No promise is made about other editor versions, localized menus, model quality, or animation correctness.
