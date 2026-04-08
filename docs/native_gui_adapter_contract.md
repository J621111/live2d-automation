# Native GUI Adapter Contract

The native GUI execution PoC can delegate Cubism automation steps to an external adapter through the `LIVE2D_NATIVE_GUI_ADAPTER_COMMAND` environment variable.

## Invocation

Set `LIVE2D_NATIVE_GUI_ADAPTER_COMMAND` to a launcher prefix such as:

```bash
python path/to/adapter.py
```

The MCP server appends an action name plus flags:

```text
<launcher...> <action> --output-dir <dir> --model-name <name> --template-id <template> [--editor-path <path>] [--psd-path <path>]
```

## Actions

Supported action names used by the current PoC:

- `launch_editor`
- `import_psd`
- `apply_template`
- `export_embedded_data`

## Flags

Always provided:

- `--output-dir`: Cubism working directory under the project `output/` tree
- `--model-name`: target model name
- `--template-id`: selected Cubism template id

Conditionally provided:

- `--editor-path`: editor executable path when available
- `--psd-path`: generated PSD path for `import_psd`

## Exit Codes

- `0`: action completed successfully
- `64`: action is unsupported by this adapter and MCP should fall back to record-only behavior for that step
- Any other non-zero code: action failed and MCP should report an execution error

Only `apply_template` and `export_embedded_data` currently use the `64 => fallback` rule. `launch_editor` and `import_psd` treat non-zero exit codes as hard failures.

## Export Expectations

If the adapter handles `export_embedded_data`, it should leave a minimal Cubism runtime bundle in `--output-dir` so local validation can succeed:

- `<model-name>.moc3`
- `model3.json`
- `textures/` directory with any referenced textures

`model3.json` should reference the generated moc3 filename and texture paths relative to `--output-dir`.

## Notes

- The adapter contract is intended for the current execution PoC, not a stable public API yet.
- `validate_export_bundle` still runs locally inside MCP after export.
- `opencli` execution is not wired to this contract yet.
