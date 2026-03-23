# Live2D Automation Maintenance Guide

## Project Layout

```text
core/                         Core mesh, parameter, and rigging helpers
mcp_server/
  server.py                   Stable MCP entrypoint
  secure_server_impl.py       Session-aware hardened server implementation
  tools/                      Image, face, physics, motion, and export helpers
tests/                        Pytest coverage for server security and pipeline contracts
README.md                     User-facing usage guide
requirements.txt              Runtime dependency list
pyproject.toml                Packaging and tool configuration
```

## Development

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Checks

- Format: `black .`
- Lint: `ruff check .`
- Type check: `mypy .`

### Tests

Primary automated coverage lives in the `tests/` package.

```bash
python -m pytest -q
```

Legacy script-style validation is still available for manual checks:

```bash
python test_simple.py
python test_live2d_pipeline.py
python test_optimized.py
```

## Runtime Notes

- MCP entrypoint: `python -m mcp_server.server`
- Session-based tools now return and consume `session_id`
- `output_dir` must remain under the project `output/` directory

## Release Notes

1. Update the version in `pyproject.toml`
2. Build with `python -m build`
3. Upload with `twine upload dist/*`
