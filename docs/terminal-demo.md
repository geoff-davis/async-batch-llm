# Regenerating the Terminal Demo

The README animation is a deterministic rendering of the credential-free
[`example_callable_application.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_callable_application.py)
run. First verify the live example, then regenerate the asset:

```bash
uv run python examples/example_callable_application.py
uv run --with pillow python scripts/generate_terminal_demo.py
```

Pillow is a maintainer-only generation tool and is not a runtime dependency.
The generated `docs/assets/v0.20-quickstart.gif` must remain below 2 MB and must
not contain machine-specific paths, credentials, or terminal history.
