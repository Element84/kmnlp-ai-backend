# llm-agent
LLM-related code.

## Developing

1. Checkout the code.
1. Create/activate your Python environment of choice.
1. Install uv, let's say with pip: `pip install uv`.
1. Install dependencies: `uv sync --all-extras` (or `pip install -r requirements.txt` if you prefer).
1. Run `pre-commit install` to install pre-commit hooks.
1. Configure your editor for realtime linting:
	- For VS Code:
		- Set the correct Python environment for the workspace via `ctrl+shift+P` > `Python: Select Interpreter`.
		- Install the Pylance and Ruff extensions.
1. Make changes.
1. Verify linting passes `scripts/lint.sh`.
1. Verify tests pass `scripts/test.sh`.
1. Commit and push your changes.

## Run evaluations

Run pre-defined experiments to evaluate LLM-components and output detailed evaluation reports:

```sh
python evals/run_evals.py
```
