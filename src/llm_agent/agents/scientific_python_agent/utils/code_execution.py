"""Utils related to code execution."""

import io
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, NamedTuple


class ExecOutput(NamedTuple):
    """Output from executing some code."""

    stdout: str
    stderr: str

    def __str__(self) -> str:
        stdout = f"stdout:\n```\n{self.stdout}\n```"
        stderr = f"stderr:\n```\n{self.stderr}\n```"
        return f"{stdout}\n\n{stderr}"


class ExecEnv:
    """Execution environment."""

    namespace: dict[str, Any]

    def __init__(self, extra_imports: list[type | Callable[(...), Any]] | None = None) -> None:
        """Constructor.

        Args:
            extra_imports (list[type | Callable[(...), Any]]): Imports to
                include in the execution namespace. The idea is for
                ScientificPythonAgent.utils to be passed here. Defaults to
                None.
        """
        if extra_imports is None:
            extra_imports = []
        self.namespace = {imp.__name__: imp for imp in extra_imports}

    def exec(self, code: str) -> ExecOutput:
        """Execute Python code."""
        return exec_code(code, namespace=self.namespace)


def exec_code(code: str, namespace: dict[str, Any]) -> ExecOutput:
    """Execute Python code."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)  # noqa: S102
    except Exception:  # noqa: BLE001
        # Capture exceptions that may occur during exec() by printing to stderr
        traceback.print_exc(file=stderr_capture)

    out = ExecOutput(
        stdout=stdout_capture.getvalue(),
        stderr=stderr_capture.getvalue(),
    )
    return out
