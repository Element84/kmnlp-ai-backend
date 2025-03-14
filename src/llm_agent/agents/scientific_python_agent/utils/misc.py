"""Misc. utils."""

import inspect
import re
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, Field


class LLMCodeSolution(BaseModel):
    """Data model for an LLM's code solution to a problem."""

    thoughts: str = Field(
        "",
        description="Understanding of the problem and plan of attack.",
    )
    code: str = Field(
        "",
        description="Code to solve the problem.",
    )
    explanation: str = Field(
        "",
        description="Code explanation.",
    )


def extract_tag_content(text: str, tag: str) -> str:
    """Extracts substring enclosed inside first <tag></tag> block in text.

    Args:
        text (str): Text from which to extract.
        tag (str): Name of the tag. Text will be extracted from insied
            <tag></tag>.

    Returns:
        str: Extracted text if a <tag></tag> block is found. Otherwise an empty
        string.
    """
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if len(matches) == 0:
        return ""
    return matches[0]


def parse_code_solution(text: str) -> LLMCodeSolution:
    """Parse LLM's output into a LLMCodeSolution instance.

    The text is assumed to contain the following sections:
    - <thoughts></thoughts>
    - <code></code>
    - <explanation></explanation>

    The <code></code> section may optionally also contain a markdown
    code block.

    Args:
        text (str): Text to parse.

    Returns:
        LLMCodeSolution: The parsed code solution.
    """
    thoughts = extract_tag_content(text, "thoughts")
    code = extract_tag_content(text, "code")
    with suppress(IndexError):
        # extract code from within markdown code block, if present
        code = re.findall(r".*```(?:\w+)?(.*?)```.*", code, re.DOTALL)[0]
    explanation = extract_tag_content(text, "explanation")

    llm_solution = LLMCodeSolution(
        thoughts=thoughts.strip(),
        code=code.strip(),
        explanation=explanation.strip(),
    )
    return llm_solution


def get_function_def(func: Callable[..., Any]) -> str:
    """Extract signature and docstring of a function as a string."""
    name = func.__name__
    signature = inspect.signature(func)
    docstr = func.__doc__
    return f'def {name}{signature}:\n    """\n    {docstr}"""'
