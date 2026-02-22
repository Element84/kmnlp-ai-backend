import pytest
from pydantic import BaseModel

from llm_agent.utils.utils import (
    EnrichedLLMToolResultContent,
    maximum_length_json_string_from_string,
)


def test_extract_json_from_string_with_no_extras():
    string = """{"hello":"there"}"""
    extracted_json_string = maximum_length_json_string_from_string(string)
    assert extracted_json_string == """{"hello":"there"}"""


def test_extract_json_from_string_with_preamble():
    string = """Try this on for size:
    {"hello":"there"}
    """
    extracted_json_string = maximum_length_json_string_from_string(string)
    assert extracted_json_string == """{"hello":"there"}"""


def test_extract_json_from_string_with_postamble():
    string = """
    {"hello":"there"}

    Is what I would say.
    """
    extracted_json_string = maximum_length_json_string_from_string(string)
    assert extracted_json_string == """{"hello":"there"}"""


def test_extract_json_from_string_with_pre_and_postamble():
    string = """
    What about this?
    {"hello":"there"}

    Is what I would say.
    """
    extracted_json_string = maximum_length_json_string_from_string(string)
    assert extracted_json_string == """{"hello":"there"}"""


def test_extract_json_from_string_unbalanced_close_curly_brace():
    string = """{"hello":"there"}}"""
    extracted_json_string = maximum_length_json_string_from_string(string)
    assert extracted_json_string == """{"hello":"there"}}"""


def test_extract_json_from_string_raises_value_error_if_missing_start():
    string = """"hello":"there"}"""

    with pytest.raises(ValueError, match="substring not found"):
        maximum_length_json_string_from_string(string)


def test_extract_json_from_string_raises_value_error_if_missing_end():
    string = """{"hello":"there" """

    with pytest.raises(ValueError, match="substring not found"):
        maximum_length_json_string_from_string(string)


def test_enriched_llm_tool_result_content():
    class Foo(BaseModel):
        abc: str

    res = EnrichedLLMToolResultContent[Foo](
        id="123", content=[], status="success", data=Foo(abc="abc")
    )
    assert res.data.abc == "abc"
