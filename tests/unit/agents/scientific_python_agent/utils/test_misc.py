from llm_agent.agents.scientific_python_agent.utils.misc import (
    extract_tag_content,
    get_function_def,
    parse_code_solution,
)


def test_extract_tag_content():
    text = "abc<code>def</code>xyz"
    assert extract_tag_content(text, "code") == "def"
    text = "abc<code>def\n\nghi</code>xyz"
    assert extract_tag_content(text, "code") == "def\n\nghi"
    assert extract_tag_content(text, "abc") == ""


def test_parse_code_solution_basic():
    text = """
lorem
<thoughts>
__thoughts__
</thoughts>
ipsum
<code>__code__</code>
dolor
<explanation>
__explanation__
</explanation>
amet
"""
    parsed_soln = parse_code_solution(text)
    assert parsed_soln.thoughts == "__thoughts__"
    assert parsed_soln.code == "__code__"
    assert parsed_soln.explanation == "__explanation__"


def test_parse_code_solution_with_codeblock():
    text = """
lorem
<thoughts>
__thoughts__
</thoughts>
ipsum
<code>
```python
__code__
```
</code>
dolor
<explanation>
__explanation__
</explanation>
amet
"""
    parsed_soln = parse_code_solution(text)
    assert parsed_soln.thoughts == "__thoughts__"
    assert parsed_soln.code == "__code__"
    assert parsed_soln.explanation == "__explanation__"


def test_parse_code_solution_empty():
    text = ""
    parsed_soln = parse_code_solution(text)
    assert parsed_soln.thoughts == ""
    assert parsed_soln.code == ""
    assert parsed_soln.explanation == ""


def dummy_func(a: int, b: str) -> int:
    """Docstring.

    Args:
        a (int): a.
        b (str): b.

    Returns:
        int: return value.
    """
    raise NotImplementedError


def test_get_function_def():
    expected_output = """\
def dummy_func(a: int, b: str) -> int:
    \"\"\"
    Docstring.

    Args:
        a (int): a.
        b (str): b.

    Returns:
        int: return value.
    \"\"\""""
    assert get_function_def(dummy_func) == expected_output
