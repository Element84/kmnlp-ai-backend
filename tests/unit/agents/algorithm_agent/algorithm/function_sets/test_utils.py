from llm_agent.agents.algorithm_agent.algorithm.function_sets.utils import (
    concat_camel_case,
)


def test_concat_camel_case():
    assert concat_camel_case([]) == ""
    assert concat_camel_case(["abc", "def", "ghi"]) == "AbcDefGhi"
    assert concat_camel_case(["ABC", "DEF", "GHI"]) == "AbcDefGhi"
    assert concat_camel_case(["AB_CD", "EFG"]) == "AbcdEfg"
