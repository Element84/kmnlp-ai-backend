from llm_agent.utils import DEFAULT_LLM, DEFAULT_MODEL_NAME


def test_deafult_model_name():
    assert DEFAULT_MODEL_NAME == "Claude 3.7 Sonnet"


def test_default_llm():
    assert (
        DEFAULT_LLM.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )
