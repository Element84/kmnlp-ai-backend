from collections.abc import Sequence
from unittest.mock import Mock, patch

from e84_geoai_common.llm.core import LLMAssistantMessage, LLMInferenceConfig, LLMMessage

from llm_agent.utils.prompt_as_service import (
    ObservePromptTransformer,
    PromptServiceAdapter,
)


class TestPromptServiceAdapter:
    async def test_adapter_creates_callable_service(self):
        service = PromptServiceAdapter(service_imp)

        result = await service(([], LLMInferenceConfig()))

        assert result.content == "yo."


class TestObservePromptTransformer:
    async def test_observe_prompt_transformer(self):
        service = PromptServiceAdapter(service_imp)
        transformer = ObservePromptTransformer(prompt_name="test_name")
        composed_service = transformer.and_then_service(service)

        with patch("llm_agent.utils.prompt_as_service.langfuse") as langfuse:
            update_current_span_mock = Mock()
            update_current_span_mock.return_value = None

            get_client_mock = Mock()
            get_client_mock.update_current_span = update_current_span_mock

            langfuse.get_client.return_value = get_client_mock

            result = await composed_service(([], LLMInferenceConfig()))

            assert result.content == "yo."
            update_current_span_mock.assert_called_once_with(name="prompt_service:test_name")


def service_imp(_: Sequence[LLMMessage], __: LLMInferenceConfig) -> LLMAssistantMessage:
    return LLMAssistantMessage(content="yo.")
