from collections.abc import Sequence
from typing import assert_type

from e84_geoai_common.llm.core import (
    LLMAssistantMessage,
    LLMInferenceConfig,
    LLMMessage,
    LLMResponseMetadata,
)

from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.utils.prompt_as_service import PromptService, PromptServiceAdapter
from tests.experiment.util.token_limiting_transformer import TokenLimitingTransformer


class TestTokenLimitingTransformer:
    async def test_composes_as_expected(self):
        def prompt(_: Sequence[LLMMessage], __: LLMInferenceConfig) -> LLMAssistantMessage:
            return LLMAssistantMessage(
                content="yo.",
                metadata=LLMResponseMetadata(
                    input_tokens=0, output_tokens=0, stop_reason="end_turn"
                ),
            )

        prompt_service: PromptService = PromptServiceAdapter(prompt)

        token_limiting_filter = TokenLimitingTransformer()
        limited_prompt_service: PromptService = token_limiting_filter.and_then_service(
            prompt_service
        )

        assert_type(limited_prompt_service, PromptService)

        assert await limited_prompt_service(([], LLMInferenceConfig())) == LLMAssistantMessage(
            content="yo.",
            metadata=LLMResponseMetadata(input_tokens=0, output_tokens=0, stop_reason="end_turn"),
        )

        orchestration_agent = OrchestrationAgent(limited_prompt_service)
        assert isinstance(orchestration_agent, OrchestrationAgent)
