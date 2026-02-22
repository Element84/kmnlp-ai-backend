from collections.abc import Callable, Sequence

import langfuse
from anyio import to_thread
from e84_geoai_common.llm.core import LLMAssistantMessage, LLMInferenceConfig, LLMMessage
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from simple_service_compose.service import Service

from llm_agent.utils.service import SimpleServiceTransformer

type PromptInput = tuple[Sequence[LLMMessage], LLMInferenceConfig]

"""PromptService = (Sequence[LLMMessage], LLMInferenceConfig) -> LLMAssistantMessage"""
PromptService = Service[PromptInput, LLMAssistantMessage]


class PromptServiceAdapter:
    """Adapt a prompt that takes 2 args into a `Service`."""

    underlying_prompt: Callable[[Sequence[LLMMessage], LLMInferenceConfig], LLMAssistantMessage]

    def __init__(
        self,
        underlying_prompt: Callable[
            [Sequence[LLMMessage], LLMInferenceConfig], LLMAssistantMessage
        ],
    ) -> None:
        """Initializer."""
        self.underlying_prompt = underlying_prompt

    async def __call__(
        self, inp: tuple[Sequence[LLMMessage], LLMInferenceConfig]
    ) -> LLMAssistantMessage:
        """Call the underlying prompt(Seq[LLMMessagel], LLMInferenceConfig)."""
        messages = inp[0]
        inference_cfg = inp[1]
        return await to_thread.run_sync(self.underlying_prompt, messages, inference_cfg)


class SimplePromptServiceTransformer(
    SimpleServiceTransformer[tuple[Sequence[LLMMessage], LLMInferenceConfig], LLMAssistantMessage]
):
    """SimplePromptServiceTransformer is a Decorator on PromptService."""


class ObservePromptTransformer(SimplePromptServiceTransformer):
    """Wrap prompt call with useful Langfuse information."""

    prompt_name: str

    def __init__(self, prompt_name: str) -> None:
        """Initializer. Curry prompt name into observed service call."""
        self.prompt_name = prompt_name

    @observe(as_type="span")
    async def __call__(self, inp: PromptInput, service: PromptService) -> LLMAssistantMessage:
        """Wrap call to prompt service in observablity information."""
        langfuse.get_client().update_current_span(name=f"prompt_service:{self.prompt_name}")
        return await service(inp)
