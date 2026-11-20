import logging
from collections.abc import AsyncGenerator, Sequence
from string import Template
from typing import Any

from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockModelSettings
from pydantic_ai.usage import UsageLimits
from rich.pretty import pretty_repr

from llm_agent.context import AgentDeps, ChatContext, ProgressCallback
from llm_agent.tools.agent_tool import AgentTool
from llm_agent.tools.fetch_datasets_tool import FetchDatasetsTool
from llm_agent.utils.langfuse import stream_events_to_final_output

log = logging.getLogger(__name__)

DEFAULT_MAX_ALLOWED_TOOL_CALLS_PER_TURN = 8

DEFAULT_MAX_OUTPUT_TOKENS = 10_000

DATASET_PICKING_AGENT_SYSTEM_PROMPT = Template("""
<role>
You are responsible for finding data relevant to users' requests.

Your role is to reply in structured JSON so that a machine can use your result \
to retrieve the relevant data.
</role>

<background>
You have access to tools that fetch information stored in a RDF knowledge \
graph.

This graph describes:
* Various climate-related datasets
* The variables present in those datasets
* Miscellaneous metadata about the datasets and variables including temporal coverage and units.

Upon receiving a user request, make a tool call to find the relevant information.

IMPORTANT: You are limited to ${max_allowed_tool_calls_per_turn} tool calls per turn. If you \
exceed this limit, your turn will be aborted and you will not be able to provide a response.

</background>

<format>
You must produce a JSON response satisfying this schema:
<response_schema>
```json
${response_format}
```
</response_schema>

It is important that you do not respond with any text outside of this schema. \
All thoughts you have should be included in the explanation field.

</format>

<limits>
You should only accept requests for 3 types of questions:
* What datasets do you know about?
* What variables are available in a particular dataset?
* Given some real-world concepts, which datasets contain relevant variables?

If the request does not fall into one of the above categories, you should \
refuse to answer.
</limits>

<example>
Here is an example request you could get, along with a final response you might \
provide after tool calls.

<request>
A dataset that can be used to calculate the mean sea surface temperature in the \
Black Sea on June 1, 2024.
</request>
<response>
${example_response}
</response>

</example>

""")


class DatasetSearchResponse(BaseModel):
    """Data search response format."""

    model_config = ConfigDict(frozen=True)

    explanation: str = Field(
        description=(
            "Understanding of the problem and reasoning for the result, as "
            "well as any commentary you have on the result. The text you "
            "intend to show to the user belongs here. Since this text is "
            "part of a double-quoted JSON string, make sure to not use double "
            "quotes inside it; use single quotes instead."
        )
    )
    result: list[str] | None = Field(
        default=None,
        description="IDs of all relevant datasets to include in the returned search results.",
    )
    is_refusal: bool = Field(
        description="Whether the response indicates that you are unable to answer a prompt."
    )


EXAMPLE_RESPONSE = DatasetSearchResponse(
    explanation="""\
Search constraints that can be inferred from the query are:
- Must have sea surface temperature
- Must have full or partial coverage of the Black Sea
- Must cover a time period that includes June 01, 2024

Based on these constraints and the available datasets, there is one relevant dataset: \
'NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)'.

This dataset has the following characteristics:
1. It contains sea surface temperature data.
2. It covers a global area, which would include the Black Sea.
3. It has daily temporal resolution, which means it would have data for specific dates like June 1, 2024.
4. Its temporal coverage extends from January 1, 2020, to November 30, 2024, which includes your date of interest.

Specifically, this dataset includes a variable:
- Name: Sea Surface Temperature
- Field name: analysed_sst
- Unit: Degrees Celsius
""",  # noqa: E501
    result=["crw_5km_sst_daily"],
    is_refusal=False,
)


class DatasetSearchQuery(BaseModel):
    """Input schema for the dataset picking agent."""

    query: str = Field(
        description="The user's natural language search query for relevant datasets."
    )


class DatasetPickingAgent:
    """Identifies relevant datasets to answer queries.

    Wraps a Pydantic AI agent whose single structured output is a
    ``DatasetSearchResponse``. The agent makes one or more ``FetchDatasetsTool``
    calls and ends its turn by producing that structured output.
    """

    def __init__(
        self,
        model: Model,
        *,
        tools: Sequence[AgentTool[Any, Any]] | None = None,
        max_allowed_tool_calls_per_turn: int = DEFAULT_MAX_ALLOWED_TOOL_CALLS_PER_TURN,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0,
    ) -> None:
        """Constructor.

        Args:
            model: The Pydantic AI model to drive the agent. Build one with
                ``llm_agent.model_factory.make_bedrock_model``.
            tools: The tools to expose to the agent. If None, defaults to a
                single ``FetchDatasetsTool``.
            max_allowed_tool_calls_per_turn: Maximum allowed tool calls per turn.
                Defaults to 8.
            max_output_tokens: Maximum number of output tokens.
            temperature: Sampling temperature for the model.
        """
        self._tools = list(tools) if tools is not None else [FetchDatasetsTool()]
        self._usage_limits = UsageLimits(tool_calls_limit=max_allowed_tool_calls_per_turn)
        self._agent = Agent(
            model,
            deps_type=AgentDeps,
            output_type=DatasetSearchResponse,
            instructions=_build_system_prompt(
                max_allowed_tool_calls_per_turn=max_allowed_tool_calls_per_turn
            ),
            model_settings=BedrockModelSettings(
                temperature=temperature,
                max_tokens=max_output_tokens,
                bedrock_cache_instructions=True,
                bedrock_cache_tool_definitions=True,
            ),
            tools=[tool.as_pydantic_tool() for tool in self._tools],
        )

    @observe(as_type="agent", name=__name__)
    async def run(
        self,
        chat_context: ChatContext,
        query: DatasetSearchQuery,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> DatasetSearchResponse:
        """Run the agent and return the final structured response."""
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        result = await self._agent.run(
            _format_query(query), deps=deps, usage_limits=self._usage_limits
        )
        log.info("DatasetPickingAgent response: %s", pretty_repr(result.output))
        return result.output

    @observe(as_type="agent", name=__name__, transform_to_string=stream_events_to_final_output)
    async def run_stream(
        self,
        chat_context: ChatContext,
        query: DatasetSearchQuery,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[DatasetSearchResponse]]:
        """Run the agent, streaming each event (tool calls/results, final output) in realtime."""
        deps = AgentDeps(chat_context=chat_context, progress_callback=progress_callback)
        async with self._agent.run_stream_events(
            _format_query(query), deps=deps, usage_limits=self._usage_limits
        ) as stream:
            async for event in stream:
                yield event


def _format_query(query: DatasetSearchQuery) -> str:
    return query.query


def _build_system_prompt(max_allowed_tool_calls_per_turn: int) -> str:
    return DATASET_PICKING_AGENT_SYSTEM_PROMPT.substitute(
        max_allowed_tool_calls_per_turn=max_allowed_tool_calls_per_turn,
        response_format=pretty_repr(DatasetSearchResponse.model_json_schema(), indent_size=2),
        example_response=pretty_repr(EXAMPLE_RESPONSE.model_dump_json()),
    )
