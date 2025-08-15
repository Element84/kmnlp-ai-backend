import json
import logging
from collections.abc import AsyncGenerator, Iterable
from string import Template

from anyio import to_thread
from e84_geoai_common.llm.core import (
    LLM,
    Agent,
    Base64ImageContent,
    LLMInferenceConfig,
    LLMMessage,
    LLMMessageContentType,
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.pretty import pretty_repr

from llm_agent.dao import knowledge_graph_dao
from llm_agent.tools.fetch_datasets_tool import fetch_datasets_tool
from llm_agent.utils import EnrichedValidationError

log = logging.getLogger(__name__)

MAX_LLM_PROMPT_TURNS = 3

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

""")  # noqa: E501


class MaxPromptTurnsExceededError(Exception):
    """Max allowed number of tool-use turns exceeded."""

    def __init__(self, conversation: list[LLMMessage]) -> None:
        """Constructor.

        Args:
            conversation (list[LLMMessage]):
                The conversation with too many prompt turns.

        Returns:
            None
        """
        self.conversation = conversation


class UnexpectedResponseTypeError(Exception):
    """Received unexpected response type from LLM."""

    def __init__(
        self, content: (Base64ImageContent | LLMToolResultContent)
    ) -> None:
        """Constructor.

        Args:
            content (Base64ImageContent() | LLMToolResultContent()):
                The content that is of an unexpected type.

        Returns:
            None
        """
        self.content = content


class DatasetResponseTextContent(BaseModel):
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
        description="IRIs of all relevant datasets to include in the returned "
        "search results.",
    )

    is_refusal: bool = Field(
        description=(
            "Whether the response indicates that your are unable to answer a "
            "prompt."
        )
    )


class DatasetPickingAgentMessage(BaseModel):
    """Wrapper for contents in a single message."""

    parsed_content: list[DatasetResponseTextContent | LLMToolUseContent] = (
        Field(
            description=(
                "Parsed content types present in an LLM assistant response."
            )
        )
    )


EXAMPLE_RESPONSE = DatasetResponseTextContent(
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
    result=["http://kmnlp-demo.noaa.gov/crw_daily_5km"],
    is_refusal=False,
)


# NOTE: We have decided to defer unifying the implementations of running
# queries and handling tool call between the `OrchestrationAgent` and the
# `DatasetPickingAgent`. However, the logic is highly similar and should be
# consolidated in something like a `ToolUsingAgent` the next time we write
# similar functionality.
class DatasetPickingAgent(Agent):
    """Identifies relevant datasets to answer queries."""

    llm: LLM
    inference_cfg: LLMInferenceConfig
    tools: dict[str, LLMTool]

    def __init__(
        self,
        llm: LLM,
        tools: dict[str, LLMTool] | None = None,
        *,
        inference_cfg: LLMInferenceConfig | None = None,
    ) -> None:
        """Constructor.

        Args:
            llm (LLM):
                An LLM instance.

            tools (dict[str, LLMTool] | None):
                The tools to make available to the agent. If None, will use
                the default set, which includes:
                    - fetch_datasets_tool

            inference_cfg: LLMInferenceConfig | None:
                The inference_cfg to use when prompting the LLM. If None, a
                default config will be provided.
        """
        self.llm = llm
        self.tools = tools or {
            fetch_datasets_tool.name: fetch_datasets_tool,
        }
        self.knowledge_graph_dao = knowledge_graph_dao

        if (
            inference_cfg is not None
            and inference_cfg.system_prompt is not None
        ):
            message = (
                "system_prompt should not be set in the provided "
                "LLMInferenceConfig"
            )
            raise ValueError(message)

        default_inference_cfg = LLMInferenceConfig(
            temperature=0,
            response_prefix="{",
            tools=list(self.tools.values()),
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            system_prompt=DATASET_PICKING_AGENT_SYSTEM_PROMPT.substitute(
                response_format=pretty_repr(
                    DatasetResponseTextContent.model_json_schema(),
                    indent_size=2,
                ),
                example_response=EXAMPLE_RESPONSE.model_dump(mode="json"),
            ),
        )

        if inference_cfg is None:
            self.inference_cfg = default_inference_cfg
        else:
            self.inference_cfg = default_inference_cfg.model_copy(
                update=inference_cfg.model_dump(
                    exclude={"system_prompt"},
                    exclude_unset=True,
                    exclude_none=True,
                )
            )

    async def run(
        self, query: str
    ) -> AsyncGenerator[DatasetPickingAgentMessage]:
        """Prompt the agent with a natural language query for relevant data.

        If the LLM's response does not contain any tool-use requests, the
        returned iterator will only yield a single message.

        If the response does contain tool-use requests, the returned iterator
        will yield all responses from the LLM as they become available. This
        includes intermediate messages, tool use requests, and final messages.

        There may be several rounds of these if the LLM issues new tool-use
        requests in response to results of previous tool-use requests.
        """
        # Keep track of conversation messages throughout this method.
        # The user's query is the first message in the conversation.
        convo_messages = [LLMMessage(role="user", content=query)]

        for _ in range(MAX_LLM_PROMPT_TURNS):
            # Get response to user's query, yield it, and track it to include
            # in future context
            assistant_message = await self.run_single(convo_messages)

            yield assistant_message

            convo_messages.append(
                _assistant_message_to_llm_message(assistant_message)
            )

            # Extract too use requests
            tool_use_requests = [
                c
                for c in assistant_message.parsed_content
                if isinstance(c, LLMToolUseContent)
            ]

            # If there are no tool use requests, stop executing!
            if not tool_use_requests:
                break

            log.info("Using tools: %s", pretty_repr(tool_use_requests))

            # Execute tools and gather results
            tool_results = [
                tool_result
                async for tool_result in self._use_tools(tool_use_requests)
            ]

            # Add all tool results as messages to the conversation,
            # then let the loop run again.
            tool_results_message = LLMMessage(
                role="user",
                content=list(tool_results),
            )
            convo_messages.append(tool_results_message)
        else:
            raise MaxPromptTurnsExceededError(conversation=convo_messages)

    async def run_single(
        self, convo_messages: list[LLMMessage]
    ) -> DatasetPickingAgentMessage:
        """Run a single prompt and return the parsed contents.

        Args:
            convo_messages (list[LLMMessage]):
                All of the messages in the conversation so far. It is expected
                that the roles of the conversation messages will alternate
                between "user" and "assistant" and represent a
                question/response flow.

                In some cases, a single `LLMMessage`'s `content` may contain >
                1 `LLMMessageContentType`. These case typically occur in
                messages from the assistant. In the context of this agent, it
                typically occurs when the assistant replies with some
                explanation and a tool call.

        Returns:
            list[DatasetResponseTextContent | LLMToolUseContent]:
                The parsed `LLMMessageContentType` elements.
        """

        # Useful because `to_thread.run_sync` takes args of (func, *args),
        # which breaks kwargs and can be hard to reason about.
        def _llm_prompt_shim(messages: list[LLMMessage]) -> LLMMessage:
            return self.llm.prompt(
                messages=messages, inference_cfg=self.inference_cfg
            )

        log.debug("Request inference cfg: %s", self.inference_cfg.model_dump())
        log.debug("Request convo messages: %s", convo_messages)
        response_message = await to_thread.run_sync(
            _llm_prompt_shim, convo_messages
        )
        log.debug("Response from agent: %s", response_message)
        content = response_message.content

        # If content is plain string, try to parse it as a response.
        if isinstance(content, str):
            log.warning(
                "Received an assistant message with plain string content: %s",
                content,
            )

            # NOTE: It is important to `json.loads` our string rather than and
            # now passing it directly to `.model_validate_json`. There is no
            # way to get Pydantic  to allow control characters (e.g. newline
            # literals) inside of string values, and the LLM loves to spit
            # those out.
            return DatasetPickingAgentMessage(
                parsed_content=[_parse_response_text_content(content)]
            )

        # Content is now assured to be a Sequence[LLMMessageContentType]
        return DatasetPickingAgentMessage(
            parsed_content=[_parse_content(c) for c in content]
        )

    async def _use_tools(
        self, tool_use_requests: Iterable[LLMToolUseContent]
    ) -> AsyncGenerator[LLMToolResultContent]:
        """Execute tools and yield results as they become available."""
        for tool_use_request in tool_use_requests:
            tool = self.tools[tool_use_request.name]
            tool_result = await to_thread.run_sync(
                tool.execute, tool_use_request
            )

            log.debug("Tool result: %s", pretty_repr(tool_result))

            yield tool_result

    @property
    def prompt_template(self) -> str:
        """The prompt template used by the agent."""
        return ""


###########
# Helpers #
###########


def _parse_content(
    content: LLMMessageContentType,
) -> DatasetResponseTextContent | LLMToolUseContent:
    match content:
        case TextContent():
            # NOTE: It is important to `json.loads` our string rather than
            # and now passing it directly to `.model_validate_json`. There
            # is no way to get Pydantic  to allow control characters (e.g.
            # newline literals) inside of string values, and the LLM loves
            # to spit those out.
            return _parse_response_text_content(content.text)

        case LLMToolUseContent():
            return content

        case Base64ImageContent() | LLMToolResultContent():
            log.warning("Unexpected response content type: %s", content)
            raise UnexpectedResponseTypeError(content)


def _parse_response_text_content(value: str) -> DatasetResponseTextContent:
    try:
        return DatasetResponseTextContent.model_validate(
            json.loads(value, strict=False), strict=False
        )
    except ValidationError as e:
        raise EnrichedValidationError(e, data=value) from None


def _assistant_message_to_llm_message(
    response: DatasetPickingAgentMessage,
) -> LLMMessage:
    return LLMMessage(
        role="assistant",
        content=[_serialize_content(c) for c in response.parsed_content],
    )


def _serialize_content(
    parsed_content: DatasetResponseTextContent | LLMToolUseContent,
) -> LLMMessageContentType:
    match parsed_content:
        case DatasetResponseTextContent():
            return TextContent(text=parsed_content.model_dump_json())
        case LLMToolUseContent():
            return parsed_content
