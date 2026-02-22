import logging
from importlib.resources import files
from string import Template

from langfuse import observe  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict, Field

import llm_agent.data
from llm_agent.context import ToolCallContext
from llm_agent.tools.llm_tool_input import LLMToolInput
from llm_agent.tools.llm_tool_typed import LLMToolTyped, ToolOutput

log = logging.getLogger(__name__)

SHACL_FILE_PATH = files(llm_agent.data).joinpath("dataset-constraints.shacl")

TOOL_DESCRIPTION = Template("""
Execute an arbitrary SPARQL read query against the knowledge graph containing
information about datasets and their metadata.

The shape of the knowledge graph is defined by the following SHACL constraints:
```turtle
${shacl}
````
""")


class _ExecuteSparqlQueryToolQuery(LLMToolInput):
    """Query arguments for the `ExecuteSparqlQueryTool`.

    Execute the provided SPARQL read query on the system's knowledge graph.
    """

    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="ExecuteSparqlQueryToolQuery",
        extra="forbid",
        frozen=True,
    )

    sparql_query: str = Field(description="The SPARQL read query to execute.")


class _ExecuteSparqlQueryToolResult(BaseModel):
    """Result of the SparqlQuery."""

    model_config = ConfigDict(
        # Title in the JSON schema, w/o leading underscore to reduce confusion.
        title="ExecuteSparqlQueryToolResult",
        extra="forbid",
        frozen=True,
    )

    result: list[str] = Field(
        description="Dictionary containing the results of the executed query."
    )


class ExecuteSparqlQueryTool(
    LLMToolTyped[
        _ExecuteSparqlQueryToolQuery, _ExecuteSparqlQueryToolResult, _ExecuteSparqlQueryToolResult
    ]
):
    """Tool to execute arbitrary Sparql queries on the knowledge graph."""

    def __init__(self) -> None:
        """Initializer."""
        shacl = SHACL_FILE_PATH.read_text()

        super().__init__(
            name="ExecuteSparqlQueryTool",
            description=TOOL_DESCRIPTION.substitute({"shacl": shacl}),
            input_model=_ExecuteSparqlQueryToolQuery,
            user_output_model=_ExecuteSparqlQueryToolResult,
            llm_output_model=_ExecuteSparqlQueryToolResult,
        )

    @observe(as_type="tool", name=__name__)
    def _execute_typed(
        self,
        context: ToolCallContext,
        tool_input: _ExecuteSparqlQueryToolQuery,
    ) -> ToolOutput[_ExecuteSparqlQueryToolResult, _ExecuteSparqlQueryToolResult]:
        res = _ExecuteSparqlQueryToolResult(
            result=context.global_context.knowledge_graph_dao.execute_read_only_query(
                tool_input.sparql_query
            )
        )

        log.debug("Tool call result: %s", res)

        return ToolOutput(user=res, llm=res)
