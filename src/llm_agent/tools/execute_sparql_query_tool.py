import logging
from importlib.resources import files
from string import Template

from anyio.to_thread import run_sync
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

import llm_agent.data
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool, ToolInput

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


class _ExecuteSparqlQueryToolQuery(ToolInput):
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
    AgentTool[_ExecuteSparqlQueryToolQuery, _ExecuteSparqlQueryToolResult]
):
    """Tool to execute arbitrary Sparql queries on the knowledge graph."""

    name = "ExecuteSparqlQueryTool"
    input_model = _ExecuteSparqlQueryToolQuery
    output_type = _ExecuteSparqlQueryToolResult

    def __init__(self) -> None:
        """Initializer."""
        shacl = SHACL_FILE_PATH.read_text()
        self.description = TOOL_DESCRIPTION.substitute({"shacl": shacl})

    async def __call__(
        self,
        ctx: RunContext[AgentDeps],
        tool_input: _ExecuteSparqlQueryToolQuery,
        /,
    ) -> _ExecuteSparqlQueryToolResult:
        """Execute the provided SPARQL read query against the knowledge graph."""
        knowledge_graph_dao = ctx.deps.chat_context.global_context.knowledge_graph_dao
        # The knowledge graph query is blocking; run it off the event loop.
        result = await run_sync(
            knowledge_graph_dao.execute_read_only_query, tool_input.sparql_query
        )
        return _ExecuteSparqlQueryToolResult(result=result)
