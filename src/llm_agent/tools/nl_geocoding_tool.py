import logging
from typing import Any

from natural_language_geocoding.errors import GeocodeError
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext
from shapely.geometry import mapping

from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    geocode_natural_language_description,
)
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool, ErrorResponse, ToolInput
from llm_agent.tools.data_analysis_tool.types import GeometryRef

log = logging.getLogger(__name__)

_TOOL_DESCRIPTION = """\
Tool for geocoding natural language descriptions into geometries in lat/lon coordinates.
Rather than being limited to simple place names, it supports complex spatial descriptions by
leveraging LLMs to generate graphs of spatial operations, including place lookups, boundary
calculations, coastline analysis, unions, intersections, differences, buffers, and more.

Examples of complex queries that are supported:
- "Within 2 km of the coast of Maui west of Kahului"
- "India, south of Mumbai"
- "Pennsylvania, excluding Philadelphia and Pittsburgh"
"""


class NaturalLanguageGeocodingToolInput(ToolInput):
    """Input for NaturalLanguageGeocodingTool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        description="Natural language description of the geographic location to geocode."
    )


class GeocodingToolAttemptUpdate(BaseModel):
    """Update for an attempt to geocode a natural language description."""

    query: str


class GeocodingToolSuccessUpdate(BaseModel):
    """Update for a successful geocoding."""

    query: str
    geojson: dict[str, Any]


class NaturalLanguageGeocodingTool(
    AgentTool[NaturalLanguageGeocodingToolInput, GeometryRef | ErrorResponse]
):
    """Tool for geocoding natural language descriptions into geographic geometries."""

    name = "NaturalLanguageGeocodingTool"
    description = _TOOL_DESCRIPTION
    input_model = NaturalLanguageGeocodingToolInput
    output_type = GeometryRef | ErrorResponse

    async def __call__(
        self, ctx: RunContext[AgentDeps], tool_input: NaturalLanguageGeocodingToolInput, /
    ) -> GeometryRef | ErrorResponse:
        """Geocode natural language description into a geometry and return a reference."""
        ctx.deps.update_progress(ctx, GeocodingToolAttemptUpdate(query=tool_input.query))

        try:
            geometry = await geocode_natural_language_description(
                ctx.deps.chat_context.global_context.place_lookup,
                ctx.deps.chat_context.global_context.bedrock_client,
                tool_input.query,
            )
        except GeocodeError as e:
            return ErrorResponse(message=f"Failed to geocode '{tool_input.query}': {e!r}")

        ctx.deps.update_progress(
            ctx, GeocodingToolSuccessUpdate(query=tool_input.query, geojson=mapping(geometry))
        )

        ref_key = ctx.deps.chat_context.persist_data(geometry, data_type="geometry")
        ref = GeometryRef(
            key=ref_key,
            name=tool_input.query[:20].replace(" ", "_").lower() + "_geom",
            description=f"Geocoded geometry for query '{tool_input.query}'",
            geometry_type=geometry.geom_type,  # pyright: ignore[reportArgumentType]
            bbox=geometry.bounds,
        )
        return ref
