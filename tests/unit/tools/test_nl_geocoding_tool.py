from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from pydantic_ai import RunContext

from llm_agent.context import AgentDeps, ToolProgressUpdate
from llm_agent.tools.data_analysis_tool.types import GeometryRef
from llm_agent.tools.nl_geocoding_tool import (
    GeocodingToolAttemptUpdate,
    GeocodingToolSuccessUpdate,
    NaturalLanguageGeocodingTool,
    NaturalLanguageGeocodingToolInput,
)
from tests.utils.mocks.context import MockChatContext

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


nl_geocoding_tool = NaturalLanguageGeocodingTool()


class TestNaturalLanguageGeocodingTool:
    def test_no_error_on_repr(self):
        with does_not_raise():
            _ = repr(nl_geocoding_tool)

    async def test_execution(
        self,
        chat_context: MockChatContext,
        make_run_context: Callable[..., RunContext[AgentDeps]],
        black_sea_geometry: "BaseGeometry",
    ):
        progress_updates: list[ToolProgressUpdate[Any]] = []
        ctx = make_run_context(chat_context, progress_callback=progress_updates.append)

        with (
            patch(
                "llm_agent.tools.nl_geocoding_tool.geocode_natural_language_description",
                return_value=black_sea_geometry,
            ),
        ):
            result = await nl_geocoding_tool(
                ctx, NaturalLanguageGeocodingToolInput(query="black sea")
            )

        assert isinstance(result, GeometryRef)
        assert result.geometry_type == "Polygon"
        assert result.bbox == black_sea_geometry.bounds

        assert {type(update.update) for update in progress_updates} == {
            GeocodingToolAttemptUpdate,
            GeocodingToolSuccessUpdate,
        }
