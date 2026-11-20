import json
from typing import Any

import shapely
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from llm_agent.context import ChatContext, DataRef, LLMAgentContext
from llm_agent.tools.data_analysis_tool.types import GeometryRef, ImageRef


class MockChatContext(ChatContext):
    """A mock ChatContext for testing purposes."""

    def __init__(self, global_context: LLMAgentContext) -> None:
        self._global_context = global_context
        self.data_store: dict[str, Any] = {}

    @property
    def global_context(self) -> LLMAgentContext:
        """Return the global context."""
        return self._global_context

    def persist_data(self, data_or_path: object, *, data_type: str) -> str:
        """Mock persist_data method that returns a dummy data reference."""
        key = f"mock_data_ref_for_{data_type}_{hash(data_or_path)}"
        match data_type:
            case "geometry":
                if isinstance(data_or_path, BaseGeometry):
                    data_or_path = json.dumps(mapping(data_or_path))
                self.data_store[key] = data_or_path
            case "image":
                match data_or_path:
                    case str():
                        with open(data_or_path, "rb") as f:
                            self.data_store[key] = f.read()
                    case bytes():
                        self.data_store[key] = data_or_path
                    case _:
                        raise TypeError(
                            "Expected data_or_path to be a file path or bytes for image data."
                        )
            case _:
                self.data_store[key] = data_or_path
        return key

    def retrieve_data(self, data_ref: DataRef) -> Any:  # noqa: ANN401
        """Mock retrieve_data method that returns a dummy data object."""
        data = self.data_store[data_ref.key]
        match data_ref:
            case GeometryRef():
                return shapely.from_geojson(data)  # type: ignore[reportUnknownMemberType]
            case ImageRef():
                return data  # bytes
            case _:
                return data
