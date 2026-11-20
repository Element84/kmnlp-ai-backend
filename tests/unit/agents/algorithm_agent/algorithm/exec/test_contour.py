from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ExecutionConfig,
    ExecutionContext,
    eval_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import DataArrayShapeError
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    MakePlot,
    MapProjectionSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.contour import ContourMapPlotSpec
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import ImageRef
from tests.utils.mocks.context import MockChatContext

default_exec_config = ExecutionConfig()


@pytest.fixture
def context(chat_context: MockChatContext) -> ToolCallContext:
    return ToolCallContext(tool_call_id="test", chat_context=chat_context)


def _make_latlon(nlat: int = 12, nlon: int = 16, seed: int = 42) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    lat = np.linspace(-60, 60, nlat)
    lon = np.linspace(-120, 120, nlon)
    # Smooth field so contours are meaningful.
    field = np.sin(np.radians(lat))[:, None] + np.cos(np.radians(lon))[None, :]
    field = field + rng.normal(0, 0.05, size=(nlat, nlon))
    return xr.DataArray(field, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_contour_map_filled(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = _make_latlon()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=ContourMapPlotSpec(
            data_array=VariableReference(name="da"),
            filled=True,
            levels=8,
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_contour_map_lines_with_labels(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = _make_latlon()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=ContourMapPlotSpec(
            data_array=VariableReference(name="da"),
            filled=False,
            label_contours=True,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_contour_map_non_latlon_raises(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = xr.DataArray(
        np.zeros((4, 5)),
        dims=["y", "x"],
        coords={"y": np.arange(4), "x": np.arange(5)},
    )
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=ContourMapPlotSpec(data_array=VariableReference(name="da")),
    )
    with pytest.raises(DataArrayShapeError, match="lat"):
        await eval_expression(expr, exec_ctx)
