from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from matplotlib.animation import FuncAnimation

# register MakePlot handlers
import llm_agent.agents.algorithm_agent.algorithm.exec.plot  # pyright: ignore[reportUnusedImport]
import llm_agent.agents.algorithm_agent.algorithm.exec.wind  # pyright: ignore[reportUnusedImport] # noqa: F401
from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.animation import make_animation
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ExecutionConfig,
    ExecutionContext,
    eval_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnimationSpec,
    MakePlot,
    MapProjectionSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.wind import WindVectorPlotSpec
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import ImageRef
from tests.utils.mocks.context import MockChatContext


@pytest.fixture
def context(chat_context: MockChatContext) -> ToolCallContext:
    return ToolCallContext(tool_call_id="test", chat_context=chat_context)


def _make_wind_component(nlat: int = 8, nlon: int = 10, seed: int = 1) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(-10, 10, size=(nlat, nlon)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-60, 60, nlat), "lon": np.linspace(-120, 120, nlon)},
    )


def _make_wind_component_3d(
    ntime: int = 4, nlat: int = 8, nlon: int = 10, seed: int = 1
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(-10, 10, size=(ntime, nlat, nlon)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(ntime),
            "lat": np.linspace(-60, 60, nlat),
            "lon": np.linspace(-120, 120, nlon),
        },
    )


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_wind_vector_plot_arrows(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    u = _make_wind_component(seed=1)
    v = _make_wind_component(seed=2)
    exec_ctx = ExecutionContext(context, input_vars={"u": u, "v": v}, config=ExecutionConfig())
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=WindVectorPlotSpec(
            u_data_array=VariableReference(name="u"),
            v_data_array=VariableReference(name="v"),
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_wind_vector_plot_barbs_and_subsample(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    u = _make_wind_component(seed=1)
    v = _make_wind_component(seed=2)
    exec_ctx = ExecutionContext(context, input_vars={"u": u, "v": v}, config=ExecutionConfig())
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=WindVectorPlotSpec(
            u_data_array=VariableReference(name="u"),
            v_data_array=VariableReference(name="v"),
            barbs=True,
            subsample=2,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_wind_vector_plot_mismatched_grids_raises(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    u = _make_wind_component(nlat=8, nlon=10)
    v = _make_wind_component(nlat=6, nlon=10)
    exec_ctx = ExecutionContext(context, input_vars={"u": u, "v": v}, config=ExecutionConfig())
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=WindVectorPlotSpec(
            u_data_array=VariableReference(name="u"),
            v_data_array=VariableReference(name="v"),
        ),
    )
    with pytest.raises(Exception, match="same grid"):
        await eval_expression(expr, exec_ctx)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_wind_animation(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    u = _make_wind_component_3d(seed=1)
    v = _make_wind_component_3d(seed=2)
    exec_ctx = ExecutionContext(context, input_vars={"u": u, "v": v}, config=ExecutionConfig())
    spec = AnimationSpec[WindVectorPlotSpec](
        plot_spec=WindVectorPlotSpec(
            u_data_array=VariableReference(name="u"),
            v_data_array=VariableReference(name="v"),
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_wind_animation_barbs(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    u = _make_wind_component_3d(seed=1)
    v = _make_wind_component_3d(seed=2)
    exec_ctx = ExecutionContext(context, input_vars={"u": u, "v": v}, config=ExecutionConfig())
    spec = AnimationSpec[WindVectorPlotSpec](
        plot_spec=WindVectorPlotSpec(
            u_data_array=VariableReference(name="u"),
            v_data_array=VariableReference(name="v"),
            barbs=True,
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
