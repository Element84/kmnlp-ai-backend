from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from matplotlib.animation import FuncAnimation

# register MakePlot handlers
import llm_agent.agents.algorithm_agent.algorithm.exec.plot  # pyright: ignore[reportUnusedImport]
import llm_agent.agents.algorithm_agent.algorithm.exec.sst  # pyright: ignore[reportUnusedImport] # noqa: F401
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
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.sst import (
    SSTAnomalyPlotSpec,
    SSTPlotSpec,
)
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import ImageRef
from tests.utils.mocks.context import MockChatContext


@pytest.fixture
def context(chat_context: MockChatContext) -> ToolCallContext:
    return ToolCallContext(tool_call_id="test", chat_context=chat_context)


def _make_3d_latlon(ntime: int = 4, nlat: int = 8, nlon: int = 10, seed: int = 42) -> xr.DataArray:
    """3D array: (time, lat, lon) — suitable for geospatial raster animations."""
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(0, 30, size=(ntime, nlat, nlon)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(ntime),
            "lat": np.linspace(-90, 90, nlat),
            "lon": np.linspace(-180, 180, nlon),
        },
    )


def _make_latlon(nlat: int = 10, nlon: int = 15, seed: int = 42) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(0, 30, size=(nlat, nlon)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, nlat), "lon": np.linspace(-180, 180, nlon)},
    )


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_sst_plot(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = _make_latlon()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=SSTPlotSpec(
            sst_data_array=VariableReference(name="da"),
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_sst_plot_with_sea_ice_concentration(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    sst = _make_latlon()
    rng = np.random.default_rng(99)
    ice = xr.DataArray(
        rng.uniform(0, 1, size=(10, 15)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 15)},
    )
    exec_ctx = ExecutionContext(
        context, input_vars={"sst": sst, "ice": ice}, config=ExecutionConfig()
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=SSTPlotSpec(
            sst_data_array=VariableReference(name="sst"),
            sea_ice_data_array=VariableReference(name="ice"),
            ice_threshold=0.15,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_sst_plot_with_sea_ice_mask(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    sst = _make_latlon()
    rng = np.random.default_rng(99)
    ice_mask = xr.DataArray(
        rng.choice([False, True], size=(10, 15)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 15)},
    )
    exec_ctx = ExecutionContext(
        context, input_vars={"sst": sst, "ice": ice_mask}, config=ExecutionConfig()
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=SSTPlotSpec(
            sst_data_array=VariableReference(name="sst"),
            sea_ice_data_array=VariableReference(name="ice"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_ssta_plot(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(-3, 3, size=(10, 15)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 15)},
    )
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=SSTAnomalyPlotSpec(
            ssta_data_array=VariableReference(name="da"),
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_sst_animation(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = _make_3d_latlon()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    spec = AnimationSpec[SSTPlotSpec](
        plot_spec=SSTPlotSpec(
            sst_data_array=VariableReference(name="da"),
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_sst_animation_with_sea_ice(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    sst = _make_3d_latlon()
    rng = np.random.default_rng(99)
    ice = xr.DataArray(
        rng.uniform(0, 1, size=(4, 8, 10)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(4),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(-180, 180, 10),
        },
    )
    exec_ctx = ExecutionContext(
        context, input_vars={"sst": sst, "ice": ice}, config=ExecutionConfig()
    )
    spec = AnimationSpec[SSTPlotSpec](
        plot_spec=SSTPlotSpec(
            sst_data_array=VariableReference(name="sst"),
            sea_ice_data_array=VariableReference(name="ice"),
            ice_threshold=0.15,
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_ssta_animation(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(-3, 3, size=(4, 8, 10)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(4),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(-180, 180, 10),
        },
    )
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    spec = AnimationSpec[SSTAnomalyPlotSpec](
        plot_spec=SSTAnomalyPlotSpec(
            ssta_data_array=VariableReference(name="da"),
            map_projection=MapProjectionSpec(projection="PlateCarree"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
