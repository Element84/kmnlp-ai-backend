from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.animation import make_animation
from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionConfig, ExecutionContext
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnimationSpec,
    AnnotationSpec,
    BarPlotSpec,
    GeneralRasterPlotSpec,
    GeospatialRasterPlotSpec,
    LinePlotSpec,
    MultiLinePlotSpec,
)
from llm_agent.context import ToolCallContext
from tests.utils.mocks.context import MockChatContext


@pytest.fixture
def context(chat_context: MockChatContext) -> ToolCallContext:
    return ToolCallContext(tool_call_id="test", chat_context=chat_context)


def _make_2d_time_series(ntime: int = 5, nx: int = 10, seed: int = 42) -> xr.DataArray:
    """2D array: (time, x) — suitable for line animations."""
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(0, 10, size=(ntime, nx)),
        dims=["time", "x"],
        coords={"time": np.arange(ntime), "x": np.arange(nx)},
    )


def _make_3d_raster(ntime: int = 4, ny: int = 8, nx: int = 10, seed: int = 42) -> xr.DataArray:
    """3D array: (time, y, x) — suitable for raster animations."""
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(0, 10, size=(ntime, ny, nx)),
        dims=["time", "y", "x"],
        coords={"time": np.arange(ntime), "y": np.arange(ny), "x": np.arange(nx)},
    )


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


async def test_line_animation(context: ToolCallContext) -> None:
    da = _make_2d_time_series()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    spec = AnimationSpec[LinePlotSpec](
        plot_spec=LinePlotSpec(
            data_array=VariableReference(name="da"),
            annotation=AnnotationSpec(title="Line Anim"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


async def test_line_animation_with_shaded_error(context: ToolCallContext) -> None:
    da = _make_2d_time_series()
    err = xr.DataArray(
        np.full((5, 10), 0.5),
        dims=["time", "x"],
        coords={"time": np.arange(5), "x": np.arange(10)},
    )
    exec_ctx = ExecutionContext(
        context, input_vars={"da": da, "err": err}, config=ExecutionConfig()
    )
    spec = AnimationSpec[LinePlotSpec](
        plot_spec=LinePlotSpec(
            data_array=VariableReference(name="da"),
            data_array_error=VariableReference(name="err"),
            error_display_type="shaded",
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


async def test_line_animation_with_errorbars(context: ToolCallContext) -> None:
    da = _make_2d_time_series()
    err = xr.DataArray(
        np.full((5, 10), 0.5),
        dims=["time", "x"],
        coords={"time": np.arange(5), "x": np.arange(10)},
    )
    exec_ctx = ExecutionContext(
        context, input_vars={"da": da, "err": err}, config=ExecutionConfig()
    )
    spec = AnimationSpec[LinePlotSpec](
        plot_spec=LinePlotSpec(
            data_array=VariableReference(name="da"),
            data_array_error=VariableReference(name="err"),
            error_display_type="errorbars",
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


async def test_multi_line_animation(context: ToolCallContext) -> None:
    da1 = _make_2d_time_series(seed=1)
    da2 = _make_2d_time_series(seed=2)
    exec_ctx = ExecutionContext(
        context, input_vars={"da1": da1, "da2": da2}, config=ExecutionConfig()
    )
    spec = AnimationSpec[MultiLinePlotSpec](
        plot_spec=MultiLinePlotSpec(
            specs=[
                LinePlotSpec(
                    data_array=VariableReference(name="da1"),
                    annotation=AnnotationSpec(title="L1"),
                ),
                LinePlotSpec(
                    data_array=VariableReference(name="da2"),
                    annotation=AnnotationSpec(title="L2"),
                ),
            ],
            annotation=AnnotationSpec(title="Multi"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


async def test_bar_animation(context: ToolCallContext) -> None:
    h1 = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=["time"],
        coords={"time": np.arange(3)},
    )
    h2 = xr.DataArray(
        np.array([3.0, 2.0, 1.0]),
        dims=["time"],
        coords={"time": np.arange(3)},
    )
    exec_ctx = ExecutionContext(context, input_vars={"h1": h1, "h2": h2}, config=ExecutionConfig())
    spec = AnimationSpec[BarPlotSpec](
        plot_spec=BarPlotSpec(
            heights=[VariableReference(name="h1"), VariableReference(name="h2")],
            labels=["A", "B"],
            annotation=AnnotationSpec(title="Bar Anim"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


async def test_general_raster_animation(context: ToolCallContext) -> None:
    da = _make_3d_raster()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    spec = AnimationSpec[GeneralRasterPlotSpec](
        plot_spec=GeneralRasterPlotSpec(
            data_array=VariableReference(name="da"),
            annotation=AnnotationSpec(title="Raster Anim"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_geospatial_raster_animation(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = _make_3d_latlon()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=ExecutionConfig())
    spec = AnimationSpec[GeospatialRasterPlotSpec](
        plot_spec=GeospatialRasterPlotSpec(
            data_array=VariableReference(name="da"),
            annotation=AnnotationSpec(title="Geo Anim"),
        ),
        frame_dim="time",
    )
    anim = await make_animation(spec, exec_ctx)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")
