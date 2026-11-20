from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

# register MakePlot handler
import llm_agent.agents.algorithm_agent.algorithm.exec.plot  # pyright: ignore[reportUnusedImport] # noqa: F401
from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    AlgorithmExecutionError,
    ExecutionConfig,
    ExecutionContext,
    eval_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import DataArrayShapeError
from llm_agent.agents.algorithm_agent.algorithm.exec.plot import (
    _antimeridian_lon_extent,  # type: ignore[reportPrivateUsage]
    make_general_raster_plot,
    make_histogram_plot,
    setup_geospatial_axes,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot_utils import (
    deoverlap_x_tick_labels,
    format_coordinate_label,
    geospatial_cbar_kwargs,
    timedelta_to_numeric,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    BarPlotSpec,
    GeneralRasterPlotSpec,
    GeospatialRasterPlotSpec,
    GeospatialSpec,
    HistogramPlotSpec,
    LinePlotSpec,
    MakePlot,
    MapProjectionSpec,
    MultiLinePlotSpec,
    PlotGridSpec,
    RasterStyleSpec,
    ScatterPlotSpec,
)
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import ImageRef
from llm_agent.utils.dask import DataSizeTooLargeError
from tests.utils.mocks.context import MockChatContext

default_exec_config = ExecutionConfig()


@pytest.fixture
def context(chat_context: MockChatContext) -> ToolCallContext:
    return ToolCallContext(tool_call_id="test", chat_context=chat_context)


def test_timedelta_to_numeric_uses_hours_for_short_ranges():
    lead = np.arange(0, 12).astype("timedelta64[h]").astype("timedelta64[ns]")
    values, unit = timedelta_to_numeric(lead)
    assert unit == "hours"
    np.testing.assert_allclose(values, np.arange(0, 12))


def test_timedelta_to_numeric_uses_days_for_long_ranges():
    lead = (np.arange(0, 16) * 24).astype("timedelta64[h]").astype("timedelta64[ns]")
    values, unit = timedelta_to_numeric(lead)
    assert unit == "days"
    np.testing.assert_allclose(values, np.arange(0, 16))


def test_format_coordinate_label_timedelta_hours_and_days():
    assert format_coordinate_label(np.timedelta64(6, "h")) == "6 h"
    assert format_coordinate_label(np.timedelta64(3, "D")) == "3 days"


def test_format_coordinate_label_datetime_and_scalar():
    assert format_coordinate_label(np.datetime64("2023-08-01T00")) == "2023-08-01T00"
    assert format_coordinate_label(5) == "5"


def test_geospatial_cbar_kwargs_vertical_for_tall_map():
    da = xr.DataArray(
        np.zeros((10, 4)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-40, 40, 10), "lon": np.linspace(-5, 5, 4)},
    )
    assert geospatial_cbar_kwargs(da)["orientation"] == "vertical"


def test_geospatial_cbar_kwargs_horizontal_for_wide_map():
    da = xr.DataArray(
        np.zeros((4, 40)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 4), "lon": np.linspace(-120, 120, 40)},
    )
    kw = geospatial_cbar_kwargs(da, label="speed", extend="both")
    assert kw["orientation"] == "horizontal"
    assert kw["label"] == "speed"
    assert kw["extend"] == "both"


def test_deoverlap_rotates_crowded_x_labels():
    fig, ax = plt.subplots(figsize=(2.5, 2))  # type: ignore[reportUnknownMemberType]
    ax.plot(range(50), range(50))  # type: ignore[reportUnknownMemberType]
    ax.set_xticks(range(0, 50, 3))  # type: ignore[reportUnknownMemberType]
    ax.set_xticklabels([f"2020-{i:02d}" for i in range(0, 50, 3)])  # type: ignore[reportUnknownMemberType]
    deoverlap_x_tick_labels(fig)
    rotations = {t.get_rotation() for t in ax.get_xticklabels() if t.get_text()}  # type: ignore[reportUnknownMemberType]
    assert rotations == {30.0}
    plt.close(fig)


def test_deoverlap_leaves_sparse_x_labels_horizontal():
    fig, ax = plt.subplots(figsize=(8, 3))  # type: ignore[reportUnknownMemberType]
    ax.plot([0, 1, 2], [0, 1, 2])  # type: ignore[reportUnknownMemberType]
    ax.set_xticks([0, 1, 2])  # type: ignore[reportUnknownMemberType]
    ax.set_xticklabels(["a", "b", "c"])  # type: ignore[reportUnknownMemberType]
    deoverlap_x_tick_labels(fig)
    assert all(t.get_rotation() == 0 for t in ax.get_xticklabels() if t.get_text())  # type: ignore[reportUnknownMemberType]
    plt.close(fig)


def _chukchi_like_data_array() -> xr.DataArray:
    """Data valid only near +/-180 (wrapping the antimeridian), NaN elsewhere."""
    lon = np.arange(-180, 180, 5.0)
    lat = np.arange(55, 80, 5.0)
    data = np.full((lat.size, lon.size), np.nan)
    data[:, (lon >= 165) | (lon <= -165)] = 1.0
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


def test_antimeridian_lon_extent_detects_wrap():
    assert _antimeridian_lon_extent(_chukchi_like_data_array()) == (165.0, 195.0)


def test_antimeridian_lon_extent_none_for_normal_region():
    lon = np.arange(-180, 180, 5.0)
    lat = np.arange(0, 20, 5.0)
    data = np.full((lat.size, lon.size), np.nan)
    data[:, (lon >= -20) & (lon <= 20)] = 1.0
    da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    assert _antimeridian_lon_extent(da) is None


@pytest.mark.parametrize("projection", ["PlateCarree", "Robinson", "EqualEarth", "AlbersEqualArea"])
@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
def test_setup_geospatial_axes_recenters_antimeridian_region(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    projection: str,
) -> None:
    da = _chukchi_like_data_array()
    spec = GeospatialSpec(map_projection=MapProjectionSpec(projection=projection))  # pyright: ignore[reportArgumentType]
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]
    ax = setup_geospatial_axes(da, spec, lambda **k: fig.add_subplot(1, 1, 1, **k))  # type: ignore[reportUnknownArgumentType]
    # The projection is recentered onto the region near the antimeridian (~180).
    lon_0 = float(ax.projection.proj4_params["lon_0"])  # type: ignore[reportUnknownArgumentType]
    assert abs(abs(lon_0) - 180.0) < 1.0
    plt.close(fig)  # type: ignore[reportUnknownMemberType]


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
def test_setup_geospatial_axes_polar_not_recentered(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
) -> None:
    da = _chukchi_like_data_array()
    spec = GeospatialSpec(map_projection=MapProjectionSpec(projection="NorthPolarStereo"))
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]
    ax = setup_geospatial_axes(da, spec, lambda **k: fig.add_subplot(1, 1, 1, **k))  # type: ignore[reportUnknownArgumentType]
    # Polar projections show the full disc, so they are left at the default center.
    assert float(ax.projection.proj4_params["lon_0"]) == 0.0  # type: ignore[reportUnknownArgumentType]
    plt.close(fig)  # type: ignore[reportUnknownMemberType]


def test_deoverlap_skips_hidden_x_labels():
    # Crowded labels, but the x-axis is hidden -> nothing to rotate.
    fig, ax = plt.subplots(figsize=(2.5, 2))  # type: ignore[reportUnknownMemberType]
    ax.plot(range(50), range(50))  # type: ignore[reportUnknownMemberType]
    ax.set_xticks(range(0, 50, 3))  # type: ignore[reportUnknownMemberType]
    ax.set_xticklabels([f"2020-{i:02d}" for i in range(0, 50, 3)])  # type: ignore[reportUnknownMemberType]
    ax.xaxis.set_visible(False)  # type: ignore[reportUnknownMemberType]
    deoverlap_x_tick_labels(fig)
    assert all(t.get_rotation() == 0 for t in ax.get_xticklabels() if t.get_text())  # type: ignore[reportUnknownMemberType]
    plt.close(fig)


def _make_1d(n: int = 20, seed: int = 42) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(rng.uniform(0, 10, size=n), dims=["x"], coords={"x": np.arange(n)})


def _make_2d(ny: int = 10, nx: int = 15, seed: int = 42) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(0, 10, size=(ny, nx)),
        dims=["y", "x"],
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )


def _make_latlon(nlat: int = 10, nlon: int = 15, seed: int = 42) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(0, 30, size=(nlat, nlon)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, nlat), "lon": np.linspace(-180, 180, nlon)},
    )


async def test_make_line_plot(context: ToolCallContext) -> None:
    da = _make_1d()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=LinePlotSpec(
            data_array=VariableReference(name="da"),
            annotation=AnnotationSpec(title="Line", xlabel="x", ylabel="y"),
            add_linear_trendline=True,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_line_plot_with_shaded_error(context: ToolCallContext) -> None:
    da = _make_1d()
    err = xr.DataArray(np.full(20, 0.5), dims=["x"], coords={"x": np.arange(20)})
    exec_ctx = ExecutionContext(
        context, input_vars={"da": da, "err": err}, config=default_exec_config
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=LinePlotSpec(
            data_array=VariableReference(name="da"),
            data_array_error=VariableReference(name="err"),
            error_display_type="shaded",
            add_linear_trendline=True,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_line_plot_with_errorbars(context: ToolCallContext) -> None:
    da = _make_1d()
    err = xr.DataArray(np.full(20, 0.5), dims=["x"], coords={"x": np.arange(20)})
    exec_ctx = ExecutionContext(
        context, input_vars={"da": da, "err": err}, config=default_exec_config
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=LinePlotSpec(
            data_array=VariableReference(name="da"),
            data_array_error=VariableReference(name="err"),
            error_display_type="errorbars",
            add_linear_trendline=True,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


@pytest.mark.parametrize("density", [False, True])
async def test_make_histogram_plot(context: ToolCallContext, density: bool) -> None:  # noqa: FBT001
    da = _make_2d()  # multi-dim: values are flattened across all dims
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=HistogramPlotSpec(
            data_array=VariableReference(name="da"),
            bins=15,
            density=density,
            annotation=AnnotationSpec(title="Distribution", xlabel="value", ylabel="count"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_histogram_plot_ignores_nans(context: ToolCallContext) -> None:
    da = _make_1d()
    da = da.where(da > 5)  # introduce NaNs
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=HistogramPlotSpec(data_array=VariableReference(name="da")),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def _histogram_bar_heights(
    da: xr.DataArray,
    context: ToolCallContext,
    *,
    bins: int = 10,
    density: bool = False,
    config: ExecutionConfig | None = None,
) -> np.ndarray:
    """Run the histogram plot directly and read back the drawn bar heights."""
    exec_ctx = ExecutionContext(
        context, input_vars={"da": da}, config=config or default_exec_config
    )
    spec = HistogramPlotSpec(data_array=VariableReference(name="da"), bins=bins, density=density)
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]
    try:
        ax = await make_histogram_plot(spec, exec_ctx, lambda **kw: fig.add_subplot(1, 1, 1, **kw))  # type: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        return np.array([patch.get_height() for patch in ax.patches])  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("density", [False, True])
async def test_make_histogram_plot_chunked_matches_numpy(
    context: ToolCallContext,
    density: bool,  # noqa: FBT001
) -> None:
    # Binning a chunked array streams over chunks; the counts must still match numpy exactly.
    da = _make_2d()
    expected, _ = np.histogram(
        np.asarray(da.values).ravel(),
        bins=10,
        range=(float(da.min()), float(da.max())),
        density=density,
    )
    got = await _histogram_bar_heights(da.chunk({"y": 3, "x": 4}), context, density=density)  # type: ignore[reportUnknownMemberType]
    np.testing.assert_allclose(got, expected)


async def test_make_histogram_plot_chunked_matches_eager(context: ToolCallContext) -> None:
    da = _make_2d()
    np.testing.assert_allclose(
        await _histogram_bar_heights(da.chunk({"y": 3, "x": 4}), context),  # type: ignore[reportUnknownMemberType]
        await _histogram_bar_heights(da, context),
    )


async def test_make_histogram_plot_constant_field(context: ToolCallContext) -> None:
    # min == max would leave zero-width bins; the range must be widened like numpy does.
    da = xr.DataArray(np.full((6, 6), 3.0), dims=["y", "x"]).chunk({"y": 3})  # type: ignore[reportUnknownMemberType]
    heights = await _histogram_bar_heights(da, context)
    assert heights.sum() == 36


async def test_make_histogram_plot_all_nan_raises(context: ToolCallContext) -> None:
    da = xr.DataArray(np.full((4, 4), np.nan), dims=["y", "x"])
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=HistogramPlotSpec(data_array=VariableReference(name="da")),
    )
    with pytest.raises(AlgorithmExecutionError, match="finite"):
        await eval_expression(expr, exec_ctx)


async def test_make_histogram_plot_allows_larger_than_base_limit(context: ToolCallContext) -> None:
    # Binning streams over chunks, so it gets the same relaxed limit as streamable reductions.
    da = _make_2d(ny=100, nx=100).chunk({"y": 25})  # type: ignore[reportUnknownMemberType]
    config = ExecutionConfig(data_array_compute_size_limit=da.nbytes // 2)
    heights = await _histogram_bar_heights(da, context, config=config)
    assert heights.sum() == 100 * 100


async def test_make_histogram_plot_rejects_beyond_relaxed_limit(context: ToolCallContext) -> None:
    da = _make_2d(ny=100, nx=100).chunk({"y": 25})  # type: ignore[reportUnknownMemberType]
    exec_ctx = ExecutionContext(
        context,
        input_vars={"da": da},
        config=ExecutionConfig(data_array_compute_size_limit=da.nbytes // 10),
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=HistogramPlotSpec(data_array=VariableReference(name="da")),
    )
    with pytest.raises(DataSizeTooLargeError):
        await eval_expression(expr, exec_ctx)


@pytest.mark.parametrize("add_regression_line", [False, True])
async def test_make_scatter_plot(
    context: ToolCallContext,
    add_regression_line: bool,  # noqa: FBT001
) -> None:
    x = _make_1d(seed=1)
    y = _make_1d(seed=2)
    exec_ctx = ExecutionContext(context, input_vars={"x": x, "y": y}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=ScatterPlotSpec(
            x_data_array=VariableReference(name="x"),
            y_data_array=VariableReference(name="y"),
            add_regression_line=add_regression_line,
            annotation=AnnotationSpec(title="Scatter", xlabel="x", ylabel="y"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_scatter_plot_length_mismatch_raises(context: ToolCallContext) -> None:
    x = _make_1d(n=20)
    y = _make_1d(n=10)
    exec_ctx = ExecutionContext(context, input_vars={"x": x, "y": y}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=ScatterPlotSpec(
            x_data_array=VariableReference(name="x"),
            y_data_array=VariableReference(name="y"),
        ),
    )
    with pytest.raises(DataArrayShapeError):
        await eval_expression(expr, exec_ctx)


async def test_make_multi_line_plot(context: ToolCallContext) -> None:
    da1 = _make_1d(seed=1)
    da2 = _make_1d(seed=2)
    exec_ctx = ExecutionContext(
        context, input_vars={"da1": da1, "da2": da2}, config=default_exec_config
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=MultiLinePlotSpec(
            specs=[
                LinePlotSpec(
                    data_array=VariableReference(name="da1"),
                    annotation=AnnotationSpec(title="Line 1"),
                    add_linear_trendline=True,
                ),
                LinePlotSpec(
                    data_array=VariableReference(name="da2"),
                    annotation=AnnotationSpec(title="Line 2"),
                    add_linear_trendline=True,
                ),
            ],
            annotation=AnnotationSpec(title="Multi"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_bar_plot(context: ToolCallContext) -> None:
    exec_ctx = ExecutionContext(
        context, input_vars={"h1": 3.5, "h2": 7.2}, config=default_exec_config
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=BarPlotSpec(
            heights=[VariableReference(name="h1"), VariableReference(name="h2")],
            labels=["A", "B"],
            annotation=AnnotationSpec(title="Bar"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_general_raster_plot(context: ToolCallContext) -> None:
    da = _make_2d()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=GeneralRasterPlotSpec(
            data_array=VariableReference(name="da"),
            annotation=AnnotationSpec(title="Raster"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_general_raster_plot_uses_colorbar_label(context: ToolCallContext) -> None:
    da = _make_2d()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    spec = GeneralRasterPlotSpec(
        data_array=VariableReference(name="da"),
        style=RasterStyleSpec(colorbar_label="My Units"),
    )
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]

    def create_ax(**kwargs: object) -> object:
        return fig.add_subplot(1, 1, 1, **kwargs)  # type: ignore[reportUnknownMemberType]

    _, im = await make_general_raster_plot(spec, exec_ctx, create_ax)  # type: ignore[arg-type]
    assert im.colorbar is not None
    assert im.colorbar.ax.get_ylabel() == "My Units"
    plt.close(fig)


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
@patch("cartopy.mpl.geoaxes.GeoAxes.coastlines")
@patch("cartopy.mpl.geoaxes.GeoAxes.stock_img")
async def test_make_geospatial_raster_plot(
    mock_stock_img: MagicMock,  # noqa: ARG001
    mock_coastlines: MagicMock,  # noqa: ARG001
    mock_add_feature: MagicMock,  # noqa: ARG001
    context: ToolCallContext,
) -> None:
    da = _make_latlon()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=GeospatialRasterPlotSpec(
            data_array=VariableReference(name="da"),
            annotation=AnnotationSpec(title="Geospatial"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_plot_grid(context: ToolCallContext) -> None:
    da1 = _make_1d(seed=1)
    da2 = _make_1d(seed=2)
    exec_ctx = ExecutionContext(
        context, input_vars={"da1": da1, "da2": da2}, config=default_exec_config
    )
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=PlotGridSpec(
            ncols=2,
            subplot_specs=[
                LinePlotSpec(
                    data_array=VariableReference(name="da1"),
                    annotation=AnnotationSpec(title="Left"),
                ),
                LinePlotSpec(
                    data_array=VariableReference(name="da2"),
                    annotation=AnnotationSpec(title="Right"),
                ),
            ],
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)
