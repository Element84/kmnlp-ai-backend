import numpy as np
import xarray as xr

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    VariableAssignment,
    VariableReference,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    ReduceDataArray,
    SliceDataArray,
)
from llm_agent.tools.data_analysis_tool.utils import algorithm_to_pseudocode, normalized_data_array


def test_leaves_normal_data_array_alone(lat_lon_data_array: xr.DataArray):
    initial_data_array = lat_lon_data_array
    modified_data_array = normalized_data_array(initial_data_array)
    assert initial_data_array.equals(modified_data_array)


def test_normalizes_data_array_with_strange_args():
    # Stub data array, 2D
    numpy_array = np.arange(100).reshape(10, 10).astype(float)  # min=0.0, max=99.0

    # Define spatial coordinates
    x_coords = np.linspace(0, 360, 10)  # longitude
    y_coords = np.linspace(-90, 90, 10)  # latitude
    y_coords = y_coords[::-1]  # Reverse to be in descending order, to test sorting

    # Create DataArray with spatial dimensions and coordinates
    initial_data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"latitude": y_coords, "longitude": x_coords},
        dims=("latitude", "longitude"),
        name="some name",
        attrs={"units": "example unit"},
    )

    # Set CRS and spatial information using rioxarray
    initial_data_array.rio.write_crs("EPSG:4326", inplace=True)
    initial_data_array.rio.set_spatial_dims(x_dim="latitude", y_dim="longitude", inplace=True)
    initial_data_array.rio.write_coordinate_system(inplace=True)

    modified_data_array = normalized_data_array(initial_data_array)

    assert "lat" in modified_data_array.dims
    assert "lon" in modified_data_array.dims

    assert "latitude" not in modified_data_array.dims
    assert "longitude" not in modified_data_array.dims

    assert modified_data_array.lat.min().item() == -90
    assert modified_data_array.lat.max().item() == 90

    assert modified_data_array.lon.min().item() == -160
    assert modified_data_array.lon.max().item() == 160

    # Check coords sorted
    np.testing.assert_array_equal(
        np.sort(modified_data_array.lat.values), modified_data_array.lat.values
    )
    np.testing.assert_array_equal(
        np.sort(modified_data_array.lon.values), modified_data_array.lon.values
    )


def _lon_encoded_data_array(lon: np.ndarray, *, chunk: bool = False) -> xr.DataArray:
    """Build a (lat, lon) array whose values equal their lon coordinate, for alignment checks."""
    lat = np.array([-10.0, 0.0, 10.0])
    data = np.broadcast_to(lon, (lat.size, lon.size)).astype(float).copy()
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return da.chunk({"lon": 2}) if chunk else da  # pyright: ignore[reportUnknownMemberType]


def test_rotated_longitude_roll_preserves_alignment():
    # 0-360 grid: after conversion to -180..180 the axis is rotated-ascending (rolled, not sorted).
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    result = normalized_data_array(_lon_encoded_data_array(lon))

    # Ascending, and each value still matches its (converted) longitude -> alignment preserved.
    np.testing.assert_array_equal(result.lon.values, np.array([-180.0, -90.0, 0.0, 90.0]))
    for lon_val, expected in [(-180.0, 180.0), (-90.0, 270.0), (0.0, 0.0), (90.0, 90.0)]:
        assert result.sel(lon=lon_val).isel(lat=0).item() == expected  # pyright: ignore[reportUnknownMemberType]


def test_descending_latitude_reversed_preserves_alignment():
    lat = np.array([90.0, 45.0, 0.0, -45.0, -90.0])
    data = np.broadcast_to(lat[:, None], (lat.size, 2)).astype(float).copy()
    da = xr.DataArray(data, coords={"lat": lat, "lon": [-10.0, 10.0]}, dims=("lat", "lon"))

    result = normalized_data_array(da)

    np.testing.assert_array_equal(result.lat.values, np.array([-90.0, -45.0, 0.0, 45.0, 90.0]))
    for lat_val in lat:
        assert result.sel(lat=lat_val).isel(lon=0).item() == lat_val  # pyright: ignore[reportUnknownMemberType]


def test_already_ascending_longitude_unchanged():
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    da = _lon_encoded_data_array(lon)
    result = normalized_data_array(da)
    assert result.equals(da)


def test_normalization_stays_lazy_on_dask_input():
    # A rotated-ascending longitude on a Dask-backed array must stay lazy (no eager compute).
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    da = _lon_encoded_data_array(lon, chunk=True)
    result = normalized_data_array(da)

    assert result.chunks is not None  # still Dask-backed
    np.testing.assert_array_equal(result.lon.values, np.array([-180.0, -90.0, 0.0, 90.0]))
    assert result.sel(lon=-90.0).isel(lat=0).compute().item() == 270.0  # pyright: ignore[reportUnknownMemberType]


def test_algorithm_to_pseudocode():
    slice_func = SliceDataArray.with_constrained_dims(["x", "y"])
    reduce_func = ReduceDataArray.with_constrained_dims(["x", "y"])
    func_call_model = make_function_call_model([slice_func, reduce_func])
    algorithm = Algorithm[func_call_model](
        statements=[
            VariableAssignment(
                name="filtered",
                value=func_call_model(
                    func=slice_func(
                        func_name="SliceDataArray_XY",
                        data_array=VariableReference(name="data"),
                        dim_slices={"x": (1, 2), "y": (3, 4)},
                    )
                ),
                description="(Long comment:) Filter the data. Lorem ipsum dolor sit amet "
                "consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore "
                "magna aliqua.",
            ),
            VariableAssignment(
                name="out",
                value=func_call_model(
                    func=reduce_func(
                        func_name="ReduceDataArray_XY",
                        data_array=VariableReference(name="filtered"),
                        op="mean",
                        dim=("x", "y"),
                    )
                ),
                description="Reduce the filtered data to obtain the mean.",
            ),
        ],
        output_variables=[VariableReference(name="out")],
    )
    pseudocode = algorithm_to_pseudocode(algorithm)
    expected_pseudocode = """\
# (Long comment:) Filter the data. Lorem ipsum dolor sit amet consectetur
# adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna
# aliqua.
filtered = SliceDataArray_XY(
    data_array=data,
    dim_slices={x: (1.0, 2.0), y: (3.0, 4.0)},
    dim_index_slices={},
    method="nearest",
)

# Reduce the filtered data to obtain the mean.
out = ReduceDataArray_XY(data_array=filtered, op="mean", dim=("x", "y"), q=None)
"""
    assert pseudocode == expected_pseudocode
