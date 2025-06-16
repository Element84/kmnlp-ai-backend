"""Utils for use by the agent in its code."""

import asyncio
import os
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr
from dask.distributed import (
    Future,
    get_client,  # type: ignore[reportUnknownVariableType]
)
from e84_geoai_common.llm.models import BedrockClaudeLLM
from natural_language_geocoding import extract_geometry_from_text
from rasterio.features import (  # type: ignore[reportMissingTypeStubs]
    rasterize,  # type: ignore[reportUnknownVariableType]
)
from rasterio.transform import (  # type: ignore[reportMissingTypeStubs]
    from_bounds,  # type: ignore[reportUnknownVariableType]
)
from rastervision.core.data import RasterioCRSTransformer

if TYPE_CHECKING:
    from shapely.geometry import Polygon
    from shapely.geometry.base import BaseGeometry


class DaskArrayComputeError(Exception):
    """Failure to compute values from a dask DataArray."""


# This wraps compute_dask_array() to customize the docstring which is part of
# ScientificPythonAgent's prompt.
def compute(data_array: xr.DataArray) -> xr.DataArray:
    """Wraps data_array.compute() to correctly handle async dask clients.

    Always use this function instead of directly calling .compute(). Also,
    always call this function on a data_array before calling .values or
    .to_numpy() or .as_numpy() or .item() etc.
    """
    return compute_dask_array(data_array)


def compute_dask_array(data_array: xr.DataArray) -> xr.DataArray:
    """Convert to non-lazy array, accounting for possibly async dask client.

    - If data_array is not a dask array, return it as is.
    - If data_array is already computed, return it as is.
    - If there is no active dask client, call data_array.compute() directly.
    - If dask client is asynchronous, use asyncio.run() to run computation.
    - If dask client is synchronous, use client.sync() to run computation.
    - If computation does not result in a DataArray, raise an error.

    Raises:
        DaskArrayComputeError: If computation fails for any reason.
    """
    if data_array.chunks is None:
        # not a dask array
        return data_array

    try:
        client = get_client()
    except ValueError:
        # no active Dask client, compute directly
        return data_array.compute()  # type: ignore[reportUnknownMemberType]

    # client.compute()'s docs say that it returns a Future or list of Futures,
    # but its inferred return type is:
    #     Unknown |
    #     Generator[Generator[Unknown, None, None] |
    #     CoroutineType[Any, Any, Unknown] | Unknown, None, None] |
    #     CoroutineType[Any, Any, Unknown] |
    #     list[Unknown]  # noqa: ERA001
    # Hence the extra type checks below.
    compute_output = client.compute(data_array)  # type: ignore[reportUnknownMemberType]

    if isinstance(compute_output, xr.DataArray):
        return compute_output

    if not isinstance(compute_output, Future):
        msg = (
            "client.compute() returned unexpected type "
            f"{type(compute_output)}."  # type: ignore[reportUnknownArgumentType]
        )
        raise DaskArrayComputeError(msg)

    try:
        # sync or await result depending on client state
        if client.asynchronous:
            coroutine = compute_output.result()  # type: ignore[reportUnknownMemberType]
            result = asyncio.run(coroutine)  # type: ignore[reportUnknownArgumentType]
        else:
            result = client.sync(compute_output.result)  # type: ignore[reportUnknownMemberType]
    except Exception as e:
        method = "asyncio.run()" if client.asynchronous else "client.sync()"
        msg = f"Error encountered when calling {method} on .compute() output."
        raise DaskArrayComputeError(msg) from e

    if not isinstance(result, xr.DataArray):
        method = "asyncio.run()" if client.asynchronous else "client.sync()"
        msg = f"{method} returned unexpected type {type(result)}."  # type: ignore[reportUnknownArgumentType]
        raise DaskArrayComputeError(msg)

    return result


def geographic_coords_to_decimal_coords(coords: str) -> float:
    """Convert geographic coord to decimal coords.

    Args:
        coords: Geographic coords for a single direction. E.g. "20N" or "50W".

    Returns:
        Decimal coords in degrees.
    """
    from lat_lon_parser import parse  # type: ignore[reportUnknownVariableType]

    return parse(coords)


# support for type args for slice was added in Python 3.13
def filter_data_array(
    data_array: xr.DataArray,
    time: "slice[str, str, None] | None" = None,  # type: ignore[reportInvalidTypeArguments]
    lat: "slice[float, float, None] | None" = None,  # type: ignore[reportInvalidTypeArguments]
    lon: "slice[float, float, None] | None" = None,  # type: ignore[reportInvalidTypeArguments]
) -> xr.DataArray:
    """Filter a (time, lat, lon) DataArray by slicing along given dims.

    Use this function instead of data_array.sel() wherever possible. To filter
    to a single value, pass a slice where start and end are the same.

    Args:
        data_array: The DataArray to slice.
        time: Slice to apply to the time dim. Start and stop must be date
            strings of the form "YYYY-MM-DD" or ISO 8601 datetimes.
            Defaults to None (no slicing).
        lat: Slice to apply to the lat dim. Start and stop must be floats in
            the range [-90, 90]. Defaults to None (no slicing).
        lon: Slice to apply to the lon dim. Start and stop must be floats in
            the range [-180, 180]. Defaults to None (no slicing).

    Returns:
        The sliced DataArray.
    """
    indexers = {"time": time, "lat": lat, "lon": lon}
    indexers = {k: v for k, v in indexers.items() if v is not None}
    return data_array.sel(indexers)  # type: ignore[reportUnknownMemberType]


def max_coords(data_array: xr.DataArray) -> dict[str, str | float]:
    """Return the coords, along each dim, of the maximum value in data_array.

    Also returns the actual value at the point.

    Args:
        data_array: The DataArray to apply the operation to.

    Returns:
        A dictionary with dim names as keys and coords as values.
        One entry for each dim in the data_array.

    Sample output:
    {
        'time': '2024-10-01T12:00:00.000000000',
        'lat': 20.675352,
        'lon': -90.825652,
        'value': 35.2
    }
    """
    tmp: dict[str, xr.DataArray] = data_array.argmax(dim=...)  # type: ignore[reportAssignmentType]
    tmp = {k: compute(v) for k, v in tmp.items()}
    out = {  # type: ignore[reportUnknownVariableType]
        k: compute(data_array.coords[k][v.to_numpy()]).to_numpy()  # type: ignore[reportAssignmentType]
        for k, v in tmp.items()
    }
    out["time"] = str(np.datetime_as_string(out["time"]))  # type: ignore[reportArgumentType]
    out["lat"] = round(out["lat"].item(), 6)  # type: ignore[reportArgumentType]
    out["lon"] = round(out["lon"].item(), 6)  # type: ignore[reportArgumentType]
    out["value"] = compute(data_array.max()).values.item()  # type: ignore[reportMemberType]
    return out  # type: ignore[reportUnknownVariableType]


def min_coords(data_array: xr.DataArray) -> dict[str, str | float]:
    """Return the coords, along each dim, of the minimum value in data_array.

    Also returns the actual value at the point.

    Args:
        data_array: The DataArray to apply the operation to.

    Returns:
        A dictionary with dim names as keys and coords as values.
        One entry for each dim in the data_array.

    Sample output:
    {
        'time': '2024-10-01T12:00:00.000000000',
        'lat': 20.675352,
        'lon': -90.825652,
        'value': 4.7
    }
    """
    tmp: dict[str, xr.DataArray] = data_array.argmin(dim=...)  # type: ignore[reportAssignmentType]
    tmp = {k: compute(v) for k, v in tmp.items()}
    out = {  # type: ignore[reportUnknownVariableType]
        k: compute(data_array.coords[k][v.to_numpy()]).to_numpy()  # type: ignore[reportAssignmentType]
        for k, v in tmp.items()
    }
    out["time"] = str(np.datetime_as_string(out["time"]))  # type: ignore[reportArgumentType]
    out["lat"] = round(out["lat"].item(), 6)  # type: ignore[reportArgumentType]
    out["lon"] = round(out["lon"].item(), 6)  # type: ignore[reportArgumentType]
    out["value"] = compute(data_array.min()).values.item()  # type: ignore[reportMemberType]
    return out  # type: ignore[reportUnknownVariableType]


def geocode_natural_language_description(text: str) -> "BaseGeometry":
    """Geocode natural language description of an area into a shapely geometry.

    Args:
        text: Natural language description of a location. Example inputs:
            "Caspian Sea", "within 1 mile of the coast of Boston",
            "Florida north of Miami", etc.

    Returns:
        Shapely Polygon in EPSG:4326 coords representing the location.
    """
    os.environ["NOMINATIM_USER_AGENT"] = "e84_nl_geocoding"
    llm = BedrockClaudeLLM()
    geometry = extract_geometry_from_text(llm, text=text)
    if geometry.geom_type.lower() == "point":
        geometry = geometry.buffer(1)
    return geometry


def polygon_to_mask(
    polygon: "Polygon",
    data_array: xr.DataArray,
) -> npt.NDArray[np.bool_]:
    """Create spatial mask from polygon for (time, lat, lon) data_array.

    Args:
        polygon: Shapely Polygon in EPSG:4326 coords.
        data_array: DataArray of shape (time, lat, lon).

    Returns:
        2D mask as boolean numpy array of shape (lat, lon).
    """
    _, array_height, array_width = data_array.shape
    lat_min, lat_max = data_array.lat.min(), data_array.lat.max()
    lon_min, lon_max = data_array.lon.min(), data_array.lon.max()

    transform = from_bounds(  # type: ignore[reportUnknownVariableType]
        lon_min, lat_max, lon_max, lat_min, array_width, array_height
    )
    crs_tf = RasterioCRSTransformer(
        transform=transform,  # type: ignore[reportUnknownArgumentType]
        image_crs="EPSG:4326",
        map_crs="EPSG:4326",
        round_pixels=False,
    )
    aoi_pixel = crs_tf.map_to_pixel(polygon)  # type: ignore[reportUnknownVariableType]

    mask = rasterize(  # type: ignore[reportUnknownVariableType]
        shapes=[(aoi_pixel, 1)],
        out_shape=(array_height, array_width),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask  # type: ignore[reportUnknownVariableType]


def mask_data_array_with_polygon(
    data_array: xr.DataArray,
    polygon: "Polygon",
    *,
    crop: bool = False,
) -> xr.DataArray:
    """Mask out data_array with polygon, optionally cropping to polygon bounds.

    Equivalent to converting the polygon to a binary mask and then calling
    data_array.where(mask, drop=True).

    Args:
        polygon: Shapely Polygon in EPSG:4326 coords.
        data_array: DataArray of shape (time, lat, lon).
        crop: If True, crop data_array to the rectangular bounds of the
            polygon. Defaults to False.

    Returns:
        A DataArray with NaN values for coords falling outside the polygon. If
        crop=False, the output shape is the same as the input shape. If
        crop=True the output shape might be smaller in the spatial dimensions.
    """
    mask = polygon_to_mask(polygon, data_array)
    mask_da = xr.DataArray(
        mask,
        dims=("lat", "lon"),
        coords={"lat": data_array["lat"], "lon": data_array["lon"]},
    )
    out = data_array.where(mask_da, drop=crop)
    return out
