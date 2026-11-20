"""Utils for use by the agent in its code."""

import logging
import os

import numpy as np
import numpy.typing as npt
import xarray as xr
from anyio import to_thread
from e84_geoai_common.llm.models import BedrockClaudeLLM
from langfuse import observe  # type: ignore[reportUnknownVariableType]
from lat_lon_parser import parse  # type: ignore[reportUnknownVariableType]
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from natural_language_geocoding import extract_geometry_from_text
from natural_language_geocoding.place_lookup import PlaceLookup
from rasterio.features import (  # type: ignore[reportMissingTypeStubs]
    rasterize,  # type: ignore[reportUnknownVariableType]
)
from rasterio.transform import (  # type: ignore[reportMissingTypeStubs]
    from_bounds,  # type: ignore[reportUnknownVariableType]
)
from rastervision.core.data import RasterioCRSTransformer
from requests.exceptions import Timeout
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from llm_agent.utils.dask import (
    SPATIAL_VALID_FRACTION_ATTR,
    MaterializedDataArray,
    SpatiallyMaskedDataArray,
    compute_dask_array,
)

log = logging.getLogger(__name__)


# This wraps compute_dask_array() to customize the docstring which is part of
# ScientificPythonAgent's prompt.
async def compute(data_array: xr.DataArray) -> MaterializedDataArray:
    """Wraps data_array.compute() to correctly handle async dask clients.

    Always use this function instead of directly calling .compute(). Also,
    always call this function on a data_array before calling .values or
    .to_numpy() or .as_numpy() or .item() etc.
    """
    return await compute_dask_array(data_array)


def geographic_coords_to_decimal_coords(coords: str) -> float:
    """Convert geographic coord to decimal coords.

    Args:
        coords: Geographic coords for a single direction. E.g. "20N" or "50W".

    Returns:
        Decimal coords in degrees.
    """
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


async def max_coords(data_array: xr.DataArray) -> dict[str, str | float]:
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
    tmp = {k: await compute_dask_array(v) for k, v in tmp.items()}
    out = {  # type: ignore[reportUnknownVariableType]
        k: (await compute_dask_array(data_array.coords[k][v.to_numpy()])).to_numpy()  # type: ignore[reportAssignmentType]
        for k, v in tmp.items()
    }
    out["time"] = str(np.datetime_as_string(out["time"]))  # type: ignore[reportArgumentType]
    out["lat"] = round(out["lat"].item(), 6)  # type: ignore[reportArgumentType]
    out["lon"] = round(out["lon"].item(), 6)  # type: ignore[reportArgumentType]
    out["value"] = (await compute_dask_array(data_array.max())).values.item()  # type: ignore[reportMemberType]
    return out  # type: ignore[reportUnknownVariableType]


async def min_coords(data_array: xr.DataArray) -> dict[str, str | float]:
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
    tmp = {k: await compute_dask_array(v) for k, v in tmp.items()}
    out = {  # type: ignore[reportUnknownVariableType]
        k: (await compute_dask_array(data_array.coords[k][v.to_numpy()])).to_numpy()  # type: ignore[reportAssignmentType]
        for k, v in tmp.items()
    }
    out["time"] = str(np.datetime_as_string(out["time"]))  # type: ignore[reportArgumentType]
    out["lat"] = round(out["lat"].item(), 6)  # type: ignore[reportArgumentType]
    out["lon"] = round(out["lon"].item(), 6)  # type: ignore[reportArgumentType]
    out["value"] = (await compute_dask_array(data_array.min())).values.item()  # type: ignore[reportMemberType]
    return out  # type: ignore[reportUnknownVariableType]


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, Timeout)),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
@observe()
async def geocode_natural_language_description(
    place_lookup: PlaceLookup, bedrock_client: BedrockRuntimeClient, text: str
) -> BaseGeometry:
    """Geocode natural language description of an area into a shapely geometry.

    Args:
        place_lookup (PlaceLookup): Implementation of place lookup for natural
            language geocoding.
        bedrock_client (BedrockRuntimeClient): Bedrock Client to be use in the place lookup.
        text (str): Natural language description of a location. Example inputs:
            "Caspian Sea", "within 1 mile of the coast of Boston", "Florida
            north of Miami", etc.

    Returns:
        Shapely geometry in EPSG:4326 coords representing the location.
    """
    os.environ["NOMINATIM_USER_AGENT"] = "e84_nl_geocoding"
    llm = BedrockClaudeLLM(client=bedrock_client)
    return await to_thread.run_sync(extract_geometry_from_text, llm, text, place_lookup)


def polygon_to_mask(
    polygon: Polygon | MultiPolygon, data_array: xr.DataArray
) -> npt.NDArray[np.bool_]:
    """Create spatial mask from polygon for (time, lat, lon) data_array.

    Args:
        polygon: Shapely Polygon in EPSG:4326 coords.
        data_array: DataArray of shape (time, lat, lon).

    Returns:
        2D mask as boolean numpy array of shape (lat, lon).
    """
    array_height = data_array.sizes["lat"]
    array_width = data_array.sizes["lon"]
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
    aoi_in_pixel_coords = crs_tf.map_to_pixel(polygon)  # type: ignore[reportUnknownVariableType]

    mask = rasterize(  # type: ignore[reportUnknownVariableType]
        shapes=[(aoi_in_pixel_coords, 1)],
        out_shape=(array_height, array_width),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask  # type: ignore[reportUnknownVariableType]


def mask_data_array_with_polygon(
    data_array: xr.DataArray, polygon: Polygon | MultiPolygon
) -> SpatiallyMaskedDataArray:
    """Mask out data_array with polygon, without cropping.

    Equivalent to converting the polygon to a binary mask and then calling
    data_array.where(mask, drop=False).

    Args:
        polygon: Shapely Polygon in EPSG:4326 coords.
        data_array: DataArray of shape (time, lat, lon).

    Returns:
        A DataArray with NaN values for coords falling outside the polygon. The
        output shape is the same as the input shape. The fraction of lat/lon cells kept is recorded
        in the ``SPATIAL_VALID_FRACTION_ATTR`` attribute so downstream size checks can count only
        the masked region rather than the full (mostly-NaN) bounding box.
    """
    mask = polygon_to_mask(polygon, data_array)
    mask_da = xr.DataArray(
        mask,
        dims=("lat", "lon"),
        coords={"lat": data_array["lat"], "lon": data_array["lon"]},
    )
    # drop=False leaves shape unchanged, fills NaNs outside mask
    out = data_array.where(mask_da, drop=False)
    # `mask` is an in-memory numpy array, so this is cheap (no Dask compute).
    out.attrs[SPATIAL_VALID_FRACTION_ATTR] = float(mask.mean())
    return SpatiallyMaskedDataArray(out)
