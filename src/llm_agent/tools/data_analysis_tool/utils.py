import logging

import xarray as xr

log = logging.getLogger(__name__)

DIM_NAME_MAPPING = {
    "latitude": "lat",
    "longitude": "lon",
}


def normalized_data_array(data_array: xr.DataArray) -> xr.DataArray:
    """Return a normalized version of the provided data array.

    Our code makes assumptions about the `DataArray`s we build from the virtual
    Zarr stores. Those assumptions are:
    * The dimension names will be standard: "lat" and "lon" rather than their
        spelled-out versions.
    * Longitude, if present, will be -180 to 180 degrees, rather than 0 to 360
        degrees.
    * Latitude, if present, will be from -90 to 90 degrees. This is not
        enforced at this time.

    This function makes the required changes (if any), and returns a new data
    array.

    Args: data_array (xr.DataArray): The data array to normalize

    Returns: A new data_array, normalized in the ways we expect.
    """
    data_array = data_array.unify_chunks()

    data_array = _ensure_standard_dim_names(data_array)
    data_array = _ensure_standard_longitude_range(data_array)
    data_array = _ensure_lat_lon_sorted(data_array)

    return data_array


def _ensure_standard_dim_names(da: xr.DataArray) -> xr.DataArray:
    for original_dim, mapped_dim in DIM_NAME_MAPPING.items():
        if original_dim in da.dims and mapped_dim not in da.dims:
            da = da.rename({original_dim: mapped_dim})
    return da


def _ensure_standard_longitude_range(da: xr.DataArray) -> xr.DataArray:
    """If longitude range is 0 to 360, convert to -180 to 180."""
    if "lon" in da.dims:
        lon = da["lon"]
        if lon.max() > 180:  # noqa: PLR2004
            # https://gis.stackexchange.com/a/201793
            lon = ((lon + 180) % 360) - 180
            da = da.assign_coords(lon=lon)  # pyright: ignore[reportUnknownMemberType]
    return da


def _ensure_lat_lon_sorted(da: xr.DataArray) -> xr.DataArray:
    """Ensure that lat and lon dimensions are sorted in ascending order."""
    if "lat" in da.dims:
        da = da.sortby("lat")  # pyright: ignore[reportUnknownMemberType]
    if "lon" in da.dims:
        da = da.sortby("lon")  # pyright: ignore[reportUnknownMemberType]
    return da
