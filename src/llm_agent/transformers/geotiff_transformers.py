import logging
from tempfile import NamedTemporaryFile

import numpy as np
import rioxarray  # noqa: F401 # type: ignore[unused-ignore]
import xarray as xr

from llm_agent.context import ChatContext
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

WORLD_GEODETIC_SYSTEM_1984 = "EPSG:4326"
PSEUDO_MERCATOR_SYSTEM = "EPSG:3857"


class InvalidShapeForGeoTIFFError(Exception):
    """Raised when the input data array has an invalid shape for GeoTIFF generation."""

    def __init__(self, dims: tuple[str, ...], shape: tuple[int, ...]) -> None:  # noqa: D107
        shape_dict = dict(zip(dims, shape, strict=False))
        msg = f"Expected a DataArray squeezable to 2D, but got shape: {shape_dict}."
        super().__init__(msg)


async def data_array_to_geotiff_ref(data_array: xr.DataArray, context: ChatContext) -> GeoTIFFRef:
    """Create a GeoTIFF for a data array and return its file path.

    Args:
        data_array (xr.DataArray): A 2D data array to render as a GeoTIFF
        context (ChatContext): The conversation context

    Returns:
        A GeoTIFFRef
    """
    log.debug("data array coords and dims: %s, %s", data_array.dims, data_array.coords)  # type: ignore[reportUnknownMemberType]
    data_array = _preprocess_data_array(data_array)

    with NamedTemporaryFile(prefix="geotiff_", suffix=".tif") as tmpfile:
        _save_data_array_as_geotiff(data_array, tmpfile.name)
        ref_key = context.persist_data(data_or_path=tmpfile.name, data_type="geotiff")

    return GeoTIFFRef(
        key=ref_key,
        name=(str(data_array.name) if data_array.name else "data"),
        units=data_array.attrs.get("units"),
        min_value=data_array.min(skipna=True).values.item(),
        max_value=data_array.max(skipna=True).values.item(),
    )


def _save_data_array_as_geotiff(data_array: xr.DataArray, path: str) -> None:
    """Write a 2D data array to a GeoTIFF file.

    Args:
        data_array (xr.DataArray): A 2D data array to render as a GeoTIFF
        path (str): The file path to save the GeoTIFF
    """
    # Set spatial dims and write CRS
    data_array_with_crs = data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    data_array_with_crs = data_array_with_crs.rio.write_crs(WORLD_GEODETIC_SYSTEM_1984)

    # Reproject to EPSG:3857 (OpenStreetMap)
    data_array_3857 = data_array_with_crs.rio.reproject(PSEUDO_MERCATOR_SYSTEM)

    # Write OpenStreetMap data array to file
    data_array_3857.rio.to_raster(path)


def _preprocess_data_array(data_array: xr.DataArray) -> xr.DataArray:
    """Preprocess the data array before converting to GeoTIFF."""
    data_array = data_array.squeeze()

    if data_array.ndim != 2:  # noqa: PLR2004
        raise InvalidShapeForGeoTIFFError(data_array.dims, data_array.shape)  # pyright: ignore[reportArgumentType]

    # Trim NaN values from the edges of the data array
    data_array = data_array.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

    # if float64, convert to float32 to reduce GeoTIFF size
    if data_array.dtype == np.float64:
        data_array = data_array.astype("float32")  # pyright: ignore[reportUnknownMemberType]

    return data_array
