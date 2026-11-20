import logging
import subprocess
from textwrap import wrap
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel
from ruff.__main__ import find_ruff_bin  # pyright: ignore[reportMissingTypeStubs]

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    Function,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnimationSpec,
    PlotGridSpec,
    PlotSpec,
)

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
    """Ensure that lat and lon dimensions are in ascending order.

    Uses structure-preserving operations instead of a full ``sortby``: a lazy strided reverse for a
    descending axis, and a roll for a rotated-ascending axis (e.g. longitude after a 0-360 ->
    -180-180 conversion, which yields two ascending runs that only need swapping). A general
    ``sortby`` triggers an expensive cross-chunk shuffle on large Dask-backed forecast arrays and
    can exhaust worker memory; the operations here are cheap and lazy. Falls back to ``sortby`` only
    for genuinely irregular coordinates.
    """
    for dim in ("lat", "lon"):
        if dim in da.dims:
            da = _ensure_ascending_along_dim(da, dim)
    return da


def _ensure_ascending_along_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Return ``da`` with ``dim`` in ascending order, avoiding a full sort where possible."""
    values = np.asarray(da[dim].values)  # type: ignore[reportUnknownMemberType]
    if values.size < 2:  # noqa: PLR2004
        return da

    diffs = np.diff(values)
    if bool(np.all(diffs > 0)):
        return da  # already ascending
    if bool(np.all(diffs < 0)):
        return da.isel({dim: slice(None, None, -1)})  # descending -> lazy reverse

    # A single descent means the axis is a rotated ascending sequence (e.g. longitude wrapped from
    # 0-360 to -180-180). Roll it back into ascending order without a full cross-chunk shuffle.
    if int(np.count_nonzero(diffs < 0)) == 1:
        split = int(np.argmin(values))
        return da.roll({dim: values.size - split}, roll_coords=True)  # type: ignore[reportUnknownMemberType]

    # Irregular coordinate: fall back to a general sort.
    return da.sortby(dim)  # pyright: ignore[reportUnknownMemberType]


def algorithm_to_pseudocode(algorithm: Algorithm[BaseFunctionCall]) -> str:
    """Convert an algorithm to its pseudocode representation."""
    blocks: list[str] = []
    for var_assignment in algorithm.statements:
        var_name = var_assignment.name
        value = _value_to_str(var_assignment.value)
        comment = var_assignment.description
        comment_lines = [f"# {ln}" for ln in wrap(comment, width=80)]

        lines: list[str] = []
        lines.extend(comment_lines)
        lines.append(f"{var_name} = {value}")
        block = "\n".join(lines)
        blocks.append(block)

    code = "\n\n".join(blocks)
    return _format_code(code)


def _base_model_to_str(val: BaseModel, ignore_keys: set[str] | None = None) -> str:
    ignore_keys = ignore_keys or set()
    args = {k: _value_to_str(v) for k, v in val if k not in ignore_keys}
    class_name = type(val).__name__
    class_name = class_name.split("[")[0]  # remove type params
    return f"{class_name}({', '.join(f'{k}={v}' for k, v in args.items())})"


def _value_to_str(val: Any) -> str:  # noqa: ANN401, PLR0911
    """Convert a function argument value to its string representation."""
    match val:
        case BaseFunctionCall():
            return _value_to_str(val.func)
        case Function():
            return _base_model_to_str(val, ignore_keys={"type", "func_name"})  # pyright: ignore[reportUnknownArgumentType]
        case VariableReference(name=name):
            return name
        case PlotSpec() | PlotGridSpec() | AnimationSpec():
            return _base_model_to_str(val, ignore_keys={"spec_name"})  # pyright: ignore[reportUnknownArgumentType]
        case BaseModel():
            return _base_model_to_str(val)
        case list():
            return f"[{', '.join(_value_to_str(v) for v in val)}]"  # pyright: ignore[reportUnknownVariableType]
        case tuple():
            return f"({', '.join(_value_to_str(v) for v in val)})"  # pyright: ignore[reportUnknownVariableType]
        case dict():
            return f"{{{', '.join(f'{k}: {_value_to_str(v)}' for k, v in val.items())}}}"  # pyright: ignore[reportUnknownVariableType]
        case _:
            return repr(val)


def _format_code(code: str) -> str:
    # Locate the embedded ruff binary executable
    ruff_bin = find_ruff_bin()
    result = subprocess.run(  # noqa: S603
        [ruff_bin, "format", "-", "--line-length", "80", "--config", "format.quote-style='double'"],
        input=code,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
