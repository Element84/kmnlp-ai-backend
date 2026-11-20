from typing import TYPE_CHECKING

import xarray as xr
from shapely import Polygon

from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    mask_data_array_with_polygon,
)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def test_mask_sst_data_array_with_black_sea_polygon(
    sst_data_array: xr.DataArray, black_sea_geometry: "BaseGeometry"
):
    data_array = sst_data_array

    assert isinstance(black_sea_geometry, Polygon)

    masked_data_array = mask_data_array_with_polygon(data_array, black_sea_geometry)

    assert masked_data_array.sizes["lon"] > 0
    assert masked_data_array.sizes["lat"] > 0


def test_mask_sst_data_array_with_gulf_of_mexico_polygon(
    sst_data_array: xr.DataArray, gulf_of_mexico_geometry: "BaseGeometry"
):
    data_array = sst_data_array

    assert isinstance(gulf_of_mexico_geometry, Polygon)

    masked_data_array = mask_data_array_with_polygon(data_array, gulf_of_mexico_geometry)

    assert masked_data_array.sizes["lon"] > 0
    assert masked_data_array.sizes["lat"] > 0


def test_mask_wind_speed_data_array_with_black_sea_polygon(
    wind_speed_data_array: xr.DataArray, black_sea_geometry: "BaseGeometry"
):
    data_array = wind_speed_data_array

    assert isinstance(black_sea_geometry, Polygon)

    masked_data_array = mask_data_array_with_polygon(data_array, black_sea_geometry)

    assert masked_data_array.sizes["lon"] > 0
    assert masked_data_array.sizes["lat"] > 0


def test_mask_wind_speed_data_array_with_gulf_of_mexico_polygon(
    wind_speed_data_array: xr.DataArray, gulf_of_mexico_geometry: "BaseGeometry"
):
    data_array = wind_speed_data_array

    assert isinstance(gulf_of_mexico_geometry, Polygon)

    masked_data_array = mask_data_array_with_polygon(data_array, gulf_of_mexico_geometry)

    assert masked_data_array.sizes["lon"] > 0
    assert masked_data_array.sizes["lat"] > 0
