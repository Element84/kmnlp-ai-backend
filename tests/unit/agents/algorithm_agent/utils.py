# ruff: noqa: E501
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any, TypedDict

import numpy as np
import pandas as pd  # type: ignore[reportMissingStubs]
import rioxarray  # noqa: F401 # type: ignore[unused-ignore]
import xarray as xr

from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    GeometryVariable,
    OutputVariable,
    SpatioTemporalDataArrayVariable,
)
from llm_agent.model.dimension import Dimension


def _geometry_input(name: str, description: str) -> GeometryVariable:
    """A pre-resolved area-of-interest polygon, as the orchestration layer passes to this agent."""
    return GeometryVariable(name=name, description=description, geometry_type="polygon")


SST_DESCRIPTION = """\
The SST (Sea Surface Temperature) variable is a 3-dimensional array of data representing global sea surface temperatures. The dimensions of this array are:

Time: Covering a period from January 1, 2024, to December 31, 2024, with daily measurements.
Latitude: Ranging from approximately -90° (South Pole) to +90° (North Pole), with 360 evenly spaced points.
Longitude: Ranging from -180° to +180°, covering the entire globe with 720 evenly spaced points.

Each point in this 3D array represents the sea surface temperature at a specific location (defined by latitude and longitude) and time. The temperatures are measured in degrees Celsius, with valid values ranging from -20°C to +50°C.

This dataset provides a comprehensive, high-resolution (5km) global view of sea surface temperatures over time. It's designed for use in various oceanographic and climate studies, particularly for monitoring coral reef environments as indicated by its association with the NOAA Coral Reef Watch program.\
"""


class DimCoords(TypedDict):
    start: float | int | str
    end: float | int | str | None
    num: int
    freq: str | None


def make_test_data_array(
    *,
    name: str = "SomeVariable",
    dims: list[tuple[str, DimCoords]],
    val_range: tuple[float, float],
    random_seed: int = 42,
) -> xr.DataArray:
    """Create a DataArray filled with random values."""
    rng = np.random.default_rng(random_seed)
    coords: list[tuple[str, Any]] = []
    for dim, coords_info in dims:
        if dim == "time":
            assert isinstance(coords_info["start"], str)
            assert isinstance(coords_info["end"], str | None)
            assert isinstance(coords_info["freq"], str)
            c = pd.date_range(  # type: ignore[reportUnknownMemberType]
                start=coords_info["start"],
                end=coords_info["end"],
                periods=coords_info["num"],
                freq=coords_info["freq"],
            ).tolist()
        else:
            assert isinstance(coords_info["start"], float | int)
            assert isinstance(coords_info["end"], float | int)
            assert coords_info["freq"] is None
            c = np.linspace(coords_info["start"], coords_info["end"], coords_info["num"])
        coords.append((dim, c))  # type: ignore[reportUnknownArgumentType]

    data = rng.uniform(*val_range, size=[len(c) for _, c in coords])
    da = xr.DataArray(data, coords=coords, name=name)
    return da


@lru_cache(maxsize=10)
def make_test_data_array_spatiotemporal(
    *,
    start_date: str = "2024-01-01",
    num_time_steps: int = 10,
    lat_shape: int = 360,
    lon_shape: int = 720,
    random_seed: int = 42,
) -> xr.DataArray:
    """Create a (time, lat, lon) DataArray filled with random values."""
    da = make_test_data_array(
        name="analysed_sst",
        dims=[
            ("time", DimCoords(start=start_date, end=None, num=num_time_steps, freq="D")),
            ("lat", DimCoords(start=-90, end=90, num=lat_shape, freq=None)),
            ("lon", DimCoords(start=-180, end=180, num=lon_shape, freq=None)),
        ],
        val_range=(-20, 50),
        random_seed=random_seed,
    )

    return (
        da.rio.write_crs("EPSG:4326", inplace=True)
        .rio.set_spatial_dims(y_dim="lat", x_dim="lon", inplace=True)
        .rio.write_coordinate_system(inplace=True)
    )


test_input_vars = [
    SpatioTemporalDataArrayVariable(
        name="sst",
        dataset_name="NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
        variable_type="DataArray",
        description="Sea Surface Temperature (SST)",
        standard_name="sea_surface_temperature",
        dimensions=[
            Dimension(field_name="time", id="", cf_standard_name="time"),
            Dimension(field_name="lat", id="", cf_standard_name="latitude"),
            Dimension(field_name="lon", id="", cf_standard_name="longitude"),
        ],
        time_dim="time",
        units="degreeC",
        spatial_coverage="global",
        temporal_coverage=(
            datetime(year=2024, month=1, day=1, tzinfo=UTC),
            datetime(year=2025, month=1, day=1, tzinfo=UTC),
        ),
        temporal_resolution=timedelta(days=1),
    )
]


class AlgoTestParams(TypedDict):
    problem: AlgorithmProblemSpec
    dummy_solution: dict[str, Any]


test_params: dict[str, AlgoTestParams] = {
    "mean": {
        "problem": AlgorithmProblemSpec(
            statement="What is the mean SST value over the entire dataset?",
            inputs_provided=test_input_vars,
            outputs_requested=[
                OutputVariable(
                    name="sst_mean_scalar",
                    variable_type="scalar_float",
                    description="The mean value of the SST.",
                )
            ],
        ),
        "dummy_solution": {
            "analysis": "Lorem ipsum.",
            "pseudocode": "result = func(input)",
            "algorithm": {
                "statements": [
                    {
                        "type": "VariableAssignment",
                        "name": "sst_mean_scalar",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "ReduceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "op": "mean",
                            },
                        },
                        "description": "Mean SST value for the Black Sea on June 01, 2024 as a scalar",
                    },
                ],
                "output_variables": [{"type": "VariableReference", "name": "sst_mean_scalar"}],
            },
            "explanation": "Dolor sit amet.",
        },
    },
    "black_sea_mean": {
        "problem": AlgorithmProblemSpec(
            statement="What was the mean SST in the Black Sea on June 01, 2024.",
            inputs_provided=[
                *test_input_vars,
                _geometry_input("black_sea_polygon", "Black Sea"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="sst_mean_scalar",
                    variable_type="scalar_float",
                    description="The mean value of the SST.",
                )
            ],
        ),
        "dummy_solution": {
            "analysis": "__analysis__",
            "pseudocode": "result = func(input)",
            "algorithm": {
                "statements": [
                    {
                        "type": "VariableAssignment",
                        "name": "sst_filtered_time",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "SliceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "dim_slices": {"time": ["2024-06-01", "2024-06-01"]},
                            },
                        },
                        "description": "SST data filtered to June 01, 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_black_sea",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "FilterDataArrayByPolygon",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_filtered_time",
                                },
                                "polygon": {
                                    "type": "VariableReference",
                                    "name": "black_sea_polygon",
                                },
                            },
                        },
                        "description": "SST data for the Black Sea on June 01, 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_mean_scalar",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "ReduceDataArray_LatLonTime",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_black_sea",
                                },
                                "op": "mean",
                            },
                        },
                        "description": "Mean SST value for the Black Sea on June 01, 2024 as a scalar",
                    },
                ],
                "output_variables": [{"type": "VariableReference", "name": "sst_mean_scalar"}],
            },
            "explanation": "__explanation__",
        },
    },
    "time_series": {
        "problem": AlgorithmProblemSpec(
            statement="Show the daily average SST within the Chesapeake Bay "
            "over the the months Jan 2024 and March 2024 as two separate time "
            "series on the same plot. The x-axis should be agnostic of the "
            "year so that the two time series are on top of each other.",
            inputs_provided=[
                *test_input_vars,
                _geometry_input("chesapeake_bay", "Chesapeake Bay"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="time_series_plot",
                    variable_type="line_plot",
                    description="The time series plot.",
                )
            ],
        ),
        "dummy_solution": {
            "analysis": "__analysis__",
            "pseudocode": "result = func(input)",
            "algorithm": {
                "statements": [
                    {
                        "type": "VariableAssignment",
                        "name": "sst_jan_2024",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "SliceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "dim_slices": {"time": ["2024-01-01", "2024-01-31"]},
                            },
                        },
                        "description": "SST data for January 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_mar_2024",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "SliceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "dim_slices": {"time": ["2024-03-01", "2024-03-31"]},
                            },
                        },
                        "description": "SST data for March 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_jan_2024_chesapeake",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "FilterDataArrayByPolygon",
                                "data_array": {"type": "VariableReference", "name": "sst_jan_2024"},
                                "polygon": {"type": "VariableReference", "name": "chesapeake_bay"},
                            },
                        },
                        "description": "SST data for January 2024 in Chesapeake Bay",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_mar_2024_chesapeake",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "FilterDataArrayByPolygon",
                                "data_array": {"type": "VariableReference", "name": "sst_mar_2024"},
                                "polygon": {"type": "VariableReference", "name": "chesapeake_bay"},
                            },
                        },
                        "description": "SST data for March 2024 in Chesapeake Bay",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_jan_2024_chesapeake_daily_avg",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "GroupByTimeAndReduceDataArray",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_jan_2024_chesapeake",
                                },
                                "op": "mean",
                                "time_unit": "dayofmonth",
                            },
                        },
                        "description": "Daily average SST for March 2024 in Chesapeake Bay",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_mar_2024_chesapeake_daily_avg",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "GroupByTimeAndReduceDataArray",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_mar_2024_chesapeake",
                                },
                                "op": "mean",
                                "time_unit": "dayofmonth",
                            },
                        },
                        "description": "Daily average SST for March 2024 in Chesapeake Bay",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "time_series_plot",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "MakePlot",
                                "plot_spec": {
                                    "spec_name": "MultiLinePlotSpec",
                                    "specs": [
                                        {
                                            "spec_name": "LinePlotSpec",
                                            "data_array": {
                                                "type": "VariableReference",
                                                "name": "sst_jan_2024_chesapeake_daily_avg",
                                            },
                                        },
                                        {
                                            "spec_name": "LinePlotSpec",
                                            "data_array": {
                                                "type": "VariableReference",
                                                "name": "sst_mar_2024_chesapeake_daily_avg",
                                            },
                                        },
                                    ],
                                    "annotation": {
                                        "title": "Daily Average SST in Chesapeake Bay: Mar 2024 vs Mar 2024",
                                        "xlabel": "Day of Month",
                                        "ylabel": "Daily Average SST (°C)",
                                    },
                                },
                            },
                        },
                        "description": "Time series plot of daily average SST in Chesapeake Bay for Mar 2024 and Mar 2024",
                    },
                ],
                "output_variables": [{"type": "VariableReference", "name": "time_series_plot"}],
            },
            "explanation": "__explanation__",
        },
    },
    "bar_chart": {
        "problem": AlgorithmProblemSpec(
            statement="Draw a bar chart comparing the max SST's in "
            "Chesapeake Bay, Delaware Bay, and San Pablo Bay in the month of "
            "June 2024.",
            inputs_provided=[
                *test_input_vars,
                _geometry_input("chesapeake_bay", "Chesapeake Bay"),
                _geometry_input("delaware_bay", "Delaware Bay"),
                _geometry_input("san_pablo_bay", "San Pablo Bay"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="bar_plot",
                    variable_type="bar_plot",
                    description="The bar plot.",
                )
            ],
        ),
        "dummy_solution": {
            "analysis": "__analysis__",
            "pseudocode": "result = func(input)",
            "algorithm": {
                "statements": [
                    {
                        "type": "VariableAssignment",
                        "name": "sst_june_2024",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "SliceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "dim_slices": {"time": ["2024-06-01", "2024-06-30"]},
                            },
                        },
                        "description": "SST data filtered for June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "chesapeake_sst",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "FilterDataArrayByPolygon",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_june_2024",
                                },
                                "polygon": {"type": "VariableReference", "name": "chesapeake_bay"},
                            },
                        },
                        "description": "SST data for Chesapeake Bay in June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "delaware_sst",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "FilterDataArrayByPolygon",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_june_2024",
                                },
                                "polygon": {"type": "VariableReference", "name": "delaware_bay"},
                            },
                        },
                        "description": "SST data for Delaware Bay in June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "san_pablo_sst",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "FilterDataArrayByPolygon",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "sst_june_2024",
                                },
                                "polygon": {"type": "VariableReference", "name": "san_pablo_bay"},
                            },
                        },
                        "description": "SST data for San Pablo Bay in June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "chesapeake_max_sst",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "ReduceDataArray_LatLonTime",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "chesapeake_sst",
                                },
                                "op": "max",
                            },
                        },
                        "description": "Maximum SST for Chesapeake Bay in June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "delaware_max_sst",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "ReduceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "delaware_sst"},
                                "op": "max",
                            },
                        },
                        "description": "Maximum SST for Delaware Bay in June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "san_pablo_max_sst",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "ReduceDataArray_LatLonTime",
                                "data_array": {
                                    "type": "VariableReference",
                                    "name": "san_pablo_sst",
                                },
                                "op": "max",
                            },
                        },
                        "description": "Maximum SST for San Pablo Bay in June 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "bar_plot",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "MakePlot",
                                "plot_spec": {
                                    "spec_name": "BarPlotSpec",
                                    "labels": ["Chesapeake Bay", "Delaware Bay", "San Pablo Bay"],
                                    "heights": [
                                        {
                                            "type": "VariableReference",
                                            "name": "chesapeake_max_sst",
                                        },
                                        {
                                            "type": "VariableReference",
                                            "name": "delaware_max_sst",
                                        },
                                        {
                                            "type": "VariableReference",
                                            "name": "san_pablo_max_sst",
                                        },
                                    ],
                                    "annotation": {
                                        "title": "Maximum SST in June 2024 for Selected Bays"
                                    },
                                },
                            },
                        },
                        "description": "Bar chart comparing max SSTs in Chesapeake Bay, Delaware Bay, and San Pablo Bay in June 2024",
                    },
                ],
                "output_variables": [{"type": "VariableReference", "name": "bar_plot"}],
            },
            "explanation": "__explanation__",
        },
    },
    "raster": {
        "problem": AlgorithmProblemSpec(
            statement="Plot the SST maps for Jan 1, 2024 and Aug 1, 2024 side by side.",
            inputs_provided=test_input_vars,
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="plot_grid",
                    description="The requested plot.",
                )
            ],
        ),
        "dummy_solution": {
            "analysis": "__analysis__",
            "pseudocode": "result = func(input)",
            "algorithm": {
                "statements": [
                    {
                        "type": "VariableAssignment",
                        "name": "sst_jan_2024",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "SliceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "dim_slices": {"time": ["2024-01-01", "2024-01-01"]},
                            },
                        },
                        "description": "SST data for January 1, 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "sst_aug_2024",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "SliceDataArray_LatLonTime",
                                "data_array": {"type": "VariableReference", "name": "sst"},
                                "dim_slices": {"time": ["2024-08-01", "2024-08-01"]},
                            },
                        },
                        "description": "SST data for August 1, 2024",
                    },
                    {
                        "type": "VariableAssignment",
                        "name": "plot",
                        "value": {
                            "type": "FunctionCall",
                            "func": {
                                "type": "Function",
                                "func_name": "MakePlot",
                                "plot_spec": {
                                    "spec_name": "PlotGridSpec",
                                    "ncols": 2,
                                    "subplot_specs": [
                                        {
                                            "spec_name": "SSTPlotSpec",
                                            "sst_data_array": {
                                                "type": "VariableReference",
                                                "name": "sst_jan_2024",
                                            },
                                        },
                                        {
                                            "spec_name": "SSTPlotSpec",
                                            "sst_data_array": {
                                                "type": "VariableReference",
                                                "name": "sst_aug_2024",
                                            },
                                        },
                                    ],
                                },
                            },
                        },
                        "description": "SST for Jan",
                    },
                ],
                "output_variables": [{"type": "VariableReference", "name": "plot"}],
            },
            "explanation": "__explanation__",
        },
    },
}
