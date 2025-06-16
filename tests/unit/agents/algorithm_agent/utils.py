from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import TypedDict

import numpy as np
import pandas as pd  # type: ignore[reportMissingStubs]
import xarray as xr

from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    OutputVariable,
    SpatioTemporalDataArrayVariable,
)

SST_DESCRIPTION = """\
The SST (Sea Surface Temperature) variable is a 3-dimensional array of data representing global sea surface temperatures. The dimensions of this array are:

Time: Covering a period from January 1, 2024, to December 31, 2024, with daily measurements.
Latitude: Ranging from approximately -90° (South Pole) to +90° (North Pole), with 360 evenly spaced points.
Longitude: Ranging from -180° to +180°, covering the entire globe with 720 evenly spaced points.

Each point in this 3D array represents the sea surface temperature at a specific location (defined by latitude and longitude) and time. The temperatures are measured in degrees Celsius, with valid values ranging from -20°C to +50°C.

This dataset provides a comprehensive, high-resolution (5km) global view of sea surface temperatures over time. It's designed for use in various oceanographic and climate studies, particularly for monitoring coral reef environments as indicated by its association with the NOAA Coral Reef Watch program.\
"""  # noqa: E501


@lru_cache(maxsize=10)
def make_dummy_data_array(
    *,
    start_date: str = "2024-01-01",
    num_time_steps: int = 10,
    lat_shape: int = 360,
    lon_shape: int = 720,
    random_seed: int = 42,
) -> xr.DataArray:
    """Create a (time, lat, lon) DataArray filled with random values."""
    rng = np.random.default_rng(random_seed)
    time = pd.date_range(  # type: ignore[reportUnknownMemberType]
        start=start_date, periods=num_time_steps, freq="D"
    ).tolist()
    lat = np.linspace(-90, 90, lat_shape)
    lon = np.linspace(-180, 180, lon_shape)
    data = rng.uniform(-20, 50, size=(num_time_steps, lat_shape, lon_shape))

    da = xr.DataArray(
        data,
        coords=[("time", time), ("lat", lat), ("lon", lon)],
        name="analysed_sst",
    )
    return da


test_input_vars = [
    SpatioTemporalDataArrayVariable(
        name="sst",
        variable_type="DataArray",
        description="Sea Surface Temperature (SST)",
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
    dummy_solution: str


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
        "dummy_solution": """{
  "thoughts": "Lorem ipsum.",
  "pseudocode": "result = func(input)",

  "algorithm": {
    "statements": [
      {
        "name": "sst_mean_dataarray",
        "value": {
          "func": {
            "func_name": "ReduceDataArray",
            "data_array": {"name": "sst"},
            "op": "mean"
          }
        },
        "description": "Mean SST value for the Black Sea on June 01, 2024 as a DataArray"
      },
      {
        "name": "sst_mean_scalar",
        "value": {
          "func": {
            "func_name": "ZeroDimDataArrayToScalar",
            "data_array": {"name": "sst_mean_dataarray"}
          }
        },
        "description": "Mean SST value for the Black Sea on June 01, 2024 as a scalar"
      }
    ],
    "output_variables": [
      {"name": "sst_mean_scalar"}
    ]
  },

  "explanation": "Dolor sit amet."
}
""",  # noqa: E501
    },
    "black_sea_mean": {
        "problem": AlgorithmProblemSpec(
            statement="What was the mean SST in the Black Sea on "
            "June 01, 2024.",
            inputs_provided=test_input_vars,
            outputs_requested=[
                OutputVariable(
                    name="sst_mean_scalar",
                    variable_type="scalar_float",
                    description="The mean value of the SST.",
                )
            ],
        ),
        "dummy_solution": """{
  "thoughts": "__thoughts__",
  "pseudocode": "result = func(input)",
  "algorithm": {
    "statements": [
      {
        "name": "black_sea_polygon",
        "value": {
          "func": {
            "func_name": "GeocodeNaturalLanguageDescription",
            "text": "Black Sea"
          }
        },
        "description": "Polygon representing the Black Sea area"
      },
      {
        "name": "sst_filtered_time",
        "value": {
          "func": {
            "func_name": "FilterDataArray",
            "data_array": {"name": "sst"},
            "time_range": ["2024-06-01", "2024-06-01"]
          }
        },
        "description": "SST data filtered to June 01, 2024"
      },
      {
        "name": "sst_black_sea",
        "value": {
          "func": {
            "func_name": "FilterDataArrayByPolygon",
            "data_array": {"name": "sst_filtered_time"},
            "polygon": {"name": "black_sea_polygon"}
          }
        },
        "description": "SST data for the Black Sea on June 01, 2024"
      },
      {
        "name": "sst_mean_dataarray",
        "value": {
          "func": {
            "func_name": "ReduceDataArray",
            "data_array": {"name": "sst_black_sea"},
            "op": "mean"
          }
        },
        "description": "Mean SST value for the Black Sea on June 01, 2024 as a DataArray"
      },
      {
        "name": "sst_mean_scalar",
        "value": {
          "func": {
            "func_name": "ZeroDimDataArrayToScalar",
            "data_array": {"name": "sst_mean_dataarray"}
          }
        },
        "description": "Mean SST value for the Black Sea on June 01, 2024 as a scalar"
      }
    ],
    "output_variables": [
      {"name": "sst_mean_scalar"}
    ]
  },
  "explanation": "__explanation__"
}""",  # noqa: E501
    },
    "time_series": {
        "problem": AlgorithmProblemSpec(
            statement="Show the daily average SST within the Chesapeake Bay "
            "over the the months Jan 2024 and Feb 2024 as two separate time "
            "series on the same plot. The x-axis should be agnostic of the "
            "year so that the two time series are on top of each other.",
            inputs_provided=test_input_vars,
            outputs_requested=[
                OutputVariable(
                    name="time_series_plot",
                    variable_type="line_plot",
                    description="The time series plot.",
                )
            ],
        ),
        "dummy_solution": """{
  "thoughts": "__thoughts__",
  "pseudocode": "result = func(input)",
  "algorithm": {
    "statements": [
      {
        "name": "chesapeake_bay",
        "value": {
          "func": {
            "func_name": "GeocodeNaturalLanguageDescription",
            "text": "Chesapeake Bay"
          }
        },
        "description": "Polygon representing the Chesapeake Bay area"
      },
      {
        "name": "sst_jan_2024",
        "value": {
          "func": {
            "func_name": "FilterDataArray",
            "data_array": {"name": "sst"},
            "time_range": ["2024-01-01", "2024-01-31"]
          }
        },
        "description": "SST data for January 2024"
      },
      {
        "name": "sst_feb_2024",
        "value": {
          "func": {
            "func_name": "FilterDataArray",
            "data_array": {"name": "sst"},
            "time_range": ["2024-02-01", "2024-02-29"]
          }
        },
        "description": "SST data for February 2024"
      },
      {
        "name": "sst_jan_2024_chesapeake",
        "value": {
          "func": {
            "func_name": "FilterDataArrayByPolygon",
            "data_array": {"name": "sst_jan_2024"},
            "polygon": {"name": "chesapeake_bay"}
          }
        },
        "description": "SST data for January 2024 in Chesapeake Bay"
      },
      {
        "name": "sst_feb_2024_chesapeake",
        "value": {
          "func": {
            "func_name": "FilterDataArrayByPolygon",
            "data_array": {"name": "sst_feb_2024"},
            "polygon": {"name": "chesapeake_bay"}
          }
        },
        "description": "SST data for February 2024 in Chesapeake Bay"
      },
      {
        "name": "sst_jan_2024_chesapeake_daily_avg",
        "value": {
          "func": {
            "func_name": "GroupByTimeAndReduceDataArray",
            "data_array": {"name": "sst_jan_2024_chesapeake"},
            "op": "mean",
            "time_unit": "dayofmonth"
          }
        },
        "description": "Daily average SST for January 2024 in Chesapeake Bay"
      },
      {
        "name": "sst_feb_2024_chesapeake_daily_avg",
        "value": {
          "func": {
            "func_name": "GroupByTimeAndReduceDataArray",
            "data_array": {"name": "sst_feb_2024_chesapeake"},
            "op": "mean",
            "time_unit": "dayofmonth"
          }
        },
        "description": "Daily average SST for February 2024 in Chesapeake Bay"
      },
      {
        "name": "time_series_plot",
        "value": {
          "func": {
            "func_name": "PlotDataArraysAsTimeSeries",
            "data_arrays": [
              {"name": "sst_jan_2024_chesapeake_daily_avg"},
              {"name": "sst_feb_2024_chesapeake_daily_avg"}
            ],
            "labels": [
                "sst_jan_2024_chesapeake_daily_avg",
                "sst_feb_2024_chesapeake_daily_avg"
            ],
            "title": "Daily Average SST in Chesapeake Bay: Jan 2024 vs Feb 2024"
          }
        },
        "description": "Time series plot of daily average SST in Chesapeake Bay for Jan 2024 and Feb 2024"
      }
    ],
    "output_variables": [
      {"name": "time_series_plot"}
    ]
  },
  "explanation": "__explanation__"
}""",  # noqa: E501
    },
    "bar_chart": {
        "problem": AlgorithmProblemSpec(
            statement="Draw a bar chart comparing the max SST's in "
            "Chesapeake Bay, Delaware Bay, and San Pablo Bay in the month of "
            "June 2024.",
            inputs_provided=test_input_vars,
            outputs_requested=[
                OutputVariable(
                    name="bar_plot",
                    variable_type="bar_plot",
                    description="The bar plot.",
                )
            ],
        ),
        "dummy_solution": """{
  "thoughts": "__thoughts__",
  "pseudocode": "result = func(input)",
  "algorithm": {
    "statements": [
      {
        "name": "chesapeake_bay",
        "value": {
          "func": {
            "func_name": "GeocodeNaturalLanguageDescription",
            "text": "Chesapeake Bay"
          }
        },
        "description": "Polygon representing Chesapeake Bay"
      },
      {
        "name": "delaware_bay",
        "value": {
          "func": {
            "func_name": "GeocodeNaturalLanguageDescription",
            "text": "Delaware Bay"
          }
        },
        "description": "Polygon representing Delaware Bay"
      },
      {
        "name": "san_pablo_bay",
        "value": {
          "func": {
            "func_name": "GeocodeNaturalLanguageDescription",
            "text": "San Pablo Bay"
          }
        },
        "description": "Polygon representing San Pablo Bay"
      },
      {
        "name": "sst_june_2024",
        "value": {
          "func": {
            "func_name": "FilterDataArray",
            "data_array": {"name": "sst"},
            "time_range": ["2024-06-01", "2024-06-30"]
          }
        },
        "description": "SST data filtered for June 2024"
      },
      {
        "name": "chesapeake_sst",
        "value": {
          "func": {
            "func_name": "FilterDataArrayByPolygon",
            "data_array": {"name": "sst_june_2024"},
            "polygon": {"name": "chesapeake_bay"}
          }
        },
        "description": "SST data for Chesapeake Bay in June 2024"
      },
      {
        "name": "delaware_sst",
        "value": {
          "func": {
            "func_name": "FilterDataArrayByPolygon",
            "data_array": {"name": "sst_june_2024"},
            "polygon": {"name": "delaware_bay"}
          }
        },
        "description": "SST data for Delaware Bay in June 2024"
      },
      {
        "name": "san_pablo_sst",
        "value": {
          "func": {
            "func_name": "FilterDataArrayByPolygon",
            "data_array": {"name": "sst_june_2024"},
            "polygon": {"name": "san_pablo_bay"}
          }
        },
        "description": "SST data for San Pablo Bay in June 2024"
      },
      {
        "name": "chesapeake_max_sst",
        "value": {
          "func": {
            "func_name": "ReduceDataArray",
            "data_array": {"name": "chesapeake_sst"},
            "op": "max"
          }
        },
        "description": "Maximum SST for Chesapeake Bay in June 2024"
      },
      {
        "name": "delaware_max_sst",
        "value": {
          "func": {
            "func_name": "ReduceDataArray",
            "data_array": {"name": "delaware_sst"},
            "op": "max"
          }
        },
        "description": "Maximum SST for Delaware Bay in June 2024"
      },
      {
        "name": "san_pablo_max_sst",
        "value": {
          "func": {
            "func_name": "ReduceDataArray",
            "data_array": {"name": "san_pablo_sst"},
            "op": "max"
          }
        },
        "description": "Maximum SST for San Pablo Bay in June 2024"
      },
      {
        "name": "bar_plot",
        "value": {
          "func": {
            "func_name": "PlotBarChart",
            "labels": ["Chesapeake Bay", "Delaware Bay", "San Pablo Bay"],
            "heights": [
              {"name": "chesapeake_max_sst"},
              {"name": "delaware_max_sst"},
              {"name": "san_pablo_max_sst"}
            ],
            "title": "Maximum SST in June 2024 for Selected Bays"
          }
        },
        "description": "Bar chart comparing max SSTs in Chesapeake Bay, Delaware Bay, and San Pablo Bay in June 2024"
      }
    ],
    "output_variables": [
      {
        "name": "bar_plot"
      }
    ]
  },
  "explanation": "__explanation__"
}""",  # noqa: E501
    },
    "raster": {
        "problem": AlgorithmProblemSpec(
            statement="Plot the SST maps for Jan 1, 2024 and Aug 1, 2024 side "
            "by side.",
            inputs_provided=test_input_vars,
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="raster_plot",
                    description="The requested plot.",
                )
            ],
        ),
        "dummy_solution": """{
  "thoughts": "__thoughts__",
  "pseudocode": "result = func(input)",
  "algorithm": {
    "statements": [
      {
        "name": "sst_jan_2024",
        "value": {
          "func": {
            "func_name": "FilterDataArray",
            "data_array": {"name": "sst"},
            "time_range": ["2024-01-01", "2024-01-01"]
          }
        },
        "description": "SST data for January 1, 2024"
      },
      {
        "name": "sst_aug_2024",
        "value": {
          "func": {
            "func_name": "FilterDataArray",
            "data_array": {"name": "sst"},
            "time_range": ["2024-08-01", "2024-08-01"]
          }
        },
        "description": "SST data for August 1, 2024"
      },
      {
        "name": "plot",
        "value": {
          "func": {
            "func_name": "PlotDataArraysAsRaster",
            "data_arrays": [
              {"name": "sst_jan_2024"},
              {"name": "sst_aug_2024"}
            ],
            "subplot_grid_shape": [1, 2],
            "title": "SST Maps for Jan 1, 2024 and Aug 1, 2024"
          }
        },
        "description": "Side-by-side plot of SST maps"
      }
    ],
    "output_variables": [
      {"name": "plot"}
    ]
  },
  "explanation": "__explanation__"
}""",
    },
}
