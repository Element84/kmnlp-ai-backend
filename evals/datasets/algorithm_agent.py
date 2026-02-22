# Ignore "line too long" errors.
# ruff: noqa: E501

from datetime import UTC, datetime, timedelta
from typing import TypedDict, cast

import llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions as da_ops
import llm_agent.agents.algorithm_agent.algorithm.function_sets.general_plot_functions as plot_ops
from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    Function,
    VariableAssignment,
    VariableReference,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    FunctionSet,
)
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    OutputVariable,
    SpatioTemporalDataArrayVariable,
    TemporalDataArrayVariable,
    input_vars_to_function_set,
)
from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvalInput
from llm_agent.evaluation.base_eval import EvalDataset
from llm_agent.model.dimension import Dimension

dims = {
    "time": Dimension(id="time", field_name="time", cf_standard_name="time"),
    "lat": Dimension(id="lat", field_name="lat", cf_standard_name="latitude"),
    "lon": Dimension(id="lon", field_name="lon", cf_standard_name="longitude"),
    "wavelength": Dimension(id="wavelength", field_name="wavelength"),
}

_input_vars = {
    "sst": SpatioTemporalDataArrayVariable(
        name="analysed_sst",
        variable_type="DataArray",
        description="Sea Surface Temperature (SST)",
        dimensions=[dims["time"], dims["lat"], dims["lon"]],
        time_dim="time",
        units="degreeC",
        spatial_coverage="global",
        temporal_coverage=(
            datetime(year=2020, month=1, day=1, tzinfo=UTC),
            datetime(year=2024, month=11, day=30, tzinfo=UTC),
        ),
        temporal_resolution=timedelta(days=1),
    ),
    "air_temperature": SpatioTemporalDataArrayVariable(
        name="air_temperature",
        variable_type="DataArray",
        description="NOAA Climate Data Record of Ocean Near-Surface Air Temperature",
        dimensions=[dims["time"], dims["lat"], dims["lon"]],
        time_dim="time",
        units="degreeC",
        spatial_coverage="global",
        temporal_coverage=(
            datetime(year=1988, month=1, day=1, tzinfo=UTC),
            datetime(year=2021, month=8, day=31, tzinfo=UTC),
        ),
        temporal_resolution=timedelta(hours=3),
    ),
    "wind_speed": SpatioTemporalDataArrayVariable(
        name="wind_speed",
        variable_type="DataArray",
        description="NOAA Climate Data Record of Ocean Near-Surface Wind Speed",
        dimensions=[dims["time"], dims["lat"], dims["lon"]],
        time_dim="time",
        units="m s-1",
        spatial_coverage="global",
        temporal_coverage=(
            datetime(year=1988, month=1, day=1, tzinfo=UTC),
            datetime(year=2021, month=8, day=31, tzinfo=UTC),
        ),
        temporal_resolution=timedelta(hours=3),
    ),
    "specific_humidity": SpatioTemporalDataArrayVariable(
        name="specific_humidity",
        variable_type="DataArray",
        description="NOAA Climate Data Record of Ocean Near-Surface Specific Humidity",
        dimensions=[dims["time"], dims["lat"], dims["lon"]],
        time_dim="time",
        units="g kg-1",
        spatial_coverage="global",
        temporal_coverage=(
            datetime(year=1988, month=1, day=1, tzinfo=UTC),
            datetime(year=2021, month=8, day=31, tzinfo=UTC),
        ),
        temporal_resolution=timedelta(hours=3),
    ),
    "SSI": TemporalDataArrayVariable(
        name="SSI",
        variable_type="DataArray",
        description="NOAA Climate Data Record of Yearly Solar Spectral Irradiance (W m-2 nm-1)",
        dimensions=[dims["time"], dims["wavelength"]],
        time_dim="time",
        units="W m-2 nm-1",
        temporal_coverage=(
            datetime(year=1610, month=1, day=1, tzinfo=UTC),
            datetime(year=2023, month=12, day=31, tzinfo=UTC),
        ),
        temporal_resolution=None,
    ),
    "TSI": TemporalDataArrayVariable(
        name="TSI",
        variable_type="DataArray",
        description="NOAA Climate Data Record of Yearly Total Solar Irradiance (W m-2)",
        dimensions=[dims["time"]],
        time_dim="time",
        units="W m-2",
        temporal_coverage=(
            datetime(year=1610, month=1, day=1, tzinfo=UTC),
            datetime(year=2024, month=12, day=31, tzinfo=UTC),
        ),
        temporal_resolution=None,
    ),
}


_func_sets: dict[str, FunctionSet] = {
    "st": input_vars_to_function_set([_input_vars["sst"]]),
    "time_wvl": input_vars_to_function_set([_input_vars["SSI"]]),
    "time_only": input_vars_to_function_set([_input_vars["TSI"]]),
}


_FunctionCall: dict[str, type[BaseFunctionCall]] = {
    "st": make_function_call_model(_func_sets["st"].functions),
    "time_wvl": make_function_call_model(_func_sets["time_wvl"].functions),
    "time_only": make_function_call_model(_func_sets["time_only"].functions),
}


class _ConstrainedFunc[T: Function](TypedDict):
    func: type[T]
    func_name: str


_SliceDataArray: dict[str, _ConstrainedFunc[da_ops.SliceDataArray]] = {
    "st": {
        "func": cast(
            "type[da_ops.SliceDataArray]",
            _func_sets["st"]["SliceDataArray_LatLonTime"],
        ),
        "func_name": "SliceDataArray_LatLonTime",
    },
    "time_wvl": {
        "func": cast(
            "type[da_ops.SliceDataArray]",
            _func_sets["time_wvl"]["SliceDataArray_TimeWavelength"],
        ),
        "func_name": "SliceDataArray_TimeWavelength",
    },
    "time_only": {
        "func": cast(
            "type[da_ops.SliceDataArray]",
            _func_sets["time_only"]["SliceDataArray_Time"],
        ),
        "func_name": "SliceDataArray_Time",
    },
}

_ReduceDataArray: dict[str, _ConstrainedFunc[da_ops.ReduceDataArray]] = {
    "st": {
        "func": cast(
            "type[da_ops.ReduceDataArray]",
            _func_sets["st"]["ReduceDataArray_LatLonTime"],
        ),
        "func_name": "ReduceDataArray_LatLonTime",
    },
    "time_wvl": {
        "func": cast(
            "type[da_ops.ReduceDataArray]",
            _func_sets["time_wvl"]["ReduceDataArray_TimeWavelength"],
        ),
        "func_name": "ReduceDataArray_TimeWavelength",
    },
    "time_only": {
        "func": cast(
            "type[da_ops.ReduceDataArray]",
            _func_sets["time_only"]["ReduceDataArray_Time"],
        ),
        "func_name": "ReduceDataArray_Time",
    },
}


_dataset = [
    AlgorithmAgentEvalInput(
        id="data_query_mismatch",
        problem=AlgorithmProblemSpec(
            statement="What was the average humidity over Philadelphia on Jan 01, 2021?",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="avg_humidity",
                    variable_type="scalar_float",
                    description="Average humidity value.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[],
                output_variables=[],
            ),
            explanation="",
            error_message="Cannot be answered from the available data.",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="query_out_of_scope",
        problem=AlgorithmProblemSpec(
            statement="Create an animation of the daily SST maps in the Black Sea over the month of June 2024.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="raster_animation",
                    variable_type="raster_plot",
                    description="An animated raster plot of the daily SST in the Black Sea over the month of June 2024.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[],
                output_variables=[],
            ),
            explanation="",
            error_message="Cannot create animated plots.",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="mean_basic",
        problem=AlgorithmProblemSpec(
            statement="What is the mean SST value over the entire dataset?",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="sst_mean_scalar",
                    variable_type="scalar_float",
                    description="The mean value of the SST.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                op="mean",
                            )
                        ),
                        description="Mean value as DataArray",
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_mean_scalar",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_mean_dataarray"),
                            )
                        ),
                        description="Mean value as scalar",
                    ),
                ],
                output_variables=[VariableReference(name="sst_mean_scalar")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="mean_black_sea",
        problem=AlgorithmProblemSpec(
            statement="What was the mean SST in the Black Sea on June 01, 2024?",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="sst_value",
                    variable_type="scalar_float",
                    description="The mean value in the Black Sea on June 01, 2024 the SST.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="black_sea_polygon",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Black Sea",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="analysed_sst"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea"),
                                dim_slices={"time": ("2024-06-01", "2024-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_value",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_mean_dataarray"),
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="sst_value")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="raster_plot",
        problem=AlgorithmProblemSpec(
            statement="Plot the SST maps for Jan 1, 2023 and Aug 1, 2023 stacked vertically.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="raster_plot",
                    description="SST maps for Jan 1, 2023 and Aug 1, 2023 stacked vertically.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_jan1_2023",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2023-01-01", "2023-01-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_aug1_2023",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2023-08-01", "2023-08-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsRaster(
                                func_name="PlotDataArraysAsRaster",
                                data_arrays=[
                                    VariableReference(name="sst_jan1_2023"),
                                    VariableReference(name="sst_aug1_2023"),
                                ],
                                subplot_grid_shape=(2, 1),
                                title="SST maps for Jan 1, 2023 and Aug 1, 2023",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="time_series_1",
        problem=AlgorithmProblemSpec(
            statement="Show the daily average SST within the Chesapeake Bay over the year 2024 as a time series.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="line_plot",
                    description="Daily average SST within the Chesapeake Bay over the year 2024.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2024-01-01", "2024-12-31")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Chesapeake Bay",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_chesapeake_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="daily_mean_sst",
                        value=_FunctionCall["st"](
                            func=da_ops.GroupByTimeAndReduceDataArray(
                                func_name="GroupByTimeAndReduceDataArray",
                                data_array=VariableReference(name="sst_chesapeake_bay"),
                                op="mean",
                                time_unit="dayofyear",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsTimeSeries(
                                func_name="PlotDataArraysAsTimeSeries",
                                data_arrays=[
                                    VariableReference(name="daily_mean_sst"),
                                ],
                                labels=["Chesapeake Bay"],
                                title="Daily mean SST in 2024",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="time_series_2",
        problem=AlgorithmProblemSpec(
            statement="Show the monthly average SST within the Chesapeake Bay over the year 2024 as a time series.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="line_plot",
                    description="Monthly average SST within the Chesapeake Bay over the year 2024.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2024-01-01", "2024-12-31")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Chesapeake Bay",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_chesapeake_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="daily_mean_sst",
                        value=_FunctionCall["st"](
                            func=da_ops.GroupByTimeAndReduceDataArray(
                                func_name="GroupByTimeAndReduceDataArray",
                                data_array=VariableReference(name="sst_chesapeake_bay"),
                                op="mean",
                                time_unit="month",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsTimeSeries(
                                func_name="PlotDataArraysAsTimeSeries",
                                data_arrays=[
                                    VariableReference(name="daily_mean_sst"),
                                ],
                                labels=["Chesapeake Bay"],
                                title="Daily mean SST in 2024",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="time_series_3",
        problem=AlgorithmProblemSpec(
            statement="Show the daily average air temperature in the Chesapeake Bay over the month of June 2020 as a time series.",
            inputs_provided=[_input_vars["air_temperature"]],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="line_plot",
                    description="Daily average SST.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="air_temperature"),
                                dim_slices={"time": ("2020-01-01", "2020-06-30")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Chesapeake Bay",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="air_temperature_filtered"),
                                polygon=VariableReference(name="polygon_chesapeake_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="daily_mean_air_temperature",
                        value=_FunctionCall["st"](
                            func=da_ops.GroupByTimeAndReduceDataArray(
                                func_name="GroupByTimeAndReduceDataArray",
                                data_array=VariableReference(name="air_temperature_chesapeake_bay"),
                                op="mean",
                                time_unit="dayofmonth",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsTimeSeries(
                                func_name="PlotDataArraysAsTimeSeries",
                                data_arrays=[
                                    VariableReference(name="daily_mean_air_temperature"),
                                ],
                                labels=["Chesapeake Bay"],
                                title="Daily mean air temperature in June 2020",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="bar_plot",
        problem=AlgorithmProblemSpec(
            statement="Draw a bar chart comparing the max SST's in Chesapeake Bay, Delaware Bay, and San Pablo Bay in the month of June 2024.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="bar_chart",
                    variable_type="bar_plot",
                    description="Bar chart comparing the max SST's in Chesapeake Bay, Delaware Bay, and San Pablo Bay in the month of June 2024.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2024-06-01", "2024-06-30")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Chesapeake Bay",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_delaware_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Delaware Bay",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_san_pablo_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="San Pablo Bay",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_chesapeake_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_delaware_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_delaware_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_san_pablo_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_san_pablo_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_chesapeake_bay_max_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_chesapeake_bay"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_delaware_bay_max_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_delaware_bay"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_san_pablo_bay_max_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_san_pablo_bay"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_chesapeake_bay_scalar",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(
                                    name="sst_chesapeake_bay_max_dataarray"
                                ),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_delaware_bay_scalar",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_delaware_bay_max_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_san_pablo_bay_scalar",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(
                                    name="sst_san_pablo_bay_max_dataarray"
                                ),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="bar_chart",
                        value=_FunctionCall["st"](
                            func=plot_ops.PlotBarChart(
                                func_name="PlotBarChart",
                                labels=[
                                    "Chesapeake Bay",
                                    "Delaware Bay",
                                    "San Pablo Bay",
                                ],
                                heights=[
                                    VariableReference(name="sst_chesapeake_bay_scalar"),
                                    VariableReference(name="sst_delaware_bay_scalar"),
                                    VariableReference(name="sst_san_pablo_bay_scalar"),
                                ],
                                title="Max SST's",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="bar_chart")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="argmax_basic",
        problem=AlgorithmProblemSpec(
            statement="Find the max value in the dataset as well as its time and location.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="max_value",
                    variable_type="scalar_float",
                    description="The max value.",
                ),
                OutputVariable(
                    name="max_value_time",
                    variable_type="datetime_string",
                    description="The time of the max value.",
                ),
                OutputVariable(
                    name="max_value_lat",
                    variable_type="latitude_deg_float",
                    description="The latitude of the max value.",
                ),
                OutputVariable(
                    name="max_value_lon",
                    variable_type="longitude_deg_float",
                    description="The lon of the max value.",
                ),
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_argmax_coords",
                        value=_FunctionCall["st"](
                            func=da_ops.DataArrayArgmaxCoords(
                                func_name="DataArrayArgmaxCoords",
                                data_array=VariableReference(name="analysed_sst"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="max_value",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                func_name="SelectPointCoordsKey",
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="value",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="max_value_time",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                func_name="SelectPointCoordsKey",
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="time",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="max_value_lat",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                func_name="SelectPointCoordsKey",
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="lat",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="max_value_lon",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                func_name="SelectPointCoordsKey",
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="lon",
                            )
                        ),
                    ),
                ],
                output_variables=[
                    VariableReference(name="max_value"),
                    VariableReference(name="max_value_time"),
                    VariableReference(name="max_value_lat"),
                    VariableReference(name="max_value_lon"),
                ],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="multi_input_multi_temporal_res",
        problem=AlgorithmProblemSpec(
            statement="Plot the SST and air temperature rasters for the Black Sea on June 01, 2021.",
            inputs_provided=[
                _input_vars["sst"],
                _input_vars["air_temperature"],
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="raster_plot",
                    description="Black Sea SST and air temperature on June 01, 2021.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2021-01-01", "2021-01-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="air_temperature"),
                                dim_slices={"time": ("2021-01-01", "2021-01-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Black Sea",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_black_sea"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="air_temperature_filtered"),
                                polygon=VariableReference(name="polygon_black_sea"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_black_sea_day_avg",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="air_temperature_black_sea"),
                                dim="time",
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsRaster(
                                func_name="PlotDataArraysAsRaster",
                                data_arrays=[
                                    VariableReference(name="sst_black_sea"),
                                    VariableReference(name="air_temperature_black_sea_day_avg"),
                                ],
                                subplot_grid_shape=(1, 2),
                                title="Black Sea SST and air temperature on June 01, 2021",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="multi_output_values_and_plots",
        problem=AlgorithmProblemSpec(
            statement="Compute the requested stats for the Black Sea on June 01, 2024",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="mean_value",
                    variable_type="scalar_float",
                    description="The mean value.",
                ),
                OutputVariable(
                    name="std_value",
                    variable_type="scalar_float",
                    description="The standard deviation value.",
                ),
                OutputVariable(
                    name="max_value",
                    variable_type="scalar_float",
                    description="The max value.",
                ),
                OutputVariable(
                    name="min_value",
                    variable_type="scalar_float",
                    description="The min value.",
                ),
                OutputVariable(
                    name="sst_plot",
                    variable_type="raster_plot",
                    description="A raster plot of the data subset in question.",
                ),
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="black_sea_polygon",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Black Sea",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="analysed_sst"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea"),
                                dim_slices={"time": ("2024-06-01", "2024-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="mean_value",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_mean_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_std_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="std",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="std_value",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_std_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_max_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="max_value",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_max_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_min_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="min",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="min_value",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_min_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="raster_plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsRaster(
                                func_name="PlotDataArraysAsRaster",
                                data_arrays=[VariableReference(name="sst_black_sea_filtered")],
                                subplot_grid_shape=(1, 1),
                                title="Raster plot",
                            )
                        ),
                    ),
                ],
                output_variables=[
                    VariableReference(name="mean_value"),
                    VariableReference(name="std_value"),
                    VariableReference(name="max_value"),
                    VariableReference(name="min_value"),
                    VariableReference(name="raster_plot"),
                ],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="multi_input_multi_output_with_plot",
        problem=AlgorithmProblemSpec(
            statement="Draw a bar chart with values for the average SST, air temperature, wind speed, and specific humidity for the Black Sea on June 01, 2021.",
            inputs_provided=[
                _input_vars["sst"],
                _input_vars["air_temperature"],
                _input_vars["wind_speed"],
                _input_vars["specific_humidity"],
            ],
            outputs_requested=[
                OutputVariable(
                    name="mean_sst",
                    variable_type="scalar_float",
                    description="The mean SST value.",
                ),
                OutputVariable(
                    name="mean_air_temperature",
                    variable_type="scalar_float",
                    description="The mean air temperature value.",
                ),
                OutputVariable(
                    name="mean_wind_speed",
                    variable_type="scalar_float",
                    description="The mean wind speed value.",
                ),
                OutputVariable(
                    name="mean_specific_humidity",
                    variable_type="scalar_float",
                    description="The mean specific humidity value.",
                ),
                OutputVariable(
                    name="bar_plot",
                    variable_type="bar_plot",
                    description="Bar plot with the values.",
                ),
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="black_sea_polygon",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Black Sea",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="analysed_sst"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="air_temperature"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="air_temperature_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="wind_speed_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="wind_speed"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="wind_speed_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="wind_speed_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="specific_humidity_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="specific_humidity"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="specific_humidity_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="specific_humidity_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="mean_sst",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="sst_mean_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(
                                    name="air_temperature_black_sea_filtered"
                                ),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="mean_air_temperature",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="air_temperature_mean_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="wind_speed_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="wind_speed_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="mean_wind_speed",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(name="wind_speed_mean_dataarray"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="specific_humidity_mean_dataarray",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(
                                    name="specific_humidity_black_sea_filtered"
                                ),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="mean_specific_humidity",
                        value=_FunctionCall["st"](
                            func=da_ops.ZeroDimDataArrayToScalar(
                                func_name="ZeroDimDataArrayToScalar",
                                data_array=VariableReference(
                                    name="specific_humidity_mean_dataarray"
                                ),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="bar_plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.PlotBarChart(
                                func_name="PlotBarChart",
                                heights=[
                                    VariableReference(name="mean_sst"),
                                    VariableReference(name="mean_air_temperature"),
                                    VariableReference(name="mean_wind_speed"),
                                    VariableReference(name="mean_specific_humidity"),
                                ],
                                labels=[
                                    "mean_sst",
                                    "mean_air_temperature",
                                    "mean_wind_speed",
                                    "mean_specific_humidity",
                                ],
                                title="Bar plot",
                            )
                        ),
                    ),
                ],
                output_variables=[
                    VariableReference(name="mean_sst"),
                    VariableReference(name="mean_air_temperature"),
                    VariableReference(name="mean_wind_speed"),
                    VariableReference(name="mean_specific_humidity"),
                    VariableReference(name="bar_plot"),
                ],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="multi_input_complex_plot",
        problem=AlgorithmProblemSpec(
            statement="Plot the ratio of SST to air temperature for the Black Sea on June 01, 2021.",
            inputs_provided=[
                _input_vars["sst"],
                _input_vars["air_temperature"],
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="raster_plot",
                    description="Ratio of SST to air temperature for the Black Sea on June 01, 2021.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["st"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["st"]](
                statements=[
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2021-01-01", "2021-01-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                func_name=_SliceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="air_temperature"),
                                dim_slices={"time": ("2021-01-01", "2021-01-01")},
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="polygon_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.GeocodeNaturalLanguageDescription(
                                func_name="GeocodeNaturalLanguageDescription",
                                text="Black Sea",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_black_sea"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                func_name="FilterDataArrayByPolygon",
                                data_array=VariableReference(name="air_temperature_filtered"),
                                polygon=VariableReference(name="polygon_black_sea"),
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="air_temperature_black_sea_day_avg",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                func_name=_ReduceDataArray["st"]["func_name"],
                                data_array=VariableReference(name="air_temperature_black_sea"),
                                dim="time",
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="ratio",
                        value=_FunctionCall["st"](
                            func=da_ops.BinaryOperation(
                                func_name="BinaryOperation",
                                left=VariableReference(name="sst_black_sea"),
                                right=VariableReference(name="air_temperature_black_sea_day_avg"),
                                op="div",
                            )
                        ),
                    ),
                    VariableAssignment[_FunctionCall["st"]](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=da_ops.PlotDataArraysAsRaster(
                                func_name="PlotDataArraysAsRaster",
                                data_arrays=[VariableReference(name="ratio")],
                                subplot_grid_shape=(1, 1),
                                title="Black Sea SST and air temperature ratio on June 01, 2021",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="time_only",
        problem=AlgorithmProblemSpec(
            statement="Plot the Total Solar Irradiance (TSI) from 1900 to 2000 as a time series",
            inputs_provided=[_input_vars["TSI"]],
            outputs_requested=[
                OutputVariable(
                    name="tsi_plot",
                    variable_type="line_plot",
                    description="TSI time series.",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["time_only"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["time_only"]](
                statements=[
                    VariableAssignment[_FunctionCall["time_only"]](
                        name="tsi_sliced",
                        value=_FunctionCall["time_only"](
                            func=_SliceDataArray["time_only"]["func"](
                                func_name=_SliceDataArray["time_only"]["func_name"],
                                data_array=VariableReference(name="TSI"),
                                dim_slices={"time": ("1900-01-01", "2000-01-01")},
                            )
                        ),
                        description="",
                    ),
                    VariableAssignment[_FunctionCall["time_only"]](
                        name="tsi_plot",
                        value=_FunctionCall["time_only"](
                            func=da_ops.PlotDataArraysAsTimeSeries(
                                func_name="PlotDataArraysAsTimeSeries",
                                data_arrays=[VariableReference(name="tsi_sliced")],
                                labels=[""],
                                title="",
                            )
                        ),
                        description="",
                    ),
                ],
                output_variables=[VariableReference(name="tsi_plot")],
            ),
            explanation="",
        ),
    ),
    AlgorithmAgentEvalInput(
        id="time_wvl_raster",
        problem=AlgorithmProblemSpec(
            statement="Visualize SSI as a time-wavelength raster for wavelengths between 200 to 600",
            inputs_provided=[_input_vars["SSI"]],
            outputs_requested=[
                OutputVariable(
                    name="ssi_plot",
                    variable_type="raster_plot",
                    description="SSI time-wavelength raster plot",
                )
            ],
            user_provided_geometry=None,
        ),
        reference_solution=AlgorithmSolutionSpec[_FunctionCall["time_wvl"]](
            thoughts="",
            pseudocode="",
            algorithm=Algorithm[_FunctionCall["time_wvl"]](
                statements=[
                    VariableAssignment[_FunctionCall["time_wvl"]](
                        name="ssi_sliced",
                        value=_FunctionCall["time_wvl"](
                            func=_SliceDataArray["time_wvl"]["func"](
                                func_name=_SliceDataArray["time_wvl"]["func_name"],
                                data_array=VariableReference(name="TSI"),
                                dim_slices={"wavelength": (200, 600)},
                            )
                        ),
                        description="",
                    ),
                    VariableAssignment[_FunctionCall["time_wvl"]](
                        name="ssi_plot",
                        value=_FunctionCall["time_wvl"](
                            func=da_ops.PlotDataArraysAsRaster(
                                func_name="PlotDataArraysAsRaster",
                                data_arrays=[VariableReference(name="ssi_sliced")],
                                subplot_grid_shape=(1, 1),
                                title="",
                            )
                        ),
                        description="",
                    ),
                ],
                output_variables=[VariableReference(name="ssi_plot")],
            ),
            explanation="",
        ),
    ),
]


dataset = EvalDataset(
    id="default",
    version="1",
    items=[d.to_eval_input() for d in _dataset],
)
