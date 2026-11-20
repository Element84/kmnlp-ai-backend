"""Original 15 eval cases for the algorithm agent."""
# ruff: noqa: E501, D103, N806
# pyright: reportPrivateUsage=false

from collections.abc import Callable

import llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions as da_ops
import llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic as plot_ops
from llm_agent.agents.algorithm_agent.algorithm.base import (
    Algorithm,
    BaseFunctionCall,
    VariableAssignment,
    VariableReference,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.sst import SSTPlotSpec
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    OutputVariable,
)
from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvalInput

from .common import (
    _FunctionCall,
    _geometry_input,
    _input_vars,
    _ReduceDataArray,
    _SliceDataArray,
)


def data_query_mismatch() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
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
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[],
                output_variables=[],
            ),
            explanation="",
            error_message="Cannot be answered from the available data.",
        ),
    )


def mean_basic() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
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
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_mean_scalar",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                op="mean",
                            )
                        ),
                        description="Mean value as scalar",
                    ),
                ],
                output_variables=[VariableReference(name="sst_mean_scalar")],
            ),
            explanation="",
        ),
    )


def mean_black_sea() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="mean_black_sea",
        problem=AlgorithmProblemSpec(
            statement="What was the mean SST in the Black Sea on June 01, 2024?",
            inputs_provided=[_input_vars["sst"], _geometry_input("black_sea_polygon", "Black Sea")],
            outputs_requested=[
                OutputVariable(
                    name="sst_value",
                    variable_type="scalar_float",
                    description="The mean value in the Black Sea on June 01, 2024 the SST.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="analysed_sst"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea"),
                                dim_slices={"time": ("2024-06-01", "2024-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_value",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="sst_value")],
            ),
            explanation="",
        ),
    )


def raster_plot() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="raster_plot",
        problem=AlgorithmProblemSpec(
            statement="Plot the SST maps for Jan 1, 2023 and Aug 1, 2023 stacked vertically.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="plot_grid",
                    description="SST maps for Jan 1, 2023 and Aug 1, 2023 stacked vertically.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_jan1_2023",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2023-01-01", "2023-01-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_aug1_2023",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2023-08-01", "2023-08-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.PlotGridSpec(
                                    subplot_specs=[
                                        SSTPlotSpec(
                                            sst_data_array=VariableReference(name="sst_jan1_2023")
                                        ),
                                        SSTPlotSpec(
                                            sst_data_array=VariableReference(name="sst_aug1_2023")
                                        ),
                                    ],
                                    ncols=1,
                                ),
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    )


def time_series_1() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="time_series_1",
        problem=AlgorithmProblemSpec(
            statement="Show the daily average SST within the Chesapeake Bay over the year 2024 as a time series.",
            inputs_provided=[
                _input_vars["sst"],
                _geometry_input("polygon_chesapeake_bay", "Chesapeake Bay"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="line_plot",
                    description="Daily average SST within the Chesapeake Bay over the year 2024.",
                )
            ],
        ),
        reference_solution=[
            # Aggregate the daily values by day-of-year (a no-op grouping within a single year).
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2024-01-01", "2024-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_chesapeake_bay",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_chesapeake_bay"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="daily_mean_sst",
                            value=_FunctionCall["st"](
                                func=da_ops.GroupByTimeAndReduceDataArray(
                                    data_array=VariableReference(name="sst_chesapeake_bay"),
                                    op="mean",
                                    time_unit="dayofyear",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.LinePlotSpec(
                                        data_array=VariableReference(name="daily_mean_sst"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Daily mean SST in 2024",
                                            xlabel="Day of year",
                                            ylabel="SST (°C)",
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
            # SST is already daily, so a per-day spatial mean over the bay is equivalent.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2024-01-01", "2024-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_chesapeake_bay",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_chesapeake_bay"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="daily_mean_sst",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="sst_chesapeake_bay"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.LinePlotSpec(
                                        data_array=VariableReference(name="daily_mean_sst"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Daily mean SST in 2024",
                                            xlabel="Day of year",
                                            ylabel="SST (°C)",
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
        ],
    )


def time_series_2() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="time_series_2",
        problem=AlgorithmProblemSpec(
            statement="Show the monthly average SST within the Chesapeake Bay over the year 2024 as a time series.",
            inputs_provided=[
                _input_vars["sst"],
                _geometry_input("polygon_chesapeake_bay", "Chesapeake Bay"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="line_plot",
                    description="Monthly average SST within the Chesapeake Bay over the year 2024.",
                )
            ],
        ),
        reference_solution=[
            # Collapse each calendar month to a single spatial+temporal mean over the bay.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2024-01-01", "2024-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_chesapeake_bay",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_chesapeake_bay"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="monthly_mean_sst",
                            value=_FunctionCall["st"](
                                func=da_ops.GroupByTimeAndReduceDataArray(
                                    data_array=VariableReference(name="sst_chesapeake_bay"),
                                    op="mean",
                                    time_unit="month",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.LinePlotSpec(
                                        data_array=VariableReference(name="monthly_mean_sst"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Monthly mean SST in 2024",
                                            xlabel="Month",
                                            ylabel="SST (°C)",
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
            # Resample to monthly means, then take the per-month spatial mean over the bay.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2024-01-01", "2024-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_chesapeake_bay",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_chesapeake_bay"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_monthly",
                            value=_FunctionCall["st"](
                                func=da_ops.ResampleTime(
                                    data_array=VariableReference(name="sst_chesapeake_bay"),
                                    target_frequency="month",
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="monthly_mean_sst",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="sst_monthly"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.LinePlotSpec(
                                        data_array=VariableReference(name="monthly_mean_sst"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Monthly mean SST in 2024",
                                            xlabel="Month",
                                            ylabel="SST (°C)",
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
        ],
    )


def time_series_3() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="time_series_3",
        problem=AlgorithmProblemSpec(
            statement="Show the daily average air temperature in the Chesapeake Bay over the month of June 2020 as a time series.",
            inputs_provided=[
                _input_vars["air_temperature"],
                _geometry_input("polygon_chesapeake_bay", "Chesapeake Bay"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="line_plot",
                    description="Daily average air temperature.",
                )
            ],
        ),
        reference_solution=[
            # Air temperature is sub-daily, so grouping by day-of-month yields a daily mean.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="air_temperature_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature"),
                                    dim_slices={"time": ("2020-06-01", "2020-06-30")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_chesapeake_bay",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="air_temperature_filtered"),
                                    polygon=VariableReference(name="polygon_chesapeake_bay"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="daily_mean_air_temperature",
                            value=_FunctionCall["st"](
                                func=da_ops.GroupByTimeAndReduceDataArray(
                                    data_array=VariableReference(
                                        name="air_temperature_chesapeake_bay"
                                    ),
                                    op="mean",
                                    time_unit="dayofmonth",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.LinePlotSpec(
                                        data_array=VariableReference(
                                            name="daily_mean_air_temperature"
                                        ),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Daily mean air temperature in June 2020",
                                            xlabel="Day of month",
                                            ylabel="Air temperature (°C)",
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
            # Resample the sub-daily data to daily means, then take the per-day spatial mean.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="air_temperature_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature"),
                                    dim_slices={"time": ("2020-06-01", "2020-06-30")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_chesapeake_bay",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="air_temperature_filtered"),
                                    polygon=VariableReference(name="polygon_chesapeake_bay"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_daily",
                            value=_FunctionCall["st"](
                                func=da_ops.ResampleTime(
                                    data_array=VariableReference(
                                        name="air_temperature_chesapeake_bay"
                                    ),
                                    target_frequency="day",
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="daily_mean_air_temperature",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature_daily"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.LinePlotSpec(
                                        data_array=VariableReference(
                                            name="daily_mean_air_temperature"
                                        ),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Daily mean air temperature in June 2020",
                                            xlabel="Day of month",
                                            ylabel="Air temperature (°C)",
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
        ],
    )


def bar_plot() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="bar_plot",
        problem=AlgorithmProblemSpec(
            statement="Draw a bar chart comparing the max SST's in Chesapeake Bay, Delaware Bay, and San Pablo Bay in the month of June 2024.",
            inputs_provided=[
                _input_vars["sst"],
                _geometry_input("polygon_chesapeake_bay", "Chesapeake Bay"),
                _geometry_input("polygon_delaware_bay", "Delaware Bay"),
                _geometry_input("polygon_san_pablo_bay", "San Pablo Bay"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="bar_chart",
                    variable_type="bar_plot",
                    description="Bar chart comparing the max SST's in Chesapeake Bay, Delaware Bay, and San Pablo Bay in the month of June 2024.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": ("2024-06-01", "2024-06-30")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_chesapeake_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_chesapeake_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_delaware_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_delaware_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_san_pablo_bay",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_san_pablo_bay"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_chesapeake_bay_max",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_chesapeake_bay"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_delaware_bay_max",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_delaware_bay"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_san_pablo_bay_max",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_san_pablo_bay"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="bar_chart",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.BarPlotSpec(
                                    labels=[
                                        "Chesapeake Bay",
                                        "Delaware Bay",
                                        "San Pablo Bay",
                                    ],
                                    heights=[
                                        VariableReference(name="sst_chesapeake_bay_max"),
                                        VariableReference(name="sst_delaware_bay_max"),
                                        VariableReference(name="sst_san_pablo_bay_max"),
                                    ],
                                    annotation=plot_ops.AnnotationSpec(
                                        title="Max SST in June 2024",
                                        xlabel="Bay",
                                        ylabel="SST (°C)",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="bar_chart")],
            ),
            explanation="",
        ),
    )


def argmax_basic() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
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
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_argmax_coords",
                        value=_FunctionCall["st"](
                            func=da_ops.DataArrayArgmaxCoords(
                                data_array=VariableReference(name="analysed_sst"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="max_value",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="value",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="max_value_time",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="time",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="max_value_lat",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmax_coords"),
                                key="lat",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="max_value_lon",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
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
    )


def multi_input_multi_temporal_res() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="multi_input_multi_temporal_res",
        problem=AlgorithmProblemSpec(
            statement="Plot the SST and air temperature rasters for the Black Sea on June 01, 2021.",
            inputs_provided=[
                _input_vars["sst"],
                _input_vars["air_temperature"],
                _geometry_input("polygon_black_sea", "Black Sea"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="plot_grid",
                    description="Black Sea SST and air temperature on June 01, 2021.",
                )
            ],
        ),
        reference_solution=[
            # SST is daily (one step on the day); air temperature is sub-daily, so collapse its
            # timesteps to a single map with a time reduction.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="air_temperature_filtered"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_black_sea_day_avg",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature_black_sea"),
                                    dim="time",
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.PlotGridSpec(
                                        subplot_specs=[
                                            SSTPlotSpec(
                                                sst_data_array=VariableReference(
                                                    name="sst_black_sea"
                                                ),
                                            ),
                                            SSTPlotSpec(
                                                sst_data_array=VariableReference(
                                                    name="air_temperature_black_sea_day_avg"
                                                ),
                                            ),
                                        ],
                                        ncols=2,
                                    ),
                                ),
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
            # Equivalent: resample the sub-daily air temperature onto a daily cadence instead.
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="air_temperature_filtered"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_black_sea_day_avg",
                            value=_FunctionCall["st"](
                                func=da_ops.ResampleTime(
                                    data_array=VariableReference(name="air_temperature_black_sea"),
                                    target_frequency="day",
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.PlotGridSpec(
                                        subplot_specs=[
                                            SSTPlotSpec(
                                                sst_data_array=VariableReference(
                                                    name="sst_black_sea"
                                                ),
                                            ),
                                            SSTPlotSpec(
                                                sst_data_array=VariableReference(
                                                    name="air_temperature_black_sea_day_avg"
                                                ),
                                            ),
                                        ],
                                        ncols=2,
                                    ),
                                ),
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
        ],
    )


def multi_output_values_and_plots() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="multi_output_values_and_plots",
        problem=AlgorithmProblemSpec(
            statement="Compute the requested stats for the Black Sea on June 01, 2024",
            inputs_provided=[_input_vars["sst"], _geometry_input("black_sea_polygon", "Black Sea")],
            outputs_requested=[
                OutputVariable(
                    name="sst_mean_scalar",
                    variable_type="scalar_float",
                    description="The mean value.",
                ),
                OutputVariable(
                    name="sst_std_scalar",
                    variable_type="scalar_float",
                    description="The standard deviation value.",
                ),
                OutputVariable(
                    name="sst_max_scalar",
                    variable_type="scalar_float",
                    description="The max value.",
                ),
                OutputVariable(
                    name="sst_min_scalar",
                    variable_type="scalar_float",
                    description="The min value.",
                ),
                OutputVariable(
                    name="sst_plot",
                    variable_type="raster_plot",
                    description="A raster plot of the data subset in question.",
                ),
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="analysed_sst"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea"),
                                dim_slices={"time": ("2024-06-01", "2024-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_mean_scalar",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_std_scalar",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="std",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_max_scalar",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="max",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_min_scalar",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="min",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.GeneralRasterPlotSpec(
                                    data_array=VariableReference(name="sst_black_sea_filtered"),
                                    annotation=plot_ops.AnnotationSpec(
                                        title="Black Sea SST on June 01, 2024",
                                        xlabel="Longitude",
                                        ylabel="Latitude",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ],
                output_variables=[
                    VariableReference(name="sst_mean_scalar"),
                    VariableReference(name="sst_std_scalar"),
                    VariableReference(name="sst_max_scalar"),
                    VariableReference(name="sst_min_scalar"),
                    VariableReference(name="sst_plot"),
                ],
            ),
            explanation="",
        ),
    )


def multi_input_multi_output_with_plot() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="multi_input_multi_output_with_plot",
        problem=AlgorithmProblemSpec(
            statement="Draw a bar chart with values for the average SST, air temperature, wind speed, and specific humidity for the Black Sea on June 01, 2021.",
            inputs_provided=[
                _input_vars["sst"],
                _input_vars["air_temperature"],
                _input_vars["wind_speed"],
                _input_vars["specific_humidity"],
                _geometry_input("black_sea_polygon", "Black Sea"),
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
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="analysed_sst"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="air_temperature_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="air_temperature"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="air_temperature_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="air_temperature_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="wind_speed_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="wind_speed"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="wind_speed_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="wind_speed_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="specific_humidity_black_sea",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="specific_humidity"),
                                polygon=VariableReference(name="black_sea_polygon"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="specific_humidity_black_sea_filtered",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="specific_humidity_black_sea"),
                                dim_slices={"time": ("2021-06-01", "2021-06-01")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="mean_sst",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="mean_air_temperature",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(
                                    name="air_temperature_black_sea_filtered"
                                ),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="mean_wind_speed",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="wind_speed_black_sea_filtered"),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="mean_specific_humidity",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(
                                    name="specific_humidity_black_sea_filtered"
                                ),
                                op="mean",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="bar_plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.BarPlotSpec(
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
                                    annotation=plot_ops.AnnotationSpec(title="Bar plot"),
                                ),
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
    )


def multi_input_complex_plot() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="multi_input_complex_plot",
        problem=AlgorithmProblemSpec(
            statement="Plot the ratio of SST to air temperature for the Black Sea on June 01, 2021.",
            inputs_provided=[
                _input_vars["sst"],
                _input_vars["air_temperature"],
                _geometry_input("polygon_black_sea", "Black Sea"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="raster_plot",
                    description="Ratio of SST to air temperature for the Black Sea on June 01, 2021.",
                )
            ],
        ),
        reference_solution=[
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sst_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sst_filtered"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="air_temperature_filtered"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_black_sea_day_avg",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature_black_sea"),
                                    dim="time",
                                    op="mean",
                                )
                            ),
                        ),
                        # SST and air temperature are on different grids; regrid before the ratio.
                        VariableAssignment[F](
                            name="air_temperature_regridded",
                            value=_FunctionCall["st"](
                                func=da_ops.RegridToMatch(
                                    source=VariableReference(
                                        name="air_temperature_black_sea_day_avg"
                                    ),
                                    reference=VariableReference(name="sst_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="ratio",
                            value=_FunctionCall["st"](
                                func=da_ops.BinaryOperation(
                                    left=VariableReference(name="sst_black_sea"),
                                    right=VariableReference(name="air_temperature_regridded"),
                                    op="div",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.GeneralRasterPlotSpec(
                                        data_array=VariableReference(name="ratio"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Black Sea SST and air temperature ratio on June 01, 2021"
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
            AlgorithmSolutionSpec[F](
                analysis="",
                pseudocode="",
                algorithm=Algorithm[F](
                    statements=[
                        VariableAssignment[F](
                            name="sst_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="analysed_sst"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature"),
                                    dim_slices={"time": ("2021-06-01", "2021-06-01")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="air_temperature_day_avg",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="air_temperature_filtered"),
                                    dim="time",
                                    op="mean",
                                )
                            ),
                        ),
                        # SST and air temperature are on different grids; regrid before the ratio.
                        VariableAssignment[F](
                            name="air_temperature_regridded",
                            value=_FunctionCall["st"](
                                func=da_ops.RegridToMatch(
                                    source=VariableReference(name="air_temperature_day_avg"),
                                    reference=VariableReference(name="sst_filtered"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="ratio",
                            value=_FunctionCall["st"](
                                func=da_ops.BinaryOperation(
                                    left=VariableReference(name="sst_filtered"),
                                    right=VariableReference(name="air_temperature_regridded"),
                                    op="div",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="ratio_black_sea",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="ratio"),
                                    polygon=VariableReference(name="polygon_black_sea"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.GeneralRasterPlotSpec(
                                        data_array=VariableReference(name="ratio"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="Black Sea SST and air temperature ratio on June 01, 2021"
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ],
                    output_variables=[VariableReference(name="plot")],
                ),
                explanation="",
            ),
        ],
    )


def time_only() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["time_only"]
    return AlgorithmAgentEvalInput(
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
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="tsi_sliced",
                        value=_FunctionCall["time_only"](
                            func=_SliceDataArray["time_only"]["func"](
                                data_array=VariableReference(name="TSI"),
                                dim_slices={"time": ("1900-01-01", "2000-01-01")},
                            )
                        ),
                        description="",
                    ),
                    VariableAssignment[F](
                        name="tsi_plot",
                        value=_FunctionCall["time_only"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.LinePlotSpec(
                                    data_array=VariableReference(name="tsi_sliced"),
                                    annotation=plot_ops.AnnotationSpec(
                                        title="Total Solar Irradiance (TSI) from 1900 to 2000"
                                    ),
                                ),
                            )
                        ),
                        description="",
                    ),
                ],
                output_variables=[VariableReference(name="tsi_plot")],
            ),
            explanation="",
        ),
    )


def time_wvl_raster() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["time_wvl"]
    return AlgorithmAgentEvalInput(
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
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="ssi_sliced",
                        value=_FunctionCall["time_wvl"](
                            func=_SliceDataArray["time_wvl"]["func"](
                                data_array=VariableReference(name="SSI"),
                                dim_slices={"wavelength": (200, 600)},
                            )
                        ),
                        description="",
                    ),
                    VariableAssignment[F](
                        name="ssi_plot",
                        value=_FunctionCall["time_wvl"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.GeneralRasterPlotSpec(
                                    data_array=VariableReference(name="ssi_sliced"),
                                    annotation=plot_ops.AnnotationSpec(
                                        title="SSI time-wavelength raster for wavelengths between 200 to 600"
                                    ),
                                ),
                            )
                        ),
                        description="",
                    ),
                ],
                output_variables=[VariableReference(name="ssi_plot")],
            ),
            explanation="",
        ),
    )


CASES: list[Callable[[], AlgorithmAgentEvalInput[BaseFunctionCall]]] = [
    data_query_mismatch,
    mean_basic,
    mean_black_sea,
    raster_plot,
    time_series_1,
    time_series_2,
    time_series_3,
    bar_plot,
    argmax_basic,
    multi_input_multi_temporal_res,
    multi_output_values_and_plots,
    multi_input_multi_output_with_plot,
    multi_input_complex_plot,
    time_only,
    time_wvl_raster,
]
