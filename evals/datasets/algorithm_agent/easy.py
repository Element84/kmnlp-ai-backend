"""Easy-tier eval cases for the algorithm agent.

Short (1-4 op) single-dataset queries that each exercise one previously-uncovered function:
histogram, scatter, argmin, geotiff export, geospatial raster, and contour plots. Also includes
coordinate-based area-of-interest selection (bounding box and single point) where the region is
given directly in the problem statement rather than as a pre-resolved geometry input.
"""
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
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.contour import ContourMapPlotSpec
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    AlgorithmSolutionSpec,
    OutputVariable,
)
from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvalInput

from .common import _FunctionCall, _geometry_input, _input_vars, _ReduceDataArray, _SliceDataArray


def histogram_sst() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="histogram_sst",
        problem=AlgorithmProblemSpec(
            statement="Plot a histogram of daily SST values in the Gulf of Mexico during July 2024.",
            inputs_provided=[
                _input_vars["sst"],
                _geometry_input("polygon_gulf_of_mexico", "Gulf of Mexico"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="histogram",
                    description="Histogram of Gulf of Mexico SST in July 2024.",
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
                                dim_slices={"time": ("2024-07-01", "2024-07-31")},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_gulf_of_mexico",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="sst_filtered"),
                                polygon=VariableReference(name="polygon_gulf_of_mexico"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.HistogramPlotSpec(
                                    data_array=VariableReference(name="sst_gulf_of_mexico"),
                                    annotation=plot_ops.AnnotationSpec(
                                        title="Gulf of Mexico SST distribution, July 2024",
                                        xlabel="SST (°C)",
                                        ylabel="Count",
                                    ),
                                )
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="plot")],
            ),
            explanation="",
        ),
    )


def scatter_heat_fluxes() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    # ScatterPlotSpec needs 1D x/y and there is no flatten op, so scatter two regional daily-mean
    # time series (not the 2D fields cell-by-cell).
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="scatter_heat_fluxes",
        problem=AlgorithmProblemSpec(
            statement="Make a scatter plot of daily-mean sensible vs. latent heat flux over the North Atlantic during 2020.",
            inputs_provided=[
                _input_vars["sensible_heat"],
                _input_vars["latent_heat"],
                _geometry_input("polygon_north_atlantic", "North Atlantic Ocean"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="plot",
                    variable_type="scatter_plot",
                    description="Sensible vs. latent heat flux over the North Atlantic in 2020.",
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
                            name="sensible_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(
                                        name="surface_upward_sensible_heat_flux"
                                    ),
                                    dim_slices={"time": ("2020-01-01", "2020-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(
                                        name="surface_upward_latent_heat_flux"
                                    ),
                                    dim_slices={"time": ("2020-01-01", "2020-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sensible_north_atlantic",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sensible_filtered"),
                                    polygon=VariableReference(name="polygon_north_atlantic"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_north_atlantic",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="latent_filtered"),
                                    polygon=VariableReference(name="polygon_north_atlantic"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sensible_daily_mean",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="sensible_north_atlantic"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_daily_mean",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="latent_north_atlantic"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.ScatterPlotSpec(
                                        x_data_array=VariableReference(name="latent_daily_mean"),
                                        y_data_array=VariableReference(name="sensible_daily_mean"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="North Atlantic sensible vs. latent heat flux, 2020",
                                            xlabel="Latent heat flux (W/m²)",
                                            ylabel="Sensible heat flux (W/m²)",
                                        ),
                                    )
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
                            name="sensible_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(
                                        name="surface_upward_sensible_heat_flux"
                                    ),
                                    dim_slices={"time": ("2020-01-01", "2020-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_filtered",
                            value=_FunctionCall["st"](
                                func=_SliceDataArray["st"]["func"](
                                    data_array=VariableReference(
                                        name="surface_upward_latent_heat_flux"
                                    ),
                                    dim_slices={"time": ("2020-01-01", "2020-12-31")},
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sensible_north_atlantic",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="sensible_filtered"),
                                    polygon=VariableReference(name="polygon_north_atlantic"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_north_atlantic",
                            value=_FunctionCall["st"](
                                func=da_ops.FilterDataArrayByPolygon(
                                    data_array=VariableReference(name="latent_filtered"),
                                    polygon=VariableReference(name="polygon_north_atlantic"),
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sensible_daily",
                            value=_FunctionCall["st"](
                                func=da_ops.ResampleTime(
                                    data_array=VariableReference(name="sensible_north_atlantic"),
                                    target_frequency="day",
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_daily",
                            value=_FunctionCall["st"](
                                func=da_ops.ResampleTime(
                                    data_array=VariableReference(name="latent_north_atlantic"),
                                    target_frequency="day",
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="sensible_daily_mean",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="sensible_daily"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="latent_daily_mean",
                            value=_FunctionCall["st"](
                                func=_ReduceDataArray["st"]["func"](
                                    data_array=VariableReference(name="latent_daily"),
                                    dim=("lat", "lon"),
                                    op="mean",
                                )
                            ),
                        ),
                        VariableAssignment[F](
                            name="plot",
                            value=_FunctionCall["st"](
                                func=plot_ops.MakePlot(
                                    plot_spec=plot_ops.ScatterPlotSpec(
                                        x_data_array=VariableReference(name="latent_daily_mean"),
                                        y_data_array=VariableReference(name="sensible_daily_mean"),
                                        annotation=plot_ops.AnnotationSpec(
                                            title="North Atlantic sensible vs. latent heat flux, 2020",
                                            xlabel="Latent heat flux (W/m²)",
                                            ylabel="Sensible heat flux (W/m²)",
                                        ),
                                    )
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


def argmin_coldest_sst() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="argmin_coldest_sst",
        problem=AlgorithmProblemSpec(
            statement="Find the coldest SST in the dataset and report its value, date, and location.",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="min_value", variable_type="scalar_float", description="The min value."
                ),
                OutputVariable(
                    name="min_value_time",
                    variable_type="datetime_string",
                    description="The date of the min value.",
                ),
                OutputVariable(
                    name="min_value_lat",
                    variable_type="latitude_deg_float",
                    description="The latitude of the min value.",
                ),
                OutputVariable(
                    name="min_value_lon",
                    variable_type="longitude_deg_float",
                    description="The longitude of the min value.",
                ),
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_argmin_coords",
                        value=_FunctionCall["st"](
                            func=da_ops.DataArrayArgminCoords(
                                data_array=VariableReference(name="analysed_sst"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="min_value",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmin_coords"),
                                key="value",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="min_value_time",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmin_coords"),
                                key="time",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="min_value_lat",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmin_coords"),
                                key="lat",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="min_value_lon",
                        value=_FunctionCall["st"](
                            func=da_ops.SelectPointCoordsKey(
                                point_coords=VariableReference(name="sst_argmin_coords"),
                                key="lon",
                            )
                        ),
                    ),
                ],
                output_variables=[
                    VariableReference(name="min_value"),
                    VariableReference(name="min_value_time"),
                    VariableReference(name="min_value_lat"),
                    VariableReference(name="min_value_lon"),
                ],
            ),
            explanation="",
        ),
    )


def geotiff_med_sst() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    # A single-date slice drops the time dim, giving the 2D field CreateGeoTIFF expects.
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="geotiff_med_sst",
        problem=AlgorithmProblemSpec(
            statement="Export the SST map of the Mediterranean Sea on 2024-08-01 as a GeoTIFF to show on the map.",
            inputs_provided=[
                _input_vars["sst"],
                _geometry_input("polygon_mediterranean", "Mediterranean Sea"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="sst_geotiff",
                    variable_type="geotiff",
                    description="GeoTIFF of Mediterranean SST on 2024-08-01.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_aug1",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": "2024-08-01"},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_mediterranean",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="sst_aug1"),
                                polygon=VariableReference(name="polygon_mediterranean"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_geotiff",
                        value=_FunctionCall["st"](
                            func=da_ops.CreateGeoTIFF(
                                data_array=VariableReference(name="sst_mediterranean"),
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="sst_geotiff")],
            ),
            explanation="",
        ),
    )


def geospatial_ndvi() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="geospatial_ndvi",
        problem=AlgorithmProblemSpec(
            statement="Show a geospatial map of NDVI over sub-Saharan Africa on 2024-09-01.",
            inputs_provided=[
                _input_vars["ndvi"],
                _geometry_input("polygon_sub_saharan_africa", "Sub-Saharan Africa"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="ndvi_plot",
                    variable_type="raster_plot",
                    description="NDVI map over sub-Saharan Africa on 2024-09-01.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="ndvi_sep1",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="NDVI"),
                                dim_slices={"time": "2024-09-01"},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="ndvi_sub_saharan_africa",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="ndvi_sep1"),
                                polygon=VariableReference(name="polygon_sub_saharan_africa"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="ndvi_plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=plot_ops.GeospatialRasterPlotSpec(
                                    data_array=VariableReference(name="ndvi_sub_saharan_africa"),
                                    annotation=plot_ops.AnnotationSpec(
                                        title="NDVI over sub-Saharan Africa, 2024-09-01",
                                    ),
                                )
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="ndvi_plot")],
            ),
            explanation="",
        ),
    )


def contour_sst() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="contour_sst",
        problem=AlgorithmProblemSpec(
            statement="Draw a contour map of SST over the North Atlantic on 2024-01-15.",
            inputs_provided=[
                _input_vars["sst"],
                _geometry_input("polygon_north_atlantic", "North Atlantic Ocean"),
            ],
            outputs_requested=[
                OutputVariable(
                    name="sst_plot",
                    variable_type="raster_plot",
                    description="Contour map of North Atlantic SST on 2024-01-15.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_jan15",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": "2024-01-15"},
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_north_atlantic",
                        value=_FunctionCall["st"](
                            func=da_ops.FilterDataArrayByPolygon(
                                data_array=VariableReference(name="sst_jan15"),
                                polygon=VariableReference(name="polygon_north_atlantic"),
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_plot",
                        value=_FunctionCall["st"](
                            func=plot_ops.MakePlot(
                                plot_spec=ContourMapPlotSpec(
                                    data_array=VariableReference(name="sst_north_atlantic"),
                                    title="North Atlantic SST, 2024-01-15",
                                )
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="sst_plot")],
            ),
            explanation="",
        ),
    )


def bbox_mean_sst() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    # The AOI is given as coordinates in the statement, so slice lat/lon ranges directly.
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="bbox_mean_sst",
        problem=AlgorithmProblemSpec(
            statement="What was the mean SST within the bounding box 10N-20N, 60W-40W on 2024-06-01?",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="sst_mean",
                    variable_type="scalar_float",
                    description="Mean SST in the 10N-20N, 60W-40W box on 2024-06-01.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_box",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={
                                    "time": "2024-06-01",
                                    "lat": (10.0, 20.0),
                                    "lon": (-60.0, -40.0),
                                },
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_mean",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_box"),
                                op="mean",
                            )
                        ),
                    ),
                ],
                output_variables=[VariableReference(name="sst_mean")],
            ),
            explanation="",
        ),
    )


def point_sst_value() -> AlgorithmAgentEvalInput[BaseFunctionCall]:
    # A single point given as coordinates: slice single lat/lon values with nearest-neighbor.
    F = _FunctionCall["st"]
    return AlgorithmAgentEvalInput(
        id="point_sst_value",
        problem=AlgorithmProblemSpec(
            statement="What was the SST at the point 36.5N, 122.0W on 2024-08-01?",
            inputs_provided=[_input_vars["sst"]],
            outputs_requested=[
                OutputVariable(
                    name="sst_value",
                    variable_type="scalar_float",
                    description="SST at 36.5N, 122.0W on 2024-08-01.",
                )
            ],
        ),
        reference_solution=AlgorithmSolutionSpec[F](
            analysis="",
            pseudocode="",
            algorithm=Algorithm[F](
                statements=[
                    VariableAssignment[F](
                        name="sst_point",
                        value=_FunctionCall["st"](
                            func=_SliceDataArray["st"]["func"](
                                data_array=VariableReference(name="analysed_sst"),
                                dim_slices={"time": "2024-08-01", "lat": 36.5, "lon": -122.0},
                                method="nearest",
                            )
                        ),
                    ),
                    VariableAssignment[F](
                        name="sst_value",
                        value=_FunctionCall["st"](
                            func=_ReduceDataArray["st"]["func"](
                                data_array=VariableReference(name="sst_point"),
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


CASES: list[Callable[[], AlgorithmAgentEvalInput[BaseFunctionCall]]] = [
    histogram_sst,
    scatter_heat_fluxes,
    argmin_coldest_sst,
    geotiff_med_sst,
    geospatial_ndvi,
    contour_sst,
    bbox_mean_sst,
    point_sst_value,
]
