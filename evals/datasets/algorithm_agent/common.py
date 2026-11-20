# Shared setup for the algorithm-agent eval dataset (split across sibling modules):
# knowledge-graph-backed input variables, per-dimensionality function-call models, and
# handles to dimension-constrained Slice/Reduce functions.
# pyright: reportUnusedFunction=false

from typing import TypedDict, cast

import llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions as da_ops
from llm_agent.agents.algorithm_agent.algorithm.base import (
    BaseFunctionCall,
    Function,
    make_function_call_model,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets import FunctionSet
from llm_agent.agents.algorithm_agent.interface import (
    GeometryVariable,
    InputVariable,
    input_variable_from_model,
    input_vars_to_function_set,
)
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.model.dataset import Dataset
from llm_agent.model.variable import Variable

_kg_dao = KnowledgeGraphDAO()
_all_datasets = _kg_dao.fetch_datasets()
_all_variables: dict[str, Variable] = {var.id: var for ds in _all_datasets for var in ds.variables}
_var_id_to_dataset: dict[str, Dataset] = {
    var.id: ds for ds in _all_datasets for var in ds.variables
}


def _to_input_var(var_id: str) -> InputVariable:
    return input_variable_from_model(_all_variables[var_id], _var_id_to_dataset[var_id])


def _geometry_input(name: str, description: str) -> GeometryVariable:
    """A pre-resolved area-of-interest polygon, as the orchestration layer passes to this agent.

    The broader app geocodes place names upstream (via a separate geocoding tool) and hands the
    resulting geometry to the algorithm agent as an input, so cases reference this rather than
    geocoding inside the algorithm.
    """
    return GeometryVariable(name=name, description=description, geometry_type="polygon")


_input_vars: dict[str, InputVariable] = {
    # --- originally-imported variables ---
    "sst": _to_input_var("sea_surface_temperature_crw_5km_sst_daily"),
    "air_temperature": _to_input_var(
        "air_temperature_noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"
    ),
    "wind_speed": _to_input_var("wind_speed_noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"),
    "specific_humidity": _to_input_var(
        "specific_humidity_noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"
    ),
    "SSI": _to_input_var(
        "solar_irradiance_per_unit_wavelength_noaa-cdr-solar-spectral-irradiance-pds_high-resolution-yearly"
    ),
    "TSI": _to_input_var(
        "solar_irradiance_noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly"
    ),
    # --- added for the expanded case set ---
    "sst_whoi": _to_input_var("sea_surface_temperature_noaa-cdr-sea-surface-temp-whoi-pds_daily"),
    "sea_ice": _to_input_var("sea_ice_area_fraction_crw_5km_sst_daily"),
    "sst_monthly": _to_input_var("sea_surface_temperature_crw_5km_sst_monthly_mean"),
    "sst_annual_mean": _to_input_var("sea_surface_temperature_crw_5km_sst_annual_mean"),
    "precip_cmorph": _to_input_var("lwe_precipitation_rate_noaa-cdr-precip-cmorph-pds_daily"),
    "precip_gpcp": _to_input_var("lwe_precipitation_rate_noaa-cdr-precip-gpcp-daily-pds_daily"),
    "precip_gpcp_monthly": _to_input_var(
        "precipitation_amount_noaa-cdr-precip-gpcp-monthly-pds_monthly"
    ),
    "ndvi": _to_input_var("normalized_difference_vegetation_index_noaa-cdr-ndvi-pds_daily"),
    "lai": _to_input_var("leaf_area_index_noaa-cdr-leaf-area-index-fapar-pds_daily"),
    "aerosol": _to_input_var(
        "atmosphere_optical_thickness_due_to_ambient_aerosol_noaa-cdr-aerosol-optical-thickness-pds_daily"
    ),
    "latent_heat": _to_input_var(
        "surface_upward_latent_heat_flux_noaa-cdr-ocean-heatflux-pds_daily"
    ),
    "sensible_heat": _to_input_var(
        "surface_upward_sensible_heat_flux_noaa-cdr-ocean-heatflux-pds_daily"
    ),
    "wind_u": _to_input_var("wind_u_10m_noaa-gefs-analysis"),
    "wind_v": _to_input_var("wind_v_10m_noaa-gefs-analysis"),
    "temp2m": _to_input_var("temperature_2m_noaa-gefs-analysis"),
    "temp2m_ens": _to_input_var("temperature_2m_ecmwf-ifs-ens-forecast-15-day-0-25-degree"),
    "tsi_yearly": _to_input_var("solar_irradiance_noaa-cdr-total-solar-irradiance-pds_yearly"),
}


_func_sets: dict[str, FunctionSet] = {
    "st": input_vars_to_function_set([_input_vars["sst"]]),
    "time_wvl": input_vars_to_function_set([_input_vars["SSI"]]),
    "time_only": input_vars_to_function_set([_input_vars["TSI"]]),
    # ECMWF ensemble forecast: dims ensemble_member/init_time/lat/lead_time/lon.
    "ens": input_vars_to_function_set([_input_vars["temp2m_ens"]]),
    # Built from wind u/v so MakePlot exposes WindVectorPlotSpec (only present for wind-typed vars).
    "wind": input_vars_to_function_set([_input_vars["wind_u"], _input_vars["wind_v"]]),
}

_FunctionCall: dict[str, type[BaseFunctionCall]] = {
    key: make_function_call_model(fs.functions) for key, fs in _func_sets.items()
}


class _ConstrainedFunc[T: Function](TypedDict):
    func: type[T]
    func_name: str


_SliceDataArray: dict[str, _ConstrainedFunc[da_ops.SliceDataArray]] = {
    "st": {
        "func": cast("type[da_ops.SliceDataArray]", _func_sets["st"]["SliceDataArray_LatLonTime"]),
        "func_name": "SliceDataArray_LatLonTime",
    },
    "time_wvl": {
        "func": cast(
            "type[da_ops.SliceDataArray]", _func_sets["time_wvl"]["SliceDataArray_TimeWavelength"]
        ),
        "func_name": "SliceDataArray_TimeWavelength",
    },
    "time_only": {
        "func": cast("type[da_ops.SliceDataArray]", _func_sets["time_only"]["SliceDataArray_Time"]),
        "func_name": "SliceDataArray_Time",
    },
    "ens": {
        "func": cast(
            "type[da_ops.SliceDataArray]",
            _func_sets["ens"]["SliceDataArray_EnsemblememberInittimeLatLeadtimeLon"],
        ),
        "func_name": "SliceDataArray_EnsemblememberInittimeLatLeadtimeLon",
    },
    "wind": {
        "func": cast(
            "type[da_ops.SliceDataArray]", _func_sets["wind"]["SliceDataArray_LatLonTime"]
        ),
        "func_name": "SliceDataArray_LatLonTime",
    },
}

_ReduceDataArray: dict[str, _ConstrainedFunc[da_ops.ReduceDataArray]] = {
    "st": {
        "func": cast(
            "type[da_ops.ReduceDataArray]", _func_sets["st"]["ReduceDataArray_LatLonTime"]
        ),
        "func_name": "ReduceDataArray_LatLonTime",
    },
    "time_wvl": {
        "func": cast(
            "type[da_ops.ReduceDataArray]", _func_sets["time_wvl"]["ReduceDataArray_TimeWavelength"]
        ),
        "func_name": "ReduceDataArray_TimeWavelength",
    },
    "time_only": {
        "func": cast(
            "type[da_ops.ReduceDataArray]", _func_sets["time_only"]["ReduceDataArray_Time"]
        ),
        "func_name": "ReduceDataArray_Time",
    },
    "ens": {
        "func": cast(
            "type[da_ops.ReduceDataArray]",
            _func_sets["ens"]["ReduceDataArray_EnsemblememberInittimeLatLeadtimeLon"],
        ),
        "func_name": "ReduceDataArray_EnsemblememberInittimeLatLeadtimeLon",
    },
    "wind": {
        "func": cast(
            "type[da_ops.ReduceDataArray]", _func_sets["wind"]["ReduceDataArray_LatLonTime"]
        ),
        "func_name": "ReduceDataArray_LatLonTime",
    },
}

_RollingReduce: dict[str, _ConstrainedFunc[da_ops.RollingReduce]] = {
    "st": {
        "func": cast("type[da_ops.RollingReduce]", _func_sets["st"]["RollingReduce_LatLonTime"]),
        "func_name": "RollingReduce_LatLonTime",
    },
    "time_only": {
        "func": cast("type[da_ops.RollingReduce]", _func_sets["time_only"]["RollingReduce_Time"]),
        "func_name": "RollingReduce_Time",
    },
}

_CumulativeReduce: dict[str, _ConstrainedFunc[da_ops.CumulativeReduce]] = {
    "st": {
        "func": cast(
            "type[da_ops.CumulativeReduce]", _func_sets["st"]["CumulativeReduce_LatLonTime"]
        ),
        "func_name": "CumulativeReduce_LatLonTime",
    },
}

_CorrelateDataArrays: dict[str, _ConstrainedFunc[da_ops.CorrelateDataArrays]] = {
    "st": {
        "func": cast(
            "type[da_ops.CorrelateDataArrays]", _func_sets["st"]["CorrelateDataArrays_LatLonTime"]
        ),
        "func_name": "CorrelateDataArrays_LatLonTime",
    },
}
