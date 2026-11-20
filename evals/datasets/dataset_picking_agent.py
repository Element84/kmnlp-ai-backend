# Ignore "line too long" errors.
# ruff: noqa: E501

from llm_agent.evaluation.base_eval import EvalDataset
from llm_agent.evaluation.dataset_picking_agent import (
    DatasetPickingEvalInput,
    DatasetPickingResponseAttrs,
)

_dataset: list[DatasetPickingEvalInput] = [
    DatasetPickingEvalInput(
        id="nonsense-query",
        natural_language_query="...",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=True,
            tool_call_count=0,
            dataset_ids=[],
        ),
    ),
    DatasetPickingEvalInput(
        id="unladen-swallow",
        natural_language_query="What is the airspeed velocity of an unladen swallow?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=True,
            tool_call_count=0,
            dataset_ids=[],
        ),
    ),
    DatasetPickingEvalInput(
        id="simple-sst",
        natural_language_query="What datasets contain sea surface temperature?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=1,
            dataset_ids=[
                "crw_5km_sst_daily",
                "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            ],
        ),
    ),
    DatasetPickingEvalInput(
        id="sst-black-sea",
        natural_language_query="What datasets contain sea surface temperature in the Black Sea?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=1,
            dataset_ids=[
                "crw_5km_sst_daily",
                "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            ],
        ),
    ),
    DatasetPickingEvalInput(
        id="sst-black-sea-date",
        natural_language_query="What datasets contains the sea surface temperature in the Black Sea on June 1, 2024?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=1,
            dataset_ids=["crw_5km_sst_daily"],
        ),
    ),
    DatasetPickingEvalInput(
        id="cdr-atmo-dataset",
        natural_language_query="What datasets exist with the air temperature near the surface of the Black Sea?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=1,
            dataset_ids=["air_temperature_noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"],
        ),
    ),
    DatasetPickingEvalInput(
        id="cdr-heatflux-dataset",
        natural_language_query="What datasets contain the latent heat flux in the Black Sea?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=1,
            dataset_ids=["noaa-cdr-ocean-heatflux-pds_daily"],
        ),
    ),
    DatasetPickingEvalInput(
        id="cdr-cmorph-dataset",
        natural_language_query="What datasets contain precipitation rate on the coast of Alabama on June 1, 2020?",
        expected_response_attrs=DatasetPickingResponseAttrs(
            is_refusal=False,
            tool_call_count=1,
            dataset_ids=[
                "noaa-cdr-precip-cmorph-pds_daily",
                "noaa-cdr-precip-gpcp-daily-pds_daily",
                "noaa-cdr-precip-gpcp-monthly-pds_monthly",
                "noaa-cdr-precip-persiann-pds_daily",
            ],
        ),
    ),
]

dataset = EvalDataset(
    id="default",
    version="2",
    items=[d.to_eval_input() for d in _dataset],
)
