"""Langfuse eval dataset creation script for DatasetPickingAgent.

`is_refusal=True` is used for queries that should return NO datasets
(irrelevant / out-of-scope queries). For all other queries, `is_refusal=False`
and `dataset_ids` contains every dataset that should be considered a correct
match (use this as a set for recall/precision scoring, not a strict ordered list).

(Generated and modified code)
"""

from typing import NotRequired, TypedDict

from langfuse import Langfuse

langfuse = Langfuse()

DATASET_NAME = "dataset_picking_agent_evals_from_sdk"

langfuse.create_dataset(
    name=DATASET_NAME,
    description="Evals for the DatasetPickingAgent",
    metadata={"created_at": "2026-09-03"},
)


class DatasetItem(TypedDict):
    input: str
    dataset_ids: list[str]
    is_refusal: NotRequired[bool | None]


items: list[DatasetItem] = [
    # ---------------- Sea Surface Temperature (SST) ----------------
    {
        "input": "sea surface temperature data",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_monthly_mean",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
        ],
    },
    {
        "input": "daily SST readings",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
        ],
    },
    {
        "input": "annual average ocean surface temperature",
        "dataset_ids": ["crw_5km_sst_annual_mean"],
    },
    {
        "input": "what was the highest sea surface temp recorded each year",
        "dataset_ids": ["crw_5km_sst_annual_max"],
    },
    {
        "input": "lowest monthly sea surface temperature",
        "dataset_ids": ["crw_5km_sst_monthly_min"],
    },
    {
        "input": "monthly mean SST",
        "dataset_ids": ["crw_5km_sst_monthly_mean"],
    },
    {
        "input": "Coral Reef Watch sea surface temperature",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "crw_5km_sst_annual_max",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_annual_min",
            "crw_5km_sst_monthly_max",
            "crw_5km_sst_monthly_mean",
            "crw_5km_sst_monthly_min",
        ],
    },
    {
        "input": "WHOI sea surface temperature climate data record",
        "dataset_ids": ["noaa-cdr-sea-surface-temp-whoi-pds_daily"],
    },
    {
        "input": "5km resolution SST",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "crw_5km_sst_annual_max",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_annual_min",
            "crw_5km_sst_monthly_max",
            "crw_5km_sst_monthly_mean",
            "crw_5km_sst_monthly_min",
        ],
    },
    {
        "input": "What datasets contain sea surface temperature?",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_monthly_mean",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
        ],
    },
    {
        "input": "What datasets contain sea surface temperature in the Black Sea?",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_monthly_mean",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
        ],
    },
    {
        "input": "What datasets contains the sea surface temperature in the Black Sea on June 1",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
        ],
    },
    # ---------------- Weather Models (Forecast/Analysis) ----------------
    {
        "input": "15-day ensemble weather forecast",
        "dataset_ids": ["ecmwf-ifs-ens-forecast-15-day-0-25-degree"],
    },
    {
        "input": "ECMWF forecast data",
        "dataset_ids": ["ecmwf-ifs-ens-forecast-15-day-0-25-degree"],
    },
    {
        "input": "0.25 degree resolution forecast model",
        "dataset_ids": ["ecmwf-ifs-ens-forecast-15-day-0-25-degree"],
    },
    {
        "input": "GFS hourly analysis",
        "dataset_ids": ["noaa-gfs-analysis-hourly"],
    },
    {
        "input": "Global Forecast System output",
        "dataset_ids": ["noaa-gfs-forecast", "noaa-gfs-analysis-hourly"],
    },
    {
        "input": "GEFS ensemble analysis data",
        "dataset_ids": ["noaa-gefs-analysis"],
    },
    {
        "input": "high resolution rapid refresh model",
        "dataset_ids": ["noaa-hrrr-analysis", "noaa-hrrr-forecast-48-hour"],
    },
    {
        "input": "48 hour HRRR forecast",
        "dataset_ids": ["noaa-hrrr-forecast-48-hour"],
    },
    {
        "input": "short-term weather forecast for the next two days",
        "dataset_ids": ["noaa-hrrr-forecast-48-hour"],
    },
    {
        "input": "current atmospheric conditions analysis",
        "dataset_ids": [
            "noaa-gefs-analysis",
            "noaa-gfs-analysis-hourly",
            "noaa-hrrr-analysis",
        ],
    },
    {
        "input": "what will the weather be like next week",
        "dataset_ids": [
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            "noaa-gfs-forecast",
        ],
    },
    {
        "input": "What datasets exist with the air temperature near the surface of the Black Sea?",
        "dataset_ids": ["noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"],
    },
    # ---------------- Aerosols ----------------
    {
        "input": "aerosol optical thickness data",
        "dataset_ids": [
            "noaa-cdr-aerosol-optical-thickness-pds_daily",
            "noaa-cdr-aerosol-optical-thickness-pds_monthly",
        ],
    },
    {
        "input": "monthly aerosol optical depth",
        "dataset_ids": ["noaa-cdr-aerosol-optical-thickness-pds_monthly"],
    },
    {
        "input": "daily atmospheric aerosol measurements",
        "dataset_ids": ["noaa-cdr-aerosol-optical-thickness-pds_daily"],
    },
    {
        "input": "how much haze/particulate is in the atmosphere",
        "dataset_ids": ["noaa-cdr-aerosol-optical-thickness-pds_daily"],
    },
    # ---------------- Vegetation ----------------
    {
        "input": "leaf area index dataset",
        "dataset_ids": ["noaa-cdr-leaf-area-index-fapar-pds_daily"],
    },
    {
        "input": "FAPAR vegetation data",
        "dataset_ids": ["noaa-cdr-leaf-area-index-fapar-pds_daily"],
    },
    {
        "input": "NDVI vegetation index",
        "dataset_ids": ["noaa-cdr-ndvi-pds_daily"],
    },
    {
        "input": "normalized difference vegetation index",
        "dataset_ids": ["noaa-cdr-ndvi-pds_daily"],
    },
    {
        "input": "satellite measurement of plant greenness",
        "dataset_ids": [
            "noaa-cdr-ndvi-pds_daily",
            "noaa-cdr-leaf-area-index-fapar-pds_daily",
        ],
    },
    {
        "input": "fraction of absorbed photosynthetically active radiation",
        "dataset_ids": ["noaa-cdr-leaf-area-index-fapar-pds_daily"],
    },
    # ---------------- Ocean Heat Flux / Near-Surface Atmosphere ----------------
    {
        "input": "ocean heat flux dataset",
        "dataset_ids": ["noaa-cdr-ocean-heatflux-pds_daily"],
    },
    {
        "input": "sensible heat flux over the ocean",
        "dataset_ids": ["noaa-cdr-ocean-heatflux-pds_daily"],
    },
    {
        "input": "What datasets contain the latent heat flux in the Black Sea?",
        "dataset_ids": ["noaa-cdr-ocean-heatflux-pds_daily"],
    },
    {
        "input": "near-surface air temperature over open water",
        "dataset_ids": ["noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"],
    },
    {
        "input": "humidity profile near the ocean surface",
        "dataset_ids": ["noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"],
    },
    {
        "input": "marine boundary layer atmospheric profile",
        "dataset_ids": ["noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"],
    },
    # ---------------- Precipitation ----------------
    {
        "input": "CMORPH precipitation estimates",
        "dataset_ids": ["noaa-cdr-precip-cmorph-pds_daily"],
    },
    {
        "input": "GPCP daily precipitation",
        "dataset_ids": ["noaa-cdr-precip-gpcp-daily-pds_daily"],
    },
    {
        "input": "monthly global precipitation climatology",
        "dataset_ids": ["noaa-cdr-precip-gpcp-monthly-pds_monthly"],
    },
    {
        "input": "PERSIANN rainfall estimates",
        "dataset_ids": ["noaa-cdr-precip-persiann-pds_daily"],
    },
    {
        "input": "satellite-derived rainfall data",
        "dataset_ids": [
            "noaa-cdr-precip-cmorph-pds_daily",
            "noaa-cdr-precip-gpcp-daily-pds_daily",
            "noaa-cdr-precip-gpcp-monthly-pds_monthly",
            "noaa-cdr-precip-persiann-pds_daily",
        ],
    },
    {
        "input": "how much did it rain yesterday",
        "dataset_ids": [
            "noaa-cdr-precip-cmorph-pds_daily",
            "noaa-cdr-precip-gpcp-daily-pds_daily",
            "noaa-cdr-precip-persiann-pds_daily",
        ],
    },
    {
        "input": "rainfall totals by month",
        "dataset_ids": ["noaa-cdr-precip-gpcp-monthly-pds_monthly"],
    },
    {
        "input": "What datasets contain precipitation rate on the coast of Alabama on June 1",
        "dataset_ids": [
            "noaa-cdr-precip-cmorph-pds_daily",
            "noaa-cdr-precip-gpcp-daily-pds_daily",
            "noaa-cdr-precip-persiann-pds_daily",
        ],
    },
    # ---------------- Solar Irradiance ----------------
    {
        "input": "daily total solar irradiance record",
        "dataset_ids": ["noaa-cdr-total-solar-irradiance-pds_daily"],
    },
    {
        "input": "yearly TSI values",
        "dataset_ids": ["noaa-cdr-total-solar-irradiance-pds_yearly"],
    },
    {
        "input": "monthly total solar irradiance",
        "dataset_ids": ["noaa-cdr-total-solar-irradiance-pds_monthly"],
    },
    {
        "input": "sun's total energy output over time",
        "dataset_ids": [
            "noaa-cdr-total-solar-irradiance-pds_daily",
            "noaa-cdr-total-solar-irradiance-pds_monthly",
            "noaa-cdr-total-solar-irradiance-pds_yearly",
        ],
    },
    {
        "input": "Total solar irradiance (TSI) over time covering multiple solar cycles",
        "dataset_ids": [
            "noaa-cdr-total-solar-irradiance-pds_daily",
            "noaa-cdr-total-solar-irradiance-pds_monthly",
            "noaa-cdr-total-solar-irradiance-pds_yearly",
        ],
    },
    {
        "input": "solar spectral irradiance by wavelength",
        "dataset_ids": [
            "noaa-cdr-solar-spectral-irradiance-pds_daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_yearly",
        ],
    },
    {
        "input": "high resolution solar spectral irradiance, monthly",
        "dataset_ids": ["noaa-cdr-solar-spectral-irradiance-pds_high-resolution-monthly"],
    },
    {
        "input": "standard resolution yearly solar spectrum",
        "dataset_ids": ["noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly"],
    },
    {
        "input": "daily solar spectrum data",
        "dataset_ids": [
            "noaa-cdr-solar-spectral-irradiance-pds_daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-daily",
        ],
    },
    {
        "input": "SORCE-style spectral solar data",
        "dataset_ids": [
            "noaa-cdr-solar-spectral-irradiance-pds_daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_yearly",
        ],
    },
    # ---------------- Ambiguous / Broad Queries ----------------
    {
        "input": "NOAA climate data records",
        "dataset_ids": [
            "noaa-cdr-aerosol-optical-thickness-pds_daily",
            "noaa-cdr-aerosol-optical-thickness-pds_monthly",
            "noaa-cdr-leaf-area-index-fapar-pds_daily",
            "noaa-cdr-ndvi-pds_daily",
            "noaa-cdr-ocean-heatflux-pds_daily",
            "noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily",
            "noaa-cdr-precip-cmorph-pds_daily",
            "noaa-cdr-precip-gpcp-daily-pds_daily",
            "noaa-cdr-precip-gpcp-monthly-pds_monthly",
            "noaa-cdr-precip-persiann-pds_daily",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            "noaa-cdr-solar-spectral-irradiance-pds_daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_yearly",
            "noaa-cdr-total-solar-irradiance-pds_daily",
            "noaa-cdr-total-solar-irradiance-pds_monthly",
            "noaa-cdr-total-solar-irradiance-pds_yearly",
        ],
    },
    {
        "input": "NOAA weather model data",
        "dataset_ids": [
            "noaa-gefs-analysis",
            "noaa-gfs-analysis-hourly",
            "noaa-gfs-forecast",
            "noaa-hrrr-analysis",
            "noaa-hrrr-forecast-48-hour",
        ],
    },
    {
        "input": "ocean surface conditions",
        "dataset_ids": [
            "crw_5km_sst_daily",
            "noaa-cdr-sea-surface-temp-whoi-pds_daily",
            "noaa-cdr-ocean-heatflux-pds_daily",
            "noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily",
        ],
    },
    {
        "input": "solar data",
        "dataset_ids": [
            "noaa-cdr-solar-spectral-irradiance-pds_daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-daily",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-monthly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_yearly",
            "noaa-cdr-total-solar-irradiance-pds_daily",
            "noaa-cdr-total-solar-irradiance-pds_monthly",
            "noaa-cdr-total-solar-irradiance-pds_yearly",
        ],
    },
    # ---------------- No-Match / Irrelevant (Refusal cases) ----------------
    {
        "input": "What is the airspeed velocity of an unladen swallow?",
        "dataset_ids": [],
        "is_refusal": True,
    },
    {
        "input": "Space Weather Prediction Center SWPC solar activity sunspot geomagnetic",
        "dataset_ids": [],
        "is_refusal": True,
    },
    {
        "input": "best pizza recipe",
        "dataset_ids": [],
        "is_refusal": True,
    },
    {
        "input": "stock market closing prices today",
        "dataset_ids": [],
        "is_refusal": True,
    },
    {
        "input": "population of France",
        "dataset_ids": [],
        "is_refusal": True,
    },
    {
        "input": "how to fix a flat tire",
        "dataset_ids": [],
        "is_refusal": True,
    },
]

for item in items:
    is_refusal = item.get("is_refusal", False)
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        input=item["input"],
        expected_output={"is_refusal": is_refusal, "dataset_ids": item.get("dataset_ids")},
    )

print(f"Created {len(items)} dataset items in '{DATASET_NAME}'.")  # noqa: T201
