# ruff: noqa: E501
import os
from datetime import UTC, datetime
from textwrap import dedent
from typing import Any

import pyparsing
import pytest

from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.dao.utils import KG_TTL_PATHS, get_kg_ttl_paths

CRW_ANALYSED_SST_VARIABLE_JSON: dict[str, Any] = {
    "id": "sea_surface_temperature_crw_5km_sst_daily",
    "name": "sea_surface_temperature",
    "dataset_name": "NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
    "field_name": "analysed_sst",
    "standard_name": "http://mmisw.org/ont/cf/parameter/sea_surface_temperature",
    "description": """Other metadata:
coverage_content_type: physicalMeasurement
grid_mapping: crs
long_name: analysed sea surface temperature""",
    "units": "degrees_Celsius",
    "dimensions": [
        {
            "id": "latitude",
            "cf_standard_name": "latitude",
            "alternate_names": ["lat"],
            "field_name": "lat",
        },
        {
            "id": "longitude",
            "cf_standard_name": "longitude",
            "alternate_names": ["lon"],
            "field_name": "lon",
        },
        {
            "id": "time",
            "cf_standard_name": "time",
            "alternate_names": [],
            "field_name": "time",
        },
    ],
}

CRW_SST_DATASET_JSON: dict[str, Any] = {
    "id": "crw_5km_sst_daily",
    "source": {
        "format": "kerchunk-parquet",
        "uri": "s3://data-c6c22a2e42294c11b52ee7f0c792c071/crw_5km/sst_daily/zarr_reference_final.parquet",
    },
    "label": "NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
    "description": "CoralTemp is derived from three different but related 5km daily gap-free SST data sets and provides an internally consistent SST product that stretches from 1985 to present: OSTIA Sea Surface Temperature Reanalysis (1985-2002), Geo-Polar Blended Night-only Sea Surface Temperature Reanalysis (2002-2016), Geo-Polar Blended Night-only Sea Surface Temperature Near Real-Time (2017 to present).",
    "landing_page": "https://coralreefwatch.noaa.gov/product/5km/index.php",
    "temporal_coverage": ["1985-01-01T00:00:00Z", "2026-09-06T00:00:00Z"],
    "temporal_resolution": "P1D",
    "spatial_coverage": "global",
    "variables": [
        {
            "id": "sea_ice_area_fraction_crw_5km_sst_daily",
            "name": "sea_ice_area_fraction",
            "dataset_name": "NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
            "field_name": "sea_ice_fraction",
            "standard_name": "http://mmisw.org/ont/cf/parameter/sea_ice_area_fraction",
            "description": """Other metadata:
comment: 0 is 0% ice, 1 is 100% ice
coverage_content_type: physicalMeasurement
grid_mapping: crs
long_name: Sea Ice Fraction""",
            "units": "1",
            "dimensions": [
                {
                    "id": "latitude",
                    "cf_standard_name": "latitude",
                    "alternate_names": ["lat"],
                    "field_name": "lat",
                },
                {
                    "id": "longitude",
                    "cf_standard_name": "longitude",
                    "alternate_names": ["lon"],
                    "field_name": "lon",
                },
                {
                    "id": "time",
                    "cf_standard_name": "time",
                    "alternate_names": [],
                    "field_name": "time",
                },
            ],
        },
        CRW_ANALYSED_SST_VARIABLE_JSON,
    ],
}


class TestKnowledgeGraphDAO:
    def test_initialization(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        assert dao is not None

    def test_fetch_all_triples(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        triples = dao.fetch_all_triples()
        assert triples != []

    def test_fetch_datasets(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets()

        assert len(datasets) == 36
        assert [d.id for d in datasets] == [
            "crw_5km_sst_annual_max",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_annual_min",
            "crw_5km_sst_daily",
            "crw_5km_sst_monthly_max",
            "crw_5km_sst_monthly_mean",
            "crw_5km_sst_monthly_min",
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
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
            "noaa-gefs-analysis",
            "noaa-gfs-analysis-hourly",
            "noaa-gfs-forecast",
            "noaa-hrrr-analysis",
            "noaa-hrrr-forecast-48-hour",
        ]
        var_ids = [v.id for d in datasets for v in d.variables]
        assert len(var_ids) == 176

        # test against a literal to ensure model fields are populated as
        # expected
        for dataset in datasets:
            if dataset.id == CRW_SST_DATASET_JSON["id"]:
                assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON
                break
        else:
            raise AssertionError

    def test_fetch_datasets_with_icechunk(self):
        # Set env var for this test
        os.environ["USE_ICECHUNK_STORES"] = "true"
        try:
            dao = KnowledgeGraphDAO(get_kg_ttl_paths())
            datasets = dao.fetch_datasets()

            assert len(datasets) == 36
            assert [d.id for d in datasets] == [
                "crw_5km_sst_annual_max",
                "crw_5km_sst_annual_mean",
                "crw_5km_sst_annual_min",
                "crw_5km_sst_daily",
                "crw_5km_sst_monthly_max",
                "crw_5km_sst_monthly_mean",
                "crw_5km_sst_monthly_min",
                "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
                "noaa-cdr-aerosol-optical-thickness-pds_daily",
                "noaa-cdr-aerosol-optical-thickness-pds_monthly",
                "noaa-cdr-leaf-area-index-fapar-pds_daily",
                "noaa-cdr-ndvi-pds_daily",
                "noaa-cdr-ocean-heatflux-pds_daily",
                "noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily",
                "noaa-cdr-precip-cmorph-pds_daily",
                "noaa-cdr-precip-cmorph-pds_hourly",
                "noaa-cdr-precip-gpcp-daily-pds_daily",
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
                "noaa-gefs-analysis",
                "noaa-gfs-analysis-hourly",
                "noaa-gfs-forecast",
                "noaa-hrrr-analysis",
                "noaa-hrrr-forecast-48-hour",
            ]
            var_ids = [v.id for d in datasets for v in d.variables]
            assert len(var_ids) == 176

            # test against a literal to ensure model fields are populated as
            # expected
            for dataset in datasets:
                if dataset.id == CRW_SST_DATASET_JSON["id"]:
                    assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON
                    break
            else:
                raise AssertionError
        finally:
            # Reset env var to default
            os.environ["USE_ICECHUNK_STORES"] = "false"

    def test_fetch_datasets_with_start(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(end_time=datetime(year=1610, month=1, day=1, tzinfo=UTC))

        # test that query finds all relevant results
        assert {d.id for d in datasets} == {
            "noaa-cdr-solar-spectral-irradiance-pds_high-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_standard-resolution-yearly",
            "noaa-cdr-solar-spectral-irradiance-pds_yearly",
            "noaa-cdr-total-solar-irradiance-pds_yearly",
        }

    def test_fetch_datasets_with_end(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        # find datasets with data from 2025-04-25 23:59:59 or later
        datasets = dao.fetch_datasets(
            start_time=datetime(
                year=2025, month=4, day=25, hour=23, minute=59, second=59, tzinfo=UTC
            )
        )

        # test that query finds all relevant results
        assert {d.id for d in datasets} == {
            "crw_5km_sst_annual_max",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_annual_min",
            "crw_5km_sst_daily",
            "crw_5km_sst_monthly_max",
            "crw_5km_sst_monthly_mean",
            "crw_5km_sst_monthly_min",
            "noaa-cdr-ndvi-pds_daily",
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            "noaa-gefs-analysis",
            "noaa-gfs-forecast",
            "noaa-hrrr-analysis",
            "noaa-hrrr-forecast-48-hour",
        }

    def test_fetch_datasets_with_start_and_end(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(
            start_time=datetime(year=2025, month=1, day=1, hour=0, minute=0, second=0, tzinfo=UTC),
            end_time=datetime(
                year=2025, month=12, day=31, hour=23, minute=59, second=59, tzinfo=UTC
            ),
        )

        # test that query finds all relevant results
        assert {d.id for d in datasets} == {
            "crw_5km_sst_annual_max",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_annual_min",
            "crw_5km_sst_daily",
            "crw_5km_sst_monthly_max",
            "crw_5km_sst_monthly_mean",
            "crw_5km_sst_monthly_min",
            "noaa-cdr-ndvi-pds_daily",
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            "noaa-gefs-analysis",
            "noaa-gfs-forecast",
            "noaa-hrrr-analysis",
            "noaa-hrrr-forecast-48-hour",
            "noaa-cdr-precip-gpcp-daily-pds_daily",
            "noaa-cdr-precip-gpcp-monthly-pds_monthly",
        }

    def test_fetch_datasets_with_time_and_regex_filters(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(
            start_time=datetime(year=2025, month=1, day=1, hour=0, minute=0, second=0, tzinfo=UTC),
            end_time=datetime(
                year=2025, month=12, day=31, hour=23, minute=59, second=59, tzinfo=UTC
            ),
            regex_pattern=r"sea surface temperature|SST",
        )

        assert {d.id for d in datasets} == {
            "crw_5km_sst_annual_max",
            "crw_5km_sst_annual_mean",
            "crw_5km_sst_annual_min",
            "crw_5km_sst_daily",
            "crw_5km_sst_monthly_max",
            "crw_5km_sst_monthly_mean",
            "crw_5km_sst_monthly_min",
        }

    def test_fetch_datasets_with_ids(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(ids=["crw_5km_sst_daily"])
        assert {d.id for d in datasets} == {"crw_5km_sst_daily"}

        assert dao.fetch_datasets(ids=[]) == []

    def test_fetch_datasets_with_regex_pattern(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(regex_pattern="air temperature")
        assert {d.id for d in datasets} == {"noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"}

        datasets = dao.fetch_datasets(regex_pattern="ECMWF.*forecast.*temperature")
        assert {d.id for d in datasets} == {"ecmwf-ifs-ens-forecast-15-day-0-25-degree"}

    def test_fetch_variable_by_id(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        variable = dao.get_variable_by_id("sea_surface_temperature_crw_5km_sst_daily")
        assert variable is not None
        assert variable.model_dump(mode="json") == CRW_ANALYSED_SST_VARIABLE_JSON

    def test_fetch_variable_by_id_none(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        variable = dao.get_variable_by_id("bad uri")
        assert variable is None

    def test_get_dataset_by_variable_id(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        dataset = dao.get_dataset_by_variable_id("sea_surface_temperature_crw_5km_sst_daily")
        assert dataset is not None
        assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON

    def test_get_dataset_by_variable_id_none(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        dataset = dao.get_dataset_by_variable_id("bad uri")
        assert dataset is None

    def test_get_datasets_by_ids(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(
            ids=[
                "crw_5km_sst_daily",
                "noaa-cdr-aerosol-optical-thickness-pds_daily",
            ]
        )
        assert len(datasets) == 2
        assert {d.id for d in datasets} == {
            "crw_5km_sst_daily",
            "noaa-cdr-aerosol-optical-thickness-pds_daily",
        }

        # test against a literal to ensure model fields are populated as
        # expected
        for dataset in datasets:
            if dataset.id == CRW_SST_DATASET_JSON["id"]:
                break
        else:
            raise AssertionError
        assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON

    # FUTURE: This test could stand to validate something more useful
    def test_execute_read_only_query_fetch_all_triples(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        query = """
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o
        }
        """
        result = dao.execute_read_only_query(query)

        assert len(result) >= 1069

    def test_execute_write_query_fails(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        query = dedent("""
            PREFIX : <http://kmnlp-demo.noaa.gov/>

            INSERT DATA {
              :subject :predicate "object"
            }
        """)

        with pytest.raises(pyparsing.exceptions.ParseException):
            _ = dao.execute_read_only_query(query)
