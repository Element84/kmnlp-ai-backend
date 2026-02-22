# ruff: noqa: E501
from datetime import UTC, datetime
from textwrap import dedent
from typing import Any

import pyparsing
import pytest

from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.dao.utils import KG_TTL_PATHS
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable

CRW_SST_DATASET_JSON: dict[str, Any] = {
    "id": "crw_daily_5km",
    "source": {
        "format": "kerchunk-parquet",
        "uri": "s3://data-c6c22a2e42294c11b52ee7f0c792c071/crw/daily_5km/zarr_reference_final.parquet",
    },
    "label": "NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
    "description": "CoralTemp is derived from three different but related 5km daily gap-free SST data sets and provides an internally consistent SST product that stretches from 1985 to present: OSTIA Sea Surface Temperature Reanalysis (1985-2002), Geo-Polar Blended Night-only Sea Surface Temperature Reanalysis (2002-2016), Geo-Polar Blended Night-only Sea Surface Temperature Near Real-Time (2017 to present).",
    "landing_page": "https://coralreefwatch.noaa.gov/product/5km/index.php",
    "temporal_coverage": ["2020-01-01T00:00:00Z", "2024-12-01T00:00:00Z"],
    "temporal_resolution": "P1D",
    "spatial_coverage": "global",
    "variables": [
        {
            "id": "sea_ice_area_fraction_crw_daily_5km",
            "name": "sea_ice_area_fraction",
            "field_name": "sea_ice_fraction",
            "description": "Sea Ice Fraction",
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
        {
            "id": "sea_surface_temperature_crw_daily_5km",
            "name": "sea_surface_temperature",
            "field_name": "analysed_sst",
            "description": "analysed sea surface temperature",
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
        },
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

        # test that query finds all relevant results
        assert len(datasets) == 31
        var_ids = [v.id for d in datasets for v in d.variables]
        assert len(var_ids) == 155

        # test against a literal to ensure model fields are populated as
        # expected
        for dataset in datasets:
            if dataset.id == CRW_SST_DATASET_JSON["id"]:
                assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON
                break
        else:
            raise AssertionError

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
            "noaa-cdr-ndvi-pds_daily",
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            "noaa-gefs-analysis",
            "noaa-gefs-forecast-35-day",
            "noaa-gfs-forecast",
            "noaa-hrrr-analysis",
            "noaa-hrrr-forecast-48-hour",
        }

    def test_fetch_datasets_with_ids(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(ids=["crw_daily_5km"])
        assert {d.id for d in datasets} == {"crw_daily_5km"}

    def test_fetch_datasets_with_regex_pattern(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        datasets = dao.fetch_datasets(regex_pattern="air temperature")
        assert {d.id for d in datasets} == {"noaa-cdr-ocean-nearsurface-atmos-profiles-pds_daily"}

    def test_fetch_variable_by_id(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        variable = dao.get_variable_by_id("sea_surface_temperature_crw_daily_5km")
        assert variable == Variable(
            id="sea_surface_temperature_crw_daily_5km",
            field_name="analysed_sst",
            name="sea_surface_temperature",
            description="analysed sea surface temperature",
            units="degrees_Celsius",
            dimensions=[
                Dimension(
                    id="longitude",
                    cf_standard_name="longitude",
                    alternate_names=frozenset({"lon"}),
                    field_name="lon",
                ),
                Dimension(
                    id="time",
                    cf_standard_name="time",
                    field_name="time",
                ),
                Dimension(
                    id="latitude",
                    cf_standard_name="latitude",
                    alternate_names=frozenset({"lat"}),
                    field_name="lat",
                ),
            ],
        )

    def test_fetch_variable_by_id_none(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        variable = dao.get_variable_by_id("bad uri")
        assert variable is None

    def test_get_dataset_by_variable_id(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATHS)
        dataset = dao.get_dataset_by_variable_id("sea_surface_temperature_crw_daily_5km")
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
                "crw_daily_5km",
                "noaa-cdr-aerosol-optical-thickness-pds_daily",
            ]
        )
        assert len(datasets) == 2
        assert {d.id for d in datasets} == {
            "crw_daily_5km",
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
