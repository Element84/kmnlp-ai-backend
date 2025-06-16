# ruff: noqa: E501
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.dao.util import KG_TTL_PATH
from llm_agent.model.variable import Variable

CRW_SST_DATASET_JSON = {
    "iri": "http://kmnlp-demo.noaa.gov/crw_daily_5km",
    "source": "s3://data-c6c22a2e42294c11b52ee7f0c792c071/crw/daily_5km/zarr_reference_final.json",
    "label": "NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface Temperature (CoralTemp)",
    "description": "CoralTemp is derived from three different but related 5km daily gap-free SST data sets and provides an internally consistent SST product that stretches from 1985 to present: OSTIA Sea Surface Temperature Reanalysis (1985-2002), Geo-Polar Blended Night-only Sea Surface Temperature Reanalysis (2002-2016), Geo-Polar Blended Night-only Sea Surface Temperature Near Real-Time (2017 to present).",
    "temporal_coverage": ["2020-01-01T00:00:00Z", "2024-12-01T00:00:00Z"],
    "temporal_resolution": "P1D",
    "spatial_coverage": "global",
    "variables": [
        {
            "iri": "http://kmnlp-demo.noaa.gov/sea_ice_area_fraction_crw_daily_5km",
            "name": "sea_ice_area_fraction",
            "field_name": "sea_ice_fraction",
            "description": "Sea Ice Fraction",
            "units": "1",
            "dimensions": ["latitude", "longitude", "time"],
        },
        {
            "iri": "http://kmnlp-demo.noaa.gov/sea_surface_temperature_crw_daily_5km",
            "name": "sea_surface_temperature",
            "field_name": "analysed_sst",
            "description": "analysed sea surface temperature",
            "units": "degrees_Celsius",
            "dimensions": ["latitude", "longitude", "time"],
        },
    ],
}


class TestKnowledgeGraphDAO:
    def test_initialization(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        assert dao is not None

    def test_fetch_all_triples(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        triples = dao.fetch_all_triples()
        assert triples != []

    def test_fetch_variables(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        datasets = dao.fetch_variables()

        # test that query finds all relevant results
        assert len(datasets) == 20
        var_iris = [v.iri for d in datasets for v in d.variables]
        assert len(var_iris) == 29

        # test against a literal to ensure model fields are populated as
        # expected
        for dataset in datasets:
            if dataset.iri == CRW_SST_DATASET_JSON["iri"]:
                break
        else:
            raise AssertionError
        assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON

    def test_fetch_variable_by_iri(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        variable = dao.get_variable_by_iri(
            "http://kmnlp-demo.noaa.gov/sea_surface_temperature_crw_daily_5km"
        )
        assert variable == Variable(
            iri="http://kmnlp-demo.noaa.gov/sea_surface_temperature_crw_daily_5km",
            field_name="analysed_sst",
            name="sea_surface_temperature",
            description="analysed sea surface temperature",
            units="degrees_Celsius",
            dimensions=frozenset({"latitude", "longitude", "time"}),
        )

    def test_fetch_variable_by_iri_none(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        variable = dao.get_variable_by_iri("bad uri")
        assert variable is None

    def test_get_dataset_by_variable_iri(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        dataset = dao.get_dataset_by_variable_iri(
            "http://kmnlp-demo.noaa.gov/sea_surface_temperature_crw_daily_5km"
        )
        assert dataset is not None
        assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON

    def test_get_dataset_by_variable_iri_none(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        dataset = dao.get_dataset_by_variable_iri("bad uri")
        assert dataset is None

    def test_get_datasets_by_iris(self):
        dao = KnowledgeGraphDAO(KG_TTL_PATH)
        datasets = dao.get_datasets_by_iris(
            [
                "http://kmnlp-demo.noaa.gov/crw_daily_5km",
                "http://kmnlp-demo.noaa.gov/noaa-cdr-aerosol-optical-thickness-pds_daily",
            ]
        )
        assert len(datasets) == 2
        assert {d.iri for d in datasets} == {
            "http://kmnlp-demo.noaa.gov/crw_daily_5km",
            "http://kmnlp-demo.noaa.gov/noaa-cdr-aerosol-optical-thickness-pds_daily",
        }

        # test against a literal to ensure model fields are populated as
        # expected
        for dataset in datasets:
            if dataset.iri == CRW_SST_DATASET_JSON["iri"]:
                break
        else:
            raise AssertionError
        assert dataset.model_dump(mode="json") == CRW_SST_DATASET_JSON
