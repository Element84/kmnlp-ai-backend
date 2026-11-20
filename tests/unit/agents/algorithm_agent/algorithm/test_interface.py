from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from llm_agent.agents.algorithm_agent.algorithm.function_sets import (
    FunctionSet,
    GeneralDataArrayFunctionSet,
    GeneralPlottingFunctionSet,
    SpatioTemporalDataArrayFunctionSet,
    TemporalDataArrayFunctionSet,
)
from llm_agent.agents.algorithm_agent.interface import (
    GeneralDataArrayVariable,
    SpatioTemporalDataArrayVariable,
    TemporalDataArrayVariable,
    input_variable_from_model,
    input_vars_to_function_set,
)
from llm_agent.model.dataset import Dataset, DatasetSource
from llm_agent.model.dimension import Dimension
from llm_agent.model.variable import Variable

dims = {
    "time": Dimension(field_name="time", id="", cf_standard_name="time"),
    "lat": Dimension(field_name="lat", id="", cf_standard_name="latitude"),
    "lon": Dimension(field_name="lon", id="", cf_standard_name="longitude"),
    "wavelength": Dimension(
        field_name="wavelength", id="", cf_standard_name="radiation_wavelength"
    ),
}

stub_dataset = Dataset(
    id="id",
    source=DatasetSource(format="zarr", uri="s3://example.com/dataset.zarr"),
    label="label",
    description="description",
    landing_page="http://example.com/landing_page",
    temporal_coverage=(
        datetime(year=2025, month=1, day=1, tzinfo=UTC),
        datetime(year=2025, month=1, day=2, tzinfo=UTC),
    ),
    temporal_resolution=timedelta(days=1),
    variables=[],
)


class TestInputVarModels:
    def test_spatiotemporal_requires_lat_lon(self):
        with pytest.raises(ValidationError):
            _ = SpatioTemporalDataArrayVariable(
                name="name",
                dataset_name="dataset_name",
                variable_type="DataArray",
                description="description",
                units="unit",
                dimensions=[
                    dims["time"],
                    dims["lat"],
                    Dimension(field_name="altitude", id="", cf_standard_name="altitude"),
                ],
                time_dim="time",
                spatial_coverage="global",
                temporal_coverage=(
                    datetime(year=2025, month=1, day=1, tzinfo=UTC),
                    datetime(year=2025, month=1, day=2, tzinfo=UTC),
                ),
                temporal_resolution=timedelta(days=1),
            )


class TestInputVariableFromModel:
    def test_spatiotemporal(self):
        variable = Variable(
            id="http://some/thing/with/a/name",
            dataset_name="dataset_name",
            field_name="name",
            name="label",
            description="description",
            units="unit",
            dimensions=[dims["time"], dims["lat"], dims["lon"]],
        )
        input_variable = input_variable_from_model(variable, stub_dataset)

        assert isinstance(input_variable, SpatioTemporalDataArrayVariable)
        assert input_variable.model_dump(mode="json") == {
            "name": "name",
            "dataset_name": "dataset_name",
            "variable_type": "DataArray",
            "description": "description",
            "standard_name": None,
            "dimensions": [
                {
                    "field_name": "lat",
                    "id": "",
                    "cf_standard_name": "latitude",
                    "alternate_names": [],
                },
                {
                    "field_name": "lon",
                    "id": "",
                    "cf_standard_name": "longitude",
                    "alternate_names": [],
                },
                {
                    "field_name": "time",
                    "id": "",
                    "cf_standard_name": "time",
                    "alternate_names": [],
                },
            ],
            "time_dim": "time",
            "spatial_coverage": "global",
            "units": "unit",
            "temporal_coverage": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "temporal_resolution": "P1D",
        }

    def test_temporal(self):
        variable = Variable(
            id="http://some/thing/with/a/name",
            dataset_name="dataset_name",
            field_name="name",
            name="label",
            description="description",
            units="unit",
            dimensions=[dims["time"], dims["wavelength"]],
        )
        input_variable = input_variable_from_model(variable, stub_dataset)

        assert isinstance(input_variable, TemporalDataArrayVariable)
        assert input_variable.model_dump(mode="json") == {
            "name": "name",
            "dataset_name": "dataset_name",
            "variable_type": "DataArray",
            "description": "description",
            "standard_name": None,
            "dimensions": [
                {
                    "field_name": "time",
                    "id": "",
                    "cf_standard_name": "time",
                    "alternate_names": [],
                },
                {
                    "field_name": "wavelength",
                    "id": "",
                    "cf_standard_name": "radiation_wavelength",
                    "alternate_names": [],
                },
            ],
            "time_dim": "time",
            "units": "unit",
            "temporal_coverage": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "temporal_resolution": "P1D",
        }

    def test_general(self):
        variable = Variable(
            id="http://some/thing/with/a/name",
            dataset_name="dataset_name",
            field_name="name",
            name="label",
            description="description",
            units="unit",
            dimensions=[
                Dimension(field_name="x", id=""),
                Dimension(field_name="y", id=""),
                Dimension(field_name="z", id=""),
            ],
        )
        input_variable = input_variable_from_model(variable, stub_dataset)

        assert isinstance(input_variable, GeneralDataArrayVariable)

        assert input_variable.model_dump(mode="json") == {
            "name": "name",
            "dataset_name": "dataset_name",
            "variable_type": "DataArray",
            "description": "description",
            "standard_name": None,
            "dimensions": [
                {
                    "field_name": "x",
                    "id": "",
                    "cf_standard_name": None,
                    "alternate_names": [],
                },
                {
                    "field_name": "y",
                    "id": "",
                    "cf_standard_name": None,
                    "alternate_names": [],
                },
                {
                    "field_name": "z",
                    "id": "",
                    "cf_standard_name": None,
                    "alternate_names": [],
                },
            ],
            "units": "unit",
        }


class TestInputVarsToFunctionSet:
    def test_spatiotemporal(self):
        input_var = SpatioTemporalDataArrayVariable(
            name="name",
            dataset_name="dataset_name",
            variable_type="DataArray",
            description="description",
            dimensions=[dims["time"], dims["lat"], dims["lon"]],
            time_dim="time",
            units="unit",
            spatial_coverage="global",
            temporal_coverage=(
                datetime(year=2025, month=1, day=1, tzinfo=UTC),
                datetime(year=2025, month=1, day=2, tzinfo=UTC),
            ),
            temporal_resolution=timedelta(days=1),
        )
        function_set = input_vars_to_function_set([input_var])
        expected_function_set = FunctionSet.merge(
            [
                SpatioTemporalDataArrayFunctionSet(["time", "lat", "lon"]),
                GeneralPlottingFunctionSet(),
            ]
        )
        assert function_set.keys() == expected_function_set.keys()

    def test_temporal(self):
        input_var = TemporalDataArrayVariable(
            name="name",
            dataset_name="dataset_name",
            variable_type="DataArray",
            description="description",
            dimensions=[dims["time"], dims["wavelength"]],
            time_dim="time",
            units="unit",
            temporal_coverage=(
                datetime(year=2025, month=1, day=1, tzinfo=UTC),
                datetime(year=2025, month=1, day=2, tzinfo=UTC),
            ),
            temporal_resolution=timedelta(days=1),
        )
        function_set = input_vars_to_function_set([input_var])
        expected_function_set = FunctionSet.merge(
            [
                TemporalDataArrayFunctionSet(["time", "wavelength"]),
                GeneralPlottingFunctionSet(),
            ]
        )
        assert function_set.keys() == expected_function_set.keys()

    def test_general(self):
        input_var = GeneralDataArrayVariable(
            name="name",
            dataset_name="dataset_name",
            variable_type="DataArray",
            description="description",
            units="unit",
            dimensions=[
                Dimension(field_name="x", id=""),
                Dimension(field_name="y", id=""),
                Dimension(field_name="z", id=""),
            ],
        )
        function_set = input_vars_to_function_set([input_var])
        expected_function_set = FunctionSet.merge(
            [
                GeneralDataArrayFunctionSet(["x", "y", "z"]),
                GeneralPlottingFunctionSet(),
            ]
        )
        assert function_set.keys() == expected_function_set.keys()
