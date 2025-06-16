from datetime import UTC, datetime, timedelta

import pytest

from llm_agent.agents.algorithm_agent.interface import (
    SpatioTemporalDataArrayVariable,
    UnsupportedVariableDimsError,
    input_variable_from_model,
)
from llm_agent.model.dataset import Dataset
from llm_agent.model.variable import Variable

stub_dataset = Dataset(
    iri="iri",
    source="some source",
    label="label",
    temporal_coverage=(
        datetime(year=2025, month=1, day=1, tzinfo=UTC),
        datetime(year=2025, month=1, day=2, tzinfo=UTC),
    ),
    temporal_resolution=timedelta(days=1),
    variables=[],
)


class TestInputVariable:
    def test_spatiotemporal(self):
        variable = Variable(
            iri="http://some/thing/with/a/name",
            field_name="name",
            name="label",
            description="description",
            units="unit",
            dimensions=frozenset({"latitude", "longitude", "time"}),
        )
        input_variable = input_variable_from_model(variable, stub_dataset)

        assert isinstance(input_variable, SpatioTemporalDataArrayVariable)
        assert input_variable.name == "name"
        assert input_variable.variable_type == "DataArray"
        assert input_variable.description == "description"
        assert input_variable.spatial_coverage == "global"
        assert input_variable.units == "unit"
        assert input_variable.temporal_coverage == (
            datetime(year=2025, month=1, day=1, tzinfo=UTC),
            datetime(year=2025, month=1, day=2, tzinfo=UTC),
        )
        assert input_variable.temporal_resolution == timedelta(days=1)

    def test_error_if_not_lat_lon_time(self):
        variable = Variable(
            iri="http://some/thing/with/a/name",
            field_name="name",
            name="label",
            description="description",
            units="unit",
            dimensions=frozenset({"time", "wavelength"}),
        )
        with pytest.raises(UnsupportedVariableDimsError):
            _ = input_variable_from_model(variable, stub_dataset)
