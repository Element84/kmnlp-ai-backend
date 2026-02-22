from datetime import datetime, timedelta

from llm_agent.model.dataset import Dataset, DatasetSource


class TestDataset:
    def test_implicit_casting_of_str_to_datetime_and_timedelta(self):
        dataset = Dataset(
            id="",
            source=DatasetSource(format="zarr", uri="http://source"),
            label="",
            description="",
            landing_page=None,
            temporal_coverage=("2020-01-01T00:00:00Z", "2024-11-30T12:00:00Z"),  # type: ignore[reportArgumentType]
            temporal_resolution="P1D",  # type: ignore[reportArgumentType]
            variables=[],
        )
        assert isinstance(dataset.temporal_coverage[0], datetime)
        assert isinstance(dataset.temporal_coverage[1], datetime)
        assert isinstance(dataset.temporal_resolution, timedelta)
