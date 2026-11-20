from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from llm_agent.model.variable import Variable


class DatasetSource(BaseModel):
    """Source information for a dataset."""

    format: Literal["zarr", "icechunk-zarr", "kerchunk-parquet"] = Field(
        description="The file format of the dataset."
    )
    uri: str = Field(description="URI at which the dataset can be accessed.")


class Dataset(BaseModel):
    """A dataset represented in the knowledge graph."""

    id: str = Field(description="Dataset ID.")
    source: DatasetSource = Field(description="Source information for the dataset.")
    label: str = Field(
        description=("Natural language label for the dataset in the knowledge graph."),
    )
    description: str = Field(description=("Dataset description."))
    landing_page: str | None = Field(
        default=None, description="Official landing page for this dataset."
    )
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times of the period covered by the dataset."
    )
    temporal_resolution: timedelta | None = Field(
        default=None,
        description=(
            "The temporal resolution of the dataset: the time interval between data points."
        ),
    )
    # hard-coded for now
    spatial_coverage: Literal["global"] = Field(
        default="global", description="The spatial coverage of the dataset."
    )
    variables: list[Variable] = Field(description="The variables present in the dataset.")

    def summary_view(self) -> "DatasetSummaryView":
        """Convert to a low-token summary view of the dataset."""
        return DatasetSummaryView(
            id=self.id,
            label=self.label,
            description=self.description,
            temporal_coverage=self.temporal_coverage,
            spatial_coverage=self.spatial_coverage,
            variables=[var.name for var in self.variables],
        )


class DatasetSummaryView(BaseModel):
    """A low-token summary view of a dataset, for use with LLMs."""

    id: str = Field(description="Dataset ID.")
    label: str = Field(
        description=("Natural language label for the dataset in the knowledge graph."),
    )
    description: str = Field(description=("Dataset description."))
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times of the period covered by the dataset."
    )
    spatial_coverage: Literal["global"] = Field(
        default="global", description="The spatial coverage of the dataset."
    )
    variables: list[str] = Field(description="The names of the variables present in the dataset.")
