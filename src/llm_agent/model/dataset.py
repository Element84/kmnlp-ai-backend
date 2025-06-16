from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from llm_agent.model.variable import Variable


class Dataset(BaseModel):
    """A dataset represented in the knowledge graph."""

    iri: str = Field(
        description="The URI of the variable in the knowledge graph."
    )
    source: str = Field(
        description="The URL at which the dataset can be accessed."
    )
    label: str = Field(
        description=(
            "Natural language label for the dataset in the knowledge graph."
        ),
    )
    description: str | None = Field(
        default=None, description=("Dataset description.")
    )
    temporal_coverage: tuple[datetime, datetime] = Field(
        description="The start and end times of the period covered by the "
        "dataset."
    )
    temporal_resolution: timedelta | None = Field(
        default=None,
        description=(
            "The temporal resolution of the dataset: the time interval "
            "between data points."
        ),
    )
    # hard-coded for now
    spatial_coverage: Literal["global"] = Field(
        default="global", description="The spatial coverage of the dataset."
    )
    variables: list[Variable] = Field(
        description="The variables present in the dataset."
    )
