from pydantic import BaseModel, Field


class CanonicalDimension(BaseModel):
    """A canonical dimension represented in the knowledge graph."""

    id: str = Field(description="Dimension ID.")
    cf_standard_name: str | None = Field(
        default=None, description="CF standard name of the dimension."
    )
    alternate_names: frozenset[str] = Field(
        default_factory=frozenset, description="Alternate names for the dimension."
    )

    def is_temporal(self) -> bool:
        """Check if dimension is temporal (i.e., has a temporal standard name)."""
        return self.cf_standard_name == "time" or "time" in self.alternate_names


class Dimension(CanonicalDimension):
    """A dimension of a dataset represented in the knowledge graph."""

    field_name: str = Field(description="Name of the dimension in the dataset.")
