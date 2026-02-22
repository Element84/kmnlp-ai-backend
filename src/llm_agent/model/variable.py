from pydantic import BaseModel, Field, field_validator

from llm_agent.model.dimension import Dimension


class Variable(BaseModel):
    """A variable represented in the knowledge graph."""

    id: str = Field(description="Variable ID.")
    name: str = Field(description=("Variable name."))
    field_name: str = Field(
        description=(
            "The machine-readable name of a field or variable as used in data "
            "storage and query systems."
        )
    )
    description: str | None = Field(description="Variable description.")
    units: str | None = Field(description="Units of the data.")
    dimensions: list[Dimension] = Field(description="Dimensions of the data.")

    def is_spatiotemporal(self) -> bool:
        """Check if variable is spatiotemporal (i.e., has both spatial and temporal dimensions)."""
        return self.is_spatial() and self.is_temporal()

    def is_spatial(self) -> bool:
        """Check if variable is spatial (i.e., has lat and lon dimensions)."""
        dim_std_names = {dim.cf_standard_name for dim in self.dimensions}
        return {"latitude", "longitude"}.issubset(dim_std_names)

    def is_temporal(self) -> bool:
        """Check if variable is temporal (i.e., has a temporal dimension)."""
        return any(dim.is_temporal() for dim in self.dimensions)

    @field_validator("dimensions")
    @classmethod
    def _sort_dims(cls, dims: list[Dimension]) -> list[Dimension]:
        """Sort dimensions by their field name for consistent ordering."""
        return sorted(dims, key=lambda d: d.field_name)
