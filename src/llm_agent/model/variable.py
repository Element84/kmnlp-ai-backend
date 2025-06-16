from pydantic import BaseModel, Field, field_serializer


class Variable(BaseModel):
    """A variable represented in the knowledge graph."""

    iri: str = Field(
        description="The URI of the variable in the knowledge graph."
    )
    name: str = Field(description=("Variable name."))
    field_name: str = Field(
        description=(
            "The machine-readable name of a field or variable as used in data "
            "storage and query systems."
        )
    )
    description: str | None = Field(description="Variable description.")
    units: str | None = Field(description="Units of the data.")
    dimensions: frozenset[str] = Field(description="Dimensions of the data.")

    @field_serializer("dimensions", when_used="json")
    def _serialize_dims_sorted(self, dims: frozenset[str]) -> list[str]:
        """Sets serialize to JSON lists. Sort to make order predictable.

        This helps with testing.
        """
        return sorted(dims)
