import numpy as np
from pydantic import BaseModel


class SummaryStats(BaseModel):
    """Summary statistics."""

    mean: float
    std: float
    min: float
    max: float
    values: list[float]

    @staticmethod
    def from_values(values: list[float]) -> "SummaryStats":
        """Populate from a list of values."""
        values_arr = np.array(values)
        return SummaryStats(
            mean=float(values_arr.mean()),
            std=float(values_arr.std()),
            min=float(values_arr.min()),
            max=float(values_arr.max()),
            values=values,
        )
