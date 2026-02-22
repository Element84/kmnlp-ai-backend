from pydantic import BaseModel


class GeoTIFFRef(BaseModel):
    """Data type representing a reference to a GeoTIFF file in S3."""

    url: str
    label: str
    units: str | None
    min_value: float | None
    max_value: float | None


class GeoTIFFUploadError(Exception):
    """Failure uploading GeoTIFF to S3 location."""


class GeoTIFFGenerationError(Exception):
    """Failure generating GeoTIFF from data array."""
