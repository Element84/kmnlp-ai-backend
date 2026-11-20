from typing import Literal

from pydantic import Field

from llm_agent.context import DataRef


class GeoTIFFRef(DataRef, frozen=True):
    """Data type representing a reference to a GeoTIFF file in S3."""

    data_ref_type: Literal["geotiff"] = "geotiff"
    units: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class GeoTIFFUploadError(Exception):
    """Failure uploading GeoTIFF to S3 location."""


class GeoTIFFGenerationError(Exception):
    """Failure generating GeoTIFF from data array."""


class GeometryRef(DataRef, frozen=True):
    """Data type representing a reference to a geometry."""

    data_ref_type: Literal["geometry"] = "geometry"
    geometry_type: (
        Literal[
            "Point",
            "LineString",
            "LinearRing",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
            "FeatureCollection",
        ]
        | None
    ) = None
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box of the geometry as a (min_lon, min_lat, max_lon, max_lat) tuple.",
    )


class ImageRef(DataRef, frozen=True):
    """Data type representing a reference to an image file."""

    data_ref_type: Literal["image"] = "image"


class AnimationRef(DataRef, frozen=True):
    """Data type representing a reference to an animation file."""

    data_ref_type: Literal["animation"] = "animation"
    format: Literal["gif", "mp4"] = "gif"
