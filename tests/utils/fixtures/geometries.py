import json
from importlib.resources import files
from typing import Any

import pytest
from e84_geoai_common.geometry import geometry_from_geojson
from shapely.geometry.base import BaseGeometry

import tests.utils.fixtures.geojson

_POLYGON_DIR = files(tests.utils.fixtures.geojson)
_BLACK_SEA_POLYGON_FILE_PATH = _POLYGON_DIR / "black_sea.json"
_GULF_OF_MEXICO_POLYGON_FILE_PATH = _POLYGON_DIR / "gulf_of_mexico.json"


@pytest.fixture(scope="session")
def gulf_of_mexico_geometry() -> BaseGeometry:
    with _GULF_OF_MEXICO_POLYGON_FILE_PATH.open() as f:
        return geometry_from_geojson(f.read())


@pytest.fixture(scope="session")
def gulf_of_mexico_geojson() -> str:
    with _GULF_OF_MEXICO_POLYGON_FILE_PATH.open() as f:
        return f.read()


@pytest.fixture(scope="session")
def gulf_of_mexico_geojson_dict() -> dict[str, Any]:
    with _GULF_OF_MEXICO_POLYGON_FILE_PATH.open() as f:
        return json.loads(f.read())


@pytest.fixture(scope="session")
def black_sea_geometry() -> BaseGeometry:
    with _BLACK_SEA_POLYGON_FILE_PATH.open() as f:
        return geometry_from_geojson(f.read())
