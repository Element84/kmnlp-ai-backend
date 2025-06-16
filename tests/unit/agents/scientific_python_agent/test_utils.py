import numpy as np
import pandas as pd  # type: ignore[reportMissingStubs]
import xarray as xr
from shapely.geometry import box

from llm_agent.agents.scientific_python_agent.utils.agent_code_utils import (
    filter_data_array,
    geographic_coords_to_decimal_coords,
    max_coords,
    min_coords,
    polygon_to_mask,
)
from llm_agent.agents.scientific_python_agent.utils.misc import (
    extract_tag_content,
    get_function_def,
    parse_code_solution,
)


def build_dummy_data_array() -> xr.DataArray:
    """Returns a (10, 10, 10) shaped (time, lat, lon) data array."""
    # '2025-01-01', '2025-01-02', ..., '2025-01-10'
    time = pd.date_range("2025-01-01", periods=10, freq="D")  # type: ignore[reportUnknownMemberType]
    # -90, -70, -50, -30, -10,  10,  30,  50,  70,  90
    lat = np.linspace(-90, 90, 10)
    # -180, -140, -100,  -60,  -20,   20,   60,  100,  140, 180
    lon = np.linspace(-180, 180, 10)

    rng = np.random.default_rng()
    random_data = rng.random((len(time), len(lat), len(lon)))

    da = xr.DataArray(
        data=random_data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="dummy_data",
    )
    return da


def test_extract_tag_content():
    text = "abc<code>def</code>xyz"
    assert extract_tag_content(text, "code") == "def"
    text = "abc<code>def\n\nghi</code>xyz"
    assert extract_tag_content(text, "code") == "def\n\nghi"
    assert extract_tag_content(text, "abc") == ""


def test_parse_code_solution_basic():
    text = """
lorem
<thoughts>
__thoughts__
</thoughts>
ipsum
<code>__code__</code>
dolor
<explanation>
__explanation__
</explanation>
amet
"""
    parsed_soln = parse_code_solution(text)
    assert parsed_soln.thoughts == "__thoughts__"
    assert parsed_soln.code == "__code__"
    assert parsed_soln.explanation == "__explanation__"


def test_parse_code_solution_with_codeblock():
    text = """
lorem
<thoughts>
__thoughts__
</thoughts>
ipsum
<code>
```python
__code__
```
</code>
dolor
<explanation>
__explanation__
</explanation>
amet
"""
    parsed_soln = parse_code_solution(text)
    assert parsed_soln.thoughts == "__thoughts__"
    assert parsed_soln.code == "__code__"
    assert parsed_soln.explanation == "__explanation__"


def test_parse_code_solution_empty():
    text = ""
    parsed_soln = parse_code_solution(text)
    assert parsed_soln.thoughts == ""
    assert parsed_soln.code == ""
    assert parsed_soln.explanation == ""


def dummy_func(a: int, b: str) -> int:
    """Docstring.

    Args:
        a (int): a.
        b (str): b.

    Returns:
        int: return value.
    """
    raise NotImplementedError


def test_get_function_def():
    expected_output = """\
def dummy_func(a: int, b: str) -> int:
    \"\"\"
    Docstring.

    Args:
        a (int): a.
        b (str): b.

    Returns:
        int: return value.
    \"\"\""""
    assert get_function_def(dummy_func) == expected_output


def test_filter_data_array():
    da = build_dummy_data_array()

    da_filtered = filter_data_array(da, time=slice(None, "2025-01-05"))
    assert da_filtered.shape == (5, 10, 10)

    da_filtered = filter_data_array(da, lat=slice(-50, 50), lon=slice(-30, 30))
    assert da_filtered.shape == (10, 6, 2)


def test_max_coords():
    da = build_dummy_data_array()
    da.data[2, 1, 2] = 100.0
    expected_output = {
        "time": "2025-01-03T00:00:00.000000000",
        "lat": -70.0,
        "lon": -100.0,
        "value": 100.0,
    }
    assert max_coords(da) == expected_output


def test_min_coords():
    da = build_dummy_data_array()
    da.data[3, 4, 1] = -100
    expected_output = {
        "time": "2025-01-04T00:00:00.000000000",
        "lat": -10.0,
        "lon": -140.0,
        "value": -100.0,
    }
    assert min_coords(da) == expected_output


def test_geographic_coords_to_decimal_coords():
    assert geographic_coords_to_decimal_coords("20N") == 20.0
    assert geographic_coords_to_decimal_coords("20S") == -20.0
    assert geographic_coords_to_decimal_coords("50E") == 50.0
    assert geographic_coords_to_decimal_coords("50W") == -50.0


def test_polygon_to_mask():
    da = build_dummy_data_array()
    _, h, w = da.shape
    polygon = box(-60, 10, 60, 50)
    mask = polygon_to_mask(polygon, da)
    expected_mask = np.zeros((1, h, w), dtype=np.bool_)
    expected_mask[:, 5:8, 3:7] = True
    assert np.all(mask == expected_mask)
