import numpy as np
import pytest
import rioxarray  # noqa: F401 # type: ignore[unused-ignore]
import xarray as xr


@pytest.fixture
def lat_lon_data_array() -> xr.DataArray:
    numpy_array = np.arange(100).reshape(10, 10).astype(float)  # min=0.0, max=99.0

    # Define spatial coordinates
    lon_coords = np.linspace(0, 10, 10)  # longitude
    lat_coords = np.linspace(0, 10, 10)  # latitude

    # Create DataArray with spatial dimensions and coordinates
    data_array: xr.DataArray = xr.DataArray(
        numpy_array,
        coords={"lat": lat_coords, "lon": lon_coords},
        dims=("lat", "lon"),
        name="some name",
        attrs={"units": "example unit"},
    )

    # Set CRS and spatial information using rioxarray
    data_array.rio.write_crs("EPSG:4326", inplace=True)

    return data_array


@pytest.fixture
def sst_data_array() -> xr.DataArray:
    # Create 100 days of time data using NumPy
    time = np.arange(
        np.datetime64("2025-01-01"),
        np.datetime64("2025-01-01") + np.timedelta64(100, "D"),
        dtype="datetime64[D]",
    )
    # Black Sea region: ~27-42°E, ~41-47°N
    # Gulf of Mexico: ~80-98°W, ~18-30°N
    lat = np.linspace(-90, 90, 360)  # 0.5 degree resolution
    lon = np.linspace(-180, 180, 720)  # 0.5 degree resolution

    # Create deterministic SST values using seeded random generator
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(-2, 40, (len(time), len(lat), len(lon)))

    return xr.DataArray(
        data=data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="analysed_sst",
    )


@pytest.fixture
def wind_speed_data_array() -> xr.DataArray:
    # Create 100 days of time data using NumPy, each with 8 data points per day
    time = np.arange(
        np.datetime64("2025-01-01"),
        np.datetime64("2025-01-01") + np.timedelta64(100, "D"),
        step=np.timedelta64(3, "h"),
        dtype="datetime64[h]",
    )
    # Black Sea region: ~27-42°E, ~41-47°N
    # Gulf of Mexico: ~80-98°W, ~18-30°N
    lat = np.linspace(-90, 90, 360)  # 0.5 degree resolution
    lon = np.linspace(-180, 180, 720)  # 0.5 degree resolution

    # Create deterministic SST values using seeded random generator
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(-2, 40, (len(time), len(lat), len(lon)))

    return xr.DataArray(
        data=data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="analysed_sst",
    )
