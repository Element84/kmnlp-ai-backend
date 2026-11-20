import pytest
from pydantic_ai.models.test import TestModel

from llm_agent.agents.scientific_python_agent import ScientificPythonAgent

SETUP_CODE = """\
import os
import fsspec
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from pprint import pprint

cluster = LocalCluster()
client = cluster.get_client()

zarr_reference_path = os.getenv('ZARR_REFERENCE_PATH')
mapper = fsspec.get_mapper(
    'reference://',
    fo=zarr_reference_path,
    remote_protocol='s3',
)
ds = xr.open_zarr(mapper, consolidated=False)
ds = ds.unify_chunks()

print('Dataset:')
print(ds)
print('----------')

print('Data variables in the dataset:')
for k, v in ds.data_vars.items():
    print(k)
    print(v)
    print('----------')

print('Dataset attributes:')
pprint(ds.attrs)
"""

SETUP_OUTPUT = """\
Dataset:
<xarray.Dataset> Size: 745GB
Dimensions:           (time: 1796, lat: 3600, lon: 7200)
Coordinates:
  * lat               (lat) float32 14kB -89.97 -89.93 -89.88 ... 89.93 89.97
  * lon               (lon) float32 29kB -180.0 -179.9 -179.9 ... 179.9 180.0
  * time              (time) datetime64[ns] 14kB 2020-01-01T12:00:00 ... 2024...
Data variables:
    analysed_sst      (time, lat, lon) float64 372GB dask.array<chunksize=(1, 900, 1800), meta=np.ndarray>
    crs               (time) float32 7kB dask.array<chunksize=(1,), meta=np.ndarray>
    sea_ice_fraction  (time, lat, lon) float64 372GB dask.array<chunksize=(1, 900, 1800), meta=np.ndarray>
Attributes: (12/58)
    Conventions:                CF-1.6, ACDD-1.3
    acknowledgement:            NOAA Coral Reef Watch Program
    cdm_data_type:              Grid
    comment:                    This product is designed to improve on and re...
    contributor_name:           NOAA Coral Reef Watch Program
    contributor_role:           Collecting source data and deriving products;...
    ...                         ...
    time_coverage_duration:     P1D
    time_coverage_end:          20200102T000000Z
    time_coverage_resolution:   P1D
    time_coverage_start:        20200101T000000Z
    title:                      NOAA Coral Reef Watch Daily Global 5km Satell...
    uuid:                       36c339e6-c901-11e7-8a08-38d54702cff0
----------
Data variables in the dataset:
analysed_sst
<xarray.DataArray 'analysed_sst' (time: 1796, lat: 3600, lon: 7200)> Size: 372GB
dask.array<rechunk-merge, shape=(1796, 3600, 7200), dtype=float64, chunksize=(1, 900, 1800), chunktype=numpy.ndarray>
Coordinates:
  * lat      (lat) float32 14kB -89.97 -89.93 -89.88 ... 89.88 89.93 89.97
  * lon      (lon) float32 29kB -180.0 -179.9 -179.9 ... 179.9 179.9 180.0
  * time     (time) datetime64[ns] 14kB 2020-01-01T12:00:00 ... 2024-11-30T12...
Attributes:
    coverage_content_type:  physicalMeasurement
    grid_mapping:           crs
    long_name:              analysed sea surface temperature
    standard_name:          sea_surface_temperature
    units:                  degrees_Celsius
    valid_max:              5000
    valid_min:              -200
----------
crs
<xarray.DataArray 'crs' (time: 1796)> Size: 7kB
dask.array<open_dataset-crs, shape=(1796,), dtype=float32, chunksize=(1,), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) datetime64[ns] 14kB 2020-01-01T12:00:00 ... 2024-11-30T12...
Attributes:
    comment:             This is a container variable that describes the grid...
    epsg_code:           EPSG:32663
    grid_mapping_name:   latitude_longitude
    inverse_flattening:  298.2572021484375
    semi_major_axis:     6378137.0
----------
sea_ice_fraction
<xarray.DataArray 'sea_ice_fraction' (time: 1796, lat: 3600, lon: 7200)> Size: 372GB
dask.array<rechunk-merge, shape=(1796, 3600, 7200), dtype=float64, chunksize=(1, 900, 1800), chunktype=numpy.ndarray>
Coordinates:
  * lat      (lat) float32 14kB -89.97 -89.93 -89.88 ... 89.88 89.93 89.97
  * lon      (lon) float32 29kB -180.0 -179.9 -179.9 ... 179.9 179.9 180.0
  * time     (time) datetime64[ns] 14kB 2020-01-01T12:00:00 ... 2024-11-30T12...
Attributes:
    comment:                0 is 0% ice, 1 is 100% ice
    coverage_content_type:  physicalMeasurement
    grid_mapping:           crs
    long_name:              Sea Ice Fraction
    standard_name:          sea_ice_area_fraction
    units:                  1
    valid_max:              100
    valid_min:              0
----------
Dataset attributes:
{'Conventions': 'CF-1.6, ACDD-1.3',
 'acknowledgement': 'NOAA Coral Reef Watch Program',
 'cdm_data_type': 'Grid',
 'comment': 'This product is designed to improve on and replace the use of '
            'AVHRR Pathfinder SST for use within the Coral Reef Watch Program.',
 'contributor_name': 'NOAA Coral Reef Watch Program',
 'contributor_role': 'Collecting source data and deriving products; performing '
                     'quality control of products; disseminating, storing, and '
                     'submitting data to archive',
 'creator_email': 'coralreefwatch@noaa.gov',
 'creator_institution': 'NOAA/NESDIS/STAR Coral Reef Watch Program',
 'creator_name': 'NOAA Coral Reef Watch Program',
 'creator_type': 'group',
 'creator_url': 'https://coralreefwatch.noaa.gov/',
 'data_source': 'NOAA Daily Global 5km Geo-Polar Blended Night-only Sea '
                'Surface Temperature Analysis from the date specified in the '
                'global attribute time_coverage_start. Note, if the text of '
                'this global attribute begins with "Due to ...", one of the '
                'following situations occurred: (1) the source data file for '
                'the CoralTemp of data file for the CoralTemp of the day was '
                'missing; (2) the sea_ice_fraction data array in the source '
                'data was missing, (3) some alternation was made on the source '
                'data to derive the CoralTemp data of the day.',
 'date_created': '20180101T000000Z',
 'date_issued': '20200102T194713Z',
 'date_metadata_modified': '20190411T000000Z',
 'date_modified': '20180101T000000Z',
 'geospatial_bounds': 'POLYGON((-90.0 180.0, 90.0 180.0, 90.0 -180.0, -90.0 '
                      '-180.0, -90.0 180.0))',
 'geospatial_bounds_crs': 'EPSG:32663',
 'geospatial_lat_max': 89.9749984741211,
 'geospatial_lat_min': -89.9749984741211,
 'geospatial_lat_resolution': 0.05000000074505806,
 'geospatial_lat_units': 'degrees_north',
 'geospatial_lon_max': 179.97500610351562,
 'geospatial_lon_min': -179.97500610351562,
 'geospatial_lon_resolution': 0.05000000074505806,
 'geospatial_lon_units': 'degrees_east',
 'history': 'This is the first version of CoralTemp. It was originally called '
            'v1.0 and then renamed to v3.1 with no change to the overall '
            'product)',
 'id': 'CoralTemp-v3.1',
 'institution': 'NOAA/NESDIS/STAR Coral Reef Watch Program',
 'instrument': 'ATSR-1, ATSR-2, AATSR, AVHRR, AVHRR-2, AVHRR-3, VIIRS, GOES '
               'Imager, MTSAT Imager, MTSAT 2 Imager, AHI, ABI, SEVIRI, buoy - '
               'moored buoy, buoy - drifting buoy, buoy - TAO buoy, surface '
               'seawater intake',
 'instrument_vocabulary': 'NOAA NODC Ocean Archive System Instruments',
 'keywords': 'Oceans > Ocean Temperature > Sea Surface Temperature, Oceans > '
             'Ocean Temperature > Water Temperature, Spectral/Engineering > '
             'Infrared Wavelengths > Thermal Infrared',
 'keywords_vocabulary': 'NASA Global Change Master Directory (GCMD) Science '
                        'Keywords',
 'metadata_link': 'https://coralreefwatch.noaa.gov/satellite/bleaching5km',
 'naming_authority': 'gov.noaa.coralreefwatch',
 'ncei_template_version': 'NCEI_NetCDF_Grid_Template_v2.0',
 'platform': 'Ships, drifting buoys, moored buoys, TOGA-TAO buoy arrays, '
             'GOES-8 satellite, GOES-9 satellite, GOES-10 satellite, GOES-11 '
             'satellite, GOES-12 satellite, GOES-13 satellite, GOES-14 '
             'satellite, GOES-15 satellite, GOES-16 satellite, MTSAT-1R '
             'satellite, MTSAT-2 satellite, Himawari-8 satellite, Meteosat-8 '
             'satellite, Meteosat-9 satellite, Meteoset-10 satellite, '
             'Meteosat-11 satellite, Suomi NPP, MetOp-A satellite, MetOp-B '
             'satellite, NOAA-9 satellite, NOAA-11 satellite, NOAA-12 '
             'satellite, NOAA-14 satellite, NOAA-15 satellite, NOAA-16 '
             'satellite, NOAA-17 satellite, NOAA-18 satellite, NOAA-19 '
             'satellite.',
 'platform_vocabulary': 'NOAA NODC Ocean Archive System Platforms',
 'processing_level': 'L4',
 'product_version': '3.1',
 'program': 'NOAA Coral Reef Watch Program',
 'project': 'NOAA Coral Reef Watch Program',
 'publisher_email': 'coralreefwatch@noaa.gov',
 'publisher_institution': 'NOAA/NESDIS/STAR Coral Reef Watch Program',
 'publisher_name': 'NOAA Coral Reef Watch Program',
 'publisher_type': 'group',
 'publisher_url': 'https://coralreefwatch.noaa.gov/',
 'references': 'Donlon, et al., 2011. The Operational Sea Surface Temperature '
               'and Sea Ice analysis (OSTIA). Maturi, et al., 2017. A new '
               'high-resolution sea surface temperature analysis. '
               'https://coralreefwatch.noaa.gov/satellite/coraltemp.php',
 'source': 'OSTIA Sea Surface Temperature Reanalysis (night-only), NOAA '
           'Geo-Polar Blended Night-only Sea Surface Temperature Reanalysis, '
           'NOAA Geo-Polar Blended Night-only Sea Surface Temperature (near '
           'real-time)',
 'standard_name_vocabulary': 'CF Standard Name Table v27',
 'summary': 'CoralTemp is derived from three different but related 5km daily '
            'gap-free SST data sets and provides an internally consistent SST '
            'product that stretches from 1985 to present: OSTIA Sea Surface '
            'Temperature Reanalysis (1985-2002), Geo-Polar Blended Night-only '
            'Sea Surface Temperature Reanalysis (2002-2016), Geo-Polar Blended '
            'Night-only Sea Surface Temperature Near Real-Time (2017 to '
            'present).',
 'time_coverage_duration': 'P1D',
 'time_coverage_end': '20200102T000000Z',
 'time_coverage_resolution': 'P1D',
 'time_coverage_start': '20200101T000000Z',
 'title': 'NOAA Coral Reef Watch Daily Global 5km Satellite Sea Surface '
          'Temperature (CoralTemp)',
 'uuid': '36c339e6-c901-11e7-8a08-38d54702cff0'}
"""  # noqa: E501


@pytest.mark.asyncio
async def test_response_parsing_for_code_solution() -> None:
    response_text = """
lorem
<thoughts>__thoughts__</thoughts>
ipsum
<code>__code__</code>
dolor
<explanation>__explanation__</explanation>
amet
"""
    agent = ScientificPythonAgent(
        TestModel(custom_output_text=response_text),
        setup_code=SETUP_CODE,
        setup_output=SETUP_OUTPUT,
    )
    solution = await agent.run("What is the mean SST value over the entire dataset?")

    assert solution.thoughts == "__thoughts__"
    assert solution.code == "__code__"
    assert solution.explanation == "__explanation__"


@pytest.mark.asyncio
async def test_response_parsing_for_noncode_solution() -> None:
    response_text = """
lorem
<thoughts>__thoughts__</thoughts>
ipsum
<code></code>
dolor
<explanation>__explanation__</explanation>
amet
"""
    agent = ScientificPythonAgent(
        TestModel(custom_output_text=response_text),
        setup_code=SETUP_CODE,
        setup_output=SETUP_OUTPUT,
    )
    solution = await agent.run("Which variables are available in the dataset?")

    assert solution.thoughts == "__thoughts__"
    assert solution.code == ""
    assert solution.explanation == "__explanation__"
