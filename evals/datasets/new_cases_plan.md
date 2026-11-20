# AlgorithmAgent eval — plan for new cases

Plan for expanding `algorithm_agent.py` from 15 → ~45 cases. **Questions only** here; reference
solutions come later. Each case lists the inputs, requested outputs, the ops it is meant to
exercise, and gotchas.

## Guiding principles (recap)

- **Scoring is bag-of-ops.** The grader compares the multiset of _function_ class names
  (`type(func).__name__`), plus refusal correctness and statement/output counts. So a case is worth
  adding when it exercises a _function_ not yet covered, not merely a new phrasing. Plot-spec choice
  (`SST` vs `GeneralRaster` vs `Geospatial`) is invisible to the grader (all are `MakePlot`).
- **Cover the op families, weight by realism.** The current 15 lean on subset → reduce → plot. The
  gaps are the scientifically meaningful ops: climatology/anomaly, trend + significance,
  correlation, resample, rolling/cumulative, mask, regrid, percentile rank — plus the untouched plot
  types (histogram, scatter, multi-line, contour, wind vectors, ensemble plume, animation, geotiff).
- **Multiple reference solutions** where a query has >1 genuinely-valid op decomposition (the
  grader now scores against the best-matching reference).
- **Difficulty spread + a real negative region** (refusals / ambiguity ≈ 10–15% of the set).

## New input variables to import

All are `lat/lon/time` DataArrays unless noted. Add to `_input_vars` (suggested dict key → KG id):

| key                   | KG variable id                                                                                     | notes                                                   |
| --------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `sst_whoi`            | `sea_surface_temperature_noaa-cdr-sea-surface-temp-whoi-pds_daily`                                 | 2nd SST product (different grid → regrid to compare)    |
| `sea_ice`             | `sea_ice_area_fraction_crw_5km_sst_daily`                                                          | **same grid as `sst`** (mask without regrid)            |
| `sst_monthly`         | `sea_surface_temperature_crw_5km_sst_monthly_mean`                                                 | pre-aggregated monthly                                  |
| `sst_annual_mean`     | `sea_surface_temperature_crw_5km_sst_annual_mean`                                                  | pre-aggregated annual (clean trends)                    |
| `precip_cmorph`       | `lwe_precipitation_rate_noaa-cdr-precip-cmorph-pds_daily`                                          | mm/day                                                  |
| `precip_gpcp`         | `lwe_precipitation_rate_noaa-cdr-precip-gpcp-daily-pds_daily`                                      | mm/day (diff grid from cmorph)                          |
| `precip_gpcp_monthly` | `precipitation_amount_noaa-cdr-precip-gpcp-monthly-pds_monthly`                                    | monthly                                                 |
| `ndvi`                | `normalized_difference_vegetation_index_noaa-cdr-ndvi-pds_daily`                                   | vegetation index (unitless)                             |
| `lai`                 | `leaf_area_index_noaa-cdr-leaf-area-index-fapar-pds_daily`                                         | leaf area index                                         |
| `aerosol`             | `atmosphere_optical_thickness_due_to_ambient_aerosol_noaa-cdr-aerosol-optical-thickness-pds_daily` | AOT                                                     |
| `latent_heat`         | `surface_upward_latent_heat_flux_noaa-cdr-ocean-heatflux-pds_daily`                                | W/m²                                                    |
| `sensible_heat`       | `surface_upward_sensible_heat_flux_noaa-cdr-ocean-heatflux-pds_daily`                              | **same grid as `latent_heat`**                          |
| `wind_u`              | `wind_u_10m_noaa-gefs-analysis`                                                                    | **same grid as `wind_v`**                               |
| `wind_v`              | `wind_v_10m_noaa-gefs-analysis`                                                                    | pair for wind-vector/speed                              |
| `temp2m`              | `temperature_2m_noaa-gefs-analysis`                                                                | 2 m air temp                                            |
| `temp2m_ens`          | `temperature_2m_ecmwf-ifs-ens-forecast-15-day-0-25-degree`                                         | dims: `ensemble_member, init_time, lat, lead_time, lon` |
| `tsi_yearly`          | `solar_irradiance_noaa-cdr-total-solar-irradiance-pds_yearly`                                      | time-only, long record                                  |

## Easy tier (single dataset, 1–4 ops, one output)

| id                    | question                                                                                      | inputs                         | output                | new ops                    | notes                       |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------------ | --------------------- | -------------------------- | --------------------------- |
| `histogram_sst`       | "Plot a histogram of daily SST values in the Gulf of Mexico during July 2024."                | `sst`                          | histogram             | `HistogramPlotSpec`        |                             |
| `scatter_heat_fluxes` | "Make a scatter plot of sensible vs. latent heat flux over the North Atlantic on 2020-06-15." | `sensible_heat`, `latent_heat` | scatter_plot          | `ScatterPlotSpec`          | same grid → no regrid       |
| `argmin_coldest_sst`  | "Find the coldest SST in the dataset and report its value, date, and location."               | `sst`                          | 4× scalar/str/lat/lon | `DataArrayArgminCoords`    | mirror of `argmax_basic`    |
| `geotiff_med_sst`     | "Export the SST map of the Mediterranean Sea on 2024-08-01 as a GeoTIFF for the map view."    | `sst`                          | geotiff               | `CreateGeoTIFF`            | "show on the map" ⇒ geotiff |
| `geospatial_ndvi`     | "Show a geospatial map of NDVI over sub-Saharan Africa on 2022-09-01."                        | `ndvi`                         | raster_plot           | `GeospatialRasterPlotSpec` |                             |
| `contour_sst`         | "Draw a contour map of SST over the North Atlantic on 2024-01-15."                            | `sst`                          | raster_plot           | `ContourMapPlotSpec`       |                             |

## Medium tier (2–5 ops, one meaningful transform)

| id                          | question                                                                                                               | inputs                         | output             | new ops                                                        | notes                                           |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------ | -------------------------------------------------------------- | ----------------------------------------------- |
| `monthly_climatology_sst`   | "Compute the monthly SST climatology for the Caribbean Sea (2010–2020) and show it as a line plot."                    | `sst`                          | line_plot          | `ComputeClimatology` (+ spatial reduce)                        |                                                 |
| `sst_anomaly_map`           | "Map the August-2023 SST anomaly relative to the 2010–2020 August climatology in the North Pacific."                   | `sst`                          | raster_plot        | `ComputeClimatology` + `ComputeAnomaly` + `SSTAnomalyPlotSpec` |                                                 |
| `rolling_mean_precip`       | "Plot a 7-day rolling mean of Gulf-of-Mexico daily precipitation over 2023."                                           | `precip_gpcp`                  | line_plot          | `RollingReduce` (+ spatial reduce)                             |                                                 |
| `precip_total_map`          | "Map total accumulated precipitation over the Amazon basin during January 2024."                                       | `precip_gpcp`                  | raster_plot        | `ReduceDataArray(op=sum, dim=time)`                            | sum (not mean)                                  |
| `cumulative_precip_series`  | "Show the running-total (cumulative) precipitation over the Sahel through the 2023 wet season as a time series."       | `precip_cmorph`                | line_plot          | `CumulativeReduce` (+ spatial reduce)                          |                                                 |
| `resample_precip_monthly`   | "Resample daily CMORPH precipitation to monthly totals for the Bay of Bengal in 2023 and plot the monthly series."     | `precip_cmorph`                | line_plot          | `ResampleTime(month, sum)` (+ spatial reduce)                  | alt: `GroupByTime`                              |
| `summer_mean_sst`           | "Compute the mean summer (JJA) SST in the Mediterranean averaged across 2015–2024."                                    | `sst`                          | raster_plot/scalar | `FilterToTimePeriodAcrossYears`                                |                                                 |
| `percentile_rank_sst`       | "For the Mediterranean, rank the 2024-07-15 SST against the 2000–2024 July distribution (percentile rank) and map it." | `sst`                          | raster_plot        | `ComputePercentileRank`                                        |                                                 |
| `mask_sst_by_ice`           | "Map Arctic SST on 2024-03-01, masking out cells where sea ice is present."                                            | `sst`, `sea_ice`               | raster_plot        | `MaskWhere` (condition from another var)                       | same grid                                       |
| `multiline_precip_products` | "Compare daily Sahel precipitation from CMORPH and GPCP as two lines on one plot for August 2023."                     | `precip_cmorph`, `precip_gpcp` | line_plot          | `MultiLinePlotSpec`                                            | each reduced to a per-day series ⇒ no regrid    |
| `wind_vector_map`           | "Show a 10 m wind-vector (quiver) map over the Caribbean on 2023-09-10."                                               | `wind_u`, `wind_v`             | raster_plot        | `WindVectorPlotSpec`                                           | same grid                                       |
| `wind_speed_map`            | "Compute and map the 10 m wind speed from its u/v components over the U.S. East Coast on 2023-01-15."                  | `wind_u`, `wind_v`             | raster_plot        | `UnaryOperation(sqrt)` + `BinaryOperation`                     | speed = √(u²+v²)                                |
| `animation_monthly_sst`     | "Create an animation of monthly-mean SST over the tropical Pacific through 2023."                                      | `sst_monthly`                  | animation          | `MakeAnimation`                                                |                                                 |
| `ndvi_seasonal_anomaly`     | "Map the NDVI anomaly for spring 2022 over the U.S. Corn Belt vs. the 2015–2021 spring climatology."                   | `ndvi`                         | raster_plot        | `ComputeClimatology` + `ComputeAnomaly` (non-SST)              | reuses climatology path on a different variable |

## Hard tier (multi-step / multi-dataset / alignment)

| id                          | question                                                                                                                                           | inputs                         | output                | new ops                                                                                      | notes                                     |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | --------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `warming_trend_map`         | "Map the linear SST warming trend (°C/decade) over the North Atlantic, 2000–2024."                                                                 | `sst_annual_mean`              | raster_plot           | `ComputeLinearTrend`                                                                         | annual input ⇒ clean trend                |
| `trend_significance_masked` | "Map the North-Pacific SST trend (1990–2024), masking cells where it is not significant (p ≥ 0.05)."                                               | `sst_annual_mean`              | raster_plot           | `ComputeTrendPValue` + `ComputeLinearTrend` + `MaskWhere`                                    |                                           |
| `correlation_sst_products`  | "Map the temporal correlation between the CRW and WHOI SST products over the Atlantic during 2022 (regrid to a common grid first)."                | `sst`, `sst_whoi`              | raster_plot           | `RegridToMatch` + `CorrelateDataArrays`                                                      |                                           |
| `correlation_ndvi_temp`     | "Map the correlation between monthly NDVI and 2 m temperature over the U.S. Great Plains in 2022, aligning grids and cadence first."               | `ndvi`, `temp2m`               | raster_plot           | `ResampleTimeToMatch` + `RegridToMatch` + `CorrelateDataArrays`                              | hardest alignment case                    |
| `bowen_ratio_map`           | "Map the Bowen ratio (sensible ÷ latent heat flux) over the Gulf Stream on 2020-07-01, masking near-zero latent flux."                             | `sensible_heat`, `latent_heat` | raster_plot           | `MaskWhere` + `BinaryOperation(div)`                                                         | same grid; mask avoids divide blow-ups    |
| `heat_flux_extreme`         | "Where and when was total turbulent heat flux (latent + sensible) highest over the Kuroshio in winter 2020? Report value, date, location."         | `latent_heat`, `sensible_heat` | 4× scalar/str/lat/lon | `BinaryOperation(add)` + `DataArrayArgmaxCoords`                                             |                                           |
| `ensemble_plume_temp`       | "Show an ensemble plume of the 15-day 2 m-temperature forecast for London from the latest init."                                                   | `temp2m_ens`                   | line_plot             | `EnsemblePlumePlotSpec` (+ spatial reduce over lat/lon)                                      | dims incl. `ensemble_member`, `lead_time` |
| `ensemble_spread_map`       | "Map the ensemble spread (std across members) of the day-10 2 m-temperature forecast over Europe."                                                 | `temp2m_ens`                   | raster_plot           | `ReduceDataArray(op=std, dim=ensemble_member)`                                               | reduce over the ensemble dim              |
| `sst_dashboard`             | "Build a 2×2 dashboard for the Mediterranean (2013–2023): monthly climatology, an August anomaly map, the annual warming trend, and JJA-mean SST." | `sst`, `sst_annual_mean`       | plot_grid             | `Climatology` + `Anomaly` + `LinearTrend` + `FilterToTimePeriodAcrossYears` + `PlotGridSpec` | integration case                          |
| `rolling_tsi_trend`         | "Plot annual total solar irradiance from 1950–2020 with a 11-year rolling mean overlaid."                                                          | `tsi_yearly`                   | line_plot             | `RollingReduce` on a time-only series (+ `MultiLine`)                                        | time-only variety                         |

## Refusal / ambiguity tier

| id                          | question                                                       | inputs   | expected behavior              | notes                                    |
| --------------------------- | -------------------------------------------------------------- | -------- | ------------------------------ | ---------------------------------------- |
| `refusal_variable_absent`   | "What was the sea-surface salinity in the Baltic Sea in 2023?" | `sst`    | refuse                         | no salinity variable available           |
| `refusal_time_out_of_range` | "Map SST for the year 1850."                                   | `sst`    | refuse                         | far outside temporal coverage            |
| `ambiguous_recent_coast`    | "Show me the recent temperature near the coast."               | `temp2m` | refuse / request clarification | no region, time, or 'recent' definition  |
| `refusal_unsupported_op`    | "Fit a neural network to predict next month's SST."            | `sst`    | refuse                         | out of scope for the available functions |

## Implementation notes (for the solution-writing phase)

- **Function-call types:** most new vars are `lat/lon/time` and work with the existing `"st"` set.
  New keys are needed for differently-dimensioned inputs, e.g. `temp2m_ens`
  (`ensemble_member/init_time/lat/lead_time/lon`) — build a dedicated `input_vars_to_function_set`
  - `make_function_call_model`, and the constrained `Slice/Reduce` names will differ accordingly.
    Multi-input cases just need a `FunctionCall` type covering the union of their inputs' functions.
- **Multiple reference solutions** are expected for: the resample-vs-groupby cases
  (`resample_precip_monthly`), climatology/anomaly (single- vs two-step forms), and any case with a
  spatial-reduce-vs-`GroupByTime` ambiguity — mirror what was done for `time_series_*`.
- **Same-grid pairs** (`sst`+`sea_ice`, `latent_heat`+`sensible_heat`, `wind_u`+`wind_v`) are chosen
  deliberately so masking/ratios/vectors work without `RegridToMatch`; cross-product cases
  (`correlation_sst_products`, `correlation_ndvi_temp`) deliberately require it.
- **Count:** ~29 new cases here (6 easy, 14 medium, 10 hard, 4 refusal) → ~44 total, hitting the
  near-term target and covering every currently-unexercised op family.

## Coverage of previously-unexercised ops

`ComputeClimatology`·`ComputeAnomaly` → climatology/anomaly/dashboard/ndvi cases · `ComputeLinearTrend`
→ `warming_trend_map`,`sst_dashboard` · `ComputeTrendPValue` → `trend_significance_masked` ·
`CorrelateDataArrays` → correlation cases · `RollingReduce` → `rolling_mean_precip`,`rolling_tsi_trend`
· `CumulativeReduce` → `cumulative_precip_series` · `MaskWhere` → `mask_sst_by_ice`,`bowen_ratio_map`,
`trend_significance_masked` · `RegridToMatch` → correlation cases · `ResampleTime`/`ResampleTimeToMatch`
→ `resample_precip_monthly`,`correlation_ndvi_temp` · `FilterToTimePeriodAcrossYears` →
`summer_mean_sst`,`sst_dashboard` · `ComputePercentileRank` → `percentile_rank_sst` ·
`DataArrayArgminCoords` → `argmin_coldest_sst` · `CreateGeoTIFF` → `geotiff_med_sst` ·
`UnaryOperation` → `wind_speed_map` · plots: `Histogram`,`Scatter`,`MultiLine`,`ContourMap`,
`GeospatialRaster`,`WindVector`,`EnsemblePlume`,`MakeAnimation` → respective cases.

## Findings for future function-definition refinements

Constraints/rough edges hit while writing the cases. These don't block the eval work (there's a
workaround for each), but they're candidates for improving the DSL / function definitions later.

- **`ScatterPlotSpec` only accepts 1D `x_data_array`/`y_data_array`, and there is no
  flatten-to-1D op.** So the natural "scatter two 2D fields cell-by-cell" (e.g. sensible vs. latent
  heat flux over a region on one day) is **not expressible**. Workaround used: scatter two 1D
  _time series_ (regional daily means over a period) instead. Possible refinement: let scatter
  accept N-D arrays and flatten/pair internally (dropping NaNs), or add an explicit flatten op.
- **Plot-spec dimensionality is documented but not type-checked.** `ScatterPlotSpec`/`LinePlotSpec`
  say "1D"; `ContourMapPlotSpec`/`CreateGeoTIFF` say "2D (lat, lon)". `type_check` only verifies the
  argument is a DataArray, so a wrong-rank reference _passes_ type-checking and would only fail at
  execution. Workaround: for 2D-required specs, slice time with a **single date value** (which drops
  the time dim) rather than a length-1 range. Possible refinement: enforce expected rank/dims for
  these specs in `type_check_expression`'s `MakePlot`/`CreateGeoTIFF` handlers.
- **Data/naming gotchas (not function defs, but relevant to case authoring):** several variables
  share bare DataArray names across datasets — `sst_whoi`, `sst_monthly`, `sst_annual_mean` are all
  named `sea_surface_temperature`; `precip_cmorph`→`cmorph`, `precip_gpcp`/`precip_gpcp_monthly`→
  `precip`; `temp2m_ens`→`temperature_2m`. `VariableReference(name=...)` must use the **DataArray
  name**, not the `_input_vars` dict key. (`sst`→`analysed_sst`, `sea_ice`→`sea_ice_fraction`,
  `ndvi`→`NDVI`, `aerosol`→`aot1`, etc.)
- **Inconsistent plot-spec annotation API.** Most specs take `annotation: AnnotationSpec`
  (title+xlabel+ylabel), but `SSTAnomalyPlotSpec` and `WindVectorPlotSpec` expose a bare `title:
str` and no `annotation`. Possible refinement: standardize on `annotation` across all plot specs.
- **Type-only plot specs are gated by variable type.** `WindVectorPlotSpec` and
  `EnsemblePlumePlotSpec` are only present in `MakePlot`'s union when the function set is built from
  a matching variable (eastward/northward wind → wind plotting set; a variable with an ensemble dim
  → ensemble plotting set). So `_FunctionCall["st"]` rejects them; cases must build the call model
  from the right variable (we added `_FunctionCall["wind"]` and `_FunctionCall["ens"]`). This is a
  reasonable guardrail, but the failure mode (a pydantic `union_tag_invalid` at construction) is
  opaque — a clearer error would help.
- **Durational dims need ISO-8601 slices.** `lead_time` (a duration) must be sliced with an ISO-8601
  duration string like `"P10D"` for "day 10", not an integer or `"10 days"`.
- **`CorrelateDataArrays` needs a shared time axis, which regridding does not provide.** For two
  products with different temporal coverage/cadence, `RegridToMatch` aligns space but the
  correlation still errors on mismatched time sizes; a `ResampleTimeToMatch` (cadence alignment)
  step is also required. Possible refinement: have `CorrelateDataArrays` auto-align (inner-join) the
  shared dim, or emit a clearer "align time first" error.
