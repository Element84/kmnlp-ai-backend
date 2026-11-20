import inspect
from collections.abc import Iterable
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from llm_agent.agents.algorithm_agent.algorithm.base import Function, VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets import FunctionSet
from llm_agent.agents.algorithm_agent.algorithm.function_sets.utils import concat_camel_case

_SliceBoundsT = tuple[str, str] | tuple[float, float]


####################
# General functions
####################
class BinaryOperation(Function, frozen=True):
    """Apply a binary operation to numeric values or DataArrays, broadcasting as needed."""

    func_name: Literal["BinaryOperation"] = "BinaryOperation"
    left: int | float | VariableReference = Field(description="The left value in the operation.")
    right: int | float | VariableReference = Field(description="The right value in the operation.")
    op: Literal["add", "sub", "mul", "div", "eq", "ne", "gt", "lt", "ge", "le"] = Field(
        description="The binary operation to apply."
    )


class UnaryOperation(Function, frozen=True):
    """Apply an element-wise unary math operation to a numeric value or DataArray.

    Useful for building derived quantities, e.g. wind speed magnitude can be computed as
    sqrt(u**2 + v**2) by squaring the components with BinaryOperation, adding them, and applying
    the 'sqrt' UnaryOperation.
    """

    func_name: Literal["UnaryOperation"] = "UnaryOperation"
    operand: int | float | VariableReference = Field(
        description="The value to apply the operation to. Either a numeric literal or a reference "
        "to a scalar or DataArray variable."
    )
    op: Literal["sqrt", "square", "abs", "exp", "log", "log10", "neg"] = Field(
        description="The element-wise unary operation to apply. 'sqrt' is the square root, "
        "'square' raises to the power of 2, 'abs' is the absolute value, 'exp' is the natural "
        "exponential, 'log' is the natural logarithm, 'log10' is the base-10 logarithm, and 'neg' "
        "negates the value. Note that 'sqrt', 'log', and 'log10' are undefined for negative "
        "inputs and will produce NaNs there."
    )


class MaskWhere(Function, frozen=True):
    """Mask a DataArray, keeping values only where a boolean condition holds (NaN elsewhere).

    Useful for conditional selection, e.g. to keep sea surface temperature only where sea-ice
    concentration is below a threshold. Build the condition with BinaryOperation using a comparison
    op (e.g. 'lt'/'gt'). Masked-out points become NaN and are therefore excluded from subsequent
    reductions such as mean.
    """

    func_name: Literal["MaskWhere"] = "MaskWhere"
    data_array: VariableReference = Field(description="The DataArray to mask.")
    condition: VariableReference = Field(
        description="A boolean DataArray, typically produced by BinaryOperation with a comparison "
        "op like 'gt' or 'lt'. Values in data_array where the condition is False are set to NaN. "
        "Its dimensions must be a subset of the data_array's dimensions."
    )


class CorrelateDataArrays(Function, frozen=True):
    """Compute the Pearson correlation between two DataArrays along a shared dimension.

    Correlates `a` and `b` over `dim` (usually time), pixel by pixel, returning a correlation
    coefficient in [-1, 1]. With two spatiotemporal fields this produces a correlation map; with a
    field and a 1-D index (e.g. an ENSO index built by spatially averaging a region) it correlates
    every pixel against that index. If nothing remains after collapsing `dim`, a scalar is returned.

    IMPORTANT: to avoid the correlation being dominated by a shared seasonal cycle, deseasonalize
    both inputs first (e.g. with ComputeAnomaly). Both inputs must contain `dim`.
    """

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "CorrelateDataArrays"
    a: VariableReference = Field(description="The first DataArray.")
    b: VariableReference = Field(
        description="The second DataArray. Must share `dim` with `a`; its other dims may broadcast."
    )
    dim: str = Field(description="The shared dimension to correlate over (typically time).")

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[CorrelateDataArrays]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim"]
        allowed_dims_type = Literal[*allowed_dims]
        name = f"CorrelateDataArrays_{concat_camel_case(sorted(allowed_dims))}"
        return create_model(
            name,
            __base__=CorrelateDataArrays,
            __doc__=(
                f"{inspect.cleandoc(CorrelateDataArrays.__doc__ or '')}\n\n"
                "The dim name must strictly match an allowed dim."
            ),
            # field defs:
            func_name=Annotated[Literal[name], Field(default=name)],
            dim=Annotated[allowed_dims_type, Field(description=original_field.description)],
        )


class SliceDataArray(Function, frozen=True):
    """Slice a DataArray along given dims."""

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "SliceDataArray"
    data_array: VariableReference = Field(description="The DataArray to slice.")
    dim_slices: dict[str, _SliceBoundsT | str | float] = Field(
        default_factory=dict[str, _SliceBoundsT | str | float],
        description="Mapping from dim names to either (start, end) tuples representing slice "
        "ranges or single values. "
        "Start and end are inclusive. "
        "If the dim is temporal, values must be ISO 8601 datetimes with an optional "
        "timezone-naive time component like 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS', which must "
        "make sense WRT the temporal resolution of the data (if known). "
        "If the dim is a durational time dimension (e.g. forecast lead time), values "
        "should be 0-indexed ISO 8601 durations like ('P0D', 'P2D') for the first 3 days, etc. "
        "For non-temporal dims, the values should be floats. "
        "If a single value, the dim will not be present in the output DataArray; if a range, "
        "and start == end, the dim will be present but with length 1. ",
    )
    dim_index_slices: dict[str, tuple[int, int | None] | int] = Field(
        default_factory=dict[str, tuple[int, int | None] | int],
        description="Mapping from dim names to (start_idx, end_idx) tuples or single integer "
        "indices representing slice "
        "ranges in terms of integer 0-indexed indices. Negative indices not allowed. "
        "Start_idx is inclusive and end_idx is exclusive. "
        "end_idx can also be None to indicate slicing until the end of the dimension. "
        "This is an alternative to dim_slices that allows for slicing by integer index "
        "rather than by coordinate value. Each dimension must only appear in either dim_slices or "
        "dim_index_slices, not both.",
    )
    method: Literal["exact", "nearest"] = Field(
        default="nearest",
        description="Method to use for inexact matches when slicing by single coordinate value. "
        "Only relevant for dim_slices and only when slicing by a single value. If 'exact', an "
        "exact match is required. Defaults to 'nearest'.",
    )

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[SliceDataArray]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim_slices"]
        name = f"SliceDataArray_{concat_camel_case(sorted(allowed_dims))}"
        allowed_dims_type = Literal[*allowed_dims]
        return create_model(
            name,
            __base__=SliceDataArray,
            __doc__="Slice a DataArray along given dims. "
            "Dim names must strictly match allowed dims.",
            # field defs:
            func_name=Annotated[Literal[name], Field(default=name)],
            dim_slices=Annotated[
                # unlike parent class, the dict keys are restricted to allowed
                # dim names
                dict[allowed_dims_type, _SliceBoundsT | str | float],
                Field(
                    default_factory=dict[allowed_dims_type, _SliceBoundsT | str | float],
                    description=original_field.description,
                ),
            ],
            dim_index_slices=Annotated[
                dict[allowed_dims_type, tuple[int, int | None] | int],
                Field(
                    default_factory=dict[allowed_dims_type, tuple[int, int | None] | int],
                    description=SliceDataArray.model_fields["dim_index_slices"].description,
                ),
            ],
        )

    @model_validator(mode="after")
    def _dim_name_exclusivity(self) -> Self:
        """Validate each dim name appears in either dim_slices or dim_index_slices, not both."""
        dim_slices_keys = set(self.dim_slices.keys())
        dim_index_slices_keys = set(self.dim_index_slices.keys())
        overlap = dim_slices_keys.intersection(dim_index_slices_keys)
        if overlap:
            raise ValueError(
                f"Dim names {overlap} appear in both dim_slices and dim_index_slices. "
                "Each dim name must only appear in one."
            )
        return self


class ReduceDataArray(Function, frozen=True):
    """Reduce a DataArray along some dims using an op.

    When op is 'mean' or 'std' and the reduction collapses the latitude dimension ('lat'), the
    reduction is automatically area-weighted by the cosine of latitude. This corrects for the fact
    that lat/lon grid cells cover progressively less area toward the poles, so a plain (unweighted)
    spatial average would over-count polar regions and produce a biased result. The other ops
    (sum/min/max/median/quantile) are never weighted.
    """

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "ReduceDataArray"
    data_array: VariableReference = Field(description="The DataArray to reduce.")
    op: Literal["sum", "mean", "std", "min", "max", "median", "quantile"]
    dim: str | tuple[str, ...] | None = Field(
        default=None,
        description="Dimension(s) to reduce. If None, reduces all dims. If the result is "
        "squeezable to a single value, this function returns a scalar instead of a DataArray.",
    )
    q: float | None = Field(
        default=None,
        description="The quantile to compute, between 0 and 1 inclusive (e.g. 0.9 for the 90th "
        "percentile). Required when op is 'quantile' and must be null for every other op.",
    )

    @model_validator(mode="after")
    def _validate_quantile_arg(self) -> Self:
        """Ensure `q` is provided iff op is 'quantile', and is a valid quantile."""
        if self.op == "quantile":
            if self.q is None:
                raise ValueError("`q` is required when op is 'quantile'.")
            if not 0 <= self.q <= 1:
                raise ValueError(f"`q` must be between 0 and 1 inclusive, got {self.q}.")
        elif self.q is not None:
            raise ValueError("`q` must only be set when op is 'quantile'.")
        return self

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[ReduceDataArray]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim"]
        allowed_dims_type = Literal[*allowed_dims]
        name = f"ReduceDataArray_{concat_camel_case(sorted(allowed_dims))}"
        return create_model(
            name,
            __base__=ReduceDataArray,
            __doc__=(
                f"{inspect.cleandoc(ReduceDataArray.__doc__ or '')}\n\n"
                "Dim names must strictly match allowed dims."
            ),
            # field defs:
            func_name=Annotated[Literal[name], Field(default=name)],
            dim=Annotated[
                allowed_dims_type | tuple[allowed_dims_type, ...] | None,
                Field(
                    default=original_field.default,
                    description=original_field.description,
                ),
            ],
        )


class RollingReduce(Function, frozen=True):
    """Apply a rolling-window reduction along a dimension.

    Slides a window of `window` consecutive steps along `dim` and reduces each window with `op`,
    producing a result of the same shape as the input. Use it for temporal smoothing (e.g. a
    30-step running mean) or moving totals (a rolling 'sum' for windowed accumulation). Positions
    that do not have a full window are set to NaN.
    """

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "RollingReduce"
    data_array: VariableReference = Field(
        description="The DataArray to apply a rolling-window reduction to."
    )
    dim: str = Field(description="The dimension to roll the window along (typically time).")
    window: int = Field(
        ge=1, description="Number of consecutive steps along `dim` in each rolling window."
    )
    op: Literal["mean", "sum", "std", "min", "max"] = Field(
        description="Reduction applied within each window."
    )
    center: bool = Field(
        default=False,
        description="If True, label each window at its center rather than its trailing edge. "
        "Use for centered running means; leave False for trailing/backward-looking windows.",
    )

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[RollingReduce]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim"]
        allowed_dims_type = Literal[*allowed_dims]
        name = f"RollingReduce_{concat_camel_case(sorted(allowed_dims))}"
        return create_model(
            name,
            __base__=RollingReduce,
            __doc__=(
                f"{inspect.cleandoc(RollingReduce.__doc__ or '')}\n\n"
                "Dim names must strictly match allowed dims."
            ),
            # field defs:
            func_name=Annotated[Literal[name], Field(default=name)],
            dim=Annotated[allowed_dims_type, Field(description=original_field.description)],
        )


class CumulativeReduce(Function, frozen=True):
    """Compute a running (cumulative) reduction along a dimension.

    Accumulates values along `dim` from the start up to each position: 'sum' gives a running total
    (e.g. accumulated precipitation), while 'max'/'min' give the running record high/low so far.
    The result has the same shape as the input.
    """

    # func_name is a str here instead of a Literal so that when we narrow the
    # type to a Literal in the subclass in with_constrained_dims(), the type
    # checker doesn't complain about an incompatible variable override.
    func_name: str = "CumulativeReduce"
    data_array: VariableReference = Field(
        description="The DataArray to accumulate along a dimension."
    )
    dim: str = Field(description="The dimension to accumulate along (typically time).")
    op: Literal["sum", "max", "min"] = Field(
        description="The cumulative reduction: 'sum' is a running total, 'max' and 'min' are the "
        "running record high and low up to each position."
    )

    @classmethod
    def with_constrained_dims(cls, allowed_dims: Iterable[str]) -> "type[CumulativeReduce]":
        """Create class variant with constrained dims names."""
        original_field = cls.model_fields["dim"]
        allowed_dims_type = Literal[*allowed_dims]
        name = f"CumulativeReduce_{concat_camel_case(sorted(allowed_dims))}"
        return create_model(
            name,
            __base__=CumulativeReduce,
            __doc__=(
                f"{inspect.cleandoc(CumulativeReduce.__doc__ or '')}\n\n"
                "Dim names must strictly match allowed dims."
            ),
            # field defs:
            func_name=Annotated[Literal[name], Field(default=name)],
            dim=Annotated[allowed_dims_type, Field(description=original_field.description)],
        )


#####################
# Temporal functions
#####################
class GroupByTimeAndReduceDataArray(Function, frozen=True):
    """Group DataArray by some time unit and apply a reduction op.

    This collapses all non-time dimensions and returns a 1-dimensional time
    series.
    """

    func_name: Literal["GroupByTimeAndReduceDataArray"] = "GroupByTimeAndReduceDataArray"
    data_array: VariableReference = Field(
        description="The DataArray to group by some time unit and reduce.",
    )
    op: Literal["sum", "mean", "std", "min", "max"]
    time_unit: Literal[
        "date",
        "week",
        "month",
        "quarter",
        "year",
        "dayofweek",
        "dayofmonth",
        "dayofyear",
    ] = Field(description="Time unit to group by.")
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )


# Standard meteorological season abbreviations used in climate science.
# Each season is named after its three constituent months (e.g. "DJF" = December-January-February).
# For the Northern Hemisphere:
# - DJF → Winter
# - MAM → Spring
# - JJA → Summer
# - SON → Fall/Autumn
SeasonName = Literal[
    "DJF",  # Dec-Jan-Feb
    "MAM",  # Mar-Apr-May
    "JJA",  # Jun-Jul-Aug
    "SON",  # Sep-Oct-Nov
]


class FilterToTimePeriodAcrossYears(Function, frozen=True):
    """Filter a DataArray to a specific time period across all years.

    Keeps the time dimension (just the matching timesteps, e.g. every July across all years). To
    compute a single-period climatology/normal (e.g. the mean July map), follow this with
    ReduceDataArray over the time dim (op='mean' for the normal, op='std' for the variability). To
    get single-period anomalies, subtract that normal from the filtered data with a BinaryOperation
    (op='sub'), and divide by the std with op='div' for a standardized/z-score anomaly.
    """

    func_name: Literal["FilterToTimePeriodAcrossYears"] = "FilterToTimePeriodAcrossYears"
    data_array: VariableReference
    period_type: Literal["month", "season", "week", "dayofyear"] = Field(
        description="Type of time period to filter to."
    )
    period_value: int | list[int] | SeasonName | list[SeasonName] = Field(
        description="Value(s) of the time period to filter to."
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )

    @field_validator("period_value")
    @classmethod
    def _validate_period_value(cls, v: Any) -> Any:  # noqa: ANN401
        return v.upper() if isinstance(v, str) else v


ClimatologyPeriod = Literal["month", "season", "dayofyear", "week", "quarter"]


class ComputeClimatology(Function, frozen=True):
    """Compute a per-pixel climatology by grouping over a calendar period and averaging over time.

    Unlike GroupByTimeAndReduceDataArray (which collapses ALL non-time dims into a 1-D series),
    this preserves the non-time dimensions (e.g. lat/lon). The time dimension is replaced by a
    dimension named after `period_type` (e.g. 'month', with 12 entries), yielding a climatology
    field such as a monthly-mean map.

    The climatology is computed over whatever time range the input covers, so to use a specific
    reference/baseline period (e.g. 1991-2020) slice the DataArray to that period first. Pair with
    ComputeAnomaly to build deseasonalized anomalies.

    This computes ALL periods (e.g. all 12 months). For the normal of a SINGLE period (e.g. just
    the mean July map), do not use this: filter with FilterToTimePeriodAcrossYears then reduce over
    the time dim with ReduceDataArray (op='mean', or op='std' for variability), which is more
    direct and avoids computing the other periods.
    """

    func_name: Literal["ComputeClimatology"] = "ComputeClimatology"
    data_array: VariableReference = Field(
        description="The temporal DataArray to build a climatology from."
    )
    period_type: ClimatologyPeriod = Field(
        description="Calendar period to group by. The time dimension is replaced by a dimension "
        "of this name in the result (e.g. 'month' yields 12 groups, 'season' yields 4)."
    )
    op: Literal["mean", "std"] = Field(
        default="mean",
        description="Aggregation applied over time within each period group. Use 'mean' for the "
        "climatological normal and 'std' for the climatological variability.",
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )


class ComputeAnomaly(Function, frozen=True):
    """Compute a deseasonalized anomaly by subtracting a climatology from a temporal DataArray.

    For each timestep, the climatology value for that timestep's calendar period is subtracted,
    yielding anomalies relative to the seasonal cycle (e.g. sea surface temperature anomalies). The
    result has the same dimensions, shape, and time axis as `data_array`.

    Typical usage: slice the data to a baseline/reference period, ComputeClimatology over it, then
    ComputeAnomaly over the full record to get anomalies relative to that baseline.
    """

    func_name: Literal["ComputeAnomaly"] = "ComputeAnomaly"
    data_array: VariableReference = Field(
        description="The temporal DataArray to deseasonalize (subtract the seasonal cycle from)."
    )
    climatology: VariableReference = Field(
        description="A climatology produced by ComputeClimatology with op='mean'. Its calendar "
        "period (e.g. month) determines how the anomaly is computed, and its non-period dimensions "
        "must match the non-time dimensions of `data_array`."
    )
    climatology_std: VariableReference | None = Field(
        default=None,
        description="Optional climatology of standard deviations, produced by ComputeClimatology "
        "with op='std' over the same period as `climatology`. If provided, the anomaly is "
        "standardized (divided by the climatological standard deviation), yielding a z-score / "
        "standardized anomaly instead of an additive one. Must have the same period and non-period "
        "dimensions as `climatology`.",
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )


class ComputeLinearTrend(Function, frozen=True):
    """Compute the per-pixel linear trend (slope) of a DataArray over time via least-squares.

    Fits a straight line to each pixel's time series and returns the slope, expressed per the
    chosen time unit (e.g. per decade for climate trends). This produces a trend/warming-rate map:
    the time dimension is removed and the other dimensions (e.g. lat/lon) are preserved. If the
    input has no non-time dimensions, a single scalar slope is returned.
    """

    func_name: Literal["ComputeLinearTrend"] = "ComputeLinearTrend"
    data_array: VariableReference = Field(
        description="The temporal DataArray to fit a linear trend to along the time dimension."
    )
    time_unit: Literal["day", "year", "decade"] = Field(
        default="decade",
        description="Unit for the returned slope: 'day' gives change per day, 'year' per year, "
        "'decade' per decade. Choose based on how the result will be interpreted (e.g. 'decade' "
        "for long-term climate warming rates).",
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )


class ComputeTrendPValue(Function, frozen=True):
    """Compute the two-sided p-value of a per-pixel linear trend over time.

    For each pixel, fits a straight line over the time dimension and returns the p-value testing
    whether the slope differs from zero (ordinary least-squares t-test, df = n - 2). Small p-values
    (e.g. < 0.05) indicate a statistically significant trend. Pair with ComputeLinearTrend and
    MaskWhere to show only the slopes that are significant. The time dimension is removed from the
    result (a scalar is returned if no other dimensions remain).

    NOTE: this OLS t-test assumes independent residuals. Climate time series are autocorrelated, so
    the p-values can be optimistic (too small); treat borderline significance with caution.
    """

    func_name: Literal["ComputeTrendPValue"] = "ComputeTrendPValue"
    data_array: VariableReference = Field(
        description="The temporal DataArray to test for a linear trend along the time dimension."
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )


class ComputePercentileRank(Function, frozen=True):
    """Compute each timestep's percentile rank within its own pixel's time distribution.

    For every pixel, ranks each timestep's value against all values of that pixel's time series and
    returns the empirical percentile rank in [0, 100] (the fraction of values at or below it, times
    100). The result has the SAME dimensions and shape as the input.

    Use it to answer "how unusual is this time relative to the record?". For example, to reproduce a
    "percentile rank relative to the 1982-2025 Augusts" map: filter to August across all years with
    FilterToTimePeriodAcrossYears, apply ComputePercentileRank over time, then SliceDataArray to a
    single year. NaNs (e.g. land) are ranked per pixel among the valid values and stay NaN.
    """

    func_name: Literal["ComputePercentileRank"] = "ComputePercentileRank"
    data_array: VariableReference = Field(
        description="The temporal DataArray whose values are ranked along the time dimension."
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension to rank each value within."
    )


class ResampleTime(Function, frozen=True):
    """Resample a DataArray onto a coarser, regular calendar cadence (e.g. daily to monthly).

    Calendar-aware down-sampling: the time axis is grouped into consecutive periods of
    `target_frequency` and each period is reduced with `op`, producing a continuous coarser time
    series while PRESERVING all non-time dimensions (e.g. lat/lon). Use it to put two variables on a
    common cadence before combining them (e.g. resample a daily field to monthly means so it can be
    correlated with or subtracted from a monthly product), or simply to aggregate to monthly/annual
    values.

    This differs from GroupByTimeAndReduceDataArray (which collapses ALL non-time dims into a
    climatological 1-D series) and from RollingReduce (which keeps the original cadence). To align
    to another variable's exact time axis, prefer ResampleTimeToMatch.
    """

    func_name: Literal["ResampleTime"] = "ResampleTime"
    data_array: VariableReference = Field(
        description="The temporal DataArray to resample onto a coarser cadence."
    )
    target_frequency: Literal["hour", "day", "week", "month", "quarter", "year"] = Field(
        description="Target cadence to aggregate onto; each output step covers one such period "
        "(e.g. 'month' yields one value per calendar month)."
    )
    op: Literal["mean", "sum", "std", "min", "max"] = Field(
        description="Reduction applied within each period. Use 'mean' for average values and 'sum' "
        "for accumulations (e.g. total precipitation per month)."
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in the DataArray."
    )


class ResampleTimeToMatch(Function, frozen=True):
    """Resample a DataArray onto another DataArray's time axis so the two can be combined.

    Aggregates `source` onto the calendar cadence of `reference` (inferred from the reference's time
    coordinate) using `op`, then snaps the result onto the reference's exact time labels. Use it to
    put `source` on the same time axis as `reference` before correlating, differencing, or otherwise
    combining them (e.g. aggregate a daily field onto a monthly product's months). Non-time
    dimensions of `source` are preserved; reference periods that `source` does not cover become NaN.

    The reference must have a regular, inferable time cadence. For a fixed target cadence without a
    reference variable, use ResampleTime instead.
    """

    func_name: Literal["ResampleTimeToMatch"] = "ResampleTimeToMatch"
    source: VariableReference = Field(
        description="The DataArray to resample; must have the time dimension."
    )
    reference: VariableReference = Field(
        description="A DataArray whose time axis `source` will be resampled and snapped onto."
    )
    op: Literal["mean", "sum", "std", "min", "max"] = Field(
        description="Reduction applied within each period. Use 'mean' for average values and 'sum' "
        "for accumulations (e.g. total precipitation per month)."
    )
    time_dim: str = Field(
        default="time", description="Name of the time dimension in both DataArrays."
    )


############################
# Spatio-temporal functions
############################


class FilterDataArrayByPolygon(Function, frozen=True):
    """Spatially filter a DataArray with lat/lon dimensions using a polygon.

    Any points falling outside the polygon are masked out.

    IMPORTANT: This should almost always go AFTER any temporal filtering (if applicable), because
    the polygon is applied to each time step independently, which can be very slow for large
    datasets with many time steps.
    """

    func_name: Literal["FilterDataArrayByPolygon"] = "FilterDataArrayByPolygon"
    data_array: VariableReference = Field(description="The DataArray to filter.")
    polygon: VariableReference = Field(description="A Polygon describing some geographical area.")


class PointCoords(BaseModel):
    """Coordinates of a single point as scalar values.

    Also includes the value at that point.
    """

    coords: dict[str, str | float] = Field(
        description="Mapping from dim names to their values. "
        "E.g. {'time': '2023-01-01', 'lat': 30.0, 'lon': -90.0}"
    )
    value: float = Field(description="Value at the point represented by the coords.")


class SelectPointCoordsKey(Function, frozen=True):
    """Select an individual value from a PointCoords object, as a scalar."""

    func_name: Literal["SelectPointCoordsKey"] = "SelectPointCoordsKey"
    point_coords: VariableReference
    key: str = Field(
        description="The key in the coords dict to select. E.g. 'time', 'lat', or 'lon'. "
        "To select the value at the point rather than a coordinate, use 'value' as the key."
    )

    @classmethod
    def model_json_schema(  # type:ignore[reportIncompatibleMethodOverride]
        cls, *args: list[Any], **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Include definition of PointCoords in JSON schema."""
        schema = super().model_json_schema(*args, **kwargs)  # type: ignore[reportUnknownArgumentType]
        schema["$defs"]["PointCoords"] = PointCoords.model_json_schema()
        return schema


class DataArrayArgmaxCoords(Function, frozen=True):
    """Return location of max value in DataArray as a PointCoords object."""

    func_name: Literal["DataArrayArgmaxCoords"] = "DataArrayArgmaxCoords"
    data_array: VariableReference


class DataArrayArgminCoords(Function, frozen=True):
    """Return location of min value in DataArray as a PointCoords object."""

    func_name: Literal["DataArrayArgminCoords"] = "DataArrayArgminCoords"
    data_array: VariableReference


class GeocodeNaturalLanguageDescription(Function, frozen=True):
    """Geocode natural language description of an area into a polygon."""

    func_name: Literal["GeocodeNaturalLanguageDescription"] = "GeocodeNaturalLanguageDescription"
    text: str = Field(
        description="Natural language description of a location. "
        'Example inputs: "Caspian Sea", "Florida, north of Miami", '
        '"within 1 mile of the coast of Boston" etc.',
    )


class CreateGeoTIFF(Function, frozen=True):
    """Create a GeoTIFF file from a 2D DataArray."""

    func_name: Literal["CreateGeoTIFF"] = "CreateGeoTIFF"
    data_array: VariableReference = Field(description="A 2D DataArray with lat and lon dims.")


class RegridToMatch(Function, frozen=True):
    """Interpolate a DataArray onto another DataArray's latitude/longitude grid.

    Regrids `source` onto the lat/lon grid of `reference` so the two can be combined (e.g. added,
    differenced, correlated, or masked) even when they come from datasets with different spatial
    resolutions. Non-spatial dimensions of `source` (e.g. time) are preserved; only its lat/lon are
    resampled. Points outside the source's coverage become NaN.

    Use bilinear ('linear') interpolation for smooth fields like temperature. Note that for
    flux-like or precipitation fields, interpolation does not conserve area-integrated totals.
    """

    func_name: Literal["RegridToMatch"] = "RegridToMatch"
    source: VariableReference = Field(
        description="The DataArray to regrid; must have 'lat' and 'lon' dims."
    )
    reference: VariableReference = Field(
        description="A DataArray whose 'lat'/'lon' grid `source` will be interpolated onto."
    )
    method: Literal["linear", "nearest"] = Field(
        default="linear",
        description="Interpolation method: 'linear' (bilinear) or 'nearest' neighbor.",
    )


################
# Function sets
################


class GeneralDataArrayFunctionSet(FunctionSet):
    """Functions for DataArrays with arbitrary dims."""

    def __init__(self, allowed_dims: Iterable[str]) -> None:
        """Constructor.

        Args:
            allowed_dims (Iterable[str]): Known dim names to constrain
                functions (that take dim names) to. This enables the dim name
                arguments to be validated via pydantic at runtime.
        """
        super().__init__(
            [
                BinaryOperation,
                UnaryOperation,
                MaskWhere,
                CorrelateDataArrays.with_constrained_dims(allowed_dims),
                SliceDataArray.with_constrained_dims(allowed_dims),
                ReduceDataArray.with_constrained_dims(allowed_dims),
                RollingReduce.with_constrained_dims(allowed_dims),
                CumulativeReduce.with_constrained_dims(allowed_dims),
                DataArrayArgmaxCoords,
                DataArrayArgminCoords,
                SelectPointCoordsKey,
            ]
        )


class TemporalDataArrayFunctionSet(FunctionSet):
    """Functions for DataArrays with a time dim plus other arbitrary dims."""

    def __init__(self, allowed_dims: Iterable[str]) -> None:
        """Constructor.

        Args:
            allowed_dims (Iterable[str]): Known dim names to constrain
                functions (that take dim names) to. This enables the dim name
                arguments to be validated via pydantic at runtime.
        """
        super().__init__(
            [
                *GeneralDataArrayFunctionSet(allowed_dims).functions,
                GroupByTimeAndReduceDataArray,
                FilterToTimePeriodAcrossYears,
                ComputeClimatology,
                ComputeAnomaly,
                ComputeLinearTrend,
                ComputeTrendPValue,
                ComputePercentileRank,
                ResampleTime,
                ResampleTimeToMatch,
            ]
        )


class SpatioTemporalDataArrayFunctionSet(FunctionSet):
    """Functions for spatiotemporal DataArrays."""

    def __init__(self, allowed_dims: Iterable[str] | None = None) -> None:
        """Constructor.

        Args:
            allowed_dims (Iterable[str]): Known dim names to constrain
                functions (that take dim names) to. This enables the dim name
                arguments to be validated via pydantic at runtime. If None,
                will be set to ("time", "lat", "lon"). Defaults to None.
        """
        allowed_dims = allowed_dims or ["time", "lat", "lon"]
        super().__init__(
            [
                *TemporalDataArrayFunctionSet(allowed_dims).functions,
                FilterDataArrayByPolygon,
                CreateGeoTIFF,
                RegridToMatch,
            ]
        )
