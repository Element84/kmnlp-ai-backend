from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
        SeasonName,
    )


# Maps each month number to its standard meteorological season.
_MONTH_TO_SEASON: dict[int, "SeasonName"] = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


class TimeCoords:
    """Class representing time coordinates with frequency.

    Used for statically inferring time dim size during algorithm validation.
    """

    def __init__(self, start: datetime, end: datetime, freq: timedelta) -> None:
        """Constructor."""
        self.freq = freq
        self.times: np.ndarray = np.arange(
            np.datetime64(start),
            np.datetime64(end) + np.timedelta64(self.freq),
            np.timedelta64(self.freq),
        ).astype("datetime64[ns]")

    def slice(self, start: str | None = None, end: str | None = None) -> "TimeCoords":
        """Slice time coordinates between start and end times."""
        mask = np.ones(len(self.times), dtype=bool)
        if start:
            mask &= self.times >= np.datetime64(start)
        if end:
            mask &= self.times <= np.datetime64(end)

        out = TimeCoords.__new__(TimeCoords)
        out.freq = self.freq
        out.times = self.times[mask]
        return out

    def slice_by_index(
        self, start_idx: int | None = None, end_idx: int | None = None
    ) -> "TimeCoords":
        """Slice time coordinates between start and end indices."""
        out = TimeCoords.__new__(TimeCoords)
        out.freq = self.freq
        out.times = self.times[start_idx:end_idx]
        return out

    def filter_to_time_period_across_years(
        self,
        period_type: Literal["month", "season", "week", "dayofyear"],
        period_values: "int | Sequence[int] | SeasonName | Sequence[SeasonName]",
    ) -> "TimeCoords":
        """Filter time coordinates to a specific period, matching across years."""
        values = [period_values] if isinstance(period_values, int | str) else list(period_values)

        mask_values: list[bool] = []
        t: np.datetime64
        for t in self.times:
            dt_str = np.datetime_as_string(t, unit="D")
            y, m, d = map(int, dt_str.split("-"))
            dte = date(y, m, d)
            match period_type:
                case "month":
                    mask_values.append(m in values)
                case "season":
                    mask_values.append(_MONTH_TO_SEASON[m] in values)
                case "week":
                    mask_values.append(dte.isocalendar()[1] in values)
                case "dayofyear":
                    mask_values.append(dte.timetuple().tm_yday in values)

        mask = np.array(mask_values, dtype=bool)
        out = TimeCoords.__new__(TimeCoords)
        out.freq = self.freq
        out.times = self.times[mask]
        return out

    def get_groupby_keys(
        self,
        time_unit: Literal[
            "date",
            "week",
            "month",
            "quarter",
            "year",
            "dayofweek",
            "dayofmonth",
            "dayofyear",
            "season",
        ],
    ) -> set[int | date | str]:
        """Return unique keys from grouping by a temporal time unit."""
        return {self._extract_grouping_key(t, time_unit) for t in self.times}

    def _extract_grouping_key(  # noqa: PLR0911
        self,
        t: np.datetime64,
        time_unit: Literal[
            "date",
            "week",
            "month",
            "quarter",
            "year",
            "dayofweek",
            "dayofmonth",
            "dayofyear",
            "season",
        ],
    ) -> int | date | str:
        """Extract grouping key for datetimes."""
        dt = np.datetime_as_string(t, unit="D")
        y, m, d = map(int, dt.split("-"))

        dte = date(y, m, d)
        match time_unit:
            case "date":
                return dte
            case "week":
                return dte.isocalendar()[1]
            case "month":
                return m
            case "quarter":
                return (m - 1) // 3 + 1
            case "year":
                return y
            case "dayofweek":
                return dte.weekday()
            case "dayofmonth":
                return d
            case "dayofyear":
                return dte.timetuple().tm_yday
            case "season":
                return _MONTH_TO_SEASON[m]

    def __len__(self) -> int:
        return len(self.times)

    def __repr__(self) -> str:
        start, end = self.times[0], self.times[-1]
        return f"<TimeCoords (n={len(self.times)}, {start} → {end})>"
