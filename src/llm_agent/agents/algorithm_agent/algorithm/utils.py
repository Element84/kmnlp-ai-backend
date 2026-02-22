from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np


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

    def get_groupby_keys(
        self,
        time_unit: Literal[
            "date", "week", "month", "quarter", "year", "dayofweek", "dayofmonth", "dayofyear"
        ],
    ) -> set[int | date]:
        """Return unique keys from grouping by a temporal time unit."""
        return {self._extract_grouping_key(t, time_unit) for t in self.times}

    def _extract_grouping_key(  # noqa: PLR0911
        self,
        t: np.datetime64,
        time_unit: Literal[
            "date", "week", "month", "quarter", "year", "dayofweek", "dayofmonth", "dayofyear"
        ],
    ) -> int | date:
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

    def __len__(self) -> int:
        return len(self.times)

    def __repr__(self) -> str:
        start, end = self.times[0], self.times[-1]
        return f"<TimeCoords (n={len(self.times)}, {start} → {end})>"
