# Exchange calendar utilities (NYSE by default)
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Literal

import pandas as pd

try:
    import pandas_market_calendars as mcal
except ImportError as exc:
    raise ImportError(
        "pandas_market_calendars is required for calendar utilities. "
        "Install via: pip install pandas-market-calendars"
    ) from exc

from src.common.types import Interval, _ensure_utc

CalendarName = Literal["XNYS", "XNAS", "XETR", "ARCX", "24/7"]  # extend later if needed

# Minimal built-in mapping; extend as needed or externalize later.
_TICKER_TO_CAL: dict[str, CalendarName] = {
    "AAPL": "XNAS",
    "MSFT": "XNAS",
    "SPY": "ARCX",
}
DEFAULT_CALENDAR: CalendarName = "XNYS"


class _AlwaysOpenCalendar:
    """Synthetic 24/7 calendar (UTC)."""
    tz = timezone.utc

    def schedule(self, start_date, end_date) -> pd.DataFrame:
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D", tz=self.tz)
        return pd.DataFrame(
            {"market_open": idx, "market_close": idx + pd.Timedelta(days=1)}
        )


def calendar_name_for_symbol(symbol: str) -> CalendarName:
    """Resolve a symbol to a calendar code (fallback to DEFAULT_CALENDAR)."""
    return _TICKER_TO_CAL.get(symbol.upper(), DEFAULT_CALENDAR)


def get_calendar(name: CalendarName = "XNYS"):
    """Return a pandas-market-calendars calendar instance."""
    if name == "24/7":
        return _AlwaysOpenCalendar()
    # XNYS/XNAS/ARCX are supported ids in pandas-market-calendars
    return mcal.get_calendar(name)


def _schedule_utc(cal, start: datetime, end: datetime) -> pd.DataFrame:
    """Get schedule DataFrame with market_open/market_close in UTC tz."""
    start_d = _ensure_utc(start).date()
    end_d = _ensure_utc(end).date()
    sched = cal.schedule(start_date=start_d, end_date=end_d)
    # Ensure tz-aware UTC
    sched = sched.copy()
    sched["market_open"] = sched["market_open"].dt.tz_convert(timezone.utc)
    sched["market_close"] = sched["market_close"].dt.tz_convert(timezone.utc)
    return sched


def sessions(start: datetime, end: datetime, name: CalendarName = "XNYS") -> pd.DatetimeIndex:
    """
    Return session opens (UTC) between start and end inclusive.
    The index corresponds to exchange sessions (one per trading day).
    """
    cal = get_calendar(name)
    sched = _schedule_utc(cal, start, end)
    return pd.DatetimeIndex(sched["market_open"], name="session_open_utc")


def session_bounds(ts: datetime, name: CalendarName = "XNYS") -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Return (open, close) UTC timestamps for the session that contains `ts`.
    If `ts` is outside trading hours, returns the session that would next open after `ts`
    only if such session lies within [ts-3d, ts+7d] search window; otherwise None.
    """
    cal = get_calendar(name)
    ts_utc = _ensure_utc(ts)
    # search window around ts to find the relevant session efficiently
    start = ts_utc - timedelta(days=3)
    end = ts_utc + timedelta(days=7)
    sched = _schedule_utc(cal, start, end)

    # If ts falls within any session, pick that; else, pick the next session after ts
    mask = (sched["market_open"] <= ts_utc) & (ts_utc <= sched["market_close"])
    if mask.any():
        row = sched.loc[mask].iloc[0]
        return row["market_open"], row["market_close"]

    # next session after ts
    later = sched[sched["market_open"] > ts_utc]
    if not later.empty:
        row = later.iloc[0]
        return row["market_open"], row["market_close"]

    return None


def session_open(ts: datetime, name: CalendarName = "XNYS") -> pd.Timestamp | None:
    """Return the UTC open time of the session that contains `ts` (or next session if off-hours)."""
    bounds = session_bounds(ts, name=name)
    return bounds[0] if bounds else None


def session_close(ts: datetime, name: CalendarName = "XNYS") -> pd.Timestamp | None:
    """Return the UTC close time of the session that contains `ts` (or next session if off-hours)."""
    bounds = session_bounds(ts, name=name)
    return bounds[1] if bounds else None


def align_to_sessions(df: pd.DataFrame, interval: Interval = "1d", name: CalendarName = "XNYS") -> pd.DataFrame:
    """
    Drop rows that fall outside exchange sessions.
    - For intraday data (1m/5m/1h), keep bars where timestamp in [open, close] of a session.
    - For daily data (1d), keep only bars whose day is a valid trading session day.

    Assumes `df.index` is tz-aware; converts to UTC internally.
    """
    if df.empty or name == "24/7":
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("align_to_sessions expects a DataFrame with a DatetimeIndex")

    idx = df.index
    # Normalize to UTC, sorted
    idx = (idx.tz_localize(timezone.utc) if idx.tz is None else idx.tz_convert(timezone.utc))
    # compute a slightly padded window to cover edges
    start = pd.Timestamp(idx.min()).to_pydatetime()
    end = pd.Timestamp(idx.max()).to_pydatetime()
    cal = get_calendar(name)
    sched = _schedule_utc(cal, start - timedelta(days=2), end + timedelta(days=2))

    if interval == "1d":
        # Keep rows whose local (calendar tz) date is a valid session day
        # Convert timestamps to the calendar's local timezone to match session index semantics
        cal_tz = cal.tz  # tzinfo / pytz tz used by the calendar
        days_series = pd.Series(idx.tz_convert(cal_tz).date, index=df.index)
        valid_days = set(sched.index.date)  # schedule index corresponds to session dates
        mask = days_series.isin(valid_days).values
        return df.loc[mask]

    # Intraday: vectorized membership via merge_asof on session opens, then check close
    bounds = pd.DataFrame(
        {
            "open": sched["market_open"].sort_values().values,
            "close": sched["market_close"].sort_values().values,
        }
    ).sort_values("open")
    ts_df = pd.DataFrame({"ts": pd.DatetimeIndex(idx).tz_convert(timezone.utc)})
    merged = pd.merge_asof(ts_df.sort_values("ts"), bounds, left_on="ts", right_on="open", direction="backward")
    in_session = (merged["ts"] >= merged["open"]) & (merged["ts"] <= merged["close"])
    # Align mask back to original order
    mask = in_session.reindex(ts_df.index).fillna(False).values
    return df.loc[mask]