from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import pandas as pd
import yfinance as yf

from common.types import Interval

_REQUIRED_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        # return an empty, correctly-shaped frame with UTC index
        idx = pd.DatetimeIndex([], tz=timezone.utc, name="datetime")
        return pd.DataFrame(columns=_REQUIRED_COLS, index=idx)

    df = df.rename(columns=str.lower)
    # yfinance sometimes returns Adj Close; we ignore it here
    df = df[["open", "high", "low", "close", "volume"]]
    # Ensure UTC tz-aware ordered index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex from yfinance")
    idx = df.index
    idx = (idx.tz_localize(timezone.utc) if idx.tz is None else idx.tz_convert(timezone.utc))
    df.index = idx.sort_values()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna()
    return df

class YahooDataSource:
    """Thin Yahoo Finance OHLCV provider."""

    _VALID_INTERVALS: Final[set[Interval]] = {"1m", "5m", "1h", "1d"}

    def __init__(self, auto_adjust: bool = False, progress: bool = False) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress

    def ohlcv(self, symbol: str, start: datetime, end: datetime | None, interval: Interval) -> pd.DataFrame:
        if interval not in self._VALID_INTERVALS:
            raise ValueError(f"Unsupported interval for Yahoo: {interval}")
        # yfinance uses the same tokens for these intervals
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=self.progress,
        )
        return _normalize(df)