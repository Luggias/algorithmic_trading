# Save/load OHLCV data with an incremental Parquet cache
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.common.config import CFG
from src.common.types import Interval

# -------------------------
# Interfaces / type helpers
# -------------------------

class MarketDataSource(Protocol):
    """Abstract interface for OHLCV providers (e.g., Yahoo, IB, CEX)."""

    def ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None,
        interval: Interval,
    ) -> pd.DataFrame:
        """
        Return OHLCV with:
          - tz-aware UTC DatetimeIndex (sorted, unique)
          - columns: open, high, low, close, volume
        """
        ...


_REQUIRED_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

_INTERVAL_STEP: dict[Interval, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def _ensure_utc_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-aware UTC DateTimeIndex, sorted and unique."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Normalize to UTC
    if df.index.tz is None:
        idx = df.index.tz_localize(timezone.utc)
    else:
        idx = df.index.tz_convert(timezone.utc)

    df = df.copy()
    df.index = idx
    # Sort & de-duplicate (keep last in case of overlaps)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    return df


# -------------------------
# Incremental Parquet cache
# -------------------------

@dataclass(frozen=True, slots=True)
class ParquetOHLCVCache:
    """Simple Parquet-backed cache for OHLCV series."""
    root: Path
    namespace: str = "yahoo"

    def _path(self, symbol: str, interval: Interval) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        # Use WORKDIR/cache/ as per AGENT.md
        return (self.root / "cache" / self.namespace / f"{safe_symbol}_{interval}.parquet").resolve()

    def read(self, symbol: str, interval: Interval) -> pd.DataFrame | None:
        p = self._path(symbol, interval)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df = _ensure_utc_idx(df)
        return _validate_ohlcv(df)

    def write(self, symbol: str, interval: Interval, df: pd.DataFrame) -> None:
        p = self._path(symbol, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        _validate_ohlcv(df)
        _ensure_utc_idx(df)  # ensure invariant
        df.to_parquet(p, index=True)

    def clear(self, symbol: str, interval: Interval) -> None:
        p = self._path(symbol, interval)
        if p.exists():
            p.unlink()


def _merge_incremental(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    incoming = _ensure_utc_idx(_validate_ohlcv(incoming))
    if existing is None or existing.empty:
        return incoming
    existing = _ensure_utc_idx(_validate_ohlcv(existing))
    merged = pd.concat([existing, incoming], axis=0)
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _slice(df: pd.DataFrame, start: datetime, end: datetime | None) -> pd.DataFrame:
    df = _ensure_utc_idx(df)
    start_utc = start.astimezone(timezone.utc)
    if end is None:
        return df.loc[start_utc:]
    return df.loc[start_utc:end.astimezone(timezone.utc)]


def load_ohlcv_cached(
    source: MarketDataSource,
    symbol: str,
    start: datetime,
    end: datetime | None,
    interval: Interval,
    namespace: str = "yahoo",
) -> pd.DataFrame:
    """
    Load OHLCV from cache, updating incrementally from `source` if needed.

    Contract:
      - returned index is tz-aware UTC, strictly increasing, unique
      - required columns: open, high, low, close, volume
      - cache stored under {WORKDIR}/cache/{namespace}/{symbol}_{interval}.parquet
    """
    from src.common.config import get_root
    cache = ParquetOHLCVCache(get_root(), namespace=namespace)

    cached = cache.read(symbol, interval)
    fetch_start = start

    if cached is not None and not cached.empty:
        # Continue 1 step after the last cached bar to avoid duplicate last bar
        last_ts = cached.index[-1]
        fetch_start = max(start.astimezone(timezone.utc), last_ts + _INTERVAL_STEP[interval])

    # Fetch only if we actually need more data
    if end is not None and fetch_start >= end.astimezone(timezone.utc):
        # Nothing new to fetch
        updated = cached if cached is not None else pd.DataFrame(columns=_REQUIRED_COLS, index=pd.DatetimeIndex([], tz=timezone.utc))
    else:
        fresh = source.ohlcv(symbol=symbol, start=fetch_start, end=end, interval=interval)
        updated = _merge_incremental(cached, fresh)

    # Persist and return requested slice
    if updated is not None and not updated.empty:
        cache.write(symbol, interval, updated)

    return _slice(updated, start, end)