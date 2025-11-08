from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Literal

import polars as pl
from ib_insync import IB, Stock  # pip install ib-insync

from src.common.config import get_root

BarSize = Literal["1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min"]
What = Literal["TRADES", "MIDPOINT", "BID", "ASK"]

def _cache_path(root: Path, symbol: str, bar_size: str) -> Path:
    return root / "cache" / "ibkr" / symbol.upper() / f"{bar_size.replace(' ','_')}.parquet"

def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )

@dataclass
class IBConfig:
    host: str = os.getenv("IB_HOST", "127.0.0.1")
    port: int = int(os.getenv("IB_PORT", "7497"))
    client_id: int = int(os.getenv("IB_CLIENT_ID", "1"))
    use_delayed: bool = bool(int(os.getenv("IB_USE_DELAYED", "0")))
    cache_root: Path = get_root()  # Use consistent project root resolution

class IBKRSource:
    """Thin wrapper around ib_insync for historical bars, chunked, UTC tz-aware, using Polars."""
    def __init__(self, cfg: IBConfig = IBConfig()) -> None:
        self.cfg = cfg
        self.ib = IB()

    def connect(self) -> None:
        self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, readonly=True)
        if self.cfg.use_delayed:
            self.ib.reqMarketDataType(3)  # 1=live,2=frozen,3=delayed,4=delayed-frozen
        assert self.ib.isConnected(), "IB connection failed"

    def close(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def _contract(self, symbol: str) -> Stock:
        # SMART routing, USD, primaryExchange qualify automatically
        c = Stock(symbol.upper(), "SMART", "USD")
        self.ib.qualifyContracts(c)
        return c

    def _one_request(
        self, contract: Stock, end_dt: datetime, duration: str, bar_size: BarSize, what: What, rth: bool
    ) -> pl.DataFrame:
        """Single IB historical request. Returns tz-aware UTC Polars DataFrame."""
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration,          # e.g. "1 Y", "180 D"
            barSizeSetting=bar_size,       # e.g. "1 day"
            whatToShow=what,
            useRTH=1 if rth else 0,
            formatDate=2,                  # datetime objects
            keepUpToDate=False,
        )
        if not bars:
            return _empty_df()
        rows = []
        for bar in bars:
            dt = bar.date
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            rows.append({
                "ts": dt,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })
        df = pl.DataFrame(rows).with_columns(
            pl.col("ts").cast(pl.Datetime("us", "UTC"))
        )
        df = df.select(["ts","open","high","low","close","volume"])
        return df

    def historical_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None,
        bar_size: BarSize = "1 day",
        what: What = "TRADES",
        rth: bool = True,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Chunking rules:
          - 1 day bars: IB erlaubt bis ~1Y per request → wir loopen in Jahreschunks.
          - Intraday (<=1h): nimm kleinere duration z.B. '30 D' und iteriere.
        Returns Polars DataFrame with tz-aware UTC 'ts' column.
        """
        assert start.tzinfo is not None, "start must be tz-aware (UTC)"
        if end is None:
            end = datetime.now(timezone.utc)

        cache_fp = _cache_path(self.cfg.cache_root, symbol, bar_size)
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        if use_cache and cache_fp.exists():
            try:
                cached = pl.read_parquet(cache_fp)
                if cached.height > 0 and cached.schema["ts"].time_zone != "UTC":
                    cached = cached.with_columns(
                        pl.col("ts").dt.replace_time_zone("UTC")
                    )
            except Exception:
                cached = _empty_df()
        else:
            cached = _empty_df()

        contract = self._contract(symbol)

        frames: list[pl.DataFrame] = []
        cur_end = end

        # Choose chunk duration
        if bar_size == "1 day":
            dur = "1 Y"
            step = timedelta(days=365)
        elif bar_size in {"1 hour", "30 mins", "15 mins"}:
            dur = "30 D"
            step = timedelta(days=30)
        else:
            dur = "7 D"
            step = timedelta(days=7)

        while cur_end > start:
            cur_start = max(start, cur_end - step)
            df = self._one_request(contract, cur_end, dur, bar_size, what, rth)
            if df.is_empty():
                cur_end = cur_start
                continue
            df = df.filter((pl.col("ts") >= cur_start) & (pl.col("ts") <= cur_end))
            frames.append(df)
            cur_end = cur_start - timedelta(seconds=1)

        if not frames:
            data = _empty_df()
        else:
            data = pl.concat(frames, how="vertical")

        if cached.height > 0:
            data = pl.concat([cached, data], how="vertical")

        data = data.sort("ts")
        data = data.unique(subset=["ts"], keep="last")

        if use_cache:
            data.write_parquet(cache_fp)
        return data