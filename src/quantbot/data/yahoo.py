import time
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_ohlcv(symbol: str, start: str = "2015-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns=str.lower)[["open","high","low","close","volume"]]
    df.index = pd.to_datetime(df.index, utc=True)  # tz-aware
    return df.dropna()

def available_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of available OHLCV columns in the dataframe."""
    return list(df.columns)

def head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first n rows of the dataframe for quick inspection."""
    return df.head(n)