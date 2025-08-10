import time
import yfinance as yf
import pandas as pd

def load_ohlcv(symbol: str, start: str = "2015-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns=str.lower)[["open","high","low","close","volume"]]
    df.index = pd.to_datetime(df.index, utc=True)  # tz-aware
    return df.dropna()