"""
Single-file, executable demo: walk-forward OLS on return lags with next-bar-open execution.

Usage:
    python scripts/run_onefile_linreg.py

What it does:
- Downloads daily OHLCV via yfinance (free).
- Builds p lagged log-return features (no ML frameworks needed).
- Expanding walk-forward OLS (np.linalg.lstsq): fit on history up to t-1, predict at t.
- Generates signals (threshold in bps) → places market orders at t → fills at open(t+1)
- Applies simple cost model (per-share commission + slippage_bps).
- Tracks cash/position/equity and prints a short summary. Saves CSVs to reports/.

Notes:
- Everything uses UTC timestamps.
- Keep it simple: one file, minimal dependencies, deterministic behavior.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Parameters (edit as needed)
# ----------------------------
SYMBOL = "SPY"
START = "2018-01-01"
END: str | None = None
INTERVAL = "1d"

P_LAGS = 5
MIN_TRAIN = 250            # warmup length (trading days)
RETRAIN_EVERY = 1          # refit cadence (bars)

THRESHOLD_BPS = 5.0        # only trade if |prediction| > 5 bps
QTY = 10.0                 # shares per trade

# Cost model
PER_SHARE_FEE = 0.005      # e.g., $0.005/share
SLIPPAGE_BPS = 1.5         # bps on execution price; positive for buys, negative for sells

# Reporting
OUT_DIR = Path("reports")


# ----------------------------
# Helpers (minimal)
# ----------------------------
def _utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex from yfinance")
    idx = df.index
    idx = idx.tz_localize(timezone.utc) if idx.tz is None else idx.tz_convert(timezone.utc)
    df = df.copy()
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _download_ohlcv(symbol: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return _utc_index(df).dropna()


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _lag_features(ret: pd.Series, p: int) -> pd.DataFrame:
    """
    Build lag matrix X (t-1..t-p) and target y_next = ret.shift(-1).
    Row at time t contains information up to t (past-only), target is return of t+1.
    """
    X = pd.concat({f"lag{i}": ret.shift(i) for i in range(1, p + 1)}, axis=1)
    y = ret.shift(-1).rename("y_next")
    df = pd.concat([X, y], axis=1).dropna()
    return df


def _max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def _sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=0))


# ----------------------------
# Main (one pass, imperative)
# ----------------------------
if __name__ == "__main__":
    print(f"[info] downloading {SYMBOL} {INTERVAL} from {START} to {END or 'latest'} …")
    ohlcv = _download_ohlcv(SYMBOL, START, END, INTERVAL)

    # Features & target
    ret = _log_returns(ohlcv["close"])
    feats = _lag_features(ret, P_LAGS)  # index ⊆ ohlcv.index
    # Align prices to feature index
    prices = ohlcv.loc[feats.index, ["open", "close"]].copy()

    n = len(feats)
    X = feats.drop(columns=["y_next"]).to_numpy()
    y = feats["y_next"].to_numpy()
    preds = np.full(n, np.nan, dtype=float)

    # Walk-forward OLS: fit on [0:i), predict at i
    last_fit = -1
    w = None
    for i in range(MIN_TRAIN, n):
        if (w is None) or (i - last_fit) >= RETRAIN_EVERY:
            Xi = X[:i]
            yi = y[:i]
            Xi1 = np.c_[np.ones(len(Xi)), Xi]  # add intercept
            w, *_ = np.linalg.lstsq(Xi1, yi, rcond=None)
            last_fit = i
        xi1 = np.r_[1.0, X[i]]
        preds[i] = float(xi1 @ w)

    pred = pd.Series(preds, index=feats.index, name="pred")

    # Trading loop (next-bar-open execution, simple costs)
    cash = 100_000.0
    position = 0.0
    avg_price = 0.0

    pending_qty = 0.0  # order placed at t, executed at open(t+1)

    trades = []  # (timestamp, symbol, qty_signed, exec_price, fees)
    equity = []  # (timestamp, equity_value)

    thresh = THRESHOLD_BPS / 10_000.0

    idx = feats.index.to_list()
    for k, ts in enumerate(idx):
        # 1) Execute pending order at today's OPEN
        if pending_qty != 0.0:
            px = float(prices.loc[ts, "open"])
            # directional slippage
            slip_mult = 1.0 + math.copysign(SLIPPAGE_BPS, pending_qty) / 10_000.0
            exec_px = px * slip_mult
            fees = abs(pending_qty) * PER_SHARE_FEE

            # cash & position updates
            cash -= pending_qty * exec_px
            cash -= fees

            # realized P&L (only if we close/flip exposure)
            # (avg_price maintained only for same-symbol single position)
            if position == 0.0 or (np.sign(position) == np.sign(pending_qty)):
                # increase same-side exposure → update avg_price
                new_qty = position + pending_qty
                if new_qty != 0.0:
                    avg_price = (abs(position) * avg_price + abs(pending_qty) * exec_px) / abs(new_qty)
                position = new_qty
                realized = 0.0
            else:
                # opposite side: realize on closed portion
                close_qty = min(abs(position), abs(pending_qty))
                sign = 1.0 if position > 0 else -1.0
                realized = (exec_px - avg_price) * (close_qty * sign)
                position += pending_qty
                if position == 0.0:
                    avg_price = 0.0
            # realized goes to P&L via cash only implicitly when position flips fully; we keep it simple

            trades.append((ts, SYMBOL, pending_qty, exec_px, fees))
            pending_qty = 0.0

        # 2) Generate new signal at bar t (prediction refers to next-period return)
        p = pred.get(ts)
        if p is None or np.isnan(p):
            # skip warmup rows
            eq = cash + position * float(prices.loc[ts, "close"])
            equity.append((ts, eq))
            continue

        desired = 0.0
        if p > thresh:
            desired = +QTY
        elif p < -thresh:
            desired = -QTY

        # Submit order to move position towards desired
        delta = desired - position
        # If small (already in desired), no order
        if abs(delta) > 1e-9:
            pending_qty = delta  # will execute next bar open

        # 3) Mark-to-market equity at CLOSE
        eq = cash + position * float(prices.loc[ts, "close"])
        equity.append((ts, eq))

    # Final equity series and metrics
    equity_sr = pd.Series({t: v for t, v in equity}).sort_index()
    daily_ret = equity_sr.pct_change().dropna()
    total_return = equity_sr.iloc[-1] / equity_sr.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (252 / max(1, len(daily_ret))) - 1.0 if len(daily_ret) > 0 else 0.0
    mdd = _max_drawdown(equity_sr)
    sharpe = _sharpe(daily_ret)

    print("\n=== Run summary ===")
    print(f"Symbol:        {SYMBOL}")
    print(f"Period:        {feats.index[0].date()} → {feats.index[-1].date()}")
    print(f"Trades:        {len(trades)}")
    print(f"Final Equity:  {equity_sr.iloc[-1]:,.2f} USD")
    print(f"Total Return:  {total_return*100:,.2f}%")
    print(f"CAGR:          {cagr*100:,.2f}%")
    print(f"Max Drawdown:  {mdd*100:,.2f}%")
    print(f"Sharpe (252):  {sharpe:,.2f}")

    # Persist basic artifacts
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"simple_linreg_{SYMBOL}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(trades, columns=["timestamp", "symbol", "qty", "price", "fees"]).to_csv(out / "trades.csv", index=False)
    equity_sr.rename("equity").to_csv(out / "equity.csv")
    with open(out / "params.txt", "w") as f:
        f.write(f"SYMBOL={SYMBOL}\nSTART={START}\nEND={END}\nINTERVAL={INTERVAL}\n")
        f.write(f"P_LAGS={P_LAGS}\nMIN_TRAIN={MIN_TRAIN}\nRETRAIN_EVERY={RETRAIN_EVERY}\n")
        f.write(f"THRESHOLD_BPS={THRESHOLD_BPS}\nQTY={QTY}\nPER_SHARE_FEE={PER_SHARE_FEE}\nSLIPPAGE_BPS={SLIPPAGE_BPS}\n")

    print(f"\nArtifacts written to: {out.resolve()}")