# Entry script to start live trading
"""
CLI entrypoint: single-file walk-forward OLS (return lags) with next-bar-open execution.

Example:
    python scripts/run.py --symbol SPY --start 2018-01-01 --interval 1d \
        --lags 5 --min-train 250 --threshold-bps 5 --qty 10 \
        --slippage-bps 1.5 --per-share-fee 0.005

Artifacts:
    reports/simple_linreg_<SYMBOL>_<UTC_RUN_ID>/{trades.csv,equity.csv,params.json,metrics.json}

Notes:
    - Deterministic, UTC-only.
    - No sklearn dependency. Uses np.linalg.lstsq.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Utils
# ----------------------------
def _iso_utc(ts: pd.Timestamp | datetime) -> str:
    dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _download_ohlcv(symbol: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return _utc_index(df).dropna()


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _lag_features(ret: pd.Series, p: int) -> pd.DataFrame:
    """
    Build lag matrix X (t-1..t-p) and target y_next = ret.shift(-1).
    Row at time t contains information up to t, target is return of t+1.
    """
    X = pd.concat({f"lag{i}": ret.shift(i) for i in range(1, p + 1)}, axis=1)
    y = ret.shift(-1).rename("y_next")
    return pd.concat([X, y], axis=1).dropna()


def _max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def _sharpe(daily_returns: pd.Series) -> float:
    vol = daily_returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(np.sqrt(252) * daily_returns.mean() / vol)


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class RunConfig:
    symbol: str
    start: str
    end: Optional[str]
    interval: str
    lags: int
    min_train: int
    retrain_every: int
    threshold_bps: float
    qty: float
    slippage_bps: float
    per_share_fee: float
    out_dir: str
    seed: Optional[int] = None


# ----------------------------
# Core
# ----------------------------
def run(cfg: RunConfig) -> Path:
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    ohlcv = _download_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.interval)

    # Features & target
    ret = _log_returns(ohlcv["close"])
    feats = _lag_features(ret, cfg.lags)
    prices = ohlcv.loc[feats.index, ["open", "close"]].copy()

    n = len(feats)
    X = feats.drop(columns=["y_next"]).to_numpy()
    y = feats["y_next"].to_numpy()
    preds = np.full(n, np.nan, dtype=float)

    # Walk-forward OLS: fit on [0:i), predict at i
    last_fit = -1
    w = None
    for i in range(cfg.min_train, n):
        if (w is None) or (i - last_fit) >= cfg.retrain_every:
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

    trades: list[tuple[str, str, float, float, float]] = []  # (ts_iso, symbol, qty, price, fees)
    equity: list[tuple[str, float]] = []                    # (ts_iso, equity_value)

    thresh = cfg.threshold_bps / 10_000.0

    for ts in feats.index:
        # 1) Execute pending order at today's OPEN
        if pending_qty != 0.0:
            px = float(prices.loc[ts, "open"])
            slip_mult = 1.0 + math.copysign(cfg.slippage_bps, pending_qty) / 10_000.0
            exec_px = px * slip_mult
            fees = abs(pending_qty) * cfg.per_share_fee

            cash -= pending_qty * exec_px
            cash -= fees

            if position == 0.0 or (np.sign(position) == np.sign(pending_qty)):
                new_qty = position + pending_qty
                if new_qty != 0.0:
                    avg_price = (abs(position) * avg_price + abs(pending_qty) * exec_px) / abs(new_qty)
                position = new_qty
            else:
                close_qty = min(abs(position), abs(pending_qty))
                sign = 1.0 if position > 0 else -1.0
                _ = (exec_px - avg_price) * (close_qty * sign)  # realized P&L implicitly via cash
                position += pending_qty
                if position == 0.0:
                    avg_price = 0.0

            trades.append((_iso_utc(ts), cfg.symbol, float(pending_qty), exec_px, fees))
            pending_qty = 0.0

        # 2) Generate new signal at bar t (prediction refers to next-period return)
        p = pred.get(ts)
        if p is None or np.isnan(p):
            eq = cash + position * float(prices.loc[ts, "close"])
            equity.append((_iso_utc(ts), eq))
            continue

        desired = 0.0
        if p > thresh:
            desired = +cfg.qty
        elif p < -thresh:
            desired = -cfg.qty

        delta = desired - position
        if abs(delta) > 1e-9:
            pending_qty = delta  # will execute next bar open

        # 3) Mark-to-market equity at CLOSE
        eq = cash + position * float(prices.loc[ts, "close"])
        equity.append((_iso_utc(ts), eq))

    # Metrics
    equity_sr = pd.Series({pd.to_datetime(t): v for t, v in equity}).sort_index()
    daily_ret = equity_sr.pct_change().dropna()
    total_return = float(equity_sr.iloc[-1] / equity_sr.iloc[0] - 1.0)
    cagr = float((1.0 + total_return) ** (252 / max(1, len(daily_ret))) - 1.0) if len(daily_ret) else 0.0
    mdd = _max_drawdown(equity_sr)
    sharpe = _sharpe(daily_ret)

    # Persist artifacts
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(cfg.out_dir) / f"simple_linreg_{cfg.symbol}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(trades, columns=["timestamp", "symbol", "qty", "price", "fees"]).to_csv(out / "trades.csv", index=False)
    equity_sr.rename("equity").to_csv(out / "equity.csv")

    params = asdict(cfg) | {"run_id": run_id}
    metrics = {
        "final_equity": float(equity_sr.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe_252": sharpe,
        "num_trades": len(trades),
        "start": _iso_utc(equity_sr.index[0]),
        "end": _iso_utc(equity_sr.index[-1]),
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Run summary ===")
    print(f"Symbol:        {cfg.symbol}")
    print(f"Period:        {metrics['start']} → {metrics['end']}")
    print(f"Trades:        {metrics['num_trades']}")
    print(f"Final Equity:  {metrics['final_equity']:,.2f} USD")
    print(f"Total Return:  {metrics['total_return']*100:,.2f}%")
    print(f"CAGR:          {metrics['cagr']*100:,.2f}%")
    print(f"Max Drawdown:  {metrics['max_drawdown']*100:,.2f}%")
    print(f"Sharpe (252):  {metrics['sharpe_252']:,.2f}")
    print(f"\nArtifacts:     {out.resolve()}")

    return out


# ----------------------------
# CLI
# ----------------------------
def _parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Run walk-forward OLS (return lags) backtest.")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d", choices=["1d"])
    p.add_argument("--lags", type=int, default=5)
    p.add_argument("--min-train", type=int, default=250)
    p.add_argument("--retrain-every", type=int, default=1)
    p.add_argument("--threshold-bps", type=float, default=5.0)
    p.add_argument("--qty", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=1.5)
    p.add_argument("--per-share-fee", type=float, default=0.005)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    return RunConfig(
        symbol=a.symbol,
        start=a.start,
        end=a.end,
        interval=a.interval,
        lags=a.lags,
        min_train=a.min_train,
        retrain_every=a.retrain_every,
        threshold_bps=a.threshold_bps,
        qty=a.qty,
        slippage_bps=a.slippage_bps,
        per_share_fee=a.per_share_fee,
        out_dir=a.out_dir,
        seed=a.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    run(cfg)