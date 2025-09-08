# Readme

This is a production-minded backbone for algorithmic trading. It cleanly separates data ingestion, signal generation, portfolio/risk, deterministic backtesting, and broker execution—so you can develop multiple strategies, test them reproducibly, and deploy to paper (and later live) with minimal friction.

## Highlights

- Modular architecture: strategy-pure interfaces, adapter-based execution.
- Deterministic backtests: next-open execution modeling, explicit costs/slippage, reproducible KPIs (equity, drawdown, Sharpe).
- US equities first: Yahoo for research; paper trading via Interactive Brokers; optional crypto CEX connectors later.
- Production hygiene: unit tests, linting, CI, structured logging, simple state & scheduling.
- Cross-platform & container-friendly: runs locally or in Docker; optional Streamlit dashboard for monitoring and interactive controls.
