# Readme

This is a production-minded backbone for algorithmic trading. It cleanly separates data ingestion, signal generation, portfolio/risk, deterministic backtesting, and broker execution—so you can develop multiple strategies, test them reproducibly, and deploy to paper (and later live) with minimal friction.

## Highlights

- Modular architecture: strategy-pure interfaces, adapter-based execution.
- Deterministic backtests: next-open execution modeling, explicit costs/slippage, reproducible KPIs (equity, drawdown, Sharpe).
- US equities first: Yahoo for research; paper trading via Interactive Brokers; optional crypto CEX connectors later.
- Production hygiene: unit tests, linting, CI, structured logging, simple state & scheduling.
- Cross-platform & container-friendly: runs locally or in Docker; optional Streamlit dashboard for monitoring and interactive controls.

## Community & Privacy Guidelines

I welcome issues, ideas, and pull requests. To keep the project efficient and respectful of privacy:

- Do not post personal information (yours or anyone else’s) in issues/PRs.
- Scrub logs and screenshots: remove account IDs, tokens, hostnames, IPs.
- Security findings: do not open a public issue; follow SECURITY.md.
- Contact: please use GitHub issues/PRs only. No off-platform outreach.
- Feedback scope: technical topics related to this repository.

By contributing, you agree that your contributions are Apache 2.0 licensed.

## License

Apache License 2.0 — see LICENSE. Attributions are listed in NOTICE.
Contributions are welcome; by contributing you agree your code is licensed under Apache-2.0.

Copyright © 2025 Lukas Kapferer

This product includes software developed by Lukas Kapferer.
This distribution may inlcude third-party components.
Licensed under the Apache License, Version 2.0.
