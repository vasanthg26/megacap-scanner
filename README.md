# Mega-Cap Scanner

A Python stock scanner that models tech mega-caps and their supply-chain / infrastructure
dependents as a **directed graph**. Signals operate over edges (parent → child relationships),
not flat ticker lists.

## Features (Phase 1)

- **Dependency graph** — typed edges with weights from `config/dependencies.yaml`
- **OHLCV ingest** — 3 years of daily data into a local DuckDB file
- **Relative strength signal** — rolling return differential vs parent, scaled by edge weight
- **Walk-forward backtest** — weekly rebalance, IC as the headline metric
- **CLI** — `ingest`, `scan`, `backtest` commands via Typer

## Quick Start

```bash
pip install -e ".[dev]"

# Pull 3 years of OHLCV for the full universe (~35 tickers)
python -m scanner ingest

# Rank dependents by RS score as of today
python -m scanner scan

# Run a walk-forward backtest
python -m scanner backtest --signal rs --horizon 5

# Run tests
pytest
```

## Universe

**Mega-caps** (seed): NVDA, MSFT, AAPL, GOOGL, AMZN, META, TSLA, AVGO, ORCL, TSM

**Dependents** — see `config/dependencies.yaml` for full edge list with types and weights.

## ⚠️ Survivorship-Bias Caveat

yfinance only returns currently-listed tickers. Backtest Sharpe ratios and drawdowns
computed with yfinance data are **overstated** because bankrupt or delisted tickers are
absent from the return history. This is a prototype data source.

**Before trusting any live performance numbers**, migrate the `_fetch_ohlcv()` function
in `scanner/ingest/prices.py` to Polygon.io, which provides delisted ticker history.
The ingest layer is designed to make this swap a single-function change.

## Architecture

```
Signal protocol  →  signals/*.py  →  backtest/runner.py  →  CLI
                                          ↑
                        graph/loader.py (edges + weights)
                                          ↑
                        config/dependencies.yaml
```

## Adding a Signal

See `CLAUDE.md` — "How to Add a New Signal".

## Phase 2 Roadmap

- Form 4 insider ingest with offering-participation filter (see `ingest/insiders.py`)
- Additional signals: insider cluster buys, earnings proximity, relative volume
- FastAPI wrapper for web deployment
- Polygon.io swap for survivorship-bias-free backtesting
