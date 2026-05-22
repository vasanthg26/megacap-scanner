"""OHLCV ingest from yfinance into DuckDB.

Designed so yfinance is swappable: replace _fetch_ohlcv() with a Polygon.io
implementation and nothing else changes.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers

logger = logging.getLogger(__name__)

_LOOKBACK_YEARS = 3

# Theme basket tickers + their benchmark ETFs + SPY (not in get_all_tickers()).
_THEME_TICKERS: list[str] = [
    "BB", "NOK", "AEHR", "HUT", "FPS", "BE", "YSS", "NVTS", # theme tickers
    "IGV", "IGN", "SMH", "WGMI", "GRID", "XLI", "ITA",      # benchmark ETFs
    "SPY",                                                     # needed for theme_active computation
]


def _fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from yfinance. Returns empty DataFrame on failure."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df.columns = ["open", "high", "low", "close", "volume", "adj_close"]
        df["ticker"] = ticker
        return df.reset_index()
    except Exception as exc:
        logger.error("yfinance fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def ingest_ticker(ticker: str, conn, start: str, end: str) -> int:
    """Upsert OHLCV rows for one ticker. Returns number of rows written."""
    df = _fetch_ohlcv(ticker, start, end)
    if df.empty:
        return 0

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]]
    df = df.dropna(subset=["close", "adj_close"])

    conn.register("_staging", df)
    conn.execute("""
        INSERT OR REPLACE INTO prices
        SELECT ticker, date, open, high, low, close, volume, adj_close
        FROM _staging
    """)
    conn.unregister("_staging")
    return len(df)


def ingest_all(tickers: list[str] | None = None) -> dict[str, str]:
    """
    Pull OHLCV for all universe tickers (or a subset) into DuckDB.
    Returns a status dict: ticker -> 'ok' | 'empty' | 'error'.
    Idempotent: re-running overwrites existing rows.
    """
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=_LOOKBACK_YEARS * 365 + 5)).strftime("%Y-%m-%d")

    universe = tickers or list(dict.fromkeys(get_all_tickers() + _THEME_TICKERS))
    results: dict[str, str] = {}

    with get_connection() as conn:
        for ticker in universe:
            fetched_at = datetime.now(timezone.utc)
            try:
                rows = ingest_ticker(ticker, conn, start, end)
                status = "ok" if rows > 0 else "empty"
                error_msg = None
                logger.info("%-6s  %s  rows=%d", ticker, status, rows)
            except Exception as exc:
                rows = 0
                status = "error"
                error_msg = str(exc)
                logger.error("%-6s  error: %s", ticker, exc)

            conn.execute("""
                INSERT OR REPLACE INTO ingest_log (ticker, fetched_at, rows_written, status, error_msg)
                VALUES (?, ?, ?, ?, ?)
            """, [ticker, fetched_at, rows, status, error_msg])

            results[ticker] = status

    return results
