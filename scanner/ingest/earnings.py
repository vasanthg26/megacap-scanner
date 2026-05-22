"""Next earnings date ingest from yfinance into DuckDB."""

import logging
from datetime import datetime, timezone, date

import yfinance as yf

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers

logger = logging.getLogger(__name__)


def _fetch_next_earnings(ticker: str) -> date | None:
    """Return the next earnings date >= today from yfinance calendar, or None if unavailable."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if not cal or not isinstance(cal, dict):
            return None
        dates = cal.get("Earnings Date", [])
        if not dates:
            return None
        today = date.today()
        future = [d for d in dates if (d.date() if hasattr(d, "date") else d) >= today]
        if not future:
            return None
        earliest = min(d.date() if hasattr(d, "date") else d for d in future)
        return earliest
    except Exception as exc:
        logger.warning("earnings date fetch failed for %s: %s", ticker, exc)
        return None


def _get_theme_tickers(conn) -> list[str]:
    try:
        return [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM themes").fetchall()]
    except Exception:
        return []


def ingest_earnings(tickers: list[str] | None = None) -> dict[str, str]:
    """
    Fetch next earnings dates for universe tickers (or a subset) into DuckDB.
    Returns a status dict: ticker -> 'ok' | 'null' | 'error'.
    Idempotent: re-running overwrites existing rows.
    """
    results: dict[str, str] = {}
    ingested_at = datetime.now(timezone.utc)

    conn = get_connection()
    try:
        universe = tickers or list(dict.fromkeys(get_all_tickers() + _get_theme_tickers(conn)))
        for ticker in universe:
            try:
                next_date = _fetch_next_earnings(ticker)
                conn.execute(
                    "INSERT OR REPLACE INTO earnings_dates (ticker, next_earnings_date, ingested_at) VALUES (?, ?, ?)",
                    [ticker, next_date, ingested_at],
                )
                status = "ok" if next_date is not None else "null"
                logger.info("%-6s  %s  next_earnings=%s", ticker, status, next_date)
            except Exception as exc:
                status = "error"
                logger.error("%-6s  earnings ingest error: %s", ticker, exc)
            results[ticker] = status
    finally:
        conn.close()

    return results
