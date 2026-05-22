"""Earnings revision momentum snapshot ingest from yfinance into DuckDB.

Stores a daily snapshot of analyst estimate revision counts.
revision_score = (analysts_up - analysts_down) / (analysts_up + analysts_down)
Range: -1 to +1. NULL when no analyst coverage (up + down == 0).

yfinance provides a CURRENT snapshot only — no historical revision time series.
Run this command once per day to accumulate history in DuckDB for backtesting.
"""

import logging
import math
from datetime import date

import yfinance as yf

from scanner.db import get_connection

logger = logging.getLogger(__name__)


def _safe_int(v) -> int:
    if v is None:
        return 0
    try:
        f = float(v)
        return 0 if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _fetch_estimates(ticker: str) -> dict | None:
    """Fetch current analyst estimate revision snapshot. Returns None on failure or no data."""
    try:
        t = yf.Ticker(ticker)
        revisions = t.eps_revisions
        estimates = t.earnings_estimate
    except Exception as exc:
        logger.error("estimates fetch failed for %s: %s", ticker, exc)
        return None

    if revisions is None or revisions.empty:
        logger.debug("no eps_revisions for %s", ticker)
        return None

    if "0q" not in revisions.index:
        logger.debug("0q period missing in eps_revisions for %s", ticker)
        return None

    row_0q = revisions.loc["0q"]
    analysts_up = _safe_int(row_0q.get("upLast30days"))
    analysts_down = _safe_int(row_0q.get("downLast30days"))

    total = analysts_up + analysts_down
    revision_score: float | None = (analysts_up - analysts_down) / total if total > 0 else None

    eps_current_q: float | None = None
    eps_next_q: float | None = None
    if estimates is not None and not estimates.empty:
        if "0q" in estimates.index:
            eps_current_q = _safe_float(estimates.loc["0q"].get("avg"))
        if "+1q" in estimates.index:
            eps_next_q = _safe_float(estimates.loc["+1q"].get("avg"))

    return {
        "eps_current_q": eps_current_q,
        "eps_next_q": eps_next_q,
        "analysts_up": analysts_up,
        "analysts_down": analysts_down,
        "revision_score": revision_score,
    }


def ingest_estimates(tickers: list[str] | None = None, as_of: date | None = None) -> dict[str, str]:
    """
    Snapshot current analyst estimate revision counts for the given tickers into DuckDB.
    Returns ticker -> 'ok' | 'no_data' | 'error'.
    Idempotent: re-running on the same day overwrites the existing row.
    """
    from scanner.graph.loader import get_all_tickers
    from scanner.signals.megacap import MEGACAP_UNIVERSE

    if tickers:
        universe = list(dict.fromkeys(tickers))
    else:
        with get_connection() as _conn:
            theme_tickers = [r[0] for r in _conn.execute("SELECT DISTINCT ticker FROM themes").fetchall()]
        universe = list(dict.fromkeys(MEGACAP_UNIVERSE + get_all_tickers() + theme_tickers))

    today = as_of or date.today()
    results: dict[str, str] = {}

    with get_connection() as conn:
        for ticker in universe:
            data = _fetch_estimates(ticker)
            if data is None:
                results[ticker] = "no_data"
                logger.info("%-6s  no estimate data", ticker)
                continue
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO estimates
                        (ticker, date, eps_current_q, eps_next_q,
                         analysts_up, analysts_down, revision_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ticker,
                        today,
                        data["eps_current_q"],
                        data["eps_next_q"],
                        data["analysts_up"],
                        data["analysts_down"],
                        data["revision_score"],
                    ],
                )
                score_str = (
                    f"{data['revision_score']:+.3f}"
                    if data["revision_score"] is not None
                    else "NULL"
                )
                logger.info(
                    "%-6s  ok  up=%d  down=%d  score=%s",
                    ticker,
                    data["analysts_up"],
                    data["analysts_down"],
                    score_str,
                )
                results[ticker] = "ok"
            except Exception as exc:
                logger.error("estimates DB write failed for %s: %s", ticker, exc)
                results[ticker] = "error"

    return results
