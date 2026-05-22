"""Theme basket scanner — computes signal metrics for theme tickers vs benchmark ETFs."""

import logging
import math
from datetime import datetime, timezone

import duckdb

from scanner.signals.base import (
    calc_days_above_50ma,
    calc_price_range_pct,
    calc_rs_trend,
    calc_rvol_trend,
)
from scanner.signals.confirmation import compute_confirmation_dependent

logger = logging.getLogger(__name__)

# Tickers with validated backtest evidence — receive action labels based on rs_score.
# All others remain WATCH until evidence accumulates.
VALIDATED_THEMES: set[str] = {"HUT", "NOK"}

_RS_THRESHOLDS = [
    (0.15, "STRONG_BUY"),
    (0.05, "BUY"),
    (-0.05, "HOLD"),
]


def _action_label_from_rs(rs_score: float | None) -> str:
    if rs_score is None or math.isnan(rs_score):
        return "WATCH"
    for threshold, label in _RS_THRESHOLDS:
        if rs_score >= threshold:
            return label
    return "SELL"


def _calc_rs_vs_benchmark(
    ticker: str, benchmark: str, as_of: str, conn: duckdb.DuckDBPyConnection
) -> float:
    """20-day return of ticker minus 20-day return of benchmark ETF."""
    rows = conn.execute(
        """
        SELECT ticker, adj_close, rn FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices WHERE ticker IN (?, ?) AND date <= ?
        ) t WHERE rn <= 21
        """,
        [ticker, benchmark, as_of],
    ).fetchall()

    closes: dict[str, list[float | None]] = {}
    for tkr, c, rn in rows:
        if tkr not in closes:
            closes[tkr] = [None] * 21
        if rn <= 21:
            closes[tkr][rn - 1] = float(c) if c is not None else None

    def _ret(tkr: str) -> float:
        c = closes.get(tkr, [None] * 21)
        if c[0] is None or c[20] is None or c[20] == 0:
            return float("nan")
        return (c[0] - c[20]) / c[20]

    t_ret = _ret(ticker)
    b_ret = _ret(benchmark)
    if math.isnan(t_ret) or math.isnan(b_ret):
        return float("nan")
    return t_ret - b_ret


def _calc_rvol(ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection) -> float:
    """Today's volume / 20-day average volume."""
    rows = conn.execute(
        """
        SELECT volume, rn FROM (
            SELECT volume, ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
            FROM prices WHERE ticker = ? AND date <= ?
        ) t WHERE rn <= 21
        """,
        [ticker, as_of],
    ).fetchall()
    if len(rows) < 21:
        return float("nan")
    rows_sorted = sorted(rows, key=lambda x: x[1])
    vols = [float(r[0]) for r in rows_sorted if r[0] is not None]
    if len(vols) < 21:
        return float("nan")
    avg_20 = sum(vols[1:21]) / 20
    if avg_20 == 0:
        return float("nan")
    return vols[0] / avg_20


def _calc_theme_active(benchmark: str, as_of: str, conn: duckdb.DuckDBPyConnection) -> str:
    """ACTIVE / NEUTRAL / INACTIVE based on benchmark ETF 20-day return vs SPY."""
    spy_ret = _calc_rs_vs_benchmark(benchmark, "SPY", as_of, conn)
    if math.isnan(spy_ret):
        return "UNKNOWN"
    if spy_ret > 0:
        return "ACTIVE"
    if spy_ret >= -0.02:
        return "NEUTRAL"
    return "INACTIVE"


def _get_insider_annotation(
    ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection
) -> tuple[str, bool]:
    """Return (annotation_string, has_buying) for insider enrichment."""
    try:
        from datetime import date as _date
        from scanner.enrichment.insiders import get_insider_summary
        as_of_date = _date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        summary = get_insider_summary(ticker, as_of_date, conn)
        if not summary:
            return ("", False)
        buy_count = summary.get("buy_count", 0)
        has_buying = buy_count > 0
        if not has_buying:
            return ("", False)
        parts = []
        if summary.get("has_cluster_buy"):
            parts.append("cluster")
        if summary.get("has_officer_buys"):
            parts.append("officer")
        dollar = summary.get("total_dollar_value") or 0
        if dollar >= 1_000_000:
            parts.append(f"${dollar / 1_000_000:.1f}M")
        elif dollar >= 1_000:
            parts.append(f"${dollar / 1_000:.0f}K")
        annotation = f"★ {', '.join(parts)}" if parts else "★"
        return (annotation, True)
    except Exception as exc:
        logger.warning("Insider annotation failed for %s: %s", ticker, exc)
        return ("", False)


def _get_earnings_days(ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection) -> int | None:
    """Days until next earnings, or None if unknown."""
    try:
        row = conn.execute(
            "SELECT next_earnings_date FROM earnings_dates WHERE ticker = ?", [ticker]
        ).fetchone()
        if not row or row[0] is None:
            return None
        from datetime import date
        as_of_date = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        earnings_date = row[0].date() if hasattr(row[0], "date") and callable(row[0].date) else row[0]
        delta = (earnings_date - as_of_date).days
        return delta if delta >= 0 else None
    except Exception as exc:
        logger.debug("earnings days calc failed for %s: %s", ticker, exc)
        return None


def _get_est_rev(ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection) -> float | None:
    """Latest revision_score from estimates table, or None.

    No date <= as_of bound: estimates are ingested as today's snapshot and may
    be dated after the last price date (e.g. ingested today, prices close yesterday).
    Theme scanner is a live viewer, not a backtest — always use most recent estimate.
    """
    row = conn.execute(
        "SELECT revision_score FROM estimates WHERE ticker = ? ORDER BY date DESC LIMIT 1",
        [ticker],
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        v = float(row[0])
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _get_price(ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection) -> float | None:
    row = conn.execute(
        "SELECT adj_close FROM prices WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1",
        [ticker, as_of],
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def compute_theme_scan(conn: duckdb.DuckDBPyConnection, as_of: str) -> list[dict]:
    """Compute theme basket signals for all active themes as of `as_of` date.

    Writes results to theme_scan_results (DELETE + INSERT for idempotency).
    Returns list of result dicts.
    """
    themes = conn.execute(
        "SELECT theme_name, ticker, benchmark_etf, theme_label FROM themes WHERE is_active = true ORDER BY theme_name, ticker"
    ).fetchall()

    if not themes:
        logger.warning("No active themes found in DB")
        return []

    results = []

    for theme_name, ticker, benchmark_etf, theme_label in themes:
        try:
            price = _get_price(ticker, as_of, conn)
            rs_score = _calc_rs_vs_benchmark(ticker, benchmark_etf, as_of, conn)
            rs_trend = calc_rs_trend(ticker, benchmark_etf, as_of, conn)
            rvol = _calc_rvol(ticker, as_of, conn)
            rvol_trend = calc_rvol_trend(ticker, as_of, conn)
            days_above = calc_days_above_50ma(ticker, as_of, conn)
            price_range_pct = calc_price_range_pct(ticker, as_of, conn)
            theme_active = _calc_theme_active(benchmark_etf, as_of, conn)
            insider_annotation, has_buying = _get_insider_annotation(ticker, as_of, conn)
            earnings_days = _get_earnings_days(ticker, as_of, conn)
            est_rev = _get_est_rev(ticker, as_of, conn)

            above_50ma = days_above > 0
            confirm = compute_confirmation_dependent(
                rs_score=rs_score,
                rvol=rvol if not math.isnan(rvol) else float("nan"),
                rev_score=est_rev,
                insider_buying=has_buying,
                above_50ma=above_50ma,
            )

            result = {
                "scan_date": as_of,
                "ticker": ticker,
                "theme_name": theme_name,
                "theme_label": theme_label,
                "benchmark_etf": benchmark_etf,
                "price": price,
                "rs_score": None if math.isnan(rs_score) else rs_score,
                "rs_trend": rs_trend,
                "rvol": None if math.isnan(rvol) else rvol,
                "rvol_trend": rvol_trend,
                "confirm": confirm,
                "days_above_50ma": days_above,
                "price_range_pct": None if math.isnan(price_range_pct) else price_range_pct,
                "est_rev": est_rev,
                "insider_annotation": insider_annotation or None,
                "earnings_days": earnings_days,
                "theme_active": theme_active,
                "action_label": (
                    _action_label_from_rs(None if math.isnan(rs_score) else rs_score)
                    if ticker in VALIDATED_THEMES
                    else "WATCH"
                ),
            }
            results.append(result)
        except Exception as exc:
            logger.error("Theme scan failed for %s/%s: %s", theme_name, ticker, exc)

    if results:
        conn.execute(
            "DELETE FROM theme_scan_results WHERE scan_date = ?", [as_of]
        )
        for r in results:
            conn.execute(
                """
                INSERT INTO theme_scan_results (
                    scan_date, ticker, theme_name, benchmark_etf, price,
                    rs_score, rs_trend, rvol, rvol_trend, confirm,
                    days_above_50ma, price_range_pct, est_rev,
                    insider_annotation, earnings_days, theme_active, action_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    r["scan_date"], r["ticker"], r["theme_name"], r["benchmark_etf"],
                    r["price"], r["rs_score"], r["rs_trend"], r["rvol"], r["rvol_trend"],
                    r["confirm"], r["days_above_50ma"], r["price_range_pct"], r["est_rev"],
                    r["insider_annotation"], r["earnings_days"], r["theme_active"], r["action_label"],
                ],
            )
        logger.info("Theme scan complete: %d results written for %s", len(results), as_of)

    return results
