"""Pre-screener discovery pipeline.

Stages
======
1. discover()        — weekly: query Massive related-companies for each known parent,
                       apply quality gates, insert passing tickers as ACCUMULATING
2. accumulate_rs()   — daily: compute rolling RS score for each ACCUMULATING candidate,
                       advance to READY_FOR_BACKTEST at 60 days
3. run_discovery_backtests() — triggered: IC backtest on rs_accumulation, auto-promote
                               to dependencies.yaml on PASS, mark FAILED otherwise

Discovery source
================
Massive /v1/related-companies/{parent} — 10 calls (one per known parent).
No Claude API, no 10-K analysis, no XNAS universe sweep.

Promotion criteria (same evidentiary bar as main signal):
  ic_h1 > 0.05 AND ic_h2 > 0.05 (both halves positive)

Auto-promotion note: graph loader uses @lru_cache; restart process after promotion
to pick up new edges.
"""

import logging
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers, MEGA_CAPS, _load_edges

logger = logging.getLogger(__name__)

_MASSIVE_BASE = "https://api.massive.com"
_MASSIVE_RATE_SLEEP = 6.0   # 10 req/min starter plan

_RS_LOOKBACK = 20           # trading days for rolling return
_RS_MIN_DAYS = 60           # days of accumulation before backtest
_IC_THRESHOLD = 0.05
_FAILED_REELIGIBLE_DAYS = 90

# Quality gate thresholds
_MIN_PRICE = 5.0            # USD
_MIN_ADV = 10_000_000.0     # USD (avg daily close × volume)
_MIN_LIST_DAYS = 180        # calendar days since IPO/listing

_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_DEPS_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "dependencies.yaml"

_VALID_PARENTS = set(MEGA_CAPS)


# ---------- config helpers -------------------------------------------------------

def _get_massive_key() -> Optional[str]:
    import os
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if key:
        return key
    if _SETTINGS_PATH.exists():
        with _SETTINGS_PATH.open() as f:
            settings = yaml.safe_load(f) or {}
        key = (settings.get("MASSIVE_API_KEY") or "").strip()
        if key and key != "YOUR_MASSIVE_API_KEY_HERE":
            return key
    return None


# ---------- discovery source — related-companies ---------------------------------

def _fetch_related_companies(parent: str, api_key: str) -> list[str]:
    """Return related ticker symbols for parent from Massive related-companies endpoint."""
    url = f"{_MASSIVE_BASE}/v1/related-companies/{parent}?apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            logger.warning("%s: related-companies returned status %s", parent, data.get("status"))
            return []
        return [r["ticker"] for r in data.get("results", []) if r.get("ticker")]
    except Exception as exc:
        logger.error("%s: related-companies fetch failed: %s", parent, exc)
        return []


# ---------- quality gates --------------------------------------------------------

def _already_in_graph() -> set[str]:
    tickers = set(get_all_tickers())
    tickers.update(MEGA_CAPS)
    return tickers


def _build_exclude_sets(conn) -> tuple[set[str], set[str]]:
    """Return (active_in_discovery, recently_failed) sets for fast gate checks."""
    cutoff = (date.today() - timedelta(days=_FAILED_REELIGIBLE_DAYS)).isoformat()

    recently_failed = {
        r[0]
        for r in conn.execute(
            "SELECT ticker FROM discovery_candidates WHERE status = 'FAILED' AND discovered_date >= ?",
            [cutoff],
        ).fetchall()
    }
    active_in_discovery = {
        r[0]
        for r in conn.execute(
            "SELECT ticker FROM discovery_candidates WHERE status != 'FAILED'",
        ).fetchall()
    }
    return active_in_discovery, recently_failed


def _price_adv_from_db(ticker: str, conn) -> Optional[tuple[float, float, str]]:
    """Return (latest_close, avg_daily_value, oldest_date) from prices table, or None."""
    rows = conn.execute(
        """
        SELECT close, volume, date
        FROM prices
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 21
        """,
        [ticker],
    ).fetchall()

    if not rows:
        return None

    latest_close = rows[0][0]
    if latest_close is None:
        return None

    dollar_vols = [r[0] * r[1] for r in rows if r[0] is not None and r[1] is not None]
    avg_daily_value = sum(dollar_vols) / len(dollar_vols) if dollar_vols else 0.0

    oldest_date = str(rows[-1][2])
    return latest_close, avg_daily_value, oldest_date


def _price_adv_from_massive(ticker: str, api_key: str) -> Optional[tuple[float, float]]:
    """Fetch last 25 daily bars from Massive to compute price and ADV.

    Returns (latest_close, avg_daily_dollar_volume) or None on failure.
    """
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=40)).isoformat()
    url = (
        f"{_MASSIVE_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        f"?adjusted=true&sort=desc&limit=25&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(_MASSIVE_RATE_SLEEP)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        # results sorted desc: first is most recent
        latest_close = results[0].get("c")
        if latest_close is None:
            return None
        dollar_vols = [r["c"] * r["v"] for r in results if r.get("c") and r.get("v")]
        avg_dv = sum(dollar_vols) / len(dollar_vols) if dollar_vols else 0.0
        return float(latest_close), avg_dv
    except Exception as exc:
        logger.warning("%s: Massive aggs fetch failed: %s", ticker, exc)
        return None


def _list_date_from_massive(ticker: str, api_key: str) -> Optional[str]:
    """Fetch ticker detail from Massive to get list_date. Returns ISO date string or None."""
    url = f"{_MASSIVE_BASE}/v3/reference/tickers/{ticker}?apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(_MASSIVE_RATE_SLEEP)
        data = resp.json()
        result = data.get("results", {})
        return result.get("list_date")
    except Exception as exc:
        logger.warning("%s: Massive ticker detail fetch failed: %s", ticker, exc)
        return None


def _check_quality_gates(
    ticker: str,
    conn,
    api_key: str,
    in_graph: set[str],
    active_in_discovery: set[str],
    recently_failed: set[str],
) -> tuple[bool, str]:
    """Return (passes, skip_reason). skip_reason is empty string on pass."""
    if ticker in in_graph:
        return False, "already active"
    if ticker in active_in_discovery:
        return False, "already in discovery"
    if ticker in recently_failed:
        return False, f"failed within {_FAILED_REELIGIBLE_DAYS}d"

    # Price + ADV gate — prices table first, Massive fallback
    db_result = _price_adv_from_db(ticker, conn)
    if db_result is not None:
        latest_close, avg_dv, oldest_date = db_result
        if latest_close < _MIN_PRICE:
            return False, f"price ${latest_close:.2f} below ${_MIN_PRICE}"
        if avg_dv < _MIN_ADV:
            return False, f"ADV ${avg_dv:,.0f} below ${_MIN_ADV:,.0f}"
        # Listing age from oldest price date in DB (proxy)
        try:
            oldest = date.fromisoformat(oldest_date)
            days_listed = (date.today() - oldest).days
            if days_listed < _MIN_LIST_DAYS:
                return False, f"listed ~{days_listed}d ago (need {_MIN_LIST_DAYS}d)"
        except ValueError:
            pass
    else:
        # Not in prices table — fetch from Massive
        massive_result = _price_adv_from_massive(ticker, api_key)
        if massive_result is None:
            return False, "no price data available"
        latest_close, avg_dv = massive_result
        if latest_close < _MIN_PRICE:
            return False, f"price ${latest_close:.2f} below ${_MIN_PRICE}"
        if avg_dv < _MIN_ADV:
            return False, f"ADV ${avg_dv:,.0f} below ${_MIN_ADV:,.0f}"
        # Listing date from Massive detail endpoint
        list_date_str = _list_date_from_massive(ticker, api_key)
        if list_date_str:
            try:
                list_dt = date.fromisoformat(list_date_str)
                days_listed = (date.today() - list_dt).days
                if days_listed < _MIN_LIST_DAYS:
                    return False, f"listed {days_listed}d ago (need {_MIN_LIST_DAYS}d)"
            except ValueError:
                pass

    return True, ""


# ---------- discover() main ------------------------------------------------------

def discover(conn=None) -> int:
    """Query Massive related-companies for each known parent, apply quality gates,
    insert passing candidates as ACCUMULATING. Returns count of new insertions.
    """
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        api_key = _get_massive_key()
        if not api_key:
            logger.error("MASSIVE_API_KEY not set — discovery cannot run")
            return 0

        logger.info("Discovery: querying related companies for %d parents...", len(MEGA_CAPS))

        in_graph = _already_in_graph()
        active_in_discovery, recently_failed = _build_exclude_sets(conn)

        inserted = 0

        for parent in MEGA_CAPS:
            related = _fetch_related_companies(parent, api_key)
            logger.info("%s related: %d candidates returned", parent, len(related))
            time.sleep(_MASSIVE_RATE_SLEEP)

            for ticker in related:
                passes, reason = _check_quality_gates(
                    ticker, conn, api_key,
                    in_graph, active_in_discovery, recently_failed,
                )

                if not passes:
                    logger.info("  SKIP: %s — %s", ticker, reason)
                    continue

                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO discovery_candidates
                            (ticker, parent, dependency_strength, claude_confidence,
                             revenue_pct, evidence, source_accession,
                             discovered_date, status)
                        VALUES (?, ?, NULL, NULL, NULL, NULL,
                                'massive_related_companies', CURRENT_DATE, 'ACCUMULATING')
                        """,
                        [ticker, parent],
                    )
                    # Mark inserted in active set so duplicate parent hits are skipped
                    active_in_discovery.add(ticker)
                    inserted += 1
                    logger.info("  NEW: %s → %s (via related-companies)", ticker, parent)
                except Exception as exc:
                    logger.error("  %s → %s: insert failed: %s", ticker, parent, exc)

        logger.info("Discovery complete: %d new candidates inserted", inserted)
        return inserted

    finally:
        if _own_conn:
            conn.close()


# ---------- RS accumulation ------------------------------------------------------

def _load_price_series(ticker: str, conn) -> dict[str, float]:
    """Return {date_str: adj_close} for all available prices for ticker."""
    rows = conn.execute(
        "SELECT date, adj_close FROM prices WHERE ticker = ? ORDER BY date",
        [ticker],
    ).fetchall()
    return {str(r[0]): r[1] for r in rows if r[1] is not None}


def _compute_rs_series(
    child_prices: dict[str, float],
    parent_prices: dict[str, float],
    already_accumulated: set[str],
) -> list[tuple[str, float]]:
    """Return [(date, rs_score)] for all dates not yet accumulated.

    Loads both price series into memory and computes the rolling 20-day return
    differential in a single pass — avoids per-date DB queries during backfill.
    Only dates where both ticker and parent have at least _RS_LOOKBACK+1 bars
    of history up to that point are included.
    """
    child_dates = sorted(child_prices)
    parent_dates_set = set(parent_prices)

    results: list[tuple[str, float]] = []

    for i, d in enumerate(child_dates):
        if d in already_accumulated:
            continue
        if d not in parent_dates_set:
            continue

        # Need _RS_LOOKBACK+1 bars up to and including d for both tickers
        child_window = child_dates[max(0, i - _RS_LOOKBACK): i + 1]
        if len(child_window) < _RS_LOOKBACK + 1:
            continue

        # Build parent window: last _RS_LOOKBACK+1 parent dates <= d
        parent_window = sorted(dt for dt in parent_dates_set if dt <= d)
        if len(parent_window) < _RS_LOOKBACK + 1:
            continue
        parent_window = parent_window[-(  _RS_LOOKBACK + 1):]

        child_latest = child_prices[child_window[-1]]
        child_oldest = child_prices[child_window[0]]
        parent_latest = parent_prices[parent_window[-1]]
        parent_oldest = parent_prices[parent_window[0]]

        if not child_oldest or not parent_oldest:
            continue

        child_ret = (child_latest - child_oldest) / child_oldest
        parent_ret = (parent_latest - parent_oldest) / parent_oldest
        results.append((d, child_ret - parent_ret))

    return results


def accumulate_rs(conn=None) -> int:
    """Backfill and extend RS scores for all ACCUMULATING candidates.

    For each (ticker, parent) pair fetches full price history for both
    into memory, computes the 20-day rolling return differential for every
    date not yet in rs_accumulation, and bulk-inserts the results. On first
    run after price backfill this inserts up to ~500 rows per candidate.
    Returns total rows inserted.
    """
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        candidates = conn.execute(
            """
            SELECT ticker, parent FROM discovery_candidates
            WHERE status IN ('ACCUMULATING', 'READY_FOR_BACKTEST')
            """
        ).fetchall()

        if not candidates:
            return 0

        # Cache parent price series — multiple candidates share the same parent
        parent_cache: dict[str, dict[str, float]] = {}
        inserted = 0

        for ticker, parent in candidates:
            child_prices = _load_price_series(ticker, conn)
            if not child_prices:
                continue

            if parent not in parent_cache:
                parent_cache[parent] = _load_price_series(parent, conn)
            parent_prices = parent_cache[parent]
            if not parent_prices:
                continue

            already = {
                str(r[0])
                for r in conn.execute(
                    "SELECT date FROM rs_accumulation WHERE ticker = ? AND parent = ?",
                    [ticker, parent],
                ).fetchall()
            }

            new_rows = _compute_rs_series(child_prices, parent_prices, already)
            if not new_rows:
                continue

            conn.executemany(
                "INSERT OR IGNORE INTO rs_accumulation (ticker, parent, date, rs_score) VALUES (?, ?, ?, ?)",
                [(ticker, parent, d, score) for d, score in new_rows],
            )
            inserted += len(new_rows)
            logger.info("%s → %s: %d RS rows inserted", ticker, parent, len(new_rows))

        # Advance candidates that have reached 60 days of accumulation
        conn.execute(
            """
            UPDATE discovery_candidates
            SET status = 'READY_FOR_BACKTEST'
            WHERE status = 'ACCUMULATING'
              AND (ticker, parent) IN (
                  SELECT ticker, parent FROM rs_accumulation
                  GROUP BY ticker, parent
                  HAVING COUNT(DISTINCT date) >= ?
              )
            """,
            [_RS_MIN_DAYS],
        )

        return inserted

    finally:
        if _own_conn:
            conn.close()


# ---------- auto-backtest + promote ----------------------------------------------

def _spearman_ic(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = float(r + 1)
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / n
    std_x = math.sqrt(sum((v - mx) ** 2 for v in rx) / n)
    std_y = math.sqrt(sum((v - my) ** 2 for v in ry) / n)
    if std_x == 0 or std_y == 0:
        return float("nan")
    return cov / (std_x * std_y)


def _run_ic_backtest(ticker: str, parent: str, conn) -> dict:
    """Return IC metrics from rs_accumulation data for (ticker, parent)."""
    rows = conn.execute(
        """
        SELECT date, rs_score FROM rs_accumulation
        WHERE ticker = ? AND parent = ?
          AND rs_score IS NOT NULL
        ORDER BY date
        """,
        [ticker, parent],
    ).fetchall()

    if len(rows) < 10:
        return {"ic_h1": float("nan"), "ic_h2": float("nan"), "ic_full": float("nan"), "passed": False}

    dates = [r[0] for r in rows]
    scores = [r[1] for r in rows]

    # Compute forward 5-day return from prices for each rs_accumulation date
    pairs: list[tuple[float, float]] = []
    for d, score in zip(dates, scores):
        fwd_rows = conn.execute(
            """
            SELECT adj_close FROM prices
            WHERE ticker = ? AND date > ?
            ORDER BY date LIMIT 6
            """,
            [ticker, str(d)],
        ).fetchall()
        if len(fwd_rows) < 5:
            continue
        entry = fwd_rows[0][0]
        exit_ = fwd_rows[4][0]
        if not entry or entry == 0:
            continue
        fwd_ret = (exit_ - entry) / entry
        pairs.append((score, fwd_ret))

    if len(pairs) < 10:
        return {"ic_h1": float("nan"), "ic_h2": float("nan"), "ic_full": float("nan"), "passed": False}

    mid = len(pairs) // 2
    h1_pairs, h2_pairs = pairs[:mid], pairs[mid:]

    ic_full = _spearman_ic([p[0] for p in pairs], [p[1] for p in pairs])
    ic_h1 = (
        _spearman_ic([p[0] for p in h1_pairs], [p[1] for p in h1_pairs])
        if len(h1_pairs) >= 5 else float("nan")
    )
    ic_h2 = (
        _spearman_ic([p[0] for p in h2_pairs], [p[1] for p in h2_pairs])
        if len(h2_pairs) >= 5 else float("nan")
    )

    passed = (
        not math.isnan(ic_h1) and not math.isnan(ic_h2)
        and ic_h1 > _IC_THRESHOLD and ic_h2 > _IC_THRESHOLD
    )

    return {"ic_h1": ic_h1, "ic_h2": ic_h2, "ic_full": ic_full, "passed": passed}


def _append_to_dependencies_yaml(ticker: str, parent: str, evidence: str) -> None:
    """Append a new edge to dependencies.yaml."""
    with open(_DEPS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    new_edge = {
        "parent": parent,
        "child": ticker,
        "type": "systems",
        "weight": 0.3,
        "notes": f"Auto-promoted via discovery pipeline: {evidence[:80]}" if evidence else "Auto-promoted via discovery pipeline",
    }
    data["edges"].append(new_edge)

    with open(_DEPS_YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    _load_edges.cache_clear()
    logger.info("PROMOTED: %s → %s appended to dependencies.yaml", ticker, parent)


def run_discovery_backtests(conn=None) -> int:
    """Run IC backtest for READY_FOR_BACKTEST candidates. Returns count processed."""
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT ticker, parent, evidence
            FROM discovery_candidates
            WHERE status = 'READY_FOR_BACKTEST'
            """
        ).fetchall()

        if not rows:
            return 0

        today = date.today().isoformat()
        processed = 0

        for ticker, parent, evidence in rows:
            try:
                metrics = _run_ic_backtest(ticker, parent, conn)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO discovery_backtest_results
                        (ticker, parent, run_date, ic_h1, ic_h2, ic_full, passed, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ticker, parent, today,
                        metrics["ic_h1"], metrics["ic_h2"], metrics["ic_full"],
                        metrics["passed"], None,
                    ],
                )

                if metrics["passed"]:
                    conn.execute(
                        "UPDATE discovery_candidates SET status = 'PROMOTED' WHERE ticker = ? AND parent = ?",
                        [ticker, parent],
                    )
                    _append_to_dependencies_yaml(ticker, parent, evidence or "")
                    logger.info(
                        "PROMOTED: %s → %s  IC H1:%.3f H2:%.3f",
                        ticker, parent, metrics["ic_h1"], metrics["ic_h2"],
                    )
                else:
                    conn.execute(
                        "UPDATE discovery_candidates SET status = 'FAILED' WHERE ticker = ? AND parent = ?",
                        [ticker, parent],
                    )
                    logger.info(
                        "FAILED: %s → %s  IC H1:%s H2:%s",
                        ticker, parent,
                        f"{metrics['ic_h1']:.3f}" if not math.isnan(metrics["ic_h1"]) else "N/A",
                        f"{metrics['ic_h2']:.3f}" if not math.isnan(metrics["ic_h2"]) else "N/A",
                    )

                processed += 1

            except Exception as exc:
                logger.error("%s → %s: backtest failed: %s", ticker, parent, exc)
                continue

        return processed

    finally:
        if _own_conn:
            conn.close()


# ---------- display data helpers -------------------------------------------------

def get_discovery_display(conn) -> dict:
    """Return discovery pipeline data for UI rendering."""
    thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
    seven_days_ago = (date.today() - timedelta(days=7)).isoformat()

    accumulating_rows = conn.execute(
        """
        SELECT dc.ticker, dc.parent, dc.dependency_strength,
               COUNT(DISTINCT ra.date) AS days_accumulated,
               AVG(ra.rs_score) AS avg_rs
        FROM discovery_candidates dc
        LEFT JOIN rs_accumulation ra ON dc.ticker = ra.ticker AND dc.parent = ra.parent
        WHERE dc.status = 'ACCUMULATING'
        GROUP BY dc.ticker, dc.parent, dc.dependency_strength
        ORDER BY days_accumulated DESC
        """
    ).fetchall()

    accumulating = []
    for ticker, parent, strength, days, avg_rs in accumulating_rows:
        last5 = conn.execute(
            """
            SELECT AVG(rs_score) FROM (
                SELECT rs_score FROM rs_accumulation
                WHERE ticker = ? AND parent = ?
                ORDER BY date DESC LIMIT 5
            )
            """,
            [ticker, parent],
        ).fetchone()[0]

        prior5 = conn.execute(
            """
            SELECT AVG(rs_score) FROM (
                SELECT rs_score FROM rs_accumulation
                WHERE ticker = ? AND parent = ?
                ORDER BY date DESC LIMIT 10
            ) t WHERE rs_score NOT IN (
                SELECT rs_score FROM rs_accumulation
                WHERE ticker = ? AND parent = ?
                ORDER BY date DESC LIMIT 5
            )
            """,
            [ticker, parent, ticker, parent],
        ).fetchone()[0]

        if last5 is not None and prior5 is not None:
            trend = "↑" if last5 > prior5 + 0.002 else ("↓" if last5 < prior5 - 0.002 else "→")
        else:
            trend = "→"

        accumulating.append({
            "ticker": ticker,
            "parent": parent,
            "strength": strength,
            "days": days or 0,
            "rs": avg_rs,
            "trend": trend,
        })

    ready_rows = conn.execute(
        """
        SELECT dc.ticker, dc.parent, COUNT(DISTINCT ra.date) AS days
        FROM discovery_candidates dc
        LEFT JOIN rs_accumulation ra ON dc.ticker = ra.ticker AND dc.parent = ra.parent
        WHERE dc.status = 'READY_FOR_BACKTEST'
        GROUP BY dc.ticker, dc.parent
        """
    ).fetchall()
    ready = [{"ticker": r[0], "parent": r[1], "days": r[2]} for r in ready_rows]

    promoted_rows = conn.execute(
        """
        SELECT dbr.ticker, dbr.parent, dbr.ic_h1, dbr.ic_h2, dbr.run_date
        FROM discovery_backtest_results dbr
        JOIN discovery_candidates dc ON dbr.ticker = dc.ticker AND dbr.parent = dc.parent
        WHERE dc.status = 'PROMOTED' AND dbr.run_date >= ?
        ORDER BY dbr.run_date DESC
        """,
        [thirty_days_ago],
    ).fetchall()
    promoted = [
        {"ticker": r[0], "parent": r[1], "ic_h1": r[2], "ic_h2": r[3], "run_date": str(r[4])}
        for r in promoted_rows
    ]

    failed_rows = conn.execute(
        """
        SELECT dbr.ticker, dbr.parent, dbr.ic_h1, dbr.ic_h2
        FROM discovery_backtest_results dbr
        JOIN discovery_candidates dc ON dbr.ticker = dc.ticker AND dbr.parent = dc.parent
        WHERE dc.status = 'FAILED' AND dbr.run_date >= ?
        ORDER BY dbr.run_date DESC
        """,
        [thirty_days_ago],
    ).fetchall()
    failed = [
        {"ticker": r[0], "parent": r[1], "ic_h1": r[2], "ic_h2": r[3]}
        for r in failed_rows
    ]

    new_rows = conn.execute(
        """
        SELECT ticker, parent, claude_confidence
        FROM discovery_candidates
        WHERE discovered_date >= ?
        ORDER BY discovered_date DESC
        """,
        [seven_days_ago],
    ).fetchall()
    new_candidates = [
        {"ticker": r[0], "parent": r[1], "confidence": r[2]}
        for r in new_rows
    ]

    return {
        "accumulating": accumulating,
        "ready": ready,
        "promoted": promoted,
        "failed": failed,
        "new_candidates": new_candidates,
    }
