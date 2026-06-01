"""Pre-screener discovery pipeline.

Stages
======
1. discover()        — weekly: fetch XNAS universe, 10-K analysis via Claude Haiku,
                       insert candidates with status=ACCUMULATING
2. accumulate_rs()   — daily: compute rolling RS score for each ACCUMULATING candidate,
                       advance to READY_FOR_BACKTEST at 60 days
3. run_discovery_backtests() — triggered: IC backtest on rs_accumulation, auto-promote
                               to dependencies.yaml on PASS, mark FAILED otherwise

Promotion criteria (same evidentiary bar as main signal):
  ic_h1 > 0.05 AND ic_h2 > 0.05 (both halves positive)

Claude model: claude-haiku-4-5-20251001 (10-K analysis only)
10-K sections: business + risk_factors (customer_concentration not available via Massive)

Auto-promotion note: graph loader uses @lru_cache; restart process after promotion
to pick up new edges.
"""

import json
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
_SECTION_RATE_SLEEP = 6.0   # same endpoint, same limit
_CLAUDE_CONFIDENCE_MIN = 0.70
_VALID_STRENGTHS = {"STRONG", "MEDIUM"}
_RS_LOOKBACK = 20           # trading days for rolling return
_RS_MIN_DAYS = 60           # days of accumulation before backtest
_IC_THRESHOLD = 0.05
_FAILED_REELIGIBLE_DAYS = 90
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


def _get_anthropic_key() -> Optional[str]:
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    if _SETTINGS_PATH.exists():
        with _SETTINGS_PATH.open() as f:
            settings = yaml.safe_load(f) or {}
        key = (settings.get("anthropic_api_key") or "").strip()
        if key and key != "YOUR_ANTHROPIC_API_KEY_HERE":
            return key
    return None


# ---------- Part 2B — universe fetch ---------------------------------------------

def _fetch_xnas_tickers(api_key: str) -> list[dict]:
    """Return all active XNAS common-stock tickers from Massive reference endpoint."""
    results: list[dict] = []
    url: Optional[str] = (
        f"{_MASSIVE_BASE}/v3/reference/tickers"
        f"?market=stocks&exchange=XNAS&active=true&type=CS&limit=250"
        f"&apiKey={api_key}"
    )

    while url:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Massive reference tickers fetch failed: %s", exc)
            break

        data = resp.json()
        if data.get("status") != "OK":
            logger.error("Massive reference tickers error: %s", data.get("error"))
            break

        for r in data.get("results", []):
            if r.get("type") == "CS" and r.get("active"):
                results.append({"ticker": r["ticker"], "cik": r.get("cik", "")})

        next_url = data.get("next_url")
        url = f"{next_url}&apiKey={api_key}" if next_url else None
        time.sleep(_MASSIVE_RATE_SLEEP)

    return results


def _already_in_graph() -> set[str]:
    all_t = set(get_all_tickers())
    all_t.update(MEGA_CAPS)
    return all_t


def _already_in_discovery(conn) -> set[str]:
    rows = conn.execute("SELECT ticker FROM discovery_candidates").fetchall()
    return {r[0] for r in rows}


def _recently_failed(conn) -> set[str]:
    cutoff = (date.today() - timedelta(days=_FAILED_REELIGIBLE_DAYS)).isoformat()
    rows = conn.execute(
        """
        SELECT ticker FROM discovery_candidates
        WHERE status = 'FAILED'
          AND discovered_date >= ?
        """,
        [cutoff],
    ).fetchall()
    return {r[0] for r in rows}


def _get_universe_candidates(api_key: str, conn) -> list[str]:
    """Return XNAS tickers eligible for 10-K analysis."""
    all_xnas = _fetch_xnas_tickers(api_key)
    in_graph = _already_in_graph()
    in_discovery = _already_in_discovery(conn)
    recently_failed = _recently_failed(conn)

    exclude = in_graph | in_discovery | recently_failed
    return [r["ticker"] for r in all_xnas if r["ticker"] not in exclude]


# ---------- Part 2C — 10-K dependency analysis ----------------------------------

def _fetch_10k_section(ticker: str, section: str, api_key: str) -> str:
    """Return most recent 10-K section text for ticker, or empty string on failure."""
    url = (
        f"{_MASSIVE_BASE}/stocks/filings/10-K/vX/sections"
        f"?ticker={ticker}&section={section}&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(_SECTION_RATE_SLEEP)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""
        # Take the most recent filing
        results_sorted = sorted(results, key=lambda r: r.get("filing_date", ""), reverse=True)
        return results_sorted[0].get("text", "")
    except Exception as exc:
        logger.warning("%s: 10-K section '%s' fetch failed: %s", ticker, section, exc)
        return ""


def _analyze_10k_for_dependency(
    ticker: str, combined_text: str, anthropic_client
) -> Optional[dict]:
    """Call Claude Haiku to check if ticker has a significant revenue dependency.

    Returns dict with parent/revenue_pct/evidence/strength/confidence, or None.
    """
    if not combined_text.strip():
        return None

    # Cap text to ~6000 chars to keep Haiku cost low
    text_cap = combined_text[:6000]
    parents_list = ", ".join(MEGA_CAPS)

    prompt = (
        f"Read these 10-K filing sections for {ticker}.\n\n"
        f"Answer: Does this company derive significant revenue from any of these companies?\n"
        f"[{parents_list}]\n\n"
        f"If yes, respond with this exact JSON:\n"
        f'{{"parent": "NVDA", "revenue_pct": 35.0, "evidence": "quote max 100 chars", '
        f'"strength": "STRONG", "confidence": 0.91}}\n\n'
        f"Strength definitions:\n"
        f"STRONG: >20% revenue or named primary customer\n"
        f"MEDIUM: 10-20% revenue or named customer\n"
        f"WEAK: <10% or indirect relationship\n\n"
        f"If no dependency found, respond with:\n"
        f'{{"parent": null, "confidence": 0.0}}\n\n'
        f"Respond only in JSON. No preamble. No markdown.\n\n"
        f"Filing sections:\n{text_cap}"
    )

    try:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="You are a financial analyst. Respond only in JSON. No preamble. No markdown.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result
    except json.JSONDecodeError as exc:
        logger.warning("%s: Claude returned invalid JSON: %s", ticker, exc)
        return None
    except Exception as exc:
        logger.error("%s: Claude API call failed: %s", ticker, exc)
        return None


# ---------- Part 2B+2C — discover() main ----------------------------------------

def discover(conn=None) -> int:
    """Fetch XNAS universe, run 10-K analysis, insert ACCUMULATING candidates.

    Returns count of new candidates inserted.
    """
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        massive_key = _get_massive_key()
        if not massive_key:
            logger.error("MASSIVE_API_KEY not set — discovery cannot run")
            return 0

        anthropic_key = _get_anthropic_key()
        if not anthropic_key:
            logger.error("ANTHROPIC_API_KEY not set — discovery cannot run")
            return 0

        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key, timeout=30.0)

        candidates = _get_universe_candidates(massive_key, conn)
        logger.info("Discovery: %d candidates to analyze", len(candidates))

        inserted = 0

        for ticker in candidates:
            try:
                business_text = _fetch_10k_section(ticker, "business", massive_key)
                risk_text = _fetch_10k_section(ticker, "risk_factors", massive_key)
                combined = f"{business_text}\n\n{risk_text}".strip()

                if not combined:
                    logger.info("%s: no 10-K sections found — skipping", ticker)
                    continue

                result = _analyze_10k_for_dependency(ticker, combined, client)
                if not result:
                    continue

                parent = result.get("parent")
                confidence = float(result.get("confidence", 0.0))
                strength = result.get("strength", "WEAK")

                if parent is None or parent not in _VALID_PARENTS:
                    logger.info("%s: no qualifying dependency found (parent=%s)", ticker, parent)
                    continue
                if confidence < _CLAUDE_CONFIDENCE_MIN:
                    logger.info(
                        "%s: confidence %.2f below threshold — skipping", ticker, confidence
                    )
                    continue
                if strength not in _VALID_STRENGTHS:
                    logger.info("%s: strength %s not STRONG/MEDIUM — skipping", ticker, strength)
                    continue

                conn.execute(
                    """
                    INSERT OR IGNORE INTO discovery_candidates
                        (ticker, parent, dependency_strength, claude_confidence,
                         revenue_pct, evidence, discovered_date, status)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_DATE, 'ACCUMULATING')
                    """,
                    [
                        ticker, parent, strength, confidence,
                        result.get("revenue_pct"), result.get("evidence", "")[:200],
                    ],
                )
                inserted += 1
                logger.info(
                    "DISCOVERED: %s → %s  strength=%s  confidence=%.2f",
                    ticker, parent, strength, confidence,
                )

            except Exception as exc:
                logger.error("%s: discovery failed unexpectedly: %s", ticker, exc)
                continue

        logger.info("Discovery complete: %d new candidates inserted", inserted)
        return inserted

    finally:
        if _own_conn:
            conn.close()


# ---------- Part 2D — RS accumulation -------------------------------------------

def _rolling_return(ticker: str, as_of: str, conn) -> Optional[float]:
    """20-day rolling return for ticker as of date, no-lookahead."""
    rows = conn.execute(
        """
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC LIMIT ?
        """,
        [ticker, as_of, _RS_LOOKBACK + 1],
    ).fetchall()

    if len(rows) < _RS_LOOKBACK + 1:
        return None
    latest = rows[0][0]
    oldest = rows[-1][0]
    if not oldest or oldest == 0:
        return None
    return (latest - oldest) / oldest


def accumulate_rs(conn=None) -> int:
    """Compute daily RS score for all ACCUMULATING candidates. Returns rows inserted."""
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT ticker, parent FROM discovery_candidates
            WHERE status = 'ACCUMULATING'
            """
        ).fetchall()

        if not rows:
            return 0

        today = date.today().isoformat()
        inserted = 0

        for ticker, parent in rows:
            child_ret = _rolling_return(ticker, today, conn)
            parent_ret = _rolling_return(parent, today, conn)

            if child_ret is None or parent_ret is None:
                continue

            rs_score = child_ret - parent_ret

            conn.execute(
                """
                INSERT OR IGNORE INTO rs_accumulation (ticker, parent, date, rs_score)
                VALUES (?, ?, ?, ?)
                """,
                [ticker, parent, today, rs_score],
            )
            inserted += 1

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


# ---------- Part 2E — auto-backtest + promote ------------------------------------

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
    for i, (d, score) in enumerate(zip(dates, scores)):
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

    xs_full = [p[0] for p in pairs]
    ys_full = [p[1] for p in pairs]
    ic_full = _spearman_ic(xs_full, ys_full)

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


def _append_to_dependencies_yaml(ticker: str, parent: str, confidence: float, evidence: str) -> None:
    """Append a new WEAK edge to dependencies.yaml under the correct parent block."""
    with open(_DEPS_YAML_PATH) as f:
        content = f.read()
        data = yaml.safe_load(content)

    new_edge = {
        "parent": parent,
        "child": ticker,
        "type": "systems",
        "weight": 0.3,
        "notes": f"Auto-discovered: {evidence[:80]}" if evidence else "Auto-discovered via discovery pipeline",
    }

    data["edges"].append(new_edge)

    with open(_DEPS_YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Clear LRU cache so next graph access picks up the new edge
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
            SELECT ticker, parent, claude_confidence, evidence
            FROM discovery_candidates
            WHERE status = 'READY_FOR_BACKTEST'
            """
        ).fetchall()

        if not rows:
            return 0

        today = date.today().isoformat()
        processed = 0

        for ticker, parent, confidence, evidence in rows:
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
                    _append_to_dependencies_yaml(ticker, parent, confidence or 0.0, evidence or "")
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


# ---------- Part 2F — display data helpers --------------------------------------

def get_discovery_display(conn) -> dict:
    """Return discovery pipeline data for UI rendering."""
    today = date.today().isoformat()
    thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
    seven_days_ago = (date.today() - timedelta(days=7)).isoformat()

    # Accumulating candidates with RS day count and recent trend
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

    # RS trend: last 5 days vs prior 5 days
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

    # Ready for backtest
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

    # Recently promoted (last 30 days)
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

    # Recently failed (last 30 days)
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

    # Recently discovered (last 7 days)
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
