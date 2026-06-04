"""FastAPI web application — read-only dashboard over scanner data."""

import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache — keys are endpoint names, values are (payload, fetched_at)
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[Any, datetime]] = {}
_CACHE_TTL_SECONDS = 600  # 10 minutes

# Bump when new fields are added to scan results — forces DB cache invalidation
_SCAN_SCHEMA_VERSION = 3


def _cache_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    payload, fetched_at = entry
    if (datetime.utcnow() - fetched_at).total_seconds() > _CACHE_TTL_SECONDS:
        return None
    return payload


def _cache_set(key: str, payload: Any) -> None:
    _cache[key] = (payload, datetime.utcnow())


# ---------------------------------------------------------------------------
# Helpers shared across endpoints
# ---------------------------------------------------------------------------

def _nan_to_none(v: float) -> float | None:
    return None if math.isnan(v) else v


def _rolling_return_pct(ticker: str, lookback: int, as_of: str, conn) -> float:
    rows = conn.execute(
        """
        SELECT adj_close
        FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
        """,
        [ticker, as_of, lookback + 1],
    ).fetchall()
    if len(rows) < lookback + 1:
        return float("nan")
    latest, oldest = rows[0][0], rows[-1][0]
    if oldest == 0:
        return float("nan")
    return (latest - oldest) / oldest


def _batch_rvol(tickers: list[str], as_of: str, conn) -> dict[str, float]:
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, volume, rn
        FROM (
            SELECT ticker, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 21
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, vol, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(vol) if vol is not None else 0.0))
    result: dict[str, float] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        vols = [v for _, v in series]
        if len(vols) < 2:
            result[tkr] = float("nan")
            continue
        today_vol = vols[0]
        avg_vol = sum(vols[1:]) / len(vols[1:])
        result[tkr] = today_vol / avg_vol if avg_vol > 0 else float("nan")
    return result


def _batch_above_50ma(tickers: list[str], as_of: str, conn) -> dict[str, bool]:
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 51
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, close, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(close)))
    result: dict[str, bool] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        closes = [c for _, c in series]
        if len(closes) < 51:
            result[tkr] = False
            continue
        latest = closes[0]
        ma50 = sum(closes[1:51]) / 50
        result[tkr] = latest > ma50
    return result


def _batch_rvol_trend(tickers: list[str], as_of: str, conn) -> dict[str, str]:
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, volume, rn
        FROM (
            SELECT ticker, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 26
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, vol, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(vol) if vol is not None else 0.0))
    result: dict[str, str] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        vols = [v for _, v in series]
        if len(vols) < 21:
            result[tkr] = "—"
            continue
        today_avg = sum(vols[1:21]) / 20
        if today_avg == 0:
            result[tkr] = "—"
            continue
        today_rvol = vols[0] / today_avg
        if len(vols) < 26:
            result[tkr] = "→"
            continue
        ago_avg = sum(vols[6:26]) / 20
        if ago_avg == 0:
            result[tkr] = "→"
            continue
        ago_rvol = vols[5] / ago_avg
        if today_rvol > ago_rvol * 1.10:
            result[tkr] = "↑"
        elif today_rvol < ago_rvol * 0.90:
            result[tkr] = "↓"
        else:
            result[tkr] = "→"
    return result


def _batch_rs_trend(
    ticker_parent_pairs: list[tuple[str, str]], as_of: str, conn
) -> dict[tuple[str, str], str]:
    if not ticker_parent_pairs:
        return {}
    all_syms = list({t for t, _ in ticker_parent_pairs} | {p for _, p in ticker_parent_pairs})
    placeholders = ", ".join(["?" for _ in all_syms])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 22
        """,
        all_syms + [as_of],
    ).fetchall()
    closes: dict[str, list] = {}
    for tkr, c, rn in rows:
        if tkr not in closes:
            closes[tkr] = [None] * 22
        if rn <= 22:
            closes[tkr][rn - 1] = float(c)

    def _rs(cc: list, pc: list, i: int) -> float:
        if cc[i] is None or cc[i + 20] is None or pc[i] is None or pc[i + 20] is None:
            return float("nan")
        if cc[i + 20] == 0 or pc[i + 20] == 0:
            return float("nan")
        return (cc[i] - cc[i + 20]) / cc[i + 20] - (pc[i] - pc[i + 20]) / pc[i + 20]

    result: dict[tuple[str, str], str] = {}
    for ticker, parent in ticker_parent_pairs:
        cc = closes.get(ticker, [None] * 22)
        pc = closes.get(parent, [None] * 22)
        today_rs = _rs(cc, pc, 0)
        yest_rs = _rs(cc, pc, 1)
        if math.isnan(today_rs) or math.isnan(yest_rs):
            result[(ticker, parent)] = "—"
        elif today_rs > yest_rs + 0.01:
            result[(ticker, parent)] = "↑"
        elif today_rs < yest_rs - 0.01:
            result[(ticker, parent)] = "↓"
        else:
            result[(ticker, parent)] = "→"
    return result


def _batch_days_above_50ma_count(tickers: list[str], as_of: str, conn) -> dict[str, int]:
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 101
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, c, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(c)))
    result: dict[str, int] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        closes = [c for _, c in series]
        if len(closes) < 51:
            result[tkr] = 0
            continue
        streak = 0
        for i in range(min(51, len(closes) - 50)):
            ma50 = sum(closes[i + 1 : i + 51]) / 50
            if closes[i] > ma50:
                streak += 1
            else:
                break
        result[tkr] = streak
    return result


def _batch_price_range_pct(tickers: list[str], as_of: str, conn) -> dict[str, float]:
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 20
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, c, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(c)))
    result: dict[str, float] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        closes = [c for _, c in series]
        if len(closes) < 20:
            result[tkr] = float("nan")
            continue
        today_close = closes[0]
        low_20d = min(closes)
        high_20d = max(closes)
        if high_20d == low_20d:
            result[tkr] = float("nan")
            continue
        result[tkr] = (today_close - low_20d) / (high_20d - low_20d) * 100
    return result


def _insider_category(summary: dict) -> str:
    if not summary:
        return "none"
    buys = summary.get("buy_count", 0)
    has_cluster = summary.get("has_cluster_buy", False)
    has_officer = summary.get("has_officer_buys", False)
    has_selling = summary.get("has_notable_selling", False)
    if buys == 0 and has_selling:
        return "selling"
    if buys == 0:
        return "none"
    if has_cluster:
        return "cluster"
    if has_officer:
        return "officer"
    return "single"


def _is_market_open() -> bool:
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= mins < 16 * 60


def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    use_live = _is_market_open()
    prices: dict[str, float] = {}
    for ticker in tickers:
        try:
            fi = yf.Ticker(ticker).fast_info
            live = getattr(fi, "last_price", None)
            prev = getattr(fi, "previous_close", None)
            price = (live or prev) if use_live else (prev or live)
            if price and float(price) > 0:
                prices[ticker] = float(price)
        except Exception as exc:
            logger.warning("price fetch failed for %s: %s", ticker, exc)
    return prices


# ---------------------------------------------------------------------------
# Sync compute functions (run in threadpool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _compute_scan_sync(scan_date: str) -> dict:
    from scanner.db import get_connection
    from scanner.graph.loader import MEGA_CAPS, get_dependents
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from scanner.signals.beta_rs import compute_beta_adjusted_rs
    from scanner.enrichment.insiders import get_insider_summary
    from scanner.signals.base import (
        classify_action, score_rank, classify_regime,
        TRADEABLE_REGIMES, VALIDATED_PARENTS,
        calc_basket_zscore,
    )
    from scanner.signals.confirmation import compute_confirmation_dependent

    conn = get_connection()
    try:
        insider_count = conn.execute("SELECT COUNT(*) FROM insider_transactions").fetchone()[0]
        estimates_count = conn.execute("SELECT COUNT(*) FROM estimates").fetchone()[0]
        show_insiders = insider_count > 0
        show_estimates = estimates_count > 0

        parent_regimes: dict[str, str] = {}
        parent_returns: dict[str, float] = {}
        for mc in MEGA_CAPS:
            ret_20d = _rolling_return_pct(mc, 20, scan_date, conn)
            parent_regimes[mc] = classify_regime(ret_20d)
            parent_returns[mc] = ret_20d

        per_parent: dict[str, list[tuple[str, float]]] = {}
        adj_by_pair: dict[tuple[str, str], float | None] = {}
        for parent in MEGA_CAPS:
            edges = get_dependents(parent)
            if not edges:
                continue
            signal = RelativeStrengthVsParent(parent=parent, lookback=20)
            seen: dict[str, float] = {}
            for edge in edges:
                if edge.child in seen:
                    continue
                s = signal.compute(edge.child, scan_date, conn)
                if not math.isnan(s):
                    seen[edge.child] = s
                try:
                    adj_by_pair[(edge.child, parent)] = compute_beta_adjusted_rs(
                        edge.child, parent, scan_date, conn
                    )
                except Exception:
                    adj_by_pair[(edge.child, parent)] = None
            per_parent[parent] = sorted(seen.items(), key=lambda x: x[1], reverse=True)

        universe_scores = [s for rows in per_parent.values() for _, s in rows]
        all_tickers = list({t for scored in per_parent.values() for t, _ in scored})

        est_scores: dict[str, float | None] = {}
        if show_estimates:
            for t in all_tickers:
                row = conn.execute(
                    "SELECT revision_score FROM estimates WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                    [t],
                ).fetchone()
                est_scores[t] = row[0] if row else None

        rvol_scores = _batch_rvol(all_tickers, scan_date, conn)
        above_50ma_flags = _batch_above_50ma(all_tickers, scan_date, conn)
        rvol_trend_flags = _batch_rvol_trend(all_tickers, scan_date, conn)
        ticker_parent_pairs = [(t, p) for p, scored in per_parent.items() for t, _ in scored]
        rs_trend_flags = _batch_rs_trend(ticker_parent_pairs, scan_date, conn)
        days_50ma_counts = _batch_days_above_50ma_count(all_tickers, scan_date, conn)
        price_range_pcts = _batch_price_range_pct(all_tickers, scan_date, conn)

        insider_summaries: dict[str, dict] = {}
        if show_insiders:
            as_of_date = date.fromisoformat(scan_date)
            for t in all_tickers:
                insider_summaries[t] = get_insider_summary(t, as_of_date, conn)

        earnings_next_dates: dict[str, object] = {}
        try:
            ec = conn.execute("SELECT COUNT(*) FROM earnings_dates").fetchone()[0]
            if ec > 0:
                placeholders = ", ".join(["?" for _ in all_tickers])
                for tkr, nd in conn.execute(
                    f"SELECT ticker, next_earnings_date FROM earnings_dates WHERE ticker IN ({placeholders})",
                    all_tickers,
                ).fetchall():
                    earnings_next_dates[tkr] = nd
        except Exception as exc:
            logger.warning("earnings date batch load failed: %s", exc)

        ticker_prices = _fetch_current_prices(all_tickers)

        scan_as_of_date = date.fromisoformat(scan_date)

        def _earnings_days(ticker: str) -> int | None:
            nd = earnings_next_dates.get(ticker)
            if nd is None:
                return None
            nd_date = nd.date() if hasattr(nd, "date") else nd
            delta = (nd_date - scan_as_of_date).days
            return delta if delta >= 0 else None

        groups = []
        for parent in MEGA_CAPS:
            scored = per_parent.get(parent)
            if not scored:
                continue
            regime = parent_regimes.get(parent, "UNKNOWN")
            ret_20d = parent_returns.get(parent, float("nan"))
            is_tradeable = regime in TRADEABLE_REGIMES
            is_validated = parent in VALIDATED_PARENTS

            rows_out = []
            shown: set[str] = set()
            top3 = scored[:3]
            bot3 = [(t, s) for t, s in scored[-3:] if t not in {tt for tt, _ in top3}]

            for section, items in [("top", top3), ("bottom", bot3)]:
                for ticker, s in items:
                    shown.add(ticker)
                    action = classify_action(s, universe_scores)
                    rank, total = score_rank(s, universe_scores)

                    if not is_validated:
                        action_label = "WAIT - unvalidated"
                    elif is_tradeable:
                        action_label = str(action)
                    else:
                        action_label = "WAIT"

                    rvol_val = rvol_scores.get(ticker, float("nan"))
                    rev_score_val = est_scores.get(ticker) if show_estimates else None
                    ins = insider_summaries.get(ticker, {}) if show_insiders else {}
                    insider_buying = ins.get("buy_count", 0) > 0
                    above_50ma = above_50ma_flags.get(ticker, False)

                    confirm = compute_confirmation_dependent(
                        rs_score=s,
                        rvol=rvol_val,
                        rev_score=rev_score_val,
                        insider_buying=insider_buying,
                        above_50ma=above_50ma,
                    )

                    z = calc_basket_zscore(ticker, scan_date, conn)
                    rows_out.append({
                        "ticker": ticker,
                        "parent": parent,
                        "price": ticker_prices.get(ticker),
                        "rs_score": _nan_to_none(s),
                        "rs_adj": adj_by_pair.get((ticker, parent)),
                        "rank": rank,
                        "total": total,
                        "action_label": action_label,
                        "est_rev": rev_score_val,
                        "rvol": _nan_to_none(rvol_val),
                        "confirm": confirm,
                        "confirm_max": 5,
                        "insider_annotation": _insider_category(ins),
                        "above_50ma": above_50ma,
                        "section": section,
                        "earnings_days": _earnings_days(ticker),
                        "rvol_trend": rvol_trend_flags.get(ticker, "—"),
                        "rs_trend": rs_trend_flags.get((ticker, parent), "—"),
                        "days_above_50ma": days_50ma_counts.get(ticker, 0),
                        "price_range_pct": _nan_to_none(price_range_pcts.get(ticker, float("nan"))),
                        "z_score": z,
                    })

            groups.append({
                "parent": parent,
                "parent_regime": regime,
                "parent_20d_return": _nan_to_none(ret_20d),
                "is_tradeable": is_tradeable,
                "is_validated": is_validated,
                "rows": rows_out,
            })

        return {
            "as_of": scan_date,
            "show_insiders": show_insiders,
            "show_estimates": show_estimates,
            "groups": groups,
            "schema_version": _SCAN_SCHEMA_VERSION,
        }
    finally:
        conn.close()


def _compute_scan_megacap_sync() -> dict:
    from scanner.db import get_connection
    from scanner.enrichment.insiders import get_insider_summary
    from scanner.signals.confirmation import compute_confirmation_megacap
    from scanner.signals.megacap import compute_megacap_scores, fetch_headlines

    results = compute_megacap_scores()
    as_of_date = date.today()
    as_of_str = str(as_of_date)

    conn = get_connection()
    try:
        insider_count = conn.execute("SELECT COUNT(*) FROM insider_transactions").fetchone()[0]
        has_insiders = insider_count > 0

        revision_scores: dict[str, float | None] = {}
        insider_summaries: dict[str, dict] = {}
        for r in results:
            row = conn.execute(
                "SELECT revision_score FROM estimates WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                [r.ticker],
            ).fetchone()
            revision_scores[r.ticker] = row[0] if row else None
            if has_insiders:
                insider_summaries[r.ticker] = get_insider_summary(r.ticker, as_of_date, conn)

        earnings_next_dates_mc: dict[str, object] = {}
        try:
            ec = conn.execute("SELECT COUNT(*) FROM earnings_dates").fetchone()[0]
            if ec > 0:
                mc_tickers = [r.ticker for r in results]
                placeholders = ", ".join(["?" for _ in mc_tickers])
                for tkr, nd in conn.execute(
                    f"SELECT ticker, next_earnings_date FROM earnings_dates WHERE ticker IN ({placeholders})",
                    mc_tickers,
                ).fetchall():
                    earnings_next_dates_mc[tkr] = nd
        except Exception as exc:
            logger.warning("megacap earnings date batch load failed: %s", exc)

        mc_tickers_list = [r.ticker for r in results]
        rvol_trend_mc = _batch_rvol_trend(mc_tickers_list, as_of_str, conn)
        rs_trend_mc = _batch_rs_trend([(r.ticker, "QQQ") for r in results], as_of_str, conn)
        days_50ma_mc = _batch_days_above_50ma_count(mc_tickers_list, as_of_str, conn)
        range_pct_mc = _batch_price_range_pct(mc_tickers_list, as_of_str, conn)
    finally:
        conn.close()

    def _mc_earnings_days(ticker: str) -> int | None:
        nd = earnings_next_dates_mc.get(ticker)
        if nd is None:
            return None
        nd_date = nd.date() if hasattr(nd, "date") else nd
        delta = (nd_date - as_of_date).days
        return delta if delta >= 0 else None

    rows = []
    for r in results:
        rev_score = revision_scores.get(r.ticker)
        ins = insider_summaries.get(r.ticker, {})
        insider_buying = ins.get("buy_count", 0) > 0
        confirm = compute_confirmation_megacap(
            rs_20=r.rs_20,
            rvol=r.rvol,
            rev_score=rev_score,
            insider_buying=insider_buying,
            trend_raw=r.trend_raw,
        )
        headlines = fetch_headlines(r.ticker, max_headlines=3)
        rows.append({
            "rank": r.rank,
            "ticker": r.ticker,
            "price": _nan_to_none(r.price),
            "rs_20": _nan_to_none(r.rs_20),
            "rs_60": _nan_to_none(r.rs_60),
            "rs_120": _nan_to_none(r.rs_120),
            "trend_raw": r.trend_raw,
            "trend": _nan_to_none(r.trend),
            "vol_conviction": _nan_to_none(r.vol_conviction),
            "hist_pct": _nan_to_none(r.hist_pct),
            "rvol": _nan_to_none(r.rvol),
            "est_rev": rev_score,
            "confirm": confirm,
            "confirm_max": 5,
            "score": _nan_to_none(r.composite),
            "label": r.label,
            "insider_annotation": _insider_category(ins),
            "headlines": headlines,
            "earnings_days": _mc_earnings_days(r.ticker),
            "rvol_trend": rvol_trend_mc.get(r.ticker, "—"),
            "rs_trend": rs_trend_mc.get((r.ticker, "QQQ"), "—"),
            "days_above_50ma": days_50ma_mc.get(r.ticker, 0),
            "price_range_pct": _nan_to_none(range_pct_mc.get(r.ticker, float("nan"))),
        })

    return {"as_of": as_of_str, "rankings": rows}


def _compute_rotation_sync() -> dict:
    from scanner.db import get_connection
    from scanner.signals.rotation import compute_rotation, fetch_vix

    as_of = str(date.today())
    conn = get_connection()
    try:
        rot = compute_rotation(as_of, conn)
    finally:
        conn.close()

    return {
        "as_of": as_of,
        "xlk_rank": rot.xlk_rank,
        "xlk_status": rot.xlk_status,
        "xlk_20d": _nan_to_none(rot.xlk_20d),
        "xlk_60d": _nan_to_none(rot.xlk_60d),
        "spy_20d": _nan_to_none(rot.spy_20d),
        "xlk_vs_spy": _nan_to_none(rot.xlk_vs_spy),
        "data_available": rot.data_available,
        "vix_level": _nan_to_none(rot.vix_level),
        "vix_label": rot.vix_label,
    }


def _compute_journal_sync() -> dict:
    from scanner.db import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT scan_date, ticker, parent, parent_regime, price, rs_score,
                   rank, action_label, est_rev, rvol, confirm, insider_annotation,
                   earnings_days, parent_20d_return, notes
            FROM journal
            ORDER BY scan_date DESC, parent, ticker
            LIMIT 200
            """
        ).fetchall()
    except Exception as exc:
        logger.error("journal fetch failed: %s", exc)
        return {"entries": []}
    finally:
        conn.close()

    cols = [
        "scan_date", "ticker", "parent", "parent_regime", "price", "rs_score",
        "rank", "action_label", "est_rev", "rvol", "confirm", "insider_annotation",
        "earnings_days", "parent_20d_return", "notes",
    ]
    entries = []
    for r in rows:
        e = dict(zip(cols, r))
        if e["scan_date"] is not None:
            e["scan_date"] = str(e["scan_date"])
        entries.append(e)
    return {"entries": entries}


def _persist_scan_to_db(result: dict, scan_date: str) -> None:
    """Write scan result JSON to scan_results table — survives redeploys."""
    import json
    from scanner.db import get_connection
    try:
        conn = get_connection()
        try:
            conn.execute("DELETE FROM scan_results WHERE scan_date = ?", [scan_date])
            conn.execute(
                "INSERT INTO scan_results (scan_date, result_json, computed_at) VALUES (?, ?, ?)",
                [scan_date, json.dumps(result), datetime.utcnow()],
            )
        finally:
            conn.close()
        logger.info("  scan result persisted to DB for %s", scan_date)
    except Exception as exc:
        logger.error("Failed to persist scan result to DB: %s", exc)


def _run_scan_today_sync(as_of: str | None = None) -> None:
    """Compute scan + megacap for as_of date (defaults to most recent price date), populate cache, and persist to DB."""
    if as_of is None:
        from scanner.db import get_connection
        try:
            conn = get_connection()
            try:
                row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
            finally:
                conn.close()
            as_of = str(row[0]) if row and row[0] else str(date.today())
        except Exception:
            as_of = str(date.today())
    try:
        result = _compute_scan_sync(as_of)
        _cache_set("scan_today", result)
        _persist_scan_to_db(result, as_of)
        logger.info("  scan cache warm (as_of=%s, groups=%d)", as_of, len(result.get("groups", [])))
    except Exception as exc:
        logger.error("scan cache warm failed: %s", exc)
    try:
        mc_result = _compute_scan_megacap_sync()
        _cache_set("scan_megacap", mc_result)
        logger.info("  megacap cache warm")
    except Exception as exc:
        logger.error("megacap cache warm failed: %s", exc)
    try:
        dep_div = _compute_divergence_sync(as_of, "dependency")
        _cache_set(f"divergence_dep_{as_of}", dep_div)
        _cache_set("divergence_dep_latest", dep_div)
        theme_div = _compute_divergence_sync(as_of, "theme")
        _cache_set(f"divergence_theme_{as_of}", theme_div)
        _cache_set("divergence_theme_latest", theme_div)
        logger.info(
            "  divergence flags: %d dependency, %d theme",
            len(dep_div.get("flags", [])),
            len(theme_div.get("flags", [])),
        )
    except Exception as exc:
        logger.error("divergence cache warm failed: %s", exc)


def _warm_caches_sync() -> None:
    logger.info("Pre-warming caches at startup...")
    try:
        _cache_set("journal", _compute_journal_sync())
        logger.info("  journal cache warm")
    except Exception as exc:
        logger.error("journal cache warm failed: %s", exc)

    # Run scan only if today's price data exists
    try:
        from scanner.db import get_connection
        conn = get_connection(read_only=False)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE date = CURRENT_DATE"
            ).fetchone()[0]
        finally:
            conn.close()
        if count > 0:
            logger.info("  today's prices found (%d rows) — running scan warm", count)
            _run_scan_today_sync()
        else:
            logger.warning("  no price data for today — scan cache skipped")
    except Exception as exc:
        logger.error("startup scan check failed: %s", exc)

    logger.info("Cache pre-warm complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from scanner.web.scheduler import create_scheduler, set_scheduler

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        logger.info("ANTHROPIC_API_KEY: SET")
    else:
        logger.warning("ANTHROPIC_API_KEY: NOT SET")

    massive_key = os.environ.get("MASSIVE_API_KEY")
    if massive_key:
        logger.info("MASSIVE_API_KEY: SET")
    else:
        logger.warning("MASSIVE_API_KEY: NOT SET — prices/filings will use yfinance/EDGAR fallbacks")

    scheduler = create_scheduler()
    set_scheduler(scheduler)
    scheduler.start()
    logger.info("APScheduler started")

    # Pre-warm caches in background so startup doesn't block
    asyncio.create_task(asyncio.to_thread(_warm_caches_sync))

    yield

    scheduler.shutdown(wait=False)
    set_scheduler(None)
    logger.info("APScheduler stopped")


_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Mega-Cap Scanner", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


def _read_scan_from_db(scan_date: str) -> dict | None:
    """Read a persisted scan result from DuckDB. Returns None if not found or schema is stale."""
    import json
    from scanner.db import get_connection
    try:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT result_json FROM scan_results WHERE scan_date = ?",
                [scan_date],
            ).fetchone()
        finally:
            conn.close()
        if row:
            data = json.loads(row[0])
            if data.get("schema_version", 0) < _SCAN_SCHEMA_VERSION:
                logger.info("Scan DB result for %s is schema v%s, recomputing", scan_date, data.get("schema_version", 0))
                return None
            return data
    except Exception as exc:
        logger.warning("Failed to read scan result from DB for %s: %s", scan_date, exc)
    return None


def _latest_price_date_sync() -> str:
    """Return the most recent date in prices table, falling back to today."""
    from scanner.db import get_connection
    try:
        conn = get_connection()
        try:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
        finally:
            conn.close()
        if row and row[0]:
            return str(row[0])
    except Exception as exc:
        logger.warning("_latest_price_date_sync failed: %s", exc)
    return str(date.today())


@app.get("/api/scan")
async def api_scan(scan_date: str = Query(default=None, description="YYYY-MM-DD")):
    if scan_date:
        as_of = scan_date
    else:
        as_of = await asyncio.to_thread(_latest_price_date_sync)

    # 1. In-memory cache (keyed to latest date)
    cached = _cache_get("scan_today")
    if cached is not None and cached.get("as_of") == as_of:
        return cached

    # 2. Persisted DB result (survives redeploys)
    db_result = await asyncio.to_thread(_read_scan_from_db, as_of)
    if db_result is not None:
        _cache_set("scan_today", db_result)
        return db_result

    # 3. Compute fresh
    try:
        result = await asyncio.to_thread(_compute_scan_sync, as_of)
        _cache_set("scan_today", result)
        await asyncio.to_thread(_persist_scan_to_db, result, as_of)
        return result
    except Exception as exc:
        logger.error("api_scan failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/scan-megacap")
async def api_scan_megacap():
    cached = _cache_get("scan_megacap")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_compute_scan_megacap_sync)
        _cache_set("scan_megacap", result)
        return result
    except Exception as exc:
        logger.error("api_scan_megacap failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/rotation")
async def api_rotation():
    try:
        return await asyncio.to_thread(_compute_rotation_sync)
    except Exception as exc:
        logger.error("api_rotation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _compute_divergence_sync(as_of: str | None = None, source: str = "dependency") -> dict:
    from scanner.db import get_connection
    from scanner.signals.divergence import find_divergences, store_divergences

    conn = get_connection()
    try:
        flags = find_divergences(conn, as_of, source=source)
        effective_date = as_of
        if effective_date is None:
            row = conn.execute("SELECT MAX(date) FROM prices WHERE ticker = 'SPY'").fetchone()
            effective_date = str(row[0]) if row and row[0] else str(date.today())
        if flags:
            store_divergences(flags, effective_date, conn)
        return {"as_of": effective_date, "flags": flags, "source": source}
    finally:
        conn.close()


def _compute_divergence_history_sync() -> dict:
    """Return last 30 days of divergence flags with 5-day forward return outcomes."""
    from scanner.db import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                df.ticker, df.flag_date, df.ticker_1d, df.spy_1d, df.benchmark,
                df.benchmark_1d, df.divergence_vs_spy, df.divergence_vs_benchmark,
                df.strength, df.theme, df.parent, df.source,
                p5.outcome_5d
            FROM divergence_flags df
            LEFT JOIN (
                SELECT base.ticker, base.flag_date,
                    CASE WHEN base.price_base > 0
                         THEN (fut.price_fut - base.price_base) / base.price_base * 100
                         ELSE NULL END AS outcome_5d
                FROM (
                    SELECT df2.ticker, df2.flag_date,
                        (SELECT adj_close FROM prices p
                         WHERE p.ticker = df2.ticker AND p.date <= df2.flag_date
                         ORDER BY p.date DESC LIMIT 1) AS price_base
                    FROM divergence_flags df2
                ) base
                LEFT JOIN (
                    SELECT df3.ticker, df3.flag_date,
                        (SELECT adj_close FROM prices p
                         WHERE p.ticker = df3.ticker AND p.date > df3.flag_date
                         ORDER BY p.date ASC LIMIT 1 OFFSET 4) AS price_fut
                    FROM divergence_flags df3
                ) fut ON base.ticker = fut.ticker AND base.flag_date = fut.flag_date
            ) p5 ON df.ticker = p5.ticker AND df.flag_date = p5.flag_date
            WHERE df.flag_date >= CURRENT_DATE - INTERVAL 30 DAYS
            ORDER BY df.flag_date DESC, df.divergence_vs_benchmark DESC
            """
        ).fetchall()
    finally:
        conn.close()

    entries = []
    for row in rows:
        (ticker, flag_date, ticker_1d, spy_1d, benchmark, benchmark_1d,
         div_vs_spy, div_vs_bench, strength, theme, parent, source, outcome_5d) = row
        entries.append({
            "ticker": ticker,
            "flag_date": str(flag_date),
            "ticker_1d": ticker_1d,
            "spy_1d": spy_1d,
            "benchmark": benchmark,
            "benchmark_1d": benchmark_1d,
            "divergence_vs_spy": div_vs_spy,
            "divergence_vs_benchmark": div_vs_bench,
            "strength": strength,
            "theme": theme,
            "parent": parent,
            "source": source,
            "outcome_5d": round(outcome_5d, 2) if outcome_5d is not None else None,
        })
    return {"entries": entries}


@app.get("/api/divergence")
async def api_divergence(scan_date: str = Query(default=None, description="YYYY-MM-DD")):
    """Dependency-source divergence flags for the Scanner page."""
    cache_key = f"divergence_dep_{scan_date or 'latest'}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_compute_divergence_sync, scan_date, "dependency")
        _cache_set(cache_key, result)
        return result
    except Exception as exc:
        logger.error("api_divergence failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/divergence/themes")
async def api_divergence_themes(scan_date: str = Query(default=None, description="YYYY-MM-DD")):
    """Theme-source divergence flags for the Themes page."""
    cache_key = f"divergence_theme_{scan_date or 'latest'}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_compute_divergence_sync, scan_date, "theme")
        _cache_set(cache_key, result)
        return result
    except Exception as exc:
        logger.error("api_divergence_themes failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/divergence/history")
async def api_divergence_history():
    cached = _cache_get("divergence_history")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_compute_divergence_history_sync)
        _cache_set("divergence_history", result)
        return result
    except Exception as exc:
        logger.error("api_divergence_history failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/journal")
async def api_journal():
    cached = _cache_get("journal")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_compute_journal_sync)
    except Exception as exc:
        logger.error("api_journal failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    _cache_set("journal", result)
    return result


@app.post("/api/run/ingest")
async def api_run_ingest():
    from scanner.web.scheduler import _job_ingest, scheduler_status
    await asyncio.to_thread(_job_ingest)
    if scheduler_status.get("last_ingest_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Ingest completed with errors — check server logs"}


@app.post("/api/run/ingest-estimates")
async def api_run_ingest_estimates():
    from scanner.web.scheduler import _job_estimates, scheduler_status
    await asyncio.to_thread(_job_estimates)
    if scheduler_status.get("last_estimates_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Estimates ingest completed with errors — check server logs"}


@app.post("/api/run/ingest-insiders")
async def api_run_ingest_insiders():
    from scanner.web.scheduler import _job_insiders, scheduler_status
    await asyncio.to_thread(_job_insiders)
    if scheduler_status.get("last_insiders_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Insider ingest completed with errors — check server logs"}


@app.post("/api/run/ingest-earnings")
async def api_run_ingest_earnings():
    from scanner.web.scheduler import _job_earnings, scheduler_status
    await asyncio.to_thread(_job_earnings)
    if scheduler_status.get("last_earnings_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Earnings ingest completed with errors — check server logs"}


@app.post("/api/run/ingest-filings")
async def api_run_ingest_filings():
    from scanner.web.scheduler import _job_filings, scheduler_status
    await asyncio.to_thread(_job_filings)
    if scheduler_status.get("last_filings_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Filings ingest completed with errors — check server logs"}


@app.post("/api/run/cleanup-filings")
async def api_run_cleanup_filings():
    from scanner.web.scheduler import _job_cleanup_filings, scheduler_status
    await asyncio.to_thread(_job_cleanup_filings)
    if scheduler_status.get("last_cleanup_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Filings cleanup failed — check server logs"}


@app.post("/api/run/ingest-short")
async def api_run_ingest_short():
    from scanner.web.scheduler import _job_short_interest, scheduler_status
    await asyncio.to_thread(_job_short_interest)
    if scheduler_status.get("last_short_interest_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Short interest ingest completed with errors — check server logs"}


@app.get("/api/admin/generate-summaries")
def generate_summaries():
    from scanner.ingest.filings import generate_filing_analysis, _fetch_filing_text, _make_session, MATERIAL_ITEMS
    from scanner.db import get_connection

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}

    results = []
    session = _make_session()
    conn = get_connection()
    try:
        filings = conn.execute("""
            SELECT id, ticker, filing_url, item_numbers, title
            FROM filings_8k
            WHERE summary IS NULL
            AND filing_url IS NOT NULL
            ORDER BY filed_date DESC
            LIMIT 5
        """).fetchall()

        for id_, ticker, url, item_numbers, title in filings:
            try:
                text = _fetch_filing_text(url, session)
                if not text:
                    results.append({"ticker": ticker, "status": "failed", "reason": "no text fetched"})
                    continue
                item_labels = ", ".join(
                    MATERIAL_ITEMS.get(i.strip(), i.strip())
                    for i in (item_numbers or "").split(",")
                    if i.strip()
                )
                summary, sentiment, impact_explanation = generate_filing_analysis(item_labels, title, text, ticker)
                if summary:
                    conn.execute(
                        "UPDATE filings_8k SET summary=?, sentiment=?, impact_explanation=? WHERE id=?",
                        [summary, sentiment, impact_explanation, id_],
                    )
                    results.append({"ticker": ticker, "status": "success", "summary": summary[:100],
                                    "sentiment": sentiment})
                else:
                    results.append({"ticker": ticker, "status": "failed", "reason": "empty analysis returned"})
            except Exception as exc:
                logger.error("generate-summaries: filing %s/%s failed: %s", ticker, id_, exc)
                results.append({"ticker": ticker, "status": "error", "error": str(exc)})
    finally:
        session.close()
        conn.close()

    return {"results": results}


@app.get("/api/admin/generate-sentiments")
def generate_sentiments():
    from scanner.ingest.filings import generate_filing_analysis, _fetch_filing_text, _make_session, MATERIAL_ITEMS
    from scanner.db import get_connection

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}

    results = []
    session = _make_session()
    conn = get_connection()
    try:
        filings = conn.execute("""
            SELECT id, ticker, filing_url, item_numbers, title
            FROM filings_8k
            WHERE summary IS NOT NULL
            AND filing_url IS NOT NULL
            ORDER BY filed_date DESC
            LIMIT 10
        """).fetchall()

        for id_, ticker, url, item_numbers, title in filings:
            try:
                text = _fetch_filing_text(url, session)
                if not text:
                    results.append({"ticker": ticker, "status": "failed", "reason": "no text fetched"})
                    continue
                item_labels = ", ".join(
                    MATERIAL_ITEMS.get(i.strip(), i.strip())
                    for i in (item_numbers or "").split(",")
                    if i.strip()
                )
                _, sentiment, impact_explanation = generate_filing_analysis(item_labels, title, text, ticker)
                if sentiment:
                    conn.execute(
                        "UPDATE filings_8k SET sentiment=?, impact_explanation=? WHERE id=?",
                        [sentiment, impact_explanation, id_],
                    )
                    results.append({"ticker": ticker, "status": "success", "sentiment": sentiment})
                else:
                    results.append({"ticker": ticker, "status": "failed", "reason": "empty analysis returned"})
            except Exception as exc:
                logger.error("generate-sentiments: filing %s/%s failed: %s", ticker, id_, exc)
                results.append({"ticker": ticker, "status": "error", "error": str(exc)})
    finally:
        session.close()
        conn.close()

    return {"results": results}


@app.get("/api/admin/generate-explanations")
async def generate_explanations():
    """Generate signal explanations for all BUY/STRONG_BUY rows in the most recent scan."""
    from scanner.db import get_connection
    from scanner.signals.base import generate_signal_explanation
    import json as _json

    def _run():
        # Step 1: delete all existing explanations before regenerating
        conn_del = get_connection()
        try:
            conn_del.execute("DELETE FROM signal_explanations")
            logger.info("generate-explanations: deleted all cached signal explanations")
        finally:
            conn_del.close()

        # Step 2: generate fresh explanations
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT scan_date, result_json FROM scan_results ORDER BY scan_date DESC LIMIT 1"
            ).fetchone()
            if not row:
                return {"error": "No scan results found"}
            scan_date = str(row[0])
            data = _json.loads(row[1])
            groups = data.get("groups", [])
            results = []
            for grp in groups:
                parent = grp.get("parent", "")
                for r in grp.get("rows", []):
                    action = (r.get("action_label") or "").upper()
                    if action not in ("BUY", "STRONG_BUY"):
                        continue
                    ticker = r.get("ticker", "")
                    try:
                        explanation = generate_signal_explanation(ticker, parent, scan_date, r, conn)
                        results.append({
                            "ticker": ticker,
                            "parent": parent,
                            "action": action,
                            "status": "ok" if explanation else "failed",
                        })
                    except Exception as exc:
                        logger.error("generate-explanations: %s/%s failed: %s", ticker, parent, exc)
                        results.append({"ticker": ticker, "parent": parent, "status": "error", "error": str(exc)})
            return {"scan_date": scan_date, "results": results}
        finally:
            conn.close()

    result = await asyncio.to_thread(_run)
    return result


@app.get("/api/signal-explanation/{ticker}/{parent}/{scan_date}")
async def get_signal_explanation(ticker: str, parent: str, scan_date: str):
    """Return cached signal explanation for a given ticker/parent/scan_date."""
    from scanner.db import get_connection

    def _run():
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT explanation FROM signal_explanations WHERE ticker=? AND parent=? AND scan_date=?",
                [ticker.upper(), parent.upper(), scan_date],
            ).fetchone()
            if row is None:
                return {"explanation": None}
            return {"explanation": row[0]}
        finally:
            conn.close()

    return await asyncio.to_thread(_run)


@app.get("/api/admin/reingest-filings-14d")
async def reingest_filings_14d():
    from scanner.db import get_connection
    from scanner.ingest.filings import ingest_filings

    conn = get_connection()
    try:
        conn.execute("DELETE FROM filings_8k")
    finally:
        conn.close()

    async def _run():
        await asyncio.to_thread(ingest_filings, None, 14)

    asyncio.create_task(_run())
    return {"status": "started - check filings page in 2 minutes"}


@app.get("/api/admin/clear-all-filings")
def clear_all_filings():
    from scanner.db import get_connection
    conn = get_connection()
    try:
        conn.execute("DELETE FROM filings_8k")
    finally:
        conn.close()
    return {"status": "cleared"}


@app.get("/api/admin/clear-high-filings")
def clear_high_filings():
    from scanner.db import get_connection
    conn = get_connection()
    try:
        conn.execute("""
            DELETE FROM filings_8k
            WHERE impact = 'HIGH'
            AND (summary IS NULL OR summary = '')
        """)
    finally:
        conn.close()
    return {"deleted": "ok"}


@app.get("/api/admin/fix-theme-benchmarks")
def fix_theme_benchmarks():
    from scanner.db import get_connection
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE themes SET benchmark_etf = 'XLK' WHERE ticker IN ('BB', 'NOK')"
        )
        result = conn.execute(
            "SELECT ticker, benchmark_etf FROM themes ORDER BY ticker"
        ).fetchall()
    finally:
        conn.close()
    return {"themes": [{"ticker": r[0], "benchmark_etf": r[1]} for r in result]}


@app.post("/api/run/run-scan")
async def api_run_scan():
    from scanner.web.scheduler import _job_scan, scheduler_status
    await asyncio.to_thread(_job_scan)
    if scheduler_status.get("last_scan_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Scan job failed — check server logs"}


@app.post("/api/run/journal-capture")
async def api_run_journal_capture():
    from scanner.web.scheduler import _job_journal_capture, scheduler_status
    await asyncio.to_thread(_job_journal_capture)
    if scheduler_status.get("last_journal_capture_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Journal capture job failed — check server logs"}


def _compute_theme_scan_sync() -> list[dict]:
    from scanner.db import get_connection
    from scanner.signals.theme_scanner import compute_theme_scan

    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
        as_of = str(row[0]) if row and row[0] else str(date.today())
        return compute_theme_scan(conn, as_of)
    finally:
        conn.close()


@app.get("/api/theme-scan")
async def api_theme_scan():
    cached = _cache_get("theme_scan")
    if cached is not None:
        return cached
    results = await asyncio.to_thread(_compute_theme_scan_sync)
    # Group by theme_name for convenient frontend rendering.
    themes: dict[str, dict] = {}
    for r in results:
        tn = r["theme_name"]
        if tn not in themes:
            themes[tn] = {
                "theme_name": tn,
                "theme_label": r.get("theme_label", tn),
                "benchmark_etf": r["benchmark_etf"],
                "theme_active": r["theme_active"],
                "tickers": [],
            }
        themes[tn]["tickers"].append(r)
    payload = {"scan_date": results[0]["scan_date"] if results else None, "themes": list(themes.values())}
    _cache_set("theme_scan", payload)
    return payload


@app.post("/api/run/scan-themes")
async def api_run_scan_themes():
    from scanner.web.scheduler import _job_scan_themes, scheduler_status
    await asyncio.to_thread(_job_scan_themes)
    _cache.pop("theme_scan", None)
    if scheduler_status.get("last_theme_scan_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Theme scan job failed — check server logs"}


@app.post("/api/run/discover")
async def api_run_discover():
    from scanner.web.scheduler import _job_discovery, scheduler_status
    await asyncio.to_thread(_job_discovery)
    if scheduler_status.get("last_discovery_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "Discovery job failed — check server logs"}


@app.post("/api/run/accumulate-rs")
async def api_run_accumulate_rs():
    from scanner.web.scheduler import _job_accumulate_rs, scheduler_status
    await asyncio.to_thread(_job_accumulate_rs)
    if scheduler_status.get("last_accumulate_rs_ok"):
        return {"status": "success"}
    return {"status": "failed", "error": "RS accumulation failed — check server logs"}


@app.get("/api/discovery")
async def api_discovery():
    from scanner.ingest.discovery import get_discovery_display
    conn = get_connection()
    try:
        return get_discovery_display(conn)
    finally:
        conn.close()


@app.post("/api/run/all")
async def api_run_all():
    from scanner.web.scheduler import (
        _job_ingest, _job_estimates, _job_earnings, _job_filings, _job_insiders,
        _job_journal_capture, _job_scan_themes,
        scheduler_status,
    )
    SEQUENCE = [
        ("ingest",           _job_ingest,    "last_ingest_ok"),
        ("ingest-estimates", _job_estimates, "last_estimates_ok"),
        ("ingest-earnings",  _job_earnings,  "last_earnings_ok"),
        ("ingest-filings",   _job_filings,   "last_filings_ok"),
        ("ingest-insiders",  _job_insiders,  "last_insiders_ok"),
    ]
    results = []
    for job_name, job_fn, ok_key in SEQUENCE:
        try:
            await asyncio.to_thread(job_fn)
            ok = scheduler_status.get(ok_key, False)
        except Exception as exc:
            logger.error("run-all: %s raised: %s", job_name, exc)
            ok = False
        results.append({"job": job_name, "status": "success" if ok else "failed"})

    # Allow DuckDB file writes to flush before the scan opens a new connection.
    await asyncio.sleep(2)

    # Step 6: find latest available price date and compute scan against it.
    try:
        from scanner.db import get_connection
        conn = get_connection()
        try:
            latest_row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
        finally:
            conn.close()
        latest_date = str(latest_row[0]) if latest_row and latest_row[0] else None
        if not latest_date:
            logger.warning("run-all: no price data in DB — skipping scan")
            results.append({"job": "run-scan", "status": "skipped-no-data"})
        else:
            logger.info("run-all: latest price date=%s — running scan", latest_date)
            await asyncio.to_thread(_run_scan_today_sync, latest_date)
            results.append({"job": "run-scan", "status": "success"})
    except Exception as exc:
        logger.error("run-all: scan raised: %s", exc)
        results.append({"job": "run-scan", "status": "failed"})

    # Step 7: journal capture
    try:
        await asyncio.to_thread(_job_journal_capture)
        ok = scheduler_status.get("last_journal_capture_ok", False)
    except Exception as exc:
        logger.error("run-all: journal-capture raised: %s", exc)
        ok = False
    results.append({"job": "journal-capture", "status": "success" if ok else "failed"})

    # Step 8: theme scan
    try:
        await asyncio.to_thread(_job_scan_themes)
        _cache.pop("theme_scan", None)
        ok = scheduler_status.get("last_theme_scan_ok", False)
    except Exception as exc:
        logger.error("run-all: scan-themes raised: %s", exc)
        ok = False
    results.append({"job": "scan-themes", "status": "success" if ok else "failed"})

    any_failed = any(r["status"] != "success" for r in results)
    return {"status": "partial" if any_failed else "complete", "results": results}


def _compute_fwd_returns(entries: list[dict], conn) -> dict[tuple, dict]:
    from collections import defaultdict
    by_date: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("price") is not None:
            by_date[e["scan_date"]].append(e)
    result: dict[tuple, dict] = {}
    for scan_date, date_entries in by_date.items():
        tickers = list({e["ticker"] for e in date_entries})
        if not tickers:
            continue
        placeholders = ", ".join(["?" for _ in tickers])
        try:
            rows = conn.execute(
                f"""
                SELECT ticker, adj_close, rn
                FROM (
                    SELECT ticker, adj_close,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS rn
                    FROM prices
                    WHERE ticker IN ({placeholders}) AND date > ?
                ) t
                WHERE rn IN (5, 10, 20)
                """,
                tickers + [scan_date],
            ).fetchall()
        except Exception as exc:
            logger.warning("fwd returns query failed for %s: %s", scan_date, exc)
            continue
        fwd_prices: dict[str, dict[int, float]] = {}
        for tkr, close, rn in rows:
            fwd_prices.setdefault(tkr, {})[int(rn)] = float(close)
        for e in date_entries:
            tkr = e["ticker"]
            cap = e["price"]
            fp = fwd_prices.get(tkr, {})
            key = (e["scan_date"], tkr, e["parent"])
            result[key] = {
                "fwd_5d":  (fp[5]  - cap) / cap if 5  in fp and cap else None,
                "fwd_10d": (fp[10] - cap) / cap if 10 in fp and cap else None,
                "fwd_20d": (fp[20] - cap) / cap if 20 in fp and cap else None,
            }
    return result


def _capture_journal_sync() -> dict:
    import math

    from scanner.db import get_connection
    from scanner.graph.loader import MEGA_CAPS, get_dependents
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from scanner.enrichment.insiders import get_insider_summary
    from scanner.signals.base import (
        classify_action, score_rank, classify_regime,
        TRADEABLE_REGIMES, VALIDATED_PARENTS,
    )
    from scanner.signals.confirmation import compute_confirmation_dependent

    scan_date = str(date.today())
    conn = get_connection()
    try:
        exists = conn.execute(
            "SELECT 1 FROM journal WHERE scan_date = ? LIMIT 1", [scan_date]
        ).fetchone()
        if exists:
            return {"status": "exists", "message": f"Already captured today ({scan_date})"}

        insider_count = conn.execute("SELECT COUNT(*) FROM insider_transactions").fetchone()[0]
        show_insiders = insider_count > 0
        estimates_count = conn.execute("SELECT COUNT(*) FROM estimates").fetchone()[0]
        show_estimates = estimates_count > 0

        parent_regimes: dict[str, str] = {}
        parent_returns: dict[str, float] = {}
        for mc in MEGA_CAPS:
            ret = _rolling_return_pct(mc, 20, scan_date, conn)
            parent_regimes[mc] = classify_regime(ret)
            parent_returns[mc] = ret

        per_parent: dict[str, list[tuple[str, float]]] = {}
        for parent in MEGA_CAPS:
            edges = get_dependents(parent)
            if not edges:
                continue
            signal = RelativeStrengthVsParent(parent=parent, lookback=20)
            seen: dict[str, float] = {}
            for edge in edges:
                if edge.child in seen:
                    continue
                s = signal.compute(edge.child, scan_date, conn)
                if not math.isnan(s):
                    seen[edge.child] = s
            per_parent[parent] = sorted(seen.items(), key=lambda x: x[1], reverse=True)

        universe_scores = [s for rows in per_parent.values() for _, s in rows]
        all_tickers = list({t for scored in per_parent.values() for t, _ in scored})

        est_scores: dict[str, float | None] = {}
        if show_estimates:
            for t in all_tickers:
                row = conn.execute(
                    "SELECT revision_score FROM estimates WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                    [t],
                ).fetchone()
                est_scores[t] = row[0] if row else None

        rvol_scores = _batch_rvol(all_tickers, scan_date, conn)
        above_50ma_flags = _batch_above_50ma(all_tickers, scan_date, conn)

        insider_summaries: dict[str, dict] = {}
        if show_insiders:
            as_of_date = date.fromisoformat(scan_date)
            for t in all_tickers:
                insider_summaries[t] = get_insider_summary(t, as_of_date, conn)

        earnings_next_dates_j: dict[str, object] = {}
        try:
            ec = conn.execute("SELECT COUNT(*) FROM earnings_dates").fetchone()[0]
            if ec > 0:
                placeholders = ", ".join(["?" for _ in all_tickers])
                for tkr, nd in conn.execute(
                    f"SELECT ticker, next_earnings_date FROM earnings_dates WHERE ticker IN ({placeholders})",
                    all_tickers,
                ).fetchall():
                    earnings_next_dates_j[tkr] = nd
        except Exception as exc:
            logger.warning("journal earnings date batch load failed: %s", exc)

        ticker_prices = _fetch_current_prices(all_tickers)
        j_as_of_date = date.fromisoformat(scan_date)

        def _j_earnings_days(ticker: str) -> int | None:
            nd = earnings_next_dates_j.get(ticker)
            if nd is None:
                return None
            nd_date = nd.date() if hasattr(nd, "date") else nd
            delta = (nd_date - j_as_of_date).days
            return delta if delta >= 0 else None

        rows_added = 0
        for parent in MEGA_CAPS:
            scored = per_parent.get(parent)
            if not scored:
                continue
            regime = parent_regimes.get(parent, "UNKNOWN")
            ret_20d = parent_returns.get(parent, float("nan"))
            is_tradeable = regime in TRADEABLE_REGIMES
            is_validated = parent in VALIDATED_PARENTS

            shown: set[str] = set()
            top3 = scored[:3]
            bot3 = [(t, s) for t, s in scored[-3:] if t not in {tt for tt, _ in top3}]

            for items in [top3, bot3]:
                for ticker, s in items:
                    if ticker in shown:
                        continue
                    shown.add(ticker)
                    action = classify_action(s, universe_scores)
                    rank, total = score_rank(s, universe_scores)

                    if not is_validated:
                        action_label = "WAIT - unvalidated"
                    elif is_tradeable:
                        action_label = str(action)
                    else:
                        action_label = "WAIT"

                    ins = insider_summaries.get(ticker, {}) if show_insiders else {}
                    cat = _insider_category(ins)
                    if cat == "cluster":
                        n_buyers = ins.get("distinct_insiders", 0)
                        note = (
                            f"{n_buyers} insiders bought {ticker} in last 30d; "
                            f"not actionable until {parent} pulls back."
                        )
                    else:
                        note = ""

                    rvol_val = rvol_scores.get(ticker, float("nan"))
                    rev_score_val = est_scores.get(ticker) if show_estimates else None
                    insider_buying = ins.get("buy_count", 0) > 0
                    above_50ma = above_50ma_flags.get(ticker, False)
                    confirm = compute_confirmation_dependent(
                        rs_score=s,
                        rvol=rvol_val,
                        rev_score=rev_score_val,
                        insider_buying=insider_buying,
                        above_50ma=above_50ma,
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO journal
                           (scan_date, ticker, parent, parent_regime, price, rs_score, rank,
                            action_label, est_rev, rvol, confirm, insider_annotation,
                            earnings_days, parent_20d_return, notes)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            scan_date, ticker, parent, regime,
                            ticker_prices.get(ticker),
                            s if not math.isnan(s) else None,
                            f"({rank}/{total})", action_label,
                            rev_score_val,
                            _nan_to_none(rvol_val),
                            confirm, cat,
                            _j_earnings_days(ticker),
                            ret_20d if not math.isnan(ret_20d) else None,
                            note,
                        ],
                    )
                    rows_added += 1

        return {"status": "success", "rows_added": rows_added}
    finally:
        conn.close()


@app.post("/api/journal/capture")
async def api_journal_capture():
    try:
        result = await asyncio.to_thread(_capture_journal_sync)
        if result.get("status") == "success":
            _cache.pop("journal", None)
            _cache.pop("journal_dates", None)
            _cache.pop("journal_perf", None)
        return result
    except Exception as exc:
        logger.error("api_journal_capture failed: %s", exc)
        return {"status": "failed", "error": str(exc)}


@app.get("/api/journal/dates")
async def api_journal_dates():
    cached = _cache_get("journal_dates")
    if cached is not None:
        return cached
    from scanner.db import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT scan_date FROM journal ORDER BY scan_date DESC"
        ).fetchall()
        result = {"dates": [str(r[0]) for r in rows]}
    except Exception as exc:
        logger.error("api_journal_dates failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()
    _cache_set("journal_dates", result)
    return result


def _compute_journal_perf_sync() -> dict:
    from collections import defaultdict
    from scanner.db import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scan_date, ticker, parent, action_label, price FROM journal WHERE price IS NOT NULL ORDER BY scan_date"
        ).fetchall()
        if not rows:
            return {"buckets": {}}
        entries = [
            {"scan_date": str(r[0]), "ticker": r[1], "parent": r[2], "action_label": r[3], "price": r[4]}
            for r in rows
        ]
        fwd = _compute_fwd_returns(entries, conn)
        buckets: dict = defaultdict(lambda: {"count": 0, "r5": [], "r10": [], "r20": []})
        for e in entries:
            key = (e["scan_date"], e["ticker"], e["parent"])
            fd = fwd.get(key, {})
            b = buckets[(e["action_label"] or "").upper()]
            b["count"] += 1
            if fd.get("fwd_5d") is not None:
                b["r5"].append(fd["fwd_5d"])
            if fd.get("fwd_10d") is not None:
                b["r10"].append(fd["fwd_10d"])
            if fd.get("fwd_20d") is not None:
                b["r20"].append(fd["fwd_20d"])
        def _avg(lst): return sum(lst) / len(lst) if lst else None
        def _hr(lst): return sum(1 for x in lst if x > 0) / len(lst) if lst else None
        result: dict = {}
        for label, b in buckets.items():
            result[label] = {
                "count": b["count"],
                "avg_5d": _avg(b["r5"]),
                "avg_10d": _avg(b["r10"]),
                "avg_20d": _avg(b["r20"]),
                "hit_rate_10d": _hr(b["r10"]),
                "n_with_returns": len(b["r10"]),
            }
        return {"buckets": result}
    except Exception as exc:
        logger.error("journal perf compute failed: %s", exc)
        return {"buckets": {}}
    finally:
        conn.close()


@app.get("/api/journal/performance")
async def api_journal_performance():
    cached = _cache_get("journal_perf")
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_compute_journal_perf_sync)
    except Exception as exc:
        logger.error("api_journal_performance failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    _cache_set("journal_perf", result)
    return result


@app.get("/api/journal/{scan_date}")
async def api_journal_for_date(scan_date: str):
    from scanner.db import get_connection
    conn = get_connection()
    entries: list[dict] = []
    try:
        rows = conn.execute(
            """SELECT scan_date, ticker, parent, parent_regime, price, rs_score,
                      rank, action_label, est_rev, rvol, confirm, insider_annotation,
                      earnings_days, parent_20d_return, notes, signal_source
               FROM journal
               WHERE scan_date = ?
               ORDER BY parent, ticker""",
            [scan_date],
        ).fetchall()
        cols = ["scan_date", "ticker", "parent", "parent_regime", "price", "rs_score",
                "rank", "action_label", "est_rev", "rvol", "confirm", "insider_annotation",
                "earnings_days", "parent_20d_return", "notes", "signal_source"]
        for r in rows:
            e = dict(zip(cols, r))
            if e["scan_date"] is not None:
                e["scan_date"] = str(e["scan_date"])
            entries.append(e)
        fwd = _compute_fwd_returns(entries, conn)
        for e in entries:
            key = (e["scan_date"], e["ticker"], e["parent"])
            fd = fwd.get(key, {})
            e["fwd_5d"] = fd.get("fwd_5d")
            e["fwd_10d"] = fd.get("fwd_10d")
            e["fwd_20d"] = fd.get("fwd_20d")
    except Exception as exc:
        logger.error("api_journal_for_date failed for %s: %s", scan_date, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()
    return {"entries": entries}


class _NotesUpdate(BaseModel):
    notes: str


@app.patch("/api/journal/{scan_date}/{ticker}/{parent}/notes")
async def api_journal_patch_notes(scan_date: str, ticker: str, parent: str, body: _NotesUpdate):
    from scanner.db import get_connection
    conn = get_connection()
    try:
        result = conn.execute(
            "UPDATE journal SET notes = ? WHERE scan_date = ? AND ticker = ? AND parent = ?",
            [body.notes, scan_date, ticker, parent],
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Journal entry not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("api_journal_patch_notes failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()
    _cache.pop("journal", None)
    _cache.pop("journal_dates", None)
    return {"status": "ok"}


def _compute_activity_log_sync() -> dict:
    from scanner.db import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT run_time, job_name, status, message
            FROM scheduler_runs
            ORDER BY run_time DESC
            LIMIT 50
            """
        ).fetchall()
        return {
            "entries": [
                {
                    "logged_at": str(r[0]),
                    "job_name": r[1],
                    "status": r[2],
                    "message": r[3] or "",
                }
                for r in rows
            ]
        }
    except Exception as exc:
        logger.error("activity log fetch failed: %s", exc)
        return {"entries": []}
    finally:
        conn.close()


@app.get("/api/activity-log")
async def api_activity_log():
    try:
        return await asyncio.to_thread(_compute_activity_log_sync)
    except Exception as exc:
        logger.error("api_activity_log failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/status")
def api_status():
    from scanner.web.scheduler import get_scheduler_info
    from scanner.db import get_connection

    last_updated = None
    try:
        conn = get_connection()
        try:
            row = conn.execute(
                """
                SELECT MAX(run_time) FROM scheduler_runs
                WHERE job_name IN ('ingest', 'ingest-estimates')
                """
            ).fetchone()
            if row and row[0] is not None:
                last_updated = str(row[0])
        finally:
            conn.close()
    except Exception as exc:
        logger.error("api_status: failed to query scheduler_runs: %s", exc)

    sched = get_scheduler_info()

    return {
        "last_updated": last_updated,
        "scheduler_running": sched["running"],
        "jobs": sched["jobs"],
        "cache_keys": list(_cache.keys()),
    }


@app.get("/api/debug")
def debug():
    from scanner.db import get_connection, _DB_PATH

    actual_path = str(_DB_PATH)
    db_exists = os.path.exists(actual_path)
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        sched = conn.execute("SELECT COUNT(*) FROM scheduler_runs").fetchone()[0]
        scan_res = conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0]
        scan_row = conn.execute(
            """
            SELECT scan_date, computed_at,
                   LENGTH(result_json) AS json_size,
                   result_json[:200]   AS json_preview
            FROM scan_results
            ORDER BY computed_at DESC LIMIT 1
            """
        ).fetchone()
        price_dates = conn.execute(
            """
            SELECT date, COUNT(*) AS count
            FROM prices
            GROUP BY date
            ORDER BY date DESC
            LIMIT 3
            """
        ).fetchall()
        current_date = conn.execute("SELECT CURRENT_DATE").fetchone()[0]
        filings_count = conn.execute("SELECT COUNT(*) FROM filings_8k").fetchone()[0]
        high_filings_rows = conn.execute("""
            SELECT ticker, filed_date, impact,
                   summary IS NOT NULL AS has_summary,
                   summary
            FROM filings_8k
            WHERE impact = 'HIGH'
            ORDER BY filed_date DESC
        """).fetchall()
    finally:
        conn.close()
    return {
        "db_path": actual_path,
        "db_exists": db_exists,
        "DATABASE_PATH_env": os.environ.get("DATABASE_PATH", "NOT SET"),
        "tables": tables,
        "prices_count": prices,
        "scheduler_runs_count": sched,
        "scan_results_count": scan_res,
        "filings_count": filings_count,
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "high_filings": [
            {
                "ticker": r[0],
                "filed_date": str(r[1]),
                "impact": r[2],
                "has_summary": r[3],
                "summary": r[4],
            }
            for r in high_filings_rows
        ],
        "scan_result": {
            "scan_date": str(scan_row[0]),
            "computed_at": str(scan_row[1]),
            "json_size": scan_row[2],
            "json_preview": scan_row[3],
        } if scan_row else None,
        "price_dates": [{"date": str(r[0]), "count": r[1]} for r in price_dates],
        "db_current_date": str(current_date),
    }


# ---------------------------------------------------------------------------
# Macro Observations endpoints
# ---------------------------------------------------------------------------

class _ObservationCreate(BaseModel):
    observation_date: str
    type: str
    primary_ticker: str
    affected_tickers: list[str] = []
    note: str
    link: str = ""


class _ObservationUpdate(BaseModel):
    observation_date: str
    type: str
    primary_ticker: str
    affected_tickers: list[str] = []
    note: str
    link: str = ""


def _get_observations_sync() -> dict:
    from scanner.db import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, observation_date, type, primary_ticker,
                   affected_tickers, note, link, created_at
            FROM macro_observations
            WHERE observation_date >= CURRENT_DATE - INTERVAL 7 DAY
            ORDER BY observation_date DESC, created_at DESC
            LIMIT 10
            """
        ).fetchall()
        return {
            "observations": [
                {
                    "id": r[0],
                    "observation_date": str(r[1]),
                    "type": r[2],
                    "primary_ticker": r[3],
                    "affected_tickers": r[4].split(",") if r[4] else [],
                    "note": r[5],
                    "link": r[6] or "",
                    "created_at": str(r[7]),
                }
                for r in rows
            ]
        }
    except Exception as exc:
        logger.error("get_observations failed: %s", exc)
        return {"observations": []}
    finally:
        conn.close()


@app.get("/api/observations")
async def api_get_observations():
    try:
        return await asyncio.to_thread(_get_observations_sync)
    except Exception as exc:
        logger.error("api_get_observations failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _create_observation_sync(body: dict) -> dict:
    from scanner.db import get_connection
    conn = get_connection()
    try:
        next_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM macro_observations"
        ).fetchone()[0]
        affected_str = ",".join(body.get("affected_tickers", [])) if body.get("affected_tickers") else None
        conn.execute(
            """
            INSERT INTO macro_observations
                (id, observation_date, type, primary_ticker, affected_tickers, note, link, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                body["observation_date"],
                body["type"],
                body["primary_ticker"],
                affected_str,
                body["note"],
                body.get("link") or None,
                datetime.utcnow(),
            ],
        )
        # Append note fragment to journal entries for primary + affected tickers
        note_fragment = f"[{body['type']}] {body['note']}"
        obs_date = body["observation_date"]
        all_tickers = [body["primary_ticker"]] + list(body.get("affected_tickers") or [])
        updated = 0
        for ticker in all_tickers:
            rows = conn.execute(
                "SELECT scan_date, ticker, parent, notes FROM journal WHERE scan_date = ? AND ticker = ?",
                [obs_date, ticker],
            ).fetchall()
            for row in rows:
                existing = row[3] or ""
                new_notes = (existing + " | " + note_fragment) if existing else note_fragment
                conn.execute(
                    "UPDATE journal SET notes = ? WHERE scan_date = ? AND ticker = ? AND parent = ?",
                    [new_notes, obs_date, ticker, row[2]],
                )
                updated += 1
        _cache.pop("journal", None)
        _cache.pop("journal_dates", None)
        return {"status": "saved", "id": next_id, "journal_updated": updated}
    except Exception as exc:
        logger.error("create_observation failed: %s", exc)
        raise
    finally:
        conn.close()


@app.post("/api/observations")
async def api_create_observation(body: _ObservationCreate):
    try:
        result = await asyncio.to_thread(_create_observation_sync, body.model_dump())
        return result
    except Exception as exc:
        logger.error("api_create_observation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _update_observation_sync(obs_id: int, body: dict) -> dict:
    from scanner.db import get_connection
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM macro_observations WHERE id = ?", [obs_id]
        ).fetchone()
        if not existing:
            raise ValueError(f"Observation {obs_id} not found")
        affected_str = ",".join(body.get("affected_tickers", [])) if body.get("affected_tickers") else None
        conn.execute(
            """
            UPDATE macro_observations
            SET observation_date = ?, type = ?, primary_ticker = ?,
                affected_tickers = ?, note = ?, link = ?
            WHERE id = ?
            """,
            [
                body["observation_date"],
                body["type"],
                body["primary_ticker"],
                affected_str,
                body["note"],
                body.get("link") or None,
                obs_id,
            ],
        )
        return {"status": "saved", "id": obs_id, "journal_updated": 0}
    except ValueError:
        raise
    except Exception as exc:
        logger.error("update_observation failed: %s", exc)
        raise
    finally:
        conn.close()


@app.put("/api/observations/{obs_id}")
async def api_update_observation(obs_id: int, body: _ObservationUpdate):
    try:
        result = await asyncio.to_thread(_update_observation_sync, obs_id, body.model_dump())
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("api_update_observation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _delete_observation_sync(obs_id: int) -> dict:
    from scanner.db import get_connection
    conn = get_connection()
    try:
        result = conn.execute(
            "DELETE FROM macro_observations WHERE id = ?", [obs_id]
        )
        if result.rowcount == 0:
            raise ValueError(f"Observation {obs_id} not found")
        return {"status": "deleted"}
    except ValueError:
        raise
    except Exception as exc:
        logger.error("delete_observation failed: %s", exc)
        raise
    finally:
        conn.close()


@app.delete("/api/observations/{obs_id}")
async def api_delete_observation(obs_id: int):
    try:
        return await asyncio.to_thread(_delete_observation_sync, obs_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("api_delete_observation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Graph dependents endpoint (used by observation modal)
# ---------------------------------------------------------------------------

@app.get("/api/graph/dependents/{parent}")
def api_graph_dependents(parent: str):
    try:
        from scanner.graph.loader import get_dependents
        edges = get_dependents(parent.upper())
        return {"dependents": [e.child for e in edges]}
    except Exception as exc:
        logger.error("api_graph_dependents failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 8-K Filings endpoints
# ---------------------------------------------------------------------------

class _FilingJournalSave(BaseModel):
    note: str
    affected_tickers: list[str] = []
    observation_date: str


class _FilingImpactUpdate(BaseModel):
    impact: str
    impact_source: str = "manual"


def _get_filings_sync(universe: str, ticker_filter: str | None) -> dict:
    from scanner.db import get_connection
    from scanner.graph.loader import get_all_tickers, MEGA_CAPS

    conn = get_connection()
    try:
        # Determine which tickers to include
        all_tickers = get_all_tickers()
        megacap_set: set[str] = set(MEGA_CAPS)

        if universe == "megacap":
            ticker_scope = [t for t in all_tickers if t in megacap_set]
        elif universe == "dependent":
            ticker_scope = [t for t in all_tickers if t not in megacap_set]
        else:
            ticker_scope = all_tickers

        if ticker_filter:
            ticker_scope = [t for t in ticker_scope if t == ticker_filter.upper()]

        if not ticker_scope:
            return {"groups": []}

        placeholders = ", ".join(["?" for _ in ticker_scope])
        rows = conn.execute(
            f"""
            SELECT id, ticker, filed_date, accession_number, form_type,
                   item_numbers, title, description, filing_url,
                   saved_to_journal, ingested_at, impact, impact_source, summary,
                   sentiment, impact_explanation
            FROM filings_8k
            WHERE ticker IN ({placeholders})
              AND filed_date >= CURRENT_DATE - INTERVAL 7 DAY
            ORDER BY ticker, filed_date DESC, impact DESC
            """,
            ticker_scope,
        ).fetchall()

        # Group by ticker; megacaps first
        groups_map: dict[str, dict] = {}
        for r in rows:
            t = r[1]
            if t not in groups_map:
                groups_map[t] = {
                    "ticker": t,
                    "is_megacap": t in megacap_set,
                    "filing_count": 0,
                    "filings": [],
                }
            groups_map[t]["filing_count"] += 1
            groups_map[t]["filings"].append({
                "id": r[0],
                "ticker": r[1],
                "filed_date": str(r[2]),
                "accession_number": r[3],
                "form_type": r[4],
                "item_numbers": r[5] or "",
                "title": r[6] or "",
                "description": r[7] or "",
                "filing_url": r[8] or "",
                "saved_to_journal": bool(r[9]),
                "ingested_at": str(r[10]),
                "impact": r[11] or "LOW",
                "impact_source": r[12] or "auto",
                "summary": r[13] or "",
                "sentiment": r[14] or "",
                "impact_explanation": r[15] or "",
            })

        # Sort: megacaps first, then dependents; within each group alpha by ticker
        groups = sorted(
            groups_map.values(),
            key=lambda g: (0 if g["is_megacap"] else 1, g["ticker"]),
        )
        return {"groups": groups}
    finally:
        conn.close()


def _save_filing_to_journal_sync(filing_id: int, body: dict) -> dict:
    from scanner.db import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT ticker, item_numbers, title, form_type FROM filings_8k WHERE id = ?",
            [filing_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"Filing {filing_id} not found")

        ticker, item_numbers, title, form_type = row
        note = body["note"]
        affected = body.get("affected_tickers", [])
        obs_date = body["observation_date"]

        # Build item label for the note prefix
        first_item = (item_numbers or "").split(",")[0].strip() if item_numbers else ""
        item_label_map = {
            "1.01": "Material Agreement", "1.02": "Agreement Terminated",
            "2.01": "Asset Acquisition/Disposal", "2.02": "Results of Operations",
            "2.05": "Employee Departure", "5.02": "Executive Change",
            "7.01": "Regulation FD", "8.01": "Other Material Event",
        }
        item_label = item_label_map.get(first_item, form_type)
        note_prefix = f"[8-K: {item_label}]"
        full_note = f"{note_prefix} {note}"

        # Insert macro_observation
        next_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM macro_observations"
        ).fetchone()[0]
        affected_str = ",".join(affected) if affected else None
        conn.execute(
            """
            INSERT INTO macro_observations
                (id, observation_date, type, primary_ticker, affected_tickers, note, link, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [next_id, obs_date, "8-K", ticker, affected_str, full_note, None, datetime.utcnow()],
        )

        # Append note to journal entries for primary + affected tickers
        all_tickers = [ticker] + [t for t in affected if t != ticker]
        for t in all_tickers:
            existing = conn.execute(
                """
                SELECT notes FROM journal
                WHERE ticker = ? AND scan_date = (
                    SELECT MAX(scan_date) FROM journal WHERE ticker = ?
                )
                """,
                [t, t],
            ).fetchone()
            if existing is None:
                continue
            current_notes = existing[0] or ""
            sep = " | " if current_notes else ""
            new_notes = current_notes + sep + full_note
            conn.execute(
                """
                UPDATE journal
                SET notes = ?
                WHERE ticker = ? AND scan_date = (
                    SELECT MAX(scan_date) FROM journal WHERE ticker = ?
                )
                """,
                [new_notes, t, t],
            )

        # Mark filing as saved
        conn.execute(
            "UPDATE filings_8k SET saved_to_journal = TRUE WHERE id = ?",
            [filing_id],
        )
        for _k in [k for k in _cache if k.startswith("filings:")]:
            _cache.pop(_k, None)
        return {"status": "ok", "journal_updated": len(all_tickers)}
    finally:
        conn.close()


def _update_filing_impact_sync(filing_id: int, body: dict) -> dict:
    from scanner.db import get_connection
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM filings_8k WHERE id = ?", [filing_id]).fetchone()
        if row is None:
            raise ValueError(f"Filing {filing_id} not found")
        impact = body["impact"].upper()
        if impact not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError(f"Invalid impact value: {impact}")
        conn.execute(
            "UPDATE filings_8k SET impact = ?, impact_source = ? WHERE id = ?",
            [impact, body.get("impact_source", "manual"), filing_id],
        )
        for _k in [k for k in _cache if k.startswith("filings:")]:
            _cache.pop(_k, None)
        return {"status": "ok"}
    finally:
        conn.close()


@app.get("/api/filings")
async def api_get_filings(
    universe: str = Query(default="all", pattern="^(all|megacap|dependent)$"),
    ticker: str | None = Query(default=None),
):
    cache_key = f"filings:{universe}:{ticker or ''}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        result = await asyncio.to_thread(_get_filings_sync, universe, ticker)
        _cache_set(cache_key, result)
        return result
    except Exception as exc:
        logger.error("api_get_filings failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/filings/{filing_id}/journal")
async def api_save_filing_to_journal(filing_id: int, body: _FilingJournalSave):
    try:
        return await asyncio.to_thread(_save_filing_to_journal_sync, filing_id, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("api_save_filing_to_journal failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/api/filings/{filing_id}/impact")
async def api_update_filing_impact(filing_id: int, body: _FilingImpactUpdate):
    try:
        return await asyncio.to_thread(_update_filing_impact_sync, filing_id, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("api_update_filing_impact failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Admin: force-full ingest (one-time migration tool)
# ---------------------------------------------------------------------------

@app.post("/api/admin/force-full-ingest")
async def api_admin_force_full_ingest(x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    def _run():
        from scanner.ingest.prices import ingest_all
        logger.info("force-full-ingest: starting background ingest (force_full=True)")
        try:
            ingest_all(force_full=True)
            logger.info("force-full-ingest: completed")
        except Exception as exc:
            logger.error("force-full-ingest: failed: %s", exc)

    asyncio.create_task(asyncio.to_thread(_run))
    return {"status": "started"}
