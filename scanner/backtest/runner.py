"""Walk-forward backtester.

Design:
  - Weekly rebalance, long top quintile of scored children per mega-cap.
  - Scores are computed strictly from data available on the rebalance date.
  - Forward returns measured over `horizon` trading days.
  - Reports: Sharpe, Max Drawdown, Information Coefficient (Spearman rank correlation
    of signal score vs forward return). IC is the headline metric.

Survivorship-bias caveat: yfinance only returns currently-listed tickers.
Results will overstate performance. Migrate to Polygon.io for production use.
"""

import math
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from scanner.db import get_connection
from scanner.graph.loader import MEGA_CAPS, get_dependents
from scanner.signals.relative_strength import RelativeStrengthVsParent
from scanner.signals.base import classify_action

logger = logging.getLogger(__name__)

SIGNAL_REGISTRY = {
    "rs": "relative_strength",
    "rs_normalized": "rs_normalized",
}

_DIAG_HORIZONS = (1, 5, 10, 20)


@dataclass
class QuintileStat:
    action: str
    count: int
    avg_fwd_1d: float
    avg_fwd_5d: float
    avg_fwd_10d: float
    avg_fwd_20d: float
    hit_rate: float       # fraction of primary-horizon returns > 0
    avg_vol_20d: float    # annualised 20-day realised vol


@dataclass
class BacktestResult:
    signal: str
    horizon: int
    sharpe: float
    max_drawdown: float
    ic_mean: float
    ic_std: float
    ic_ir: float
    n_periods: int
    period_returns: list[float] = field(default_factory=list)
    ic_series: list[float] = field(default_factory=list)
    quintile_stats: list[QuintileStat] = field(default_factory=list)
    per_parent_ic: dict[str, tuple[float, int]] = field(default_factory=dict)
    parent_conditional_ic: dict[str, tuple[float, int]] = field(default_factory=dict)
    parent_regime_ic: dict[str, tuple[float, int]] = field(default_factory=dict)
    regime_ic_splits: dict[str, dict[str, tuple[float, int]]] = field(default_factory=dict)
    split_dates: tuple[str, str, str] | None = field(default=None)


def _get_trading_dates(conn, start: str, end: str) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT date FROM prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, [start, end]).fetchall()
    return [str(r[0]) for r in rows]


def _score_children_on_date(
    parent: str, date: str, conn, lookback: int, signal_name: str = "rs"
) -> dict[str, float]:
    edges = get_dependents(parent)
    if not edges:
        return {}
    if signal_name == "rs_normalized":
        from scanner.signals.rs_normalized import RSNormalized
        signal = RSNormalized(parent=parent, lookback=lookback)
    else:
        signal = RelativeStrengthVsParent(parent=parent, lookback=lookback)
    scores: dict[str, float] = {}
    for edge in edges:
        if edge.child in scores:
            continue
        score = signal.compute(edge.child, date, conn)
        if not math.isnan(score):
            scores[edge.child] = score
    return scores


def _forward_return(ticker: str, from_date: str, horizon: int, conn) -> float | None:
    """Return over next `horizon` trading days from from_date (exclusive).

    Entry is day+1 close; exit is day+horizon close. Used for IC and portfolio
    returns — consistent with the live scan convention. Do NOT use for diagnostics
    where horizon=1 would collapse to zero (same entry/exit row).
    """
    rows = conn.execute("""
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date > ?
        ORDER BY date ASC
        LIMIT ?
    """, [ticker, from_date, horizon]).fetchall()

    if len(rows) < horizon:
        return None

    entry = rows[0][0]
    exit_ = rows[-1][0]
    if entry == 0:
        return None
    return (exit_ - entry) / entry


def _multi_horizon_returns(
    ticker: str,
    from_date: str,
    horizons: tuple[int, ...],
    conn,
) -> dict[int, float | None]:
    """Forward returns at multiple horizons for diagnostic tables.

    Uses inclusive from_date so rows[0] is from_date's close (true entry),
    rows[h] is the exit after exactly h trading days. This avoids the
    horizon=1 zero-return bug present in _forward_return (which is exclusive).
    """
    max_h = max(horizons)
    rows = conn.execute("""
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date >= ?
        ORDER BY date ASC
        LIMIT ?
    """, [ticker, from_date, max_h + 1]).fetchall()

    entry = rows[0][0] if rows else 0
    result: dict[int, float | None] = {}
    for h in horizons:
        if len(rows) <= h or entry == 0:
            result[h] = None
        else:
            result[h] = (rows[h][0] - entry) / entry
    return result


def _rolling_return(ticker: str, as_of: str, lookback: int, conn) -> float:
    """lookback-day rolling return as of `as_of`."""
    rows = conn.execute("""
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
    """, [ticker, as_of, lookback + 1]).fetchall()

    if len(rows) < lookback + 1:
        return float("nan")
    oldest = rows[-1][0]
    if oldest == 0:
        return float("nan")
    return (rows[0][0] - oldest) / oldest


def _realized_vol(ticker: str, as_of: str, lookback: int, conn) -> float:
    """Annualised 20-day realised volatility."""
    rows = conn.execute("""
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
    """, [ticker, as_of, lookback + 1]).fetchall()

    if len(rows) < lookback + 1:
        return float("nan")

    closes = [r[0] for r in reversed(rows)]
    daily = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1] != 0]
    if len(daily) < 2:
        return float("nan")
    return float(np.std(daily, ddof=1) * np.sqrt(252))


def _compute_regime_ic(
    obs_list: list[dict],
    min_pairs: int = 5,
) -> dict[str, tuple[float, int]]:
    """Spearman IC per parent-return magnitude regime bucket."""
    regime_groups: dict[str, list[tuple[float, float]]] = {
        "strong": [],
        "mild": [],
        "correction": [],
        "drawdown": [],
    }
    for obs in obs_list:
        if obs["fwd"] is None:
            continue
        pr = obs.get("parent_ret_20d", float("nan"))
        if math.isnan(pr):
            continue
        if pr > 0:
            key = "strong"
        elif pr >= -0.05:
            key = "mild"
        elif pr >= -0.15:
            key = "correction"
        else:
            key = "drawdown"
        regime_groups[key].append((obs["score"], obs["fwd"]))

    result: dict[str, tuple[float, int]] = {}
    for key, pairs in regime_groups.items():
        if len(pairs) < min_pairs:
            continue
        scores_r, fwds_r = zip(*pairs)
        ic, _ = spearmanr(scores_r, fwds_r)
        if not math.isnan(ic):
            result[key] = (float(ic), len(pairs))
    return result


def _compute_diagnostics(
    all_obs: list[dict],
    horizon: int,
) -> tuple[list[QuintileStat], dict[str, tuple[float, int]], dict[str, tuple[float, int]]]:
    """Compute per-quintile stats, per-parent IC, and parent-conditional IC.

    Each obs dict has keys: action, parent, score, fwd (primary horizon),
    diag (dict horizon→float|None), vol_20d, parent_ret_20d.
    """
    action_order = ["STRONG_BUY", "BUY", "HOLD", "SELL"]

    # --- Per-quintile stats ---
    buckets: dict[str, list[dict]] = {a: [] for a in action_order}
    for obs in all_obs:
        a = obs["action"]
        if a in buckets:
            buckets[a].append(obs)

    def _avg(vals: list[float | None]) -> float:
        good = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(good)) if good else float("nan")

    quintile_stats: list[QuintileStat] = []
    for action in action_order:
        group = buckets[action]
        fwd_by_h = {h: [obs["diag"].get(h) for obs in group] for h in _DIAG_HORIZONS}
        primary_fwd = [obs["fwd"] for obs in group if obs["fwd"] is not None]
        hit_rate = sum(v > 0 for v in primary_fwd) / len(primary_fwd) if primary_fwd else float("nan")
        vols = [obs["vol_20d"] for obs in group if not math.isnan(obs.get("vol_20d", float("nan")))]
        quintile_stats.append(QuintileStat(
            action=action,
            count=len(group),
            avg_fwd_1d=_avg(fwd_by_h[1]),
            avg_fwd_5d=_avg(fwd_by_h[5]),
            avg_fwd_10d=_avg(fwd_by_h[10]),
            avg_fwd_20d=_avg(fwd_by_h[20]),
            hit_rate=hit_rate,
            avg_vol_20d=float(np.mean(vols)) if vols else float("nan"),
        ))

    # --- Per-parent IC ---
    parent_groups: dict[str, list[tuple[float, float]]] = {}
    for obs in all_obs:
        if obs["fwd"] is None:
            continue
        p = obs["parent"]
        parent_groups.setdefault(p, []).append((obs["score"], obs["fwd"]))

    per_parent_ic: dict[str, tuple[float, int]] = {}
    for parent, pairs in parent_groups.items():
        if len(pairs) < 5:
            continue
        scores_p, fwds_p = zip(*pairs)
        ic, _ = spearmanr(scores_p, fwds_p)
        if not math.isnan(ic):
            per_parent_ic[parent] = (float(ic), len(pairs))

    # --- Parent-conditional IC ---
    cond_groups: dict[str, list[tuple[float, float]]] = {"strong": [], "weak": []}
    for obs in all_obs:
        if obs["fwd"] is None:
            continue
        pr = obs.get("parent_ret_20d", float("nan"))
        if math.isnan(pr):
            continue
        key = "strong" if pr > 0 else "weak"
        cond_groups[key].append((obs["score"], obs["fwd"]))

    parent_conditional_ic: dict[str, tuple[float, int]] = {}
    for key, pairs in cond_groups.items():
        if len(pairs) < 5:
            continue
        scores_c, fwds_c = zip(*pairs)
        ic, _ = spearmanr(scores_c, fwds_c)
        if not math.isnan(ic):
            parent_conditional_ic[key] = (float(ic), len(pairs))

    # --- Parent-regime IC (magnitude breakdown of "parent down" condition) ---
    parent_regime_ic = _compute_regime_ic(all_obs, min_pairs=5)

    return quintile_stats, per_parent_ic, parent_conditional_ic, parent_regime_ic


def run_backtest(
    signal_name: str = "rs",
    horizon: int = 5,
    lookback: int = 20,
    start: str | None = None,
    end: str | None = None,
    parents: list[str] | None = None,
    tickers: list[str] | None = None,
) -> BacktestResult:
    if signal_name not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal '{signal_name}'. Available: {list(SIGNAL_REGISTRY)}")

    active_parents = parents if parents else MEGA_CAPS
    unknown = [p for p in active_parents if p not in MEGA_CAPS]
    if unknown:
        raise ValueError(f"Unknown parent(s): {unknown}. Must be one of {MEGA_CAPS}")

    ticker_filter: set[str] | None = set(tickers) if tickers else None

    conn = get_connection()
    all_dates = _get_trading_dates(conn, start or "2000-01-01", end or "2099-01-01")

    if len(all_dates) < lookback + horizon + 10:
        raise RuntimeError(f"Insufficient price history: only {len(all_dates)} trading days.")

    rebalance_dates = all_dates[lookback::5]
    rebalance_dates = rebalance_dates[:-1]  # drop last — need horizon days after

    all_ic: list[float] = []
    period_portfolio_returns: list[float] = []
    all_obs: list[dict] = []

    for rebal_date in rebalance_dates:
        # Pass A: collect scores for every parent's children this period.
        period_rows: list[dict] = []  # {parent, ticker, score, parent_ret_20d}
        period_scores_list: list[float] = []

        for parent in active_parents:
            scores = _score_children_on_date(parent, rebal_date, conn, lookback, signal_name)
            if ticker_filter is not None:
                scores = {t: s for t, s in scores.items() if t in ticker_filter}
            if not scores:
                continue
            parent_ret_20d = _rolling_return(parent, rebal_date, 20, conn)
            for ticker, score in scores.items():
                period_rows.append({
                    "parent": parent,
                    "ticker": ticker,
                    "score": score,
                    "parent_ret_20d": parent_ret_20d,
                })
                period_scores_list.append(score)

        # Pass B: forward returns, classification, diagnostics.
        period_ic_scores: list[float] = []
        period_ic_fwds: list[float] = []
        long_tickers: list[str] = []

        for row in period_rows:
            ticker = row["ticker"]

            fwd = _forward_return(ticker, rebal_date, horizon, conn)
            diag = _multi_horizon_returns(ticker, rebal_date, _DIAG_HORIZONS, conn)
            vol_20d = _realized_vol(ticker, rebal_date, 20, conn)
            action = classify_action(row["score"], period_scores_list)

            if fwd is not None:
                period_ic_scores.append(row["score"])
                period_ic_fwds.append(fwd)

            all_obs.append({
                "parent": row["parent"],
                "ticker": ticker,
                "score": row["score"],
                "action": str(action),
                "fwd": fwd,
                "diag": diag,
                "vol_20d": vol_20d,
                "parent_ret_20d": row["parent_ret_20d"],
                "rebal_date": rebal_date,
            })

        # Build long basket: top quintile per parent.
        parent_scored: dict[str, list[tuple[str, float]]] = {}
        for row in period_rows:
            parent_scored.setdefault(row["parent"], []).append((row["ticker"], row["score"]))

        for parent, pairs in parent_scored.items():
            sorted_children = sorted(pairs, key=lambda x: x[1], reverse=True)
            top_n = max(1, len(sorted_children) // 5)
            long_tickers.extend(t for t, _ in sorted_children[:top_n])

        # IC for this period.
        if len(period_ic_scores) >= 5:
            ic, _ = spearmanr(period_ic_scores, period_ic_fwds)
            if not math.isnan(ic):
                all_ic.append(ic)

        # Portfolio return: equal-weight long basket.
        basket_returns: list[float] = []
        for ticker in set(long_tickers):
            fwd = _forward_return(ticker, rebal_date, horizon, conn)
            if fwd is not None:
                basket_returns.append(fwd)

        period_portfolio_returns.append(float(np.mean(basket_returns)) if basket_returns else 0.0)

    if not period_portfolio_returns:
        raise RuntimeError("No valid rebalance periods found. Run ingest first.")

    rets = np.array(period_portfolio_returns)
    periods_per_year = 252 / horizon
    sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) if rets.std() > 0 else 0.0

    cumulative = np.cumprod(1 + rets)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdowns.min())

    ic_array = np.array(all_ic) if all_ic else np.array([0.0])
    ic_mean = float(ic_array.mean())
    ic_std = float(ic_array.std()) if len(ic_array) > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

    quintile_stats, per_parent_ic, parent_conditional_ic, parent_regime_ic = _compute_diagnostics(all_obs, horizon)

    # --- Out-of-sample time split for regime IC ---
    regime_ic_splits: dict[str, dict[str, tuple[float, int]]] = {}
    split_dates: tuple[str, str, str] | None = None

    all_rebal_dates = sorted(set(obs["rebal_date"] for obs in all_obs))
    if len(all_rebal_dates) >= 4:
        mid_idx = len(all_rebal_dates) // 2
        mid_date = all_rebal_dates[mid_idx]
        h1_obs = [obs for obs in all_obs if obs["rebal_date"] < mid_date]
        h2_obs = [obs for obs in all_obs if obs["rebal_date"] >= mid_date]
        regime_ic_splits["h1"] = _compute_regime_ic(h1_obs, min_pairs=20)
        regime_ic_splits["h2"] = _compute_regime_ic(h2_obs, min_pairs=20)
        split_dates = (all_rebal_dates[0], mid_date, all_rebal_dates[-1])

    conn.close()

    return BacktestResult(
        signal=signal_name,
        horizon=horizon,
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        n_periods=len(period_portfolio_returns),
        period_returns=list(rets),
        ic_series=list(all_ic),
        quintile_stats=quintile_stats,
        per_parent_ic=per_parent_ic,
        parent_conditional_ic=parent_conditional_ic,
        parent_regime_ic=parent_regime_ic,
        regime_ic_splits=regime_ic_splits,
        split_dates=split_dates,
    )
