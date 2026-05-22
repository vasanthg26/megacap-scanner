"""Cross-sectional Z-score backtest for AI Power Infrastructure basket.

Basket: BE, FPS, CEG, VST, HUT
Signal: 20-day return cross-sectional Z-score within basket
Combined: Z-score × RVOL ratio

No-lookahead guarantee: 20d return uses prices on and before signal_date.
Forward returns enter at signal_date's close, exit h trading days later.
"""

import bisect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import duckdb

from scanner.db import get_connection

logger = logging.getLogger(__name__)

BASKET = ["BE", "FPS", "CEG", "VST", "HUT"]
HORIZONS = (5, 10, 20)
_LOOKBACK = 20
_MIN_BASKET = 3
_POST2024_START = "2024-01-01"
_MIN_OBS_FULL = 100
_MIN_OBS_POST2024 = 30


@dataclass
class ZScoreResult:
    version: str
    start_date: str
    end_date: str
    mid_date: str
    ticker_obs: dict[str, int]
    total_obs: int
    n_h1: int
    n_h2: int
    # Z-score alone
    ic_pooled: dict[int, float] = field(default_factory=dict)
    ic_h1: dict[int, float] = field(default_factory=dict)
    ic_h2: dict[int, float] = field(default_factory=dict)
    # Z-score × RVOL
    ic_combined_pooled: dict[int, float] = field(default_factory=dict)
    ic_combined_h1: dict[int, float] = field(default_factory=dict)
    ic_combined_h2: dict[int, float] = field(default_factory=dict)


def _spearman_ic(scores: list[float], returns: list[float]) -> float:
    n = len(scores)
    if n < 3:
        return float("nan")

    def _rank(lst: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = float(r + 1)
        return ranks

    rs = _rank(scores)
    rr = _rank(returns)
    mean_s = sum(rs) / n
    mean_r = sum(rr) / n
    cov = sum((rs[i] - mean_s) * (rr[i] - mean_r) for i in range(n)) / n
    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in rs) / n)
    std_r = math.sqrt(sum((x - mean_r) ** 2 for x in rr) / n)
    if std_s == 0 or std_r == 0:
        return float("nan")
    return cov / (std_s * std_r)


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def run_zscore_backtest() -> tuple[ZScoreResult, ZScoreResult] | None:
    conn = get_connection()
    try:
        return _run(conn)
    finally:
        conn.close()


def _run(conn: duckdb.DuckDBPyConnection) -> tuple[ZScoreResult, ZScoreResult] | None:
    placeholders = ",".join("?" for _ in BASKET)
    rows = conn.execute(
        f"SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        f"FROM prices WHERE ticker IN ({placeholders}) ORDER BY ticker, date",
        BASKET,
    ).fetchall()

    if not rows:
        logger.warning("zscore backtest: no price data found for basket %s", BASKET)
        return None

    ticker_dates: dict[str, list[str]] = defaultdict(list)
    ticker_closes: dict[str, list[float]] = defaultdict(list)
    ticker_vols: dict[str, list[float]] = defaultdict(list)

    for tkr, dt, close, vol in rows:
        ticker_dates[tkr].append(dt)
        ticker_closes[tkr].append(float(close))
        ticker_vols[tkr].append(float(vol) if vol is not None else 0.0)

    all_dates = sorted({dt for dates in ticker_dates.values() for dt in dates})

    if len(all_dates) < _LOOKBACK + max(HORIZONS) + 5:
        logger.warning("zscore backtest: insufficient price history (%d dates)", len(all_dates))
        return None

    # Build observations: one entry per (signal_date, ticker) pair.
    observations: list[dict] = []

    for signal_date in all_dates:
        basket_rets: dict[str, float] = {}
        basket_rvols: dict[str, float] = {}

        for tkr in BASKET:
            dates = ticker_dates.get(tkr, [])
            closes = ticker_closes.get(tkr, [])
            vols = ticker_vols.get(tkr, [])

            idx = bisect.bisect_right(dates, signal_date) - 1
            if idx < 0 or dates[idx] != signal_date:
                continue  # no data on this date

            if idx < _LOOKBACK:
                continue  # insufficient lookback for 20d return

            close_now = closes[idx]
            close_20d_ago = closes[idx - _LOOKBACK]
            if close_20d_ago == 0:
                continue

            basket_rets[tkr] = (close_now - close_20d_ago) / close_20d_ago

            avg_vol = _mean(vols[idx - _LOOKBACK:idx])
            if avg_vol > 0:
                basket_rvols[tkr] = vols[idx] / avg_vol

        if len(basket_rets) < _MIN_BASKET:
            continue

        ret_vals = list(basket_rets.values())
        basket_mean = _mean(ret_vals)
        n = len(ret_vals)
        basket_std = math.sqrt(sum((r - basket_mean) ** 2 for r in ret_vals) / n)

        if basket_std == 0:
            continue

        for tkr, ret_20d in basket_rets.items():
            z = (ret_20d - basket_mean) / basket_std
            rvol = basket_rvols.get(tkr, float("nan"))
            combined = z * rvol if not math.isnan(rvol) else float("nan")

            dates = ticker_dates[tkr]
            closes = ticker_closes[tkr]
            idx = bisect.bisect_right(dates, signal_date) - 1
            base_close = closes[idx]

            fwds: dict[int, float | None] = {}
            for h in HORIZONS:
                # Entry: day after signal_date; exit: h days after signal_date
                entry_idx = idx + 1
                exit_idx = idx + h
                if exit_idx >= len(closes) or base_close == 0:
                    fwds[h] = None
                else:
                    fwds[h] = (closes[exit_idx] - base_close) / base_close

            observations.append({
                "date": signal_date,
                "ticker": tkr,
                "z_score": z,
                "combined": combined,
                "fwd": fwds,
            })

    if not observations:
        logger.warning("zscore backtest: no observations computed")
        return None

    def _compute_result(obs: list[dict], version: str) -> ZScoreResult | None:
        if not obs:
            return None

        dates = sorted({o["date"] for o in obs})
        mid_idx = len(dates) // 2
        mid_date = dates[mid_idx] if dates else ""
        h1_dates = set(dates[:mid_idx])

        ticker_obs: dict[str, int] = defaultdict(int)
        for o in obs:
            if o["fwd"].get(5) is not None:
                ticker_obs[o["ticker"]] += 1

        total_obs = sum(ticker_obs.values())
        n_h1 = sum(1 for o in obs if o["date"] in h1_dates and o["fwd"].get(5) is not None)
        n_h2 = total_obs - n_h1

        ic_pooled: dict[int, float] = {}
        ic_h1: dict[int, float] = {}
        ic_h2: dict[int, float] = {}
        ic_combined_pooled: dict[int, float] = {}
        ic_combined_h1: dict[int, float] = {}
        ic_combined_h2: dict[int, float] = {}

        for h in HORIZONS:
            all_z, all_fwd = [], []
            h1_z, h1_fwd = [], []
            h2_z, h2_fwd = [], []
            all_c, all_cf = [], []
            h1_c, h1_cf = [], []
            h2_c, h2_cf = [], []

            for o in obs:
                fwd = o["fwd"].get(h)
                if fwd is None:
                    continue
                z = o["z_score"]
                c = o["combined"]
                d = o["date"]

                all_z.append(z)
                all_fwd.append(fwd)
                if d in h1_dates:
                    h1_z.append(z)
                    h1_fwd.append(fwd)
                else:
                    h2_z.append(z)
                    h2_fwd.append(fwd)

                if not math.isnan(c):
                    all_c.append(c)
                    all_cf.append(fwd)
                    if d in h1_dates:
                        h1_c.append(c)
                        h1_cf.append(fwd)
                    else:
                        h2_c.append(c)
                        h2_cf.append(fwd)

            ic_pooled[h] = _spearman_ic(all_z, all_fwd)
            ic_h1[h] = _spearman_ic(h1_z, h1_fwd)
            ic_h2[h] = _spearman_ic(h2_z, h2_fwd)
            ic_combined_pooled[h] = _spearman_ic(all_c, all_cf)
            ic_combined_h1[h] = _spearman_ic(h1_c, h1_cf)
            ic_combined_h2[h] = _spearman_ic(h2_c, h2_cf)

        return ZScoreResult(
            version=version,
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
            mid_date=mid_date,
            ticker_obs=dict(ticker_obs),
            total_obs=total_obs,
            n_h1=n_h1,
            n_h2=n_h2,
            ic_pooled=ic_pooled,
            ic_h1=ic_h1,
            ic_h2=ic_h2,
            ic_combined_pooled=ic_combined_pooled,
            ic_combined_h1=ic_combined_h1,
            ic_combined_h2=ic_combined_h2,
        )

    full_result = _compute_result(observations, "Full History")
    post2024_obs = [o for o in observations if o["date"] >= _POST2024_START]
    post2024_result = _compute_result(post2024_obs, "Post-2024")

    return full_result, post2024_result
