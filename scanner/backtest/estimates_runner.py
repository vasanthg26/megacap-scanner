"""Walk-forward backtest for the earnings revision momentum signal.

Requires accumulated daily snapshots in the `estimates` DuckDB table.
Run `ingest-estimates` each day to build up history.

Minimum thresholds before IC is computed:
  _MIN_DATES: distinct snapshot dates with revision_score present
  _MIN_OBS:   total (date x ticker) observations at the 5d horizon
"""

import bisect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import duckdb

from scanner.db import get_connection

logger = logging.getLogger(__name__)

_MIN_DATES = 20
_MIN_OBS = 100
_HORIZONS = (5, 10, 20)


@dataclass
class EstimatesBacktestResult:
    n_dates: int
    n_obs: int
    start_date: str
    end_date: str
    mid_date: str
    n_h1: int
    n_h2: int
    ic_mean: dict[int, float] = field(default_factory=dict)
    ic_std: dict[int, float] = field(default_factory=dict)
    ic_ir: dict[int, float] = field(default_factory=dict)
    ic_h1_mean: dict[int, float] = field(default_factory=dict)
    ic_h2_mean: dict[int, float] = field(default_factory=dict)
    positive_avg: dict[int, float] = field(default_factory=dict)
    negative_avg: dict[int, float] = field(default_factory=dict)
    spread: dict[int, float] = field(default_factory=dict)


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


def _std(lst: list[float]) -> float:
    if len(lst) < 2:
        return float("nan")
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


def run_estimates_backtest() -> EstimatesBacktestResult | None:
    """
    Walk-forward backtest of the revision_score signal.
    Returns None if insufficient history; logs the data-availability limitation clearly.
    """
    conn = get_connection()
    try:
        return _run(conn)
    finally:
        conn.close()


def _run(conn: duckdb.DuckDBPyConnection) -> EstimatesBacktestResult | None:
    n_dates = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM estimates WHERE revision_score IS NOT NULL"
    ).fetchone()[0]

    if n_dates < _MIN_DATES:
        logger.warning(
            "estimates backtest requires %d+ snapshot dates with coverage; "
            "only %d found. Run ingest-estimates daily — backtest unlocks after ~%d more days.",
            _MIN_DATES,
            n_dates,
            _MIN_DATES - n_dates,
        )
        return None

    # Load all estimates with coverage
    est_rows = conn.execute(
        "SELECT ticker, CAST(date AS VARCHAR), revision_score "
        "FROM estimates WHERE revision_score IS NOT NULL ORDER BY date, ticker"
    ).fetchall()

    tickers = list({r[0] for r in est_rows})

    # Batch-load prices for all relevant tickers
    placeholders = ", ".join(["?" for _ in tickers])
    price_rows = conn.execute(
        f"SELECT ticker, CAST(date AS VARCHAR), adj_close "
        f"FROM prices WHERE ticker IN ({placeholders}) ORDER BY ticker, date",
        tickers,
    ).fetchall()

    # Build sorted price series per ticker for O(log n) lookups
    raw_prices: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for tkr, dt, price in price_rows:
        raw_prices[tkr].append((dt, price))

    price_dates: dict[str, list[str]] = {}
    price_vals: dict[str, list[float]] = {}
    for tkr, series in raw_prices.items():
        price_dates[tkr] = [d for d, _ in series]
        price_vals[tkr] = [p for _, p in series]

    def get_price_at(tkr: str, dt: str) -> float | None:
        dates = price_dates.get(tkr)
        if not dates:
            return None
        i = bisect.bisect_right(dates, dt) - 1
        return price_vals[tkr][i] if i >= 0 else None

    def get_price_after(tkr: str, dt: str, h: int) -> float | None:
        dates = price_dates.get(tkr)
        if not dates:
            return None
        i = bisect.bisect_right(dates, dt)   # first date strictly after dt
        target = i + h - 1
        return price_vals[tkr][target] if target < len(dates) else None

    # Group estimates by date
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for tkr, dt, score in est_rows:
        by_date[dt][tkr] = score

    all_dates = sorted(by_date.keys())
    mid_idx = len(all_dates) // 2
    mid_date = all_dates[mid_idx]
    h1_set = set(all_dates[:mid_idx])

    # Single pass: compute per-date IC and accumulate spread data
    per_date_ic: dict[int, list[tuple[str, float]]] = {h: [] for h in _HORIZONS}
    pos_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    neg_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    n_obs_5d = 0

    for dt in all_dates:
        ticker_scores = by_date[dt]
        horizon_pairs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}

        for tkr, score in ticker_scores.items():
            base = get_price_at(tkr, dt)
            if base is None or base == 0:
                continue
            for h in _HORIZONS:
                future = get_price_after(tkr, dt, h)
                if future is None:
                    continue
                fwd_ret = (future - base) / base
                horizon_pairs[h].append((score, fwd_ret))

        for h in _HORIZONS:
            pairs = horizon_pairs[h]
            if len(pairs) >= 3:
                ic = _spearman_ic([p[0] for p in pairs], [p[1] for p in pairs])
                if not math.isnan(ic):
                    per_date_ic[h].append((dt, ic))
                    if h == 5:
                        n_obs_5d += len(pairs)
            for score, fwd_ret in pairs:
                if score > 0:
                    pos_rets[h].append(fwd_ret)
                elif score < 0:
                    neg_rets[h].append(fwd_ret)

    if n_obs_5d < _MIN_OBS:
        logger.warning(
            "estimates backtest: only %d observations at 5d horizon (need %d+). "
            "Run ingest-estimates daily to accumulate history.",
            n_obs_5d,
            _MIN_OBS,
        )
        return None

    # Aggregate IC stats
    ic_mean: dict[int, float] = {}
    ic_std: dict[int, float] = {}
    ic_ir: dict[int, float] = {}
    ic_h1_mean: dict[int, float] = {}
    ic_h2_mean: dict[int, float] = {}

    for h in _HORIZONS:
        all_ics = [ic for _, ic in per_date_ic[h]]
        h1_ics = [ic for dt, ic in per_date_ic[h] if dt in h1_set]
        h2_ics = [ic for dt, ic in per_date_ic[h] if dt not in h1_set]

        m = _mean(all_ics)
        s = _std(all_ics)
        ic_mean[h] = m
        ic_std[h] = s
        ic_ir[h] = m / s if not (math.isnan(m) or math.isnan(s) or s == 0) else float("nan")
        ic_h1_mean[h] = _mean(h1_ics)
        ic_h2_mean[h] = _mean(h2_ics)

    # Label spread: positive vs negative revision_score
    positive_avg: dict[int, float] = {}
    negative_avg: dict[int, float] = {}
    spread: dict[int, float] = {}
    for h in _HORIZONS:
        pa = _mean(pos_rets[h])
        na = _mean(neg_rets[h])
        positive_avg[h] = pa
        negative_avg[h] = na
        spread[h] = pa - na if not (math.isnan(pa) or math.isnan(na)) else float("nan")

    return EstimatesBacktestResult(
        n_dates=len(all_dates),
        n_obs=n_obs_5d,
        start_date=all_dates[0],
        end_date=all_dates[-1],
        mid_date=mid_date,
        n_h1=len(h1_set),
        n_h2=len(all_dates) - len(h1_set),
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_h1_mean=ic_h1_mean,
        ic_h2_mean=ic_h2_mean,
        positive_avg=positive_avg,
        negative_avg=negative_avg,
        spread=spread,
    )
