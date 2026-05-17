"""Walk-forward backtest for Sector Rotation Detection.

Tests whether XLK's rank among its sector peers (XLF, XLE, XLV) predicts
forward returns for MSFT/META dependents in CORRECTION regime.

Primary question: does LEADING XLK improve IC vs baseline RS alone?
Comparison: LEADING vs not-LEADING (2-way avoids small-n fragmentation).

IC score: raw 20d return differential (child 20d return minus average parent
20d return across qualifying MSFT/META parents in CORRECTION).

Validation bar: IC > 0.05 with IC IR > 0.5.
Sufficiency: n >= 100 → OK, 50–99 → MARGINAL, < 50 → INSUFFICIENT.
"""

import bisect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date, timedelta

import duckdb

from scanner.db import get_connection
from scanner.signals.megacap import MEGACAP_UNIVERSE

logger = logging.getLogger(__name__)

_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV"]
_HORIZONS = (5, 10, 20)
_RS_LOOKBACK = 20
_MA_SHORT = 50
_RVOL_LOOKBACK = 20
_ROT_LOOKBACK = 61   # need 61 rows for a valid 60d composite
_CORRECTION_LOW = -0.15
_CORRECTION_HIGH = -0.05
_VALIDATED_PARENTS_SET: frozenset[str] = frozenset({"MSFT", "META"})
_INSUFFICIENT_OBS = 50
_MARGINAL_OBS = 100


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def _std(lst: list[float]) -> float:
    if len(lst) < 2:
        return float("nan")
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


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


def _sufficiency_label(n: int) -> str:
    if n >= _MARGINAL_OBS:
        return "OK"
    if n >= _INSUFFICIENT_OBS:
        return "MARGINAL"
    return "INSUFFICIENT"


@dataclass
class RotationBucketStats:
    label: str                       # "baseline", "LEADING", "not-LEADING"
    n: dict[int, int]                # obs per horizon
    ic_mean: dict[int, float] = field(default_factory=dict)
    ic_h1_mean: dict[int, float] = field(default_factory=dict)
    ic_h2_mean: dict[int, float] = field(default_factory=dict)
    ic_ir: dict[int, float] = field(default_factory=dict)
    sufficiency: str = "INSUFFICIENT"


@dataclass
class RotationBacktestResult:
    n_dates: int
    start_date: str
    end_date: str
    mid_date: str
    n_h1: int
    n_h2: int
    etf_data_available: bool
    baseline: RotationBucketStats
    leading: RotationBucketStats
    not_leading: RotationBucketStats


def _compute_xlk_rank(
    etf_closes: dict[str, list[float]],
    lookback_count: int,
) -> int | None:
    """Return XLK rank (1=best, 4=worst) from the last `lookback_count` closes, or None."""
    composites: dict[str, float] = {}
    for etf in _SECTOR_ETFS:
        closes = etf_closes.get(etf, [])
        if len(closes) < lookback_count:
            return None
        base_20 = closes[-21] if len(closes) >= 21 else None
        base_60 = closes[-61] if len(closes) >= 61 else None
        last = closes[-1]
        if last == 0:
            return None
        r20 = (last - base_20) / base_20 if (base_20 and base_20 != 0) else float("nan")
        r60 = (last - base_60) / base_60 if (base_60 and base_60 != 0) else float("nan")
        valid = [v for v in (r20, r60) if not math.isnan(v)]
        composites[etf] = sum(valid) / len(valid) if valid else float("nan")

    valid_etfs = [(e, c) for e, c in composites.items() if not math.isnan(c)]
    if len(valid_etfs) < len(_SECTOR_ETFS):
        return None
    valid_etfs.sort(key=lambda x: x[1], reverse=True)
    rank_map = {e: i + 1 for i, (e, _) in enumerate(valid_etfs)}
    return rank_map.get("XLK")


def run_rotation_backtest() -> RotationBacktestResult | None:
    conn = get_connection()
    try:
        return _run(conn)
    finally:
        conn.close()


def _run(conn: duckdb.DuckDBPyConnection) -> RotationBacktestResult | None:
    from scanner.graph.loader import get_parents

    price_rows = conn.execute(
        "SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        "FROM prices ORDER BY ticker, date"
    ).fetchall()

    if not price_rows:
        logger.warning("rotation backtest: no price data found")
        return None

    raw_series: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for tkr, dt, close, vol in price_rows:
        raw_series[tkr].append((dt, float(close), float(vol) if vol is not None else 0.0))

    ticker_dates: dict[str, list[str]] = {}
    ticker_closes: dict[str, list[float]] = {}
    ticker_vols: dict[str, list[float]] = {}
    for tkr, series in raw_series.items():
        ticker_dates[tkr] = [d for d, _, _ in series]
        ticker_closes[tkr] = [c for _, c, _ in series]
        ticker_vols[tkr] = [v for _, _, v in series]

    # Check sector ETF availability
    missing_etfs = [e for e in _SECTOR_ETFS if e not in ticker_dates]
    etf_data_available = len(missing_etfs) == 0
    if missing_etfs:
        logger.warning(
            "rotation backtest: sector ETFs missing from prices table: %s. "
            "Run: scanner ingest XLK XLF XLE XLV SPY",
            missing_etfs,
        )

    # Precompute date -> XLK rank across all dates with sufficient ETF history.
    # We build a unified date list and for each date find the row index in each ETF series.
    all_unique_dates = sorted({d for dates in ticker_dates.values() for d in dates})

    date_to_xlk_leading: dict[str, bool] = {}  # True = LEADING (rank 1 or 2)
    if etf_data_available:
        for dt in all_unique_dates:
            etf_closes_window: dict[str, list[float]] = {}
            ok = True
            for etf in _SECTOR_ETFS:
                idx = bisect.bisect_right(ticker_dates[etf], dt) - 1
                if idx < _ROT_LOOKBACK - 1:
                    ok = False
                    break
                etf_closes_window[etf] = ticker_closes[etf][idx - _ROT_LOOKBACK + 1: idx + 1]
            if not ok:
                continue
            rank = _compute_xlk_rank(etf_closes_window, _ROT_LOOKBACK)
            if rank is not None:
                date_to_xlk_leading[dt] = rank <= 2

    megacap_set = set(MEGACAP_UNIVERSE)

    # Precompute parent 20d returns per date
    parent_20d_ret: dict[str, dict[str, float]] = {}
    for parent in megacap_set:
        if parent not in ticker_dates:
            continue
        p_dates = ticker_dates[parent]
        p_closes = ticker_closes[parent]
        ret_by_date: dict[str, float] = {}
        for i in range(_RS_LOOKBACK, len(p_dates)):
            prev = p_closes[i - _RS_LOOKBACK]
            if prev > 0:
                ret_by_date[p_dates[i]] = (p_closes[i] - prev) / prev
        parent_20d_ret[parent] = ret_by_date

    # Dates where each validated parent is in CORRECTION
    date_to_correction_parents: dict[str, set[str]] = defaultdict(set)
    for parent, ret_by_date in parent_20d_ret.items():
        if parent not in _VALIDATED_PARENTS_SET:
            continue
        for dt, ret in ret_by_date.items():
            if _CORRECTION_LOW <= ret < _CORRECTION_HIGH:
                date_to_correction_parents[dt].add(parent)

    # Build child -> validated mega-cap parents map
    child_to_parents: dict[str, list[str]] = {}
    for tkr in ticker_dates:
        if tkr in megacap_set or tkr in ("QQQ", "XLK", "XLF", "XLE", "XLV", "SPY"):
            continue
        parents = [
            e.parent for e in get_parents(tkr)
            if e.parent in _VALIDATED_PARENTS_SET
        ]
        if parents:
            child_to_parents[tkr] = parents

    def get_price_after(tkr: str, dt: str, h: int) -> float | None:
        dates = ticker_dates.get(tkr)
        if not dates:
            return None
        i = bisect.bisect_right(dates, dt)
        target = i + h - 1
        return ticker_closes[tkr][target] if target < len(dates) else None

    min_rows = max(_MA_SHORT + 1, _RS_LOOKBACK + 1, _RVOL_LOOKBACK + 1)

    # Accumulate observations: (rs_score: float, fwd_ret: float, is_leading: bool | None)
    # grouped by date for per-date IC; is_leading=None means no rotation data for that date
    by_date: dict[str, list[tuple[float, dict[int, float], bool | None]]] = defaultdict(list)
    all_dates_set: set[str] = set()

    for tkr, parents in child_to_parents.items():
        if tkr not in ticker_dates:
            continue
        dates = ticker_dates[tkr]
        closes = ticker_closes[tkr]
        n = len(dates)
        parents_set = set(parents)

        for i in range(min_rows, n):
            dt = dates[i]

            correction_on_date = date_to_correction_parents.get(dt)
            if not correction_on_date:
                continue
            qualifying = correction_on_date & parents_set
            if not qualifying:
                continue

            # RS score: child 20d return minus average qualifying-parent 20d return
            prev_close = closes[i - _RS_LOOKBACK]
            if prev_close <= 0 or closes[i] <= 0:
                continue
            child_ret = (closes[i] - prev_close) / prev_close

            parent_rets = []
            for p in qualifying:
                p_ret = parent_20d_ret.get(p, {}).get(dt)
                if p_ret is not None:
                    parent_rets.append(p_ret)
            if not parent_rets:
                continue
            avg_parent_ret = sum(parent_rets) / len(parent_rets)
            rs_score = child_ret - avg_parent_ret

            base = closes[i]
            fwd: dict[int, float] = {}
            for h in _HORIZONS:
                future = get_price_after(tkr, dt, h)
                if future is not None:
                    fwd[h] = (future - base) / base

            if not fwd:
                continue

            is_leading: bool | None = date_to_xlk_leading.get(dt)

            by_date[dt].append((rs_score, fwd, is_leading))
            all_dates_set.add(dt)

    if not all_dates_set:
        logger.warning("rotation backtest: no CORRECTION MSFT/META observations computed")
        return None

    all_dates = sorted(all_dates_set)
    mid_idx = len(all_dates) // 2
    mid_date = all_dates[mid_idx]
    h1_set = set(all_dates[:mid_idx])

    # Accumulators: baseline, leading, not-leading
    def _empty_per_h() -> dict[int, list[tuple[float, float]]]:
        return {h: [] for h in _HORIZONS}

    base_obs: dict[int, list[tuple[float, float]]] = _empty_per_h()
    lead_obs: dict[int, list[tuple[float, float]]] = _empty_per_h()
    notlead_obs: dict[int, list[tuple[float, float]]] = _empty_per_h()

    base_obs_h1: dict[int, list[tuple[float, float]]] = _empty_per_h()
    base_obs_h2: dict[int, list[tuple[float, float]]] = _empty_per_h()
    lead_obs_h1: dict[int, list[tuple[float, float]]] = _empty_per_h()
    lead_obs_h2: dict[int, list[tuple[float, float]]] = _empty_per_h()
    notlead_obs_h1: dict[int, list[tuple[float, float]]] = _empty_per_h()
    notlead_obs_h2: dict[int, list[tuple[float, float]]] = _empty_per_h()

    is_h1: dict[str, bool] = {dt: (dt in h1_set) for dt in all_dates}

    for dt in all_dates:
        in_h1 = is_h1[dt]
        for rs_score, fwd, leading in by_date[dt]:
            for h in _HORIZONS:
                ret = fwd.get(h)
                if ret is None:
                    continue
                pair = (rs_score, ret)
                base_obs[h].append(pair)
                if in_h1:
                    base_obs_h1[h].append(pair)
                else:
                    base_obs_h2[h].append(pair)

                if leading is None:
                    continue
                if leading:
                    lead_obs[h].append(pair)
                    (lead_obs_h1 if in_h1 else lead_obs_h2)[h].append(pair)
                else:
                    notlead_obs[h].append(pair)
                    (notlead_obs_h1 if in_h1 else notlead_obs_h2)[h].append(pair)

    def _build_bucket(
        label: str,
        obs: dict[int, list[tuple[float, float]]],
        obs_h1: dict[int, list[tuple[float, float]]],
        obs_h2: dict[int, list[tuple[float, float]]],
    ) -> RotationBucketStats:
        n = {h: len(obs[h]) for h in _HORIZONS}
        ic_mean: dict[int, float] = {}
        ic_h1_mean: dict[int, float] = {}
        ic_h2_mean: dict[int, float] = {}
        ic_ir: dict[int, float] = {}
        for h in _HORIZONS:
            scores = [x[0] for x in obs[h]]
            rets = [x[1] for x in obs[h]]
            m = _spearman_ic(scores, rets)
            ic_mean[h] = m
            ic_h1_mean[h] = _spearman_ic(
                [x[0] for x in obs_h1[h]], [x[1] for x in obs_h1[h]]
            )
            ic_h2_mean[h] = _spearman_ic(
                [x[0] for x in obs_h2[h]], [x[1] for x in obs_h2[h]]
            )
            # IC IR: pooled IC (one obs per cross-section) not available here;
            # approximate with IC / std_of_returns as a rough IR proxy.
            all_rets = [x[1] for x in obs[h]]
            std_r = _std(all_rets)
            ic_ir[h] = (
                m / std_r
                if (not math.isnan(m) and not math.isnan(std_r) and std_r > 0)
                else float("nan")
            )
        ref_n = n.get(5, 0)
        return RotationBucketStats(
            label=label,
            n=n,
            ic_mean=ic_mean,
            ic_h1_mean=ic_h1_mean,
            ic_h2_mean=ic_h2_mean,
            ic_ir=ic_ir,
            sufficiency=_sufficiency_label(ref_n),
        )

    baseline = _build_bucket("baseline", base_obs, base_obs_h1, base_obs_h2)
    leading = _build_bucket("LEADING", lead_obs, lead_obs_h1, lead_obs_h2)
    not_leading = _build_bucket("not-LEADING", notlead_obs, notlead_obs_h1, notlead_obs_h2)

    if baseline.n.get(5, 0) < _INSUFFICIENT_OBS:
        logger.warning(
            "rotation backtest: baseline has only %d 5d-horizon observations. "
            "Results not meaningful.",
            baseline.n.get(5, 0),
        )

    return RotationBacktestResult(
        n_dates=len(all_dates),
        start_date=all_dates[0],
        end_date=all_dates[-1],
        mid_date=mid_date,
        n_h1=len(h1_set),
        n_h2=len(all_dates) - len(h1_set),
        etf_data_available=etf_data_available,
        baseline=baseline,
        leading=leading,
        not_leading=not_leading,
    )
