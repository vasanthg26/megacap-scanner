"""Walk-forward backtest for the Cross-Signal Confirmation Score.

Scoring uses compute_confirmation_backtest() — max score 4, NOT 5.
Est Rev is EXCLUDED: only 1 daily snapshot exists; backtest unlocks once 20+
daily snapshots are accumulated via `ingest-estimates`.

Score buckets: {0-1, 2, 3-4}  (max=4 because Est Rev is excluded)

Insider lookahead guard: uses filed_date <= T (not transaction_date) so that
the 2-business-day Form 4 filing window is respected. transaction_date >= T - 30d
ensures only the 30-day window is used.

QQQ fallback: if QQQ is not in the prices table, mega-cap RS uses own 20d
return > 0 as the positive/negative binary. This is noted in the result.
"""

import bisect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date, timedelta

import duckdb

from scanner.db import get_connection
from scanner.signals.confirmation import compute_confirmation_backtest
from scanner.signals.megacap import MEGACAP_UNIVERSE

logger = logging.getLogger(__name__)

_MIN_OBS = 100
_HORIZONS = (5, 10, 20)
_RVOL_LOOKBACK = 20
_MA_SHORT = 50
_RS_LOOKBACK = 20
_INSIDER_WINDOW_DAYS = 30


@dataclass
class BucketStats:
    label: str
    n: int
    avg_ret: dict[int, float] = field(default_factory=dict)


@dataclass
class ConfirmationBacktestResult:
    n_dates: int
    n_obs: int
    start_date: str
    end_date: str
    mid_date: str
    n_h1: int
    n_h2: int
    qqq_available: bool
    ic_mean: dict[int, float] = field(default_factory=dict)
    ic_std: dict[int, float] = field(default_factory=dict)
    ic_ir: dict[int, float] = field(default_factory=dict)
    ic_h1_mean: dict[int, float] = field(default_factory=dict)
    ic_h2_mean: dict[int, float] = field(default_factory=dict)
    ic_megacap: dict[int, float] = field(default_factory=dict)
    ic_dependent: dict[int, float] = field(default_factory=dict)
    n_megacap_obs: int = 0
    n_dependent_obs: int = 0
    buckets: list[BucketStats] = field(default_factory=list)
    ic_rs_alone: dict[int, float] = field(default_factory=dict)
    ic_confirm_score: dict[int, float] = field(default_factory=dict)


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


def _had_insider_buy(
    tkr: str,
    as_of_str: str,
    insider_by_ticker: dict[str, list[tuple[str, str]]],
) -> bool:
    """True if any open-market buy was filed on/before as_of and transacted within 30d."""
    buys = insider_by_ticker.get(tkr)
    if not buys:
        return False
    as_of_d = _date.fromisoformat(as_of_str)
    cutoff = as_of_d - timedelta(days=_INSIDER_WINDOW_DAYS)
    cutoff_str = str(cutoff)
    for filed_str, txn_str in buys:
        if filed_str > as_of_str:
            break
        if txn_str >= cutoff_str:
            return True
    return False


def run_confirmation_backtest() -> ConfirmationBacktestResult | None:
    conn = get_connection()
    try:
        return _run(conn)
    finally:
        conn.close()


def _run(conn: duckdb.DuckDBPyConnection) -> ConfirmationBacktestResult | None:
    # Load all price data
    price_rows = conn.execute(
        "SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        "FROM prices ORDER BY ticker, date"
    ).fetchall()

    if not price_rows:
        logger.warning("confirmation backtest: no price data found")
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

    # Check QQQ availability for mega-cap RS
    qqq_available = "QQQ" in ticker_dates
    if not qqq_available:
        logger.warning(
            "confirmation backtest: QQQ not in prices table — "
            "mega-cap RS falls back to own 20d return > 0"
        )

    megacap_set = set(MEGACAP_UNIVERSE)

    def get_price_after(tkr: str, dt: str, h: int) -> float | None:
        dates = ticker_dates.get(tkr)
        if not dates:
            return None
        i = bisect.bisect_right(dates, dt)
        target = i + h - 1
        return ticker_closes[tkr][target] if target < len(dates) else None

    # Load insider buys: use filed_date <= T AND transaction_date >= T - 30d
    # This respects the 2-business-day Form 4 filing window (no lookahead).
    insider_rows = conn.execute(
        "SELECT ticker, CAST(filed_date AS VARCHAR), CAST(transaction_date AS VARCHAR) "
        "FROM insider_transactions "
        "WHERE transaction_code = 'P' AND is_open_market = TRUE "
        "ORDER BY ticker, filed_date"
    ).fetchall()

    # Build per-ticker sorted list of (filed_date_str, transaction_date_str)
    insider_by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tkr, fd, td in insider_rows:
        insider_by_ticker[tkr].append((fd, td))

    # Pre-sort insider lists by filed_date for the break optimization in _had_insider_buy
    for tkr in insider_by_ticker:
        insider_by_ticker[tkr].sort(key=lambda x: x[0])

    # Determine all tradeable dates: need at least _MA_SHORT + _RS_LOOKBACK rows
    min_rows = max(_MA_SHORT + 1, _RS_LOOKBACK + 1, _RVOL_LOOKBACK + 1)

    # Build observations: per-ticker per-date (score, rs_binary, fwd_ret at each horizon)
    # We'll collect per-date groups for cross-sectional IC
    all_dates_set: set[str] = set()

    # tagged: date -> list of (tkr, score, rs_binary, fwd_ret_dict)
    by_date: dict[str, list[tuple[str, int, bool, dict[int, float]]]] = defaultdict(list)

    for tkr in ticker_dates:
        dates = ticker_dates[tkr]
        closes = ticker_closes[tkr]
        vols = ticker_vols[tkr]
        n = len(dates)

        is_megacap = tkr in megacap_set

        for i in range(min_rows, n):
            dt = dates[i]

            # RVOL
            avg_vol = _mean(vols[i - _RVOL_LOOKBACK:i])
            rvol = vols[i] / avg_vol if (avg_vol > 0 and vols[i] is not None) else float("nan")

            # RS (positive/negative binary)
            if is_megacap and qqq_available:
                qqq_dates = ticker_dates["QQQ"]
                qqq_closes = ticker_closes["QQQ"]
                qqq_i = bisect.bisect_right(qqq_dates, dt) - 1
                qqq_prev = qqq_i - _RS_LOOKBACK
                if qqq_prev >= 0 and qqq_closes[qqq_prev] > 0 and closes[i - _RS_LOOKBACK] > 0:
                    tkr_ret = (closes[i] - closes[i - _RS_LOOKBACK]) / closes[i - _RS_LOOKBACK]
                    qqq_ret = (qqq_closes[qqq_i] - qqq_closes[qqq_prev]) / qqq_closes[qqq_prev]
                    rs_positive = (tkr_ret - qqq_ret) > 0
                else:
                    rs_positive = False
            else:
                # Dependents (or mega-cap QQQ fallback): own 20d return > 0
                prev_close = closes[i - _RS_LOOKBACK]
                rs_positive = (prev_close > 0 and closes[i] > prev_close)

            # Above 50d MA
            if i >= _MA_SHORT:
                ma50 = _mean(closes[i - _MA_SHORT:i])
                above_50ma = closes[i] > ma50
            else:
                above_50ma = False

            # Insider: use filed_date-based lookup
            insider_buying = _had_insider_buy(tkr, dt, insider_by_ticker)

            score = compute_confirmation_backtest(
                rs_positive=rs_positive,
                rvol=rvol,
                insider_buying=insider_buying,
                above_50ma=above_50ma,
            )

            # Forward returns
            base = closes[i]
            if base == 0:
                continue
            fwd: dict[int, float] = {}
            for h in _HORIZONS:
                future = get_price_after(tkr, dt, h)
                if future is not None:
                    fwd[h] = (future - base) / base

            if not fwd:
                continue

            by_date[dt].append((tkr, score, rs_positive, fwd))
            all_dates_set.add(dt)

    if not all_dates_set:
        logger.warning("confirmation backtest: no observations computed")
        return None

    all_dates = sorted(all_dates_set)
    mid_idx = len(all_dates) // 2
    mid_date = all_dates[mid_idx]
    h1_set = set(all_dates[:mid_idx])

    # Per-date IC + pooled accumulators
    per_date_ic: dict[int, list[tuple[str, float]]] = {h: [] for h in _HORIZONS}
    mc_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    dep_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    rs_alone_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    confirm_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}

    # Bucket accumulators: keys 0-1, 2, 3-4
    bucket_rets: dict[str, dict[int, list[float]]] = {
        "0-1": {h: [] for h in _HORIZONS},
        "2":   {h: [] for h in _HORIZONS},
        "3-4": {h: [] for h in _HORIZONS},
    }
    bucket_n: dict[str, int] = {"0-1": 0, "2": 0, "3-4": 0}

    n_obs_5d = 0

    for dt in all_dates:
        entries = by_date[dt]
        horizon_scores: dict[int, list[float]] = {h: [] for h in _HORIZONS}
        horizon_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}

        for tkr, score, rs_pos, fwd in entries:
            is_mc = tkr in megacap_set
            for h in _HORIZONS:
                ret = fwd.get(h)
                if ret is None:
                    continue
                horizon_scores[h].append(float(score))
                horizon_rets[h].append(ret)

                if is_mc:
                    mc_obs[h].append((float(score), ret))
                else:
                    dep_obs[h].append((float(score), ret))

                rs_alone_obs[h].append((1.0 if rs_pos else 0.0, ret))
                confirm_obs[h].append((float(score), ret))

                # Bucket
                if score <= 1:
                    bk = "0-1"
                elif score == 2:
                    bk = "2"
                else:
                    bk = "3-4"
                bucket_rets[bk][h].append(ret)

        for h in _HORIZONS:
            if len(horizon_scores[h]) >= 3:
                ic = _spearman_ic(horizon_scores[h], horizon_rets[h])
                if not math.isnan(ic):
                    per_date_ic[h].append((dt, ic))
                    if h == 5:
                        n_obs_5d += len(horizon_scores[h])

    # Count bucket n (use 5d horizon as reference)
    for dt in all_dates:
        for tkr, score, _, fwd in by_date[dt]:
            if 5 not in fwd:
                continue
            if score <= 1:
                bucket_n["0-1"] += 1
            elif score == 2:
                bucket_n["2"] += 1
            else:
                bucket_n["3-4"] += 1

    if n_obs_5d < _MIN_OBS:
        logger.warning(
            "confirmation backtest: only %d 5d-horizon observations (need %d+).",
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

    ic_megacap: dict[int, float] = {}
    ic_dependent: dict[int, float] = {}
    ic_rs_alone: dict[int, float] = {}
    ic_confirm_score: dict[int, float] = {}

    for h in _HORIZONS:
        ic_megacap[h] = _spearman_ic(
            [x[0] for x in mc_obs[h]], [x[1] for x in mc_obs[h]]
        )
        ic_dependent[h] = _spearman_ic(
            [x[0] for x in dep_obs[h]], [x[1] for x in dep_obs[h]]
        )
        ic_rs_alone[h] = _spearman_ic(
            [x[0] for x in rs_alone_obs[h]], [x[1] for x in rs_alone_obs[h]]
        )
        ic_confirm_score[h] = _spearman_ic(
            [x[0] for x in confirm_obs[h]], [x[1] for x in confirm_obs[h]]
        )

    # Build bucket stats
    buckets: list[BucketStats] = []
    for label in ("0-1", "2", "3-4"):
        avg_ret = {h: _mean(bucket_rets[label][h]) for h in _HORIZONS}
        buckets.append(BucketStats(label=label, n=bucket_n[label], avg_ret=avg_ret))

    return ConfirmationBacktestResult(
        n_dates=len(all_dates),
        n_obs=n_obs_5d,
        start_date=all_dates[0],
        end_date=all_dates[-1],
        mid_date=mid_date,
        n_h1=len(h1_set),
        n_h2=len(all_dates) - len(h1_set),
        qqq_available=qqq_available,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_h1_mean=ic_h1_mean,
        ic_h2_mean=ic_h2_mean,
        ic_megacap=ic_megacap,
        ic_dependent=ic_dependent,
        n_megacap_obs=len(mc_obs[5]),
        n_dependent_obs=len(dep_obs[5]),
        buckets=buckets,
        ic_rs_alone=ic_rs_alone,
        ic_confirm_score=ic_confirm_score,
    )


# ---------------------------------------------------------------------------
# Regime-focused backtest: dependents in CORRECTION only
# ---------------------------------------------------------------------------

_CORRECTION_LOW = -0.15   # inclusive
_CORRECTION_HIGH = -0.05  # exclusive
_VALIDATED_PARENTS_SET: frozenset[str] = frozenset({"MSFT", "META"})


@dataclass
class ConfirmationRegimeSubset:
    """IC analysis for a filtered subset (e.g. VALIDATED_PARENTS only)."""
    label: str
    n_obs: dict[int, int]
    ic_mean: dict[int, float] = field(default_factory=dict)
    ic_h1_mean: dict[int, float] = field(default_factory=dict)
    ic_h2_mean: dict[int, float] = field(default_factory=dict)
    ic_rs_alone: dict[int, float] = field(default_factory=dict)
    ic_confirm_score: dict[int, float] = field(default_factory=dict)
    buckets: list[BucketStats] = field(default_factory=list)


@dataclass
class ConfirmationRegimeBacktestResult:
    n_dates: int
    n_obs: dict[int, int]  # per horizon
    start_date: str
    end_date: str
    mid_date: str
    n_h1: int
    n_h2: int
    qqq_available: bool
    ic_mean: dict[int, float] = field(default_factory=dict)
    ic_std: dict[int, float] = field(default_factory=dict)
    ic_ir: dict[int, float] = field(default_factory=dict)
    ic_h1_mean: dict[int, float] = field(default_factory=dict)
    ic_h2_mean: dict[int, float] = field(default_factory=dict)
    ic_rs_alone: dict[int, float] = field(default_factory=dict)
    ic_confirm_score: dict[int, float] = field(default_factory=dict)
    buckets: list[BucketStats] = field(default_factory=list)
    validated: ConfirmationRegimeSubset = field(
        default_factory=lambda: ConfirmationRegimeSubset(
            label="VALIDATED (MSFT/META)", n_obs={}
        )
    )


def run_confirmation_regime_backtest() -> ConfirmationRegimeBacktestResult | None:
    conn = get_connection()
    try:
        return _run_regime(conn)
    finally:
        conn.close()


def _run_regime(conn: duckdb.DuckDBPyConnection) -> ConfirmationRegimeBacktestResult | None:
    from scanner.graph.loader import get_parents

    price_rows = conn.execute(
        "SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        "FROM prices ORDER BY ticker, date"
    ).fetchall()

    if not price_rows:
        logger.warning("confirmation regime backtest: no price data found")
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

    qqq_available = "QQQ" in ticker_dates
    if not qqq_available:
        logger.warning(
            "confirmation regime backtest: QQQ not in prices — "
            "dependent RS uses own 20d return > 0"
        )

    megacap_set = set(MEGACAP_UNIVERSE)

    # Precompute 20d return per mega-cap parent on each of their dates
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

    # For each date, which parents are in CORRECTION? (-15% <= ret < -5%)
    date_to_correction_parents: dict[str, set[str]] = defaultdict(set)
    for parent, ret_by_date in parent_20d_ret.items():
        for dt, ret in ret_by_date.items():
            if _CORRECTION_LOW <= ret < _CORRECTION_HIGH:
                date_to_correction_parents[dt].add(parent)

    # Build child -> mega-cap parents map (dependents only)
    child_to_parents: dict[str, list[str]] = {}
    for tkr in ticker_dates:
        if tkr in megacap_set or tkr in ("QQQ", "XLK"):
            continue
        parents = [e.parent for e in get_parents(tkr) if e.parent in megacap_set]
        if parents:
            child_to_parents[tkr] = parents

    # Load insider data
    insider_rows = conn.execute(
        "SELECT ticker, CAST(filed_date AS VARCHAR), CAST(transaction_date AS VARCHAR) "
        "FROM insider_transactions "
        "WHERE transaction_code = 'P' AND is_open_market = TRUE "
        "ORDER BY ticker, filed_date"
    ).fetchall()

    insider_by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tkr, fd, td in insider_rows:
        insider_by_ticker[tkr].append((fd, td))
    for tkr in insider_by_ticker:
        insider_by_ticker[tkr].sort(key=lambda x: x[0])

    def get_price_after(tkr: str, dt: str, h: int) -> float | None:
        dates = ticker_dates.get(tkr)
        if not dates:
            return None
        i = bisect.bisect_right(dates, dt)
        target = i + h - 1
        return ticker_closes[tkr][target] if target < len(dates) else None

    min_rows = max(_MA_SHORT + 1, _RS_LOOKBACK + 1, _RVOL_LOOKBACK + 1)

    # by_date: date -> list of (tkr, score, rs_positive, fwd_rets, is_validated)
    by_date: dict[str, list[tuple[str, int, bool, dict[int, float], bool]]] = defaultdict(list)
    all_dates_set: set[str] = set()

    for tkr, parents in child_to_parents.items():
        if tkr not in ticker_dates:
            continue
        dates = ticker_dates[tkr]
        closes = ticker_closes[tkr]
        vols = ticker_vols[tkr]
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

            is_validated = bool(qualifying & _VALIDATED_PARENTS_SET)

            avg_vol = _mean(vols[i - _RVOL_LOOKBACK:i])
            rvol = (
                vols[i] / avg_vol
                if avg_vol > 0 and vols[i] is not None
                else float("nan")
            )

            # Dependent RS: own 20d return > 0
            prev_close = closes[i - _RS_LOOKBACK]
            rs_positive = prev_close > 0 and closes[i] > prev_close

            if i >= _MA_SHORT:
                ma50 = _mean(closes[i - _MA_SHORT:i])
                above_50ma = closes[i] > ma50
            else:
                above_50ma = False

            insider_buying = _had_insider_buy(tkr, dt, insider_by_ticker)

            score = compute_confirmation_backtest(
                rs_positive=rs_positive,
                rvol=rvol,
                insider_buying=insider_buying,
                above_50ma=above_50ma,
            )

            base = closes[i]
            if base == 0:
                continue
            fwd: dict[int, float] = {}
            for h in _HORIZONS:
                future = get_price_after(tkr, dt, h)
                if future is not None:
                    fwd[h] = (future - base) / base

            if not fwd:
                continue

            by_date[dt].append((tkr, score, rs_positive, fwd, is_validated))
            all_dates_set.add(dt)

    if not all_dates_set:
        logger.warning("confirmation regime backtest: no CORRECTION-regime observations computed")
        return None

    all_dates = sorted(all_dates_set)
    mid_idx = len(all_dates) // 2
    mid_date = all_dates[mid_idx]
    h1_set = set(all_dates[:mid_idx])

    # Accumulators — all qualifying dependents
    per_date_ic: dict[int, list[tuple[str, float]]] = {h: [] for h in _HORIZONS}
    rs_alone_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    confirm_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    bucket_rets: dict[str, dict[int, list[float]]] = {
        "0-1": {h: [] for h in _HORIZONS},
        "2":   {h: [] for h in _HORIZONS},
        "3-4": {h: [] for h in _HORIZONS},
    }
    n_obs: dict[int, int] = {h: 0 for h in _HORIZONS}

    # Accumulators — validated subset (MSFT/META dependents in CORRECTION)
    per_date_ic_val: dict[int, list[tuple[str, float]]] = {h: [] for h in _HORIZONS}
    rs_alone_obs_val: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    confirm_obs_val: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    bucket_rets_val: dict[str, dict[int, list[float]]] = {
        "0-1": {h: [] for h in _HORIZONS},
        "2":   {h: [] for h in _HORIZONS},
        "3-4": {h: [] for h in _HORIZONS},
    }
    n_obs_val: dict[int, int] = {h: 0 for h in _HORIZONS}

    for dt in all_dates:
        entries = by_date[dt]
        h_scores: dict[int, list[float]] = {h: [] for h in _HORIZONS}
        h_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}
        h_scores_val: dict[int, list[float]] = {h: [] for h in _HORIZONS}
        h_rets_val: dict[int, list[float]] = {h: [] for h in _HORIZONS}

        for tkr, score, rs_pos, fwd, is_val in entries:
            bk = "0-1" if score <= 1 else ("2" if score == 2 else "3-4")
            for h in _HORIZONS:
                ret = fwd.get(h)
                if ret is None:
                    continue
                h_scores[h].append(float(score))
                h_rets[h].append(ret)
                n_obs[h] += 1
                rs_alone_obs[h].append((1.0 if rs_pos else 0.0, ret))
                confirm_obs[h].append((float(score), ret))
                bucket_rets[bk][h].append(ret)

                if is_val:
                    h_scores_val[h].append(float(score))
                    h_rets_val[h].append(ret)
                    n_obs_val[h] += 1
                    rs_alone_obs_val[h].append((1.0 if rs_pos else 0.0, ret))
                    confirm_obs_val[h].append((float(score), ret))
                    bucket_rets_val[bk][h].append(ret)

        for h in _HORIZONS:
            if len(h_scores[h]) >= 3:
                ic = _spearman_ic(h_scores[h], h_rets[h])
                if not math.isnan(ic):
                    per_date_ic[h].append((dt, ic))
            if len(h_scores_val[h]) >= 3:
                ic_v = _spearman_ic(h_scores_val[h], h_rets_val[h])
                if not math.isnan(ic_v):
                    per_date_ic_val[h].append((dt, ic_v))

    if n_obs[5] < _MIN_OBS:
        logger.warning(
            "confirmation regime backtest: only %d 5d-horizon observations (need %d+).",
            n_obs[5],
            _MIN_OBS,
        )
        return None

    # Aggregate — all
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

    ic_rs_alone: dict[int, float] = {}
    ic_confirm_score: dict[int, float] = {}
    for h in _HORIZONS:
        ic_rs_alone[h] = _spearman_ic(
            [x[0] for x in rs_alone_obs[h]], [x[1] for x in rs_alone_obs[h]]
        )
        ic_confirm_score[h] = _spearman_ic(
            [x[0] for x in confirm_obs[h]], [x[1] for x in confirm_obs[h]]
        )

    # Buckets — all (n from 5d horizon)
    buckets: list[BucketStats] = []
    for label in ("0-1", "2", "3-4"):
        avg_ret = {h: _mean(bucket_rets[label][h]) for h in _HORIZONS}
        buckets.append(BucketStats(
            label=label,
            n=len(bucket_rets[label][5]),
            avg_ret=avg_ret,
        ))

    # Aggregate — validated
    ic_mean_val: dict[int, float] = {}
    ic_h1_mean_val: dict[int, float] = {}
    ic_h2_mean_val: dict[int, float] = {}
    for h in _HORIZONS:
        all_ics_v = [ic for _, ic in per_date_ic_val[h]]
        h1_ics_v = [ic for dt, ic in per_date_ic_val[h] if dt in h1_set]
        h2_ics_v = [ic for dt, ic in per_date_ic_val[h] if dt not in h1_set]
        ic_mean_val[h] = _mean(all_ics_v)
        ic_h1_mean_val[h] = _mean(h1_ics_v)
        ic_h2_mean_val[h] = _mean(h2_ics_v)

    ic_rs_alone_val: dict[int, float] = {}
    ic_confirm_score_val: dict[int, float] = {}
    for h in _HORIZONS:
        ic_rs_alone_val[h] = _spearman_ic(
            [x[0] for x in rs_alone_obs_val[h]], [x[1] for x in rs_alone_obs_val[h]]
        )
        ic_confirm_score_val[h] = _spearman_ic(
            [x[0] for x in confirm_obs_val[h]], [x[1] for x in confirm_obs_val[h]]
        )

    buckets_val: list[BucketStats] = []
    for label in ("0-1", "2", "3-4"):
        avg_ret = {h: _mean(bucket_rets_val[label][h]) for h in _HORIZONS}
        buckets_val.append(BucketStats(
            label=label,
            n=len(bucket_rets_val[label][5]),
            avg_ret=avg_ret,
        ))

    validated = ConfirmationRegimeSubset(
        label="VALIDATED (MSFT/META dependents)",
        n_obs=n_obs_val,
        ic_mean=ic_mean_val,
        ic_h1_mean=ic_h1_mean_val,
        ic_h2_mean=ic_h2_mean_val,
        ic_rs_alone=ic_rs_alone_val,
        ic_confirm_score=ic_confirm_score_val,
        buckets=buckets_val,
    )

    return ConfirmationRegimeBacktestResult(
        n_dates=len(all_dates),
        n_obs=n_obs,
        start_date=all_dates[0],
        end_date=all_dates[-1],
        mid_date=mid_date,
        n_h1=len(h1_set),
        n_h2=len(all_dates) - len(h1_set),
        qqq_available=qqq_available,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_h1_mean=ic_h1_mean,
        ic_h2_mean=ic_h2_mean,
        ic_rs_alone=ic_rs_alone,
        ic_confirm_score=ic_confirm_score,
        buckets=buckets,
        validated=validated,
    )
