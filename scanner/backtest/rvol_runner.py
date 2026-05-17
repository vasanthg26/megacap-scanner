"""Walk-forward backtest for the Relative Volume (RVOL) signal.

Hypotheses tested:
  1. Signed RVOL (rvol * sign(day_return)) predicts forward returns.
  2. RVOL magnitude alone vs direction alone -- does the combination add value?
  3. Mega-cap tickers vs dependents -- does predictive power differ?
  4. High-RVOL up-day vs high-RVOL down-day segment analysis.

Uses existing `prices` table volume column -- no additional ingest required.
"""

import bisect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import duckdb

from scanner.db import get_connection
from scanner.signals.megacap import MEGACAP_UNIVERSE

logger = logging.getLogger(__name__)

_MIN_OBS = 100
_HORIZONS = (5, 10, 20)
_RVOL_LOOKBACK = 20
_RVOL_HIGH_THRESHOLD = 2.0


@dataclass
class RvolBacktestResult:
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
    ic_unsigned: dict[int, float] = field(default_factory=dict)
    ic_direction: dict[int, float] = field(default_factory=dict)
    ic_megacap: dict[int, float] = field(default_factory=dict)
    ic_dependent: dict[int, float] = field(default_factory=dict)
    n_megacap_obs: int = 0
    n_dependent_obs: int = 0
    up_high_rvol_avg: dict[int, float] = field(default_factory=dict)
    down_high_rvol_avg: dict[int, float] = field(default_factory=dict)
    n_up_high: int = 0
    n_down_high: int = 0


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


def run_rvol_backtest() -> RvolBacktestResult | None:
    conn = get_connection()
    try:
        return _run(conn)
    finally:
        conn.close()


def _run(conn: duckdb.DuckDBPyConnection) -> RvolBacktestResult | None:
    price_rows = conn.execute(
        "SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        "FROM prices ORDER BY ticker, date"
    ).fetchall()

    if not price_rows:
        logger.warning("rvol backtest: no price data found")
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

    megacap_set = set(MEGACAP_UNIVERSE)

    def get_price_after(tkr: str, dt: str, h: int) -> float | None:
        dates = ticker_dates.get(tkr)
        if not dates:
            return None
        i = bisect.bisect_right(dates, dt)
        target = i + h - 1
        return ticker_closes[tkr][target] if target < len(dates) else None

    # Build RVOL observations per ticker per date
    tagged_obs: list[tuple[str, str, float, float, float]] = []
    for tkr in ticker_dates:
        dates = ticker_dates[tkr]
        closes = ticker_closes[tkr]
        vols = ticker_vols[tkr]
        n = len(dates)
        for i in range(_RVOL_LOOKBACK + 1, n):
            avg_vol = _mean(vols[i - _RVOL_LOOKBACK:i])
            if avg_vol == 0 or closes[i - 1] == 0:
                continue
            rvol = vols[i] / avg_vol
            day_ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            direction = 1.0 if day_ret > 0 else (-1.0 if day_ret < 0 else 0.0)
            tagged_obs.append((dates[i], tkr, rvol * direction, rvol, direction))

    if not tagged_obs:
        logger.warning("rvol backtest: no RVOL observations computed")
        return None

    all_dates = sorted({obs[0] for obs in tagged_obs})
    mid_idx = len(all_dates) // 2
    mid_date = all_dates[mid_idx]
    h1_set = set(all_dates[:mid_idx])

    by_date: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    for dt, tkr, signed_rvol, rvol_u, direction in tagged_obs:
        by_date[dt].append((tkr, signed_rvol, rvol_u, direction))

    per_date_ic: dict[int, list[tuple[str, float]]] = {h: [] for h in _HORIZONS}
    mc_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    dep_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    unsigned_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    direction_obs: dict[int, list[tuple[float, float]]] = {h: [] for h in _HORIZONS}
    up_high_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    down_high_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    n_obs_5d = 0

    for dt in all_dates:
        horizon_signed: dict[int, list[float]] = {h: [] for h in _HORIZONS}
        horizon_rets: dict[int, list[float]] = {h: [] for h in _HORIZONS}

        for tkr, signed_rvol, rvol_u, direction in by_date[dt]:
            idx = bisect.bisect_right(ticker_dates[tkr], dt) - 1
            if idx < 0:
                continue
            base = ticker_closes[tkr][idx]
            if base == 0:
                continue

            for h in _HORIZONS:
                future = get_price_after(tkr, dt, h)
                if future is None:
                    continue
                fwd_ret = (future - base) / base

                horizon_signed[h].append(signed_rvol)
                horizon_rets[h].append(fwd_ret)
                unsigned_obs[h].append((rvol_u, fwd_ret))
                direction_obs[h].append((direction, fwd_ret))

                if tkr in megacap_set:
                    mc_obs[h].append((signed_rvol, fwd_ret))
                else:
                    dep_obs[h].append((signed_rvol, fwd_ret))

                if rvol_u >= _RVOL_HIGH_THRESHOLD:
                    if direction > 0:
                        up_high_rets[h].append(fwd_ret)
                    elif direction < 0:
                        down_high_rets[h].append(fwd_ret)

        for h in _HORIZONS:
            if len(horizon_signed[h]) >= 3:
                ic = _spearman_ic(horizon_signed[h], horizon_rets[h])
                if not math.isnan(ic):
                    per_date_ic[h].append((dt, ic))
                    if h == 5:
                        n_obs_5d += len(horizon_signed[h])

    if n_obs_5d < _MIN_OBS:
        logger.warning(
            "rvol backtest: only %d 5d-horizon observations (need %d+).",
            n_obs_5d,
            _MIN_OBS,
        )
        return None

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

    ic_unsigned: dict[int, float] = {}
    ic_direction: dict[int, float] = {}
    ic_megacap: dict[int, float] = {}
    ic_dependent: dict[int, float] = {}

    for h in _HORIZONS:
        ic_unsigned[h] = _spearman_ic(
            [x[0] for x in unsigned_obs[h]], [x[1] for x in unsigned_obs[h]]
        )
        ic_direction[h] = _spearman_ic(
            [x[0] for x in direction_obs[h]], [x[1] for x in direction_obs[h]]
        )
        ic_megacap[h] = _spearman_ic(
            [x[0] for x in mc_obs[h]], [x[1] for x in mc_obs[h]]
        )
        ic_dependent[h] = _spearman_ic(
            [x[0] for x in dep_obs[h]], [x[1] for x in dep_obs[h]]
        )

    return RvolBacktestResult(
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
        ic_unsigned=ic_unsigned,
        ic_direction=ic_direction,
        ic_megacap=ic_megacap,
        ic_dependent=ic_dependent,
        n_megacap_obs=len(mc_obs[5]),
        n_dependent_obs=len(dep_obs[5]),
        up_high_rvol_avg={h: _mean(up_high_rets[h]) for h in _HORIZONS},
        down_high_rvol_avg={h: _mean(down_high_rets[h]) for h in _HORIZONS},
        n_up_high=len(up_high_rets[5]),
        n_down_high=len(down_high_rets[5]),
    )
