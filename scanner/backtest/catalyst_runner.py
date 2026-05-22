"""Alternative signal backtest for a single ticker — candidate signals.

Signals:
  1. rvol           — today's volume / 20-day average volume
  2. rs_vs_xlk      — 20-day return of ticker minus 20-day return of XLK
  3. rs_vs_benchmark — 20-day return of ticker minus 20-day return of per-ticker benchmark ETF

Supports optional start_date to restrict observations to a post-pivot window.

Forward returns sampled at 5d, 10d, 20d.
PASS: IC > 0.05 at 10d or 20d, both halves positive, n >= min_obs.
"""

import logging
import math
from dataclasses import dataclass, field

from scipy.stats import spearmanr

from scanner.db import get_connection

logger = logging.getLogger(__name__)

HORIZONS = (5, 10, 20)
LOOKBACK = 20
DEFAULT_MIN_OBS = 30

SIGNALS = ("rvol", "rs_vs_xlk", "rs_vs_benchmark")


@dataclass
class SignalResult:
    name: str
    n_obs: int
    ic: dict[int, float] = field(default_factory=dict)
    ic_h1: dict[int, float] = field(default_factory=dict)
    ic_h2: dict[int, float] = field(default_factory=dict)
    passes: bool = False


@dataclass
class CatalystBacktestResult:
    ticker: str
    benchmark: str
    start_date: str | None
    signal_results: list[SignalResult] = field(default_factory=list)


def _trading_dates(ticker: str, conn, start_date: str | None = None) -> list[str]:
    if start_date:
        rows = conn.execute(
            "SELECT DISTINCT date FROM prices WHERE ticker = ? AND date >= ? ORDER BY date",
            [ticker, start_date],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT date FROM prices WHERE ticker = ? ORDER BY date",
            [ticker],
        ).fetchall()
    return [str(r[0]) for r in rows]


def _rolling_return(ticker: str, as_of: str, conn) -> float:
    rows = conn.execute(
        """
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC LIMIT ?
        """,
        [ticker, as_of, LOOKBACK + 1],
    ).fetchall()
    if len(rows) < LOOKBACK + 1:
        return float("nan")
    oldest, latest = rows[-1][0], rows[0][0]
    if oldest is None or latest is None or float(oldest) == 0:
        return float("nan")
    return (float(latest) - float(oldest)) / float(oldest)


def _signal_rvol(ticker: str, as_of: str, conn) -> float:
    """Today's volume / 20-day average volume."""
    rows = conn.execute(
        """
        SELECT volume FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC LIMIT 21
        """,
        [ticker, as_of],
    ).fetchall()
    if len(rows) < 21:
        return float("nan")
    vols = [float(r[0]) for r in rows if r[0] is not None]
    if len(vols) < 21:
        return float("nan")
    avg_20 = sum(vols[1:21]) / 20
    if avg_20 == 0:
        return float("nan")
    return vols[0] / avg_20


def _signal_rs_vs(ticker: str, benchmark: str, as_of: str, conn) -> float:
    """20-day return of ticker minus 20-day return of benchmark."""
    t_ret = _rolling_return(ticker, as_of, conn)
    b_ret = _rolling_return(benchmark, as_of, conn)
    if math.isnan(t_ret) or math.isnan(b_ret):
        return float("nan")
    return t_ret - b_ret


def _forward_return(ticker: str, from_date: str, horizon: int, conn) -> float | None:
    rows = conn.execute(
        """
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date > ?
        ORDER BY date ASC LIMIT ?
        """,
        [ticker, from_date, horizon],
    ).fetchall()
    if len(rows) < horizon:
        return None
    entry, exit_ = rows[0][0], rows[-1][0]
    if entry is None or float(entry) == 0:
        return None
    return (float(exit_) - float(entry)) / float(entry)


def _ic(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 5:
        return float("nan")
    scores, fwds = zip(*pairs)
    val, _ = spearmanr(scores, fwds)
    return float(val) if not math.isnan(val) else float("nan")


def _eval_signal(signal_name: str, obs: list[dict], min_obs: int) -> SignalResult:
    result = SignalResult(name=signal_name, n_obs=len(obs))
    if len(obs) < min_obs:
        return result

    mid = len(obs) // 2
    h1_obs, h2_obs = obs[:mid], obs[mid:]

    for h in HORIZONS:
        full = [(o["score"], o["fwds"][h]) for o in obs if o["fwds"][h] is not None]
        h1 = [(o["score"], o["fwds"][h]) for o in h1_obs if o["fwds"][h] is not None]
        h2 = [(o["score"], o["fwds"][h]) for o in h2_obs if o["fwds"][h] is not None]
        result.ic[h] = _ic(full)
        result.ic_h1[h] = _ic(h1)
        result.ic_h2[h] = _ic(h2)

    for h in (10, 20):
        ic_full = result.ic.get(h, float("nan"))
        ic_h1 = result.ic_h1.get(h, float("nan"))
        ic_h2 = result.ic_h2.get(h, float("nan"))
        if (
            not math.isnan(ic_full) and ic_full > 0.05
            and not math.isnan(ic_h1) and ic_h1 > 0
            and not math.isnan(ic_h2) and ic_h2 > 0
        ):
            result.passes = True
            break

    return result


def run_catalyst_backtest(
    ticker: str,
    benchmark: str = "XLK",
    start_date: str | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
) -> CatalystBacktestResult:
    """Run all candidate signals for `ticker`, optionally filtered to start_date onwards.

    `benchmark` is used for Signal 3 (rs_vs_benchmark) and is ticker-specific
    (e.g. BB -> IGV, AEHR -> SMH, BE -> GRID, NOK -> XLK, HUT -> WGMI).
    """
    conn = get_connection()
    try:
        # Price dates within the post-pivot window (signals still look back LOOKBACK days
        # before the window start to compute rolling returns — no lookahead violation).
        dates = _trading_dates(ticker, conn, start_date=start_date)
        # Drop the trailing LOOKBACK dates so every signal date has room for a 20d fwd return.
        signal_dates = dates[: max(0, len(dates) - LOOKBACK)]

        signal_fns = {
            "rvol": lambda t, dt, c: _signal_rvol(t, dt, c),
            "rs_vs_xlk": lambda t, dt, c: _signal_rs_vs(t, "XLK", dt, c),
            "rs_vs_benchmark": lambda t, dt, c: _signal_rs_vs(t, benchmark, dt, c),
        }

        all_obs: dict[str, list[dict]] = {s: [] for s in SIGNALS}

        for dt in signal_dates:
            fwds = {h: _forward_return(ticker, dt, h, conn) for h in HORIZONS}
            if not any(v is not None for v in fwds.values()):
                continue
            for sig_name, sig_fn in signal_fns.items():
                score = sig_fn(ticker, dt, conn)
                if not math.isnan(score):
                    all_obs[sig_name].append({"date": dt, "score": score, "fwds": fwds})

        signal_results = [
            _eval_signal(sig_name, all_obs[sig_name], min_obs) for sig_name in SIGNALS
        ]

    finally:
        conn.close()

    return CatalystBacktestResult(
        ticker=ticker,
        benchmark=benchmark,
        start_date=start_date,
        signal_results=signal_results,
    )
