"""Theme basket backtest — RS vs benchmark ETF signal IC.

Methodology mirrors runner.py but operates per-theme-ticker rather than
per-parent-dependent. Signal is the 20-day return differential between the
theme ticker and its benchmark ETF. Forward returns sampled at 5d, 10d, 20d.

Survivorship-bias caveat: yfinance only returns currently-listed tickers.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from scanner.db import get_connection

logger = logging.getLogger(__name__)

HORIZONS = (5, 10, 20)
LOOKBACK = 20
MIN_OBS = 30


@dataclass
class ThemeTickerResult:
    ticker: str
    benchmark_etf: str
    theme_name: str
    theme_label: str
    n_obs: int
    ic: dict[int, float] = field(default_factory=dict)
    ic_h1: dict[int, float] = field(default_factory=dict)
    ic_h2: dict[int, float] = field(default_factory=dict)
    passes: bool = False   # IC > 0.05 at 10d or 20d, both halves positive, n >= MIN_OBS


@dataclass
class ThemeBacktestResult:
    ticker_results: list[ThemeTickerResult] = field(default_factory=list)
    combined_ic: dict[int, float] = field(default_factory=dict)
    n_valid: int = 0
    n_total: int = 0


def _get_trading_dates(conn, ticker: str, benchmark: str) -> list[str]:
    """Dates where both ticker AND benchmark have a closing price."""
    rows = conn.execute(
        """
        SELECT DISTINCT p.date
        FROM prices p
        JOIN prices b ON b.date = p.date AND b.ticker = ?
        WHERE p.ticker = ?
        ORDER BY p.date
        """,
        [benchmark, ticker],
    ).fetchall()
    return [str(r[0]) for r in rows]


def _rolling_return(ticker: str, as_of: str, conn) -> float:
    rows = conn.execute(
        """
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
        """,
        [ticker, as_of, LOOKBACK + 1],
    ).fetchall()
    if len(rows) < LOOKBACK + 1:
        return float("nan")
    oldest = rows[-1][0]
    latest = rows[0][0]
    if oldest is None or latest is None or float(oldest) == 0:
        return float("nan")
    return (float(latest) - float(oldest)) / float(oldest)


def _rs_score(ticker: str, benchmark: str, as_of: str, conn) -> float:
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
        ORDER BY date ASC
        LIMIT ?
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


def _collect_obs(ticker: str, benchmark: str, conn) -> list[dict]:
    """Build signal→forward-return observations for one ticker."""
    dates = _get_trading_dates(conn, ticker, benchmark)
    # Need LOOKBACK days before first signal date; drop last LOOKBACK to ensure 20d fwd room.
    signal_dates = dates[LOOKBACK: max(LOOKBACK, len(dates) - LOOKBACK)]

    obs = []
    for dt in signal_dates:
        score = _rs_score(ticker, benchmark, dt, conn)
        if math.isnan(score):
            continue
        fwds = {h: _forward_return(ticker, dt, h, conn) for h in HORIZONS}
        if any(v is not None for v in fwds.values()):
            obs.append({"date": dt, "score": score, "fwds": fwds})
    return obs


def _eval_ticker(
    ticker: str,
    benchmark: str,
    theme_name: str,
    theme_label: str,
    obs: list[dict],
) -> ThemeTickerResult:
    result = ThemeTickerResult(
        ticker=ticker,
        benchmark_etf=benchmark,
        theme_name=theme_name,
        theme_label=theme_label,
        n_obs=len(obs),
    )

    if len(obs) < MIN_OBS:
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

    # PASS: IC > 0.05 at 10d or 20d with both halves positive.
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


def run_theme_backtest() -> ThemeBacktestResult:
    """Run RS vs benchmark IC backtest for all active theme tickers."""
    conn = get_connection()
    try:
        themes = conn.execute(
            """
            SELECT theme_name, ticker, benchmark_etf, theme_label
            FROM themes WHERE is_active = true
            ORDER BY theme_name, ticker
            """
        ).fetchall()

        if not themes:
            logger.warning("No active themes in DB")
            return ThemeBacktestResult()

        ticker_results: list[ThemeTickerResult] = []
        combined_pairs: dict[int, list[tuple[float, float]]] = {h: [] for h in HORIZONS}

        for theme_name, ticker, benchmark, theme_label in themes:
            logger.info("Backtesting %s (%s vs %s)", theme_name, ticker, benchmark)
            obs = _collect_obs(ticker, benchmark, conn)
            result = _eval_ticker(ticker, benchmark, theme_name, theme_label, obs)
            ticker_results.append(result)

            for h in HORIZONS:
                for o in obs:
                    fwd = o["fwds"].get(h)
                    if fwd is not None:
                        combined_pairs[h].append((o["score"], fwd))

    finally:
        conn.close()

    combined_ic = {h: _ic(combined_pairs[h]) for h in HORIZONS}
    n_valid = sum(1 for r in ticker_results if r.passes)

    return ThemeBacktestResult(
        ticker_results=ticker_results,
        combined_ic=combined_ic,
        n_valid=n_valid,
        n_total=len(ticker_results),
    )
